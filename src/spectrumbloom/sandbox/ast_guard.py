"""Static AST security validator and code normalization for sandboxed execution."""

from __future__ import annotations

import ast
import re

from spectrumbloom.domain.exceptions import SandboxSecurityError

MAX_AST_NODES = 1500
MAX_CODE_CHARS = 24000

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "collections",
        "datetime",
        "functools",
        "itertools",
        "json",
        "math",
        "numpy",
        "pandas",
        "plotly",
        "re",
        "scipy",
        "sklearn",
        "statistics",
        "statsmodels",
    }
)

BLOCKED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "hasattr",
        "help",
        "input",
        "locals",
        "memoryview",
        "object",
        "open",
        "quit",
        "setattr",
        "super",
        "type",
        "vars",
    }
)

BLOCKED_ATTRIBUTES = frozenset(
    {
        "builtins",
        "chdir",
        "chmod",
        "chown",
        "cr_frame",
        "ctypes",
        "dump",
        "dumps",
        "environ",
        "eval",
        "execl",
        "execv",
        "execve",
        "f_back",
        "f_builtins",
        "f_globals",
        "f_locals",
        "fetch_california_housing",
        "fetch_covtype",
        "fetch_kddcup99",
        "fetch_lfw_pairs",
        "fetch_lfw_people",
        "fetch_olivetti_faces",
        "fetch_openml",
        "fetch_rcv1",
        "fetch_species_distributions",
        "fork",
        "forkpty",
        "func_globals",
        "getenv",
        "gi_frame",
        "glob",
        "im_func",
        "importlib",
        "joblib",
        "listdir",
        "load",
        "loads",
        "makedirs",
        "mkdir",
        "mro",
        "open",
        "os",
        "pathlib",
        "pickle",
        "popen",
        "Popen",
        "putenv",
        "query",
        "read_bytes",
        "read_clipboard",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_fwf",
        "read_gbq",
        "read_hdf",
        "read_html",
        "read_json",
        "read_orc",
        "read_parquet",
        "read_pickle",
        "read_sas",
        "read_spss",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_stata",
        "read_table",
        "read_text",
        "read_xml",
        "remove",
        "rename",
        "replace",
        "Request",
        "rglob",
        "rmdir",
        "save",
        "scandir",
        "shutil",
        "socket",
        "spawn",
        "spawnl",
        "spawnlp",
        "spawnv",
        "subprocess",
        "sys",
        "system",
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_html",
        "to_json",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "to_xml",
        "unlink",
        "unsetenv",
        "urlopen",
        "walk",
        "write_bytes",
        "write_html",
        "write_image",
        "write_json",
        "write_text",
    }
)

# Allow safe DataFrame/Series `.replace(...)` while still blocking os.replace/pathlib.replace?
# Wait: `df.replace` is heavily used for data cleaning; `os`/`pathlib` are already unreachable
# via import and attribute blocks (`os`, `pathlib`). So remove `replace` from BLOCKED_ATTRIBUTES
# and instead rely on import + `os`/`pathlib`/`shutil` blocks + runtime fs lockdown.
SAFE_DATA_METHOD_EXCEPTIONS = frozenset({"replace"})
EFFECTIVE_BLOCKED_ATTRIBUTES = BLOCKED_ATTRIBUTES - SAFE_DATA_METHOD_EXCEPTIONS


def is_allowed_import(module_name: str) -> bool:
    """Check if a module path belongs to the vetted scientific allowlist."""
    if not module_name:
        return False
    root = module_name.split(".", 1)[0]
    return root in ALLOWED_IMPORT_ROOTS


def normalize_generated_code(raw_code: str) -> str:
    """Strip markdown code fences safely without corrupting string literals."""
    if not isinstance(raw_code, str):
        return ""

    text = raw_code.strip()
    if not text:
        return ""

    fence_match = re.match(r"^```(?:python|py)?\s*\n([\s\S]*?)\n```$", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    elif re.match(r"^```(?:python|py)?\s*\r?\n", text, re.IGNORECASE):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        ast.parse(text, mode="exec")
        return text
    except SyntaxError:
        if "\\n" in text and "\n" not in text:
            candidate = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
            return candidate.strip()
        return text


class SandboxASTValidator(ast.NodeVisitor):
    """Strict AST validator rejecting unsafe imports, names, attributes, and oversized trees."""

    def __init__(self) -> None:
        self._node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self._node_count += 1
        if self._node_count > MAX_AST_NODES:
            raise SandboxSecurityError(
                f"Code exceeds maximum AST complexity ({MAX_AST_NODES} nodes)."
            )
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self._node_count += 1
        for alias in node.names:
            if not is_allowed_import(alias.name):
                raise SandboxSecurityError(f"Import '{alias.name}' is not permitted in sandbox.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._node_count += 1
        if node.level != 0 or not is_allowed_import(node.module or ""):
            mod = node.module or "<relative>"
            raise SandboxSecurityError(f"Import from '{mod}' is not permitted in sandbox.")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in BLOCKED_NAMES:
            raise SandboxSecurityError(f"Identifier '{node.id}' is not permitted in sandbox.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr in EFFECTIVE_BLOCKED_ATTRIBUTES:
            raise SandboxSecurityError(f"Attribute '.{node.attr}' is not permitted in sandbox.")
        self.generic_visit(node)


def validate_code_safety(raw_code: str) -> tuple[str, ast.Module]:
    """Normalize, parse, and validate generated Python code before execution."""
    code = normalize_generated_code(raw_code)
    if not code:
        raise SandboxSecurityError("No Python code provided for sandbox execution.")
    if len(code) > MAX_CODE_CHARS:
        raise SandboxSecurityError(f"Code exceeds maximum length ({MAX_CODE_CHARS} characters).")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxSecurityError(f"Syntax error in generated Python code: {exc}") from exc

    SandboxASTValidator().visit(tree)
    return code, tree
