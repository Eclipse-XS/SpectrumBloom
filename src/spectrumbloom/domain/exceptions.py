"""Typed domain exceptions for SpectrumBloom."""

from __future__ import annotations


class SpectrumBloomError(Exception):
    """Base application exception."""


class SessionNotFoundError(SpectrumBloomError):
    """Raised when a session ID does not exist or has expired."""


class DatasetNotFoundError(SpectrumBloomError):
    """Raised when an expected dataset artifact is missing."""


class DatasetValidationError(SpectrumBloomError):
    """Raised when an uploaded dataset fails size, type, or safety validation."""


class SandboxError(SpectrumBloomError):
    """Base error for sandboxed Python execution."""


class SandboxSecurityError(SandboxError):
    """Raised when generated code fails static AST security checks."""


class SandboxUnavailableError(SandboxError):
    """Raised when the configured sandbox backend cannot run (fail-closed)."""


class SandboxTimeoutError(SandboxError):
    """Raised when sandbox execution exceeds the time limit."""


class SandboxExecutionError(SandboxError):
    """Raised when sandbox execution crashes or exhausts memory."""


class LLMConfigurationError(SpectrumBloomError):
    """Raised when an LLM provider is misconfigured or missing required credentials."""


class LLMInvocationError(SpectrumBloomError):
    """Raised when calling the configured LLM backend fails."""


class MLExperimentError(SpectrumBloomError):
    """Raised when an ML experiment request cannot be executed on the dataset."""
