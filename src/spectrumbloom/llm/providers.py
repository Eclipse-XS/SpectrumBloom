"""Local-first LLM provider factory (Ollama, llama.cpp, optional external providers)."""

from __future__ import annotations

import httpx
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import LLMConfigurationError


class UnifiedLLMProvider:
    """Builds LangChain chat models with local inference as the primary path."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_chat_model(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
    ):
        chosen_provider = (provider or self.settings.llm_provider).lower().strip()
        chosen_model = (model or self.settings.llm_model).strip()
        temp = self.settings.llm_temperature if temperature is None else temperature
        ctx = self.settings.llm_num_ctx if num_ctx is None else num_ctx
        tok_limit = self.settings.llm_max_tokens if max_tokens is None else max_tokens

        if chosen_provider == "ollama":
            return ChatOllama(
                model=chosen_model,
                base_url=self.settings.ollama_base_url,
                temperature=temp,
                num_ctx=ctx,
                num_predict=tok_limit,
                reasoning=self.settings.llm_enable_thinking,
            )

        if chosen_provider == "llamacpp":
            return ChatOpenAI(
                model=chosen_model,
                base_url=self.settings.llamacpp_base_url,
                api_key=api_key or "llamacpp-local",
                temperature=temp,
                max_tokens=tok_limit,
                timeout=self.settings.llm_request_timeout_seconds,
            )

        if chosen_provider == "openai":
            resolved_key = (api_key or self.settings.openai_api_key or "").strip()
            if not resolved_key:
                raise LLMConfigurationError(
                    "OpenAI-compatible provider selected, but no API key was supplied."
                )
            kwargs = {
                "model": chosen_model,
                "api_key": resolved_key,
                "temperature": temp,
                "max_tokens": tok_limit,
                "timeout": self.settings.llm_request_timeout_seconds,
            }
            if self.settings.openai_base_url:
                kwargs["base_url"] = self.settings.openai_base_url
            return ChatOpenAI(**kwargs)

        if chosen_provider == "gemini":
            resolved_key = (api_key or self.settings.gemini_api_key or "").strip()
            if not resolved_key:
                raise LLMConfigurationError(
                    "Gemini provider selected, but no GEMINI_API_KEY was supplied."
                )
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise LLMConfigurationError(
                    "Gemini requested but 'langchain-google-genai' is not installed. "
                    "Install with `uv pip install -e '.[gemini]'`."
                ) from exc
            return ChatGoogleGenerativeAI(
                model=chosen_model,
                google_api_key=resolved_key,
                temperature=temp,
                max_output_tokens=tok_limit,
            )

        raise LLMConfigurationError(f"Unsupported LLM provider: '{chosen_provider}'.")

    def check_local_backend_health(self) -> dict[str, object]:
        """Probe configured local inference backend without pulling or downloading models."""
        provider = self.settings.llm_provider.lower()
        if provider == "ollama":
            try:
                resp = httpx.get(
                    f"{self.settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0
                )
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", [])]
                    return {
                        "provider": "ollama",
                        "status": "connected",
                        "configured_model": self.settings.llm_model,
                        "available_models": models,
                    }
                return {"provider": "ollama", "status": f"http_{resp.status_code}"}
            except Exception as exc:
                return {"provider": "ollama", "status": "unavailable", "detail": str(exc)}

        if provider == "llamacpp":
            try:
                base = self.settings.llamacpp_base_url.rstrip("/")
                resp = httpx.get(f"{base}/models", timeout=2.0)
                return {
                    "provider": "llamacpp",
                    "status": "connected"
                    if resp.status_code == 200
                    else f"http_{resp.status_code}",
                }
            except Exception as exc:
                return {"provider": "llamacpp", "status": "unavailable", "detail": str(exc)}

        return {"provider": provider, "status": "external_configured"}
