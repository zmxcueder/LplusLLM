from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # "openai" | "deepseek" | "openrouter"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MAX_OUTPUT_TOKENS: int | None = None
LLM_TEMPERATURE: float | None = None


@dataclass(slots=True)
class LLMClientConfig:
    """OpenAI-compatible client settings shared by Querier and Modeler."""

    provider: str = LLM_PROVIDER
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = LLM_MAX_OUTPUT_TOKENS
    temperature: float | None = LLM_TEMPERATURE
    use_json_schema_for_openai: bool = True

    def __post_init__(self) -> None:
        provider = str(self.provider or LLM_PROVIDER).lower().strip()
        if provider not in {"openai", "deepseek", "openrouter"}:
            raise ValueError("provider must be 'openai', 'deepseek', or 'openrouter'.")
        self.provider = provider

        if self.provider == "openai":
            self.api_key_env = self.api_key_env or "OPENAI_API_KEY"
            self.model = self.model or OPENAI_MODEL
        elif self.provider == "deepseek":
            self.api_key_env = self.api_key_env or "DEEPSEEK_API_KEY"
            self.base_url = self.base_url or DEEPSEEK_BASE_URL
            self.model = self.model or DEEPSEEK_MODEL
        else:
            self.api_key_env = self.api_key_env or "OPENROUTER_API_KEY"
            self.base_url = self.base_url or OPENROUTER_BASE_URL
            self.model = self.model or OPENROUTER_MODEL
            self.use_json_schema_for_openai = False

        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided.")

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env or "")
        if not key:
            raise RuntimeError(
                f"Missing API key for provider={self.provider!r}. "
                f"Pass api_key explicitly or set environment variable {self.api_key_env}."
            )
        return key.strip()


def configure_local_llm_api_key(base_dir: str | Path) -> None:
    """Load a local provider key file into the matching environment variable."""
    config = LLMClientConfig()
    if not config.api_key_env or os.getenv(config.api_key_env):
        return

    key_files = {
        "openai": "OpenAIAPIKey.txt",
        "deepseek": "DeepSeekAPIKey.txt",
        "openrouter": "OpenRouterAPIKey.txt",
    }
    key_file = key_files.get(config.provider)
    if not key_file:
        return

    key_path = Path(base_dir) / key_file
    if not key_path.exists():
        return

    key = key_path.read_text(encoding="utf-8").strip()
    if key:
        os.environ[config.api_key_env] = key


@dataclass(slots=True)
class LLMRunConfig:
    """L+LLM integration settings used by the stochastic learner."""

    sampling_budget_per_round: int | None = 20
    max_query_length: int = 16
    max_selected_queries: int | None = None
    samples_per_sequence: int = 1
    snapshot_max_depth: int | None = None
    snapshot_max_nodes: int | None = None
    artifact_dir: str | Path | None = None
    include_full_nodes: bool = False
    modeler_max_retries: int = 1
    modeler_artifact_dir: str | Path | None = None
    modeler_activation_sample_budget: int | None = None
    modeler_budget_unit: str = "episodes"  # episodes | steps

    def __post_init__(self) -> None:
        if self.sampling_budget_per_round is not None and self.sampling_budget_per_round <= 0:
            raise ValueError("sampling_budget_per_round must be positive when provided.")
        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be positive.")
        if self.max_selected_queries is not None and self.max_selected_queries <= 0:
            raise ValueError("max_selected_queries must be positive when provided.")
        if self.samples_per_sequence <= 0:
            raise ValueError("samples_per_sequence must be positive.")
        if self.snapshot_max_depth is not None and self.snapshot_max_depth <= 0:
            raise ValueError("snapshot_max_depth must be positive when provided.")
        if self.snapshot_max_nodes is not None and self.snapshot_max_nodes <= 0:
            raise ValueError("snapshot_max_nodes must be positive when provided.")
        if self.modeler_max_retries < 0:
            raise ValueError("modeler_max_retries must be non-negative.")
        if self.modeler_activation_sample_budget is not None and self.modeler_activation_sample_budget < 0:
            raise ValueError("modeler_activation_sample_budget must be non-negative when provided.")
        if self.modeler_budget_unit not in {"episodes", "steps"}:
            raise ValueError("modeler_budget_unit must be 'episodes' or 'steps'.")
