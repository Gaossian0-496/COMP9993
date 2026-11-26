"""
Config-driven LLM client factory.

Switch backends by changing config_local.yaml — no code changes required.

Supported backends:
  sglang  — local SGLang server (OpenAI-compatible endpoint)
  openai  — OpenAI API (GPT-4o, etc.)
  azure   — Azure AI inference (original paper's setup)
"""


def get_client(config: dict):
    """Return an OpenAI-compatible client for the backend specified in config."""
    llm_cfg = config.get("llm", {})
    backend = llm_cfg.get("backend", "sglang")

    if backend == "sglang":
        from openai import OpenAI
        # Explicit timeout/retries: the SDK default (600s x 3 tries) silently
        # discarded every generation longer than 10 min and cost 30 min per
        # timeout. Retries are handled by optimizer._chat_create instead.
        return OpenAI(
            base_url=llm_cfg.get("server_url", "http://127.0.0.1:8000/v1"),
            api_key="EMPTY",  # SGLang does not require auth
            timeout=float(llm_cfg.get("request_timeout_s", 3600)),
            max_retries=0,
        )

    elif backend == "openai":
        from openai import OpenAI
        return OpenAI(api_key=llm_cfg["api_key"])

    elif backend == "azure":
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        return ChatCompletionsClient(
            endpoint=llm_cfg["endpoint"],
            credential=AzureKeyCredential(llm_cfg["api_key"]),
        )

    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. "
                         f"Choose from: sglang, openai, azure")


def get_model_name(config: dict) -> str:
    """Return the model name string to pass in API calls."""
    return config.get("llm", {}).get("model_name", "qwen-3.5-9b")


def is_multimodal(config: dict) -> bool:
    """Return True if image inputs should be included in prompts (Runs B/D)."""
    return bool(config.get("llm", {}).get("multimodal", False))
