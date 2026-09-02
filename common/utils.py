from pathlib import Path

# Chat-template overrides served with `vllm serve --chat-template`.
CHAT_TEMPLATES_DIR = Path(__file__).resolve().parent / "chat_templates"

# The two *-Think models are "instruct" on purpose.
MODEL_NAME_TO_TYPE = {
    "meta-llama/Llama-3.1-8B": "base",
    "meta-llama/Llama-3.1-8B-Instruct": "instruct",
    "meta-llama/Llama-3.1-70B": "base",
    "meta-llama/Llama-3.1-70B-Instruct": "instruct",
    "allenai/Olmo-3-1125-32B": "base",
    "allenai/Olmo-3.1-32B-Instruct": "instruct",
    "allenai/Olmo-3.1-32B-Think": "instruct",
    "google/gemma-4-31B": "base",
    "google/gemma-4-31B-it": "instruct",
    # No base counterpart is published for Qwen3.8-27B
    "Qwen/Qwen3.8-27B": "instruct",
}
