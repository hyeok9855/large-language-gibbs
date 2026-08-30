"""The canonical model registry, shared by every experiment package.

One dict, so a model added for one experiment is spelled identically in the
others. Membership is also each experiment's "known model" gate: the runners
and sbatch preflights reject anything absent from it rather than guessing a
prompt style, so adding an entry here opts the model into every experiment.

Deliberately stdlib-only, so a SLURM script's fail-fast preflight can import it
before the heavy experiment extras are installed.
"""

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
    "allenai/Olmo-3-32B-Think": "instruct",
    "allenai/Olmo-3.1-32B-Think": "instruct",
    "google/gemma-4-31B": "base",
    "google/gemma-4-31B-it": "instruct",
    # No base counterpart is published for Qwen3.8-27B
    "Qwen/Qwen3.8-27B": "instruct",
}
