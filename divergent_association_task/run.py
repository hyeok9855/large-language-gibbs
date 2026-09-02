"""Divergent Association Task with the reference prompt of Bellemare-Pepin et al.

direct: one free-form reply per answer, as in the paper, redrawn until the
        scorer can score it.
gibbs:  start from a direct answer, then repeatedly resample one word given the
        other nine (rendered as a numbered list in an assistant prefill);
        a draw that makes the answer unscorable is rejected and the chain stays.
Both are rejection sampling from p(answer | scorable), which is the constraint
the paper applies post hoc by dropping unscorable replies -- Gibbs cannot do
that, because an unscorable state feeds itself into the next conditional.
Answers are saved unscored; evaluate.py scores them.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from common.utils import MODEL_NAME_TO_TYPE
from divergent_association_task.utils import (
    MODEL_CHAT_TEMPLATE_KWARGS,
    RESULTS_DIR,
    dat_prompt,
    is_scorable,
    load_valid_words,
    parse_words,
)

# One list entry. The tail lets the tokenizer fuse a newline/period onto the word.
ENTRY_REGEX = r" ?[A-Za-z][a-z-]{0,18}[a-z][.\n ]{0,2}"
# Redraws before giving up: a whole answer (direct) or one word (gibbs step).
MAX_ATTEMPTS = 100


class LLM:
    def __init__(self, args: argparse.Namespace):
        # trust_env=False: the cluster's proxy must not see localhost requests.
        self.client = OpenAI(
            base_url=args.base_url, api_key=args.api_key, http_client=httpx.Client(trust_env=False)
        )
        self.model_name = args.model_name
        self.base = args.model_type == "base"
        self.temperature = args.temperature
        self.template_kwargs = MODEL_CHAT_TEMPLATE_KWARGS.get(args.model_name) or {}
        self.calls = 0

    def _chat(self, messages: list[dict], **kwargs) -> str:
        extra = kwargs.pop("extra_body", {})
        if self.template_kwargs:
            extra["chat_template_kwargs"] = self.template_kwargs
        self.calls += 1
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            extra_body=extra,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        response = self.client.completions.create(
            model=self.model_name, prompt=prompt, temperature=self.temperature, **kwargs
        )
        return response.choices[0].text or ""

    def answer(self, prompt: str) -> str:
        """Free-form reply. Base models simply continue the prompt (completions API)."""
        if self.base:
            return self._complete(prompt + "\n\n", max_tokens=256)
        return self._chat([{"role": "user", "content": prompt}], max_tokens=1024)

    def next_entry(self, prompt: str, prefill: str) -> str:
        """One more list entry after `prefill`, constrained to a single word."""
        extra = {"structured_outputs": {"regex": ENTRY_REGEX}}
        if self.base:
            text = self._complete(prompt + "\n\n" + prefill, max_tokens=16, extra_body=extra)
        else:
            extra |= {"add_generation_prompt": False, "continue_final_message": True}
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefill},
            ]
            text = self._chat(messages, max_tokens=16, extra_body=extra)
        return re.search(r"[A-Za-z][a-z-]*", text).group(0)


def render(words: list[str], n_words: int) -> str:
    return "".join(f"{i + 1}. {w}\n" for i, w in enumerate(words)) + f"{n_words}."


def direct_draw(
    llm: LLM, prompt: str, args: argparse.Namespace, valid: set[str], full: bool = False
):
    """One answer, redrawn until scorable. Returns the last draw either way, so an
    exhausted budget shows up as an unscorable answer rather than a crash.

    ``full`` additionally demands all n_words entries. Scorability alone does not:
    a reply parsed into 9 words can still hold 7 embeddable ones, and it counts for
    Direct -- but a Gibbs chain needs one word per slot to resample.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = llm.answer(prompt)
        words = parse_words(text, args.n_words)
        if is_scorable(words, valid) and not (full and len(words) < args.n_words):
            return words, text, attempt
    return words, text, MAX_ATTEMPTS


def gibbs_chain(llm: LLM, prompt: str, args: argparse.Namespace, valid: set[str], seed, pbar):
    rng = np.random.default_rng(seed)
    words, _, _ = direct_draw(llm, prompt, args, valid, full=True)
    if not is_scorable(words, valid) or len(words) < args.n_words:
        raise RuntimeError(f"No scorable {args.n_words}-word draw to initialise the chain from.")

    chain, kept, order = [list(words)], [], []
    accepted = rejected = 0
    n_steps = args.burn_in + (args.n_samples // args.n_chains) * args.thinning
    for step in range(1, n_steps + 1):
        if not order:  # systematic sweep in a fresh random order
            order = rng.permutation(args.n_words).tolist()
        i = order.pop()
        others = [w for j, w in enumerate(words) if j != i]
        rng.shuffle(others)
        for _ in range(MAX_ATTEMPTS):
            proposal = list(words)
            proposal[i] = llm.next_entry(prompt, render(others, args.n_words))
            if is_scorable(proposal, valid):
                words, accepted = proposal, accepted + 1
                break
            rejected += 1
        chain.append(list(words))
        if step > args.burn_in and (step - args.burn_in) % args.thinning == 0:
            kept.append(list(words))
        pbar.update()
    return kept, chain, accepted, rejected


def save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    print(f"Saved {len(payload['samples'])} answers to {path}")


def main(args: argparse.Namespace) -> None:
    out_dir = RESULTS_DIR / f"{args.model_name.replace('/', '--')}_temp{args.temperature}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = dat_prompt(args.n_words)
    valid = load_valid_words()
    print(f"Rejecting unscorable answers against {len(valid)} embeddable words.")
    meta = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "temperature": args.temperature,
        "n_words": args.n_words,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "prompt": prompt,
    }

    path = out_dir / f"direct_n{args.n_samples}_seed{args.seed}.json"
    if "direct" in args.methods and not path.exists():
        print("--- direct ---")
        llm, t0 = LLM(args), time.time()
        with ThreadPoolExecutor(args.n_chains) as pool:
            draws = list(
                tqdm(
                    pool.map(
                        lambda _: direct_draw(llm, prompt, args, valid), range(args.n_samples)
                    ),
                    total=args.n_samples,
                )
            )
        attempts = [a for *_, a in draws]
        print(f"Attempts per answer: {np.mean(attempts):.2f} (max {max(attempts)})")
        save(
            path,
            {
                "method": "direct",
                **meta,
                "samples": [words for words, *_ in draws],
                "responses": [text for _, text, _ in draws],
                "attempts": attempts,
                "llm_calls": llm.calls,
                "duration_seconds": time.time() - t0,
            },
        )

    path = out_dir / f"gibbs_n{args.n_samples}_nc{args.n_chains}_seed{args.seed}.json"
    if "gibbs" in args.methods and not path.exists():
        print("--- gibbs ---")
        llm, t0 = LLM(args), time.time()
        n_steps = args.burn_in + (args.n_samples // args.n_chains) * args.thinning
        with tqdm(total=n_steps * args.n_chains) as pbar, ThreadPoolExecutor(args.n_chains) as pool:
            results = list(
                pool.map(
                    lambda c: gibbs_chain(llm, prompt, args, valid, args.seed * 1000 + c, pbar),
                    range(args.n_chains),
                )
            )
        accepted = sum(a for *_, a, _ in results)
        rejected = sum(r for *_, r in results)
        print(f"Acceptance rate: {accepted / (accepted + rejected):.3f}")
        save(
            path,
            {
                "method": "gibbs",
                **meta,
                "n_chains": args.n_chains,
                "burn_in": args.burn_in,
                "thinning": args.thinning,
                "samples": [s for kept, *_ in results for s in kept],
                "chains": [chain for _, chain, *_ in results],
                "accepted": accepted,
                "rejected": rejected,
                "llm_calls": llm.calls,
                "duration_seconds": time.time() - t0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default="NOT_A_KEY")
    parser.add_argument(
        "--methods", nargs="+", choices=["direct", "gibbs"], default=["direct", "gibbs"]
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=500, help="answers per method")
    parser.add_argument("--n_words", type=int, default=10)
    parser.add_argument("--n_chains", type=int, default=25, help="Gibbs chains, also concurrency")
    parser.add_argument("--burn_in", type=int, default=50, help="Gibbs steps (single-word updates)")
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_TYPE:
        parser.error(f"Unknown model {args.model_name!r}; add it to common/utils.py.")
    args.model_type = MODEL_NAME_TO_TYPE[args.model_name]
    if args.model_type == "instruct" and args.model_name not in MODEL_CHAT_TEMPLATE_KWARGS:
        parser.error(f"Add {args.model_name!r} to MODEL_CHAT_TEMPLATE_KWARGS ({{}} if none).")
    if args.n_samples % args.n_chains:
        parser.error("--n_samples must be divisible by --n_chains.")
    args.base_url = args.base_url or f"http://localhost:{args.port}/v1"
    main(args)
