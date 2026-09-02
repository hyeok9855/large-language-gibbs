"""Divergent Association Task with the reference prompt of Bellemare-Pepin et al.

direct: one free-form reply per answer, as in the paper.
gibbs:  start from a direct answer, then repeatedly resample one word given the
        other nine (rendered as a numbered list in an assistant prefill).
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
    parse_words,
)

# One list entry. The tail lets the tokenizer fuse a newline/period onto the word.
ENTRY_REGEX = r" ?[A-Za-z][a-z-]{0,18}[a-z][.\n ]{0,2}"
BASE_LEAD = "\n\nResponse:\n"


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
        """Free-form reply. Base models continue the prompt after a 'Response:' lead
        and a '1.' list marker (a bare '1.' mostly yields an empty form)."""
        if self.base:
            return "1." + self._complete(prompt + BASE_LEAD + "1.", max_tokens=256)
        return self._chat([{"role": "user", "content": prompt}], max_tokens=1024)

    def next_entry(self, prompt: str, prefill: str) -> str:
        """One more list entry after `prefill`, constrained to a single word."""
        extra = {"structured_outputs": {"regex": ENTRY_REGEX}}
        if self.base:
            text = self._complete(prompt + BASE_LEAD + prefill, max_tokens=16, extra_body=extra)
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


def gibbs_chain(llm: LLM, prompt: str, args: argparse.Namespace, seed: int, pbar: tqdm):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        words = parse_words(llm.answer(prompt), args.n_words)
        if len(words) == args.n_words:
            break
    else:
        raise RuntimeError("Direct draws never produced a full answer to initialise from.")

    chain, kept, order = [list(words)], [], []
    n_steps = args.burn_in + (args.n_samples // args.n_chains) * args.thinning
    for step in range(1, n_steps + 1):
        if not order:  # systematic sweep in a fresh random order
            order = rng.permutation(args.n_words).tolist()
        i = order.pop()
        others = [w for j, w in enumerate(words) if j != i]
        rng.shuffle(others)
        words[i] = llm.next_entry(prompt, render(others, args.n_words))
        chain.append(list(words))
        if step > args.burn_in and (step - args.burn_in) % args.thinning == 0:
            kept.append(list(words))
        pbar.update()
    return kept, chain


def save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    print(f"Saved {len(payload['samples'])} answers to {path}")


def main(args: argparse.Namespace) -> None:
    out_dir = RESULTS_DIR / f"{args.model_name.replace('/', '--')}_temp{args.temperature}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = dat_prompt(args.n_words)
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
            responses = list(
                tqdm(
                    pool.map(lambda _: llm.answer(prompt), range(args.n_samples)),
                    total=args.n_samples,
                )
            )
        samples = [parse_words(r, args.n_words) for r in responses]
        save(
            path,
            {
                "method": "direct",
                **meta,
                "samples": samples,
                "responses": responses,
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
                    lambda c: gibbs_chain(llm, prompt, args, args.seed * 1000 + c, pbar),
                    range(args.n_chains),
                )
            )
        save(
            path,
            {
                "method": "gibbs",
                **meta,
                "n_chains": args.n_chains,
                "burn_in": args.burn_in,
                "thinning": args.thinning,
                "samples": [s for kept, _ in results for s in kept],
                "chains": [chain for _, chain in results],
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
