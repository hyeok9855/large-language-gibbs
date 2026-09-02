"""Capture the first N request/reasoning/result triples of a run.

Manual reasoning is normally invisible in the results: priorbot's decision
kernels read ``choice``/``bet`` off the dict the client returned and drop the
rest, so the chain of thought is thrown away as soon as it is used. This module
snapshots it *at generate time* and writes it beside the run under a
``reasoning_traces/`` tree.

Traces live in their own tree, NOT under a ``results*/`` root: the plot drivers
glob ``*.json`` in every model dir and key methods off ``_seed(\\d+)``, so a
trace file dropped next to the samples would be parsed as a bogus method.
"""

import json
import threading
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).resolve().parent / "reasoning_traces"
DEFAULT_N_TRACES = 10


class TraceRecorder:
    """Thread-safe, capped record of the first ``n_traces`` LLM calls.

    Chains run concurrently in executor threads, so the counter is locked;
    "first N" therefore means the first N calls to complete, across chains.
    For the Gibbs-style kernels these are burn-in calls - the request index is
    stored so that stays visible.
    """

    def __init__(self, n_traces: int = DEFAULT_N_TRACES):
        self.n_traces = n_traces
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self.n_calls = 0

    @property
    def full(self) -> bool:
        return len(self._records) >= self.n_traces

    def claim(self) -> int | None:
        """Reserve a slot, returning the 0-based request index, or None if full."""
        with self._lock:
            self.n_calls += 1
            if len(self._records) >= self.n_traces:
                return None
            index = len(self._records)
            self._records.append({})
            return index

    def record(self, prompt: Any, reasoning: Any, result: Any, index: int) -> None:
        with self._lock:
            self._records[index] = {
                "request": index,
                "prompt": jsonable(prompt),
                "reasoning": reasoning,
                "result": jsonable(result),
            }

    def dump(self, path: Path, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            records = [r for r in self._records if r]
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"meta": meta or {}, "n_calls": self.n_calls, "traces": records}, f, indent=2)
        print(f"Saved {len(records)} reasoning traces to {path}")


def jsonable(value: Any) -> Any:
    """Prompts are str or (user, prefill); results are dicts/scalars."""
    if isinstance(value, tuple):
        return list(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def trace_meta(args: Namespace, family: str, method: str, result_filename: str) -> dict[str, Any]:
    """Provenance header written at the top of a trace file."""
    return {
        "family": family,
        "method": method,
        "model_name": args.model_name,
        "target": args.target,
        "manual_reasoning": args.manual_reasoning,
        "temperature": args.temperature,
        "seed": args.seed,
        "result_file": result_filename,
    }


def install_json(llm, recorder: TraceRecorder) -> None:
    """Wrap ``llm.generate`` (JSON family). The returned dict is snapshotted
    before the caller can read the answer out of it and discard the rest."""
    inner = llm.generate

    def wrapper(prompt, schema=None, verbose=False, max_trials=10):
        index = recorder.claim()
        out = inner(prompt, schema, verbose, max_trials)
        if index is not None:
            reasoning = result = None
            if isinstance(out, dict):
                snapshot = deepcopy(out)
                reasoning = snapshot.pop("reasoning", None)
                result = snapshot
            else:
                result = out
            recorder.record(prompt, reasoning, result, index)
        return out

    llm.generate = wrapper
