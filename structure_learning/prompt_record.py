"""Optional ``--save_prompt`` sidecar: record what a run actually sent.

Every sampler funnels through ``llm.generate(prompt, schema, ...)``, so one
wrapper at that boundary captures the real request for all methods -- JSON
schema, choice, or prefill regex -- without the templates knowing about it.

Two requests are kept: the first the run sent and the last. For the Gibbs
samplers that pair is informative on its own, since the first call is the
unconditional initialisation and the last is a conditional kernel step.
"""

import json
import threading
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any


class PromptRecorder:
    """Wraps ``llm.generate`` in place and remembers the first and last call.

    ``constraint_of`` maps the schema a method passes to ``generate`` onto the
    constraint the server actually receives (a JSON schema, a choice list, or a
    shape regex).
    """

    def __init__(self, llm: Any, constraint_of: Callable[[Any], Any]):
        self._generate = llm.generate
        self._constraint_of = constraint_of
        self._lock = threading.Lock()
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        # Priors only ever reach the client through this attribute.
        llm.generate = self

    def __call__(self, prompt: Any, schema: Any = None, *args: Any, **kwargs: Any) -> Any:
        record = {"prompt": prompt, "constraint": self._constraint_of(schema)}
        with self._lock:
            if self.first is None:
                self.first = record
            self.last = record
        return self._generate(prompt, schema, *args, **kwargs)


def _fence(text: str) -> str:
    return f"```\n{text}\n```\n"


def _render_request(record: dict[str, Any]) -> str:
    prompt, constraint = record["prompt"], record["constraint"]
    out = ""
    if isinstance(prompt, tuple):  # instruct prefill: (user message, assistant prefill)
        user, prefill = prompt
        out += f"#### User message\n\n{_fence(user)}\n#### Assistant prefill\n\n{_fence(prefill)}"
    else:
        out += f"#### Prompt\n\n{_fence(prompt)}"
    rendered = constraint if isinstance(constraint, str) else json.dumps(constraint, indent=2)
    out += f"\n#### Constraint\n\n{_fence(rendered)}"
    return out


def save_prompt_record(
    path: Path,
    args: Namespace,
    llm: Any,
    recorder: PromptRecorder,
    data_filename: str,
) -> None:
    """Write the Markdown sidecar next to the run's CSV."""
    if recorder.first is None:
        print(f"No requests recorded; not writing {path}")
        return

    prefilled = isinstance(recorder.first["prompt"], tuple)
    api = (
        "chat.completions (assistant prefill)"
        if prefilled
        else ("chat.completions" if args.model_type == "instruct" else "completions")
    )
    rows = [
        ("dataset", args.dataset_name),
        ("model", f"{args.model_name} ({args.model_type})"),
        ("API", api),
        ("sampling method", args.sampling_method),
        ("temperature / top_p", f"{args.temperature} / {args.top_p}"),
        ("max_tokens", str(llm.max_tokens)),
        ("samples", f"{args.n_samples} over {args.n_chains} chain(s)"),
        ("manual reasoning", str(args.manual_reasoning)),
        ("seed", str(args.seed)),
        ("data", data_filename),
    ]
    if "gibbs" in args.sampling_method:
        rows.insert(
            -1,
            (
                "burn-in / thinning / block",
                f"{args.burn_in} / {args.thinning} / {args.block_size}"
                f"{' (sweep)' if args.sweep else ''}",
            ),
        )

    body = f"# Prompt record: {args.sampling_method}\n\n"
    body += "| | |\n|---|---|\n"
    body += "".join(f"| {name} | {value} |\n" for name, value in rows)
    body += (
        "\nRecorded verbatim: the first request the run sent and the last one. "
        "For the Gibbs samplers the first is the unconditional initialisation and "
        "the last is a conditional kernel step.\n"
    )
    if llm.system_prompt:
        body += f"\n## System prompt\n\n{_fence(llm.system_prompt)}"
    body += f"\n## First request\n\n{_render_request(recorder.first)}"
    body += f"\n## Last request\n\n{_render_request(recorder.last)}"

    path.write_text(body)
    print(f"Saved prompt record to {path}")
