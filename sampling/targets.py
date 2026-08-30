"""Registry of target distributions for the sampling experiments."""

from __future__ import annotations

import math
import re
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sampling.utils import indexed_var_names

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_LIST = rf"(?:{NUMBER})(?:,(?:{NUMBER}))*"
KVAR_METHODS = ("gibbs", "barker_gibbs", "gambling_gibbs")

# Truncation windows for unbounded targets, in sigmas.
SCHEMA_REACH = 4.0  # what the model may emit
PLOT_REACH = 4.0  # what the figures show, and off-plot draws go to overflow bin.
assert SCHEMA_REACH >= PLOT_REACH


@dataclass(frozen=True)
class Target:
    """Everything the sampler and the plots need to know about a distribution."""

    name: str
    add_arguments: Callable[[ArgumentParser], None]
    description: Callable[[Namespace], str]
    value_schema: Callable[[Namespace], dict[str, Any]]
    dir_name: Callable[[Namespace], str]
    parse_dir_name: Callable[[str], dict[str, Any] | None]
    bin_edges: Callable[[dict[str, Any]], np.ndarray]
    reference: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]
    validate: Callable[[Namespace], None]

    def xlim(self, params: dict[str, Any]) -> tuple[float, float]:
        edges = self.bin_edges(params)
        return float(edges[0]), float(edges[-1])

    def object_schema(self, args: Namespace, method: str) -> dict[str, Any]:
        """JSON schema for the object the LLM should return for ``method``."""
        if method in ("indep", "batch"):
            value = self.value_schema(args)
            if method == "indep":
                properties: dict[str, Any] = {"sample": value}
            else:
                properties = {
                    "samples": {
                        "type": "array",
                        "items": value,
                        "minItems": args.n_samples_per_chain,
                        "maxItems": args.n_samples_per_chain,
                    }
                }

        elif method in KVAR_METHODS:
            properties = {
                name: self.value_schema(args) for name in indexed_var_names(args.gibbs_k_vars)
            }
        else:
            raise ValueError(f"Invalid method: {method}")

        return {"type": "object", "properties": properties, "required": list(properties)}


# --- Uniform (discrete) -----------------------------------------------------

UNIFORM_N_BINS = 50


def _uniform_args(parser: ArgumentParser) -> None:
    parser.add_argument("--minnum", type=int, default=0)
    parser.add_argument("--maxnum", type=int, default=99)


def _uniform_validate(args: Namespace) -> None:
    if args.minnum >= args.maxnum:
        raise ValueError(f"--minnum ({args.minnum}) must be < --maxnum ({args.maxnum}).")
    n_values = args.maxnum - args.minnum + 1
    if n_values % UNIFORM_N_BINS != 0:
        raise ValueError(
            f"uniform support size (maxnum - minnum + 1) must be a multiple of "
            f"{UNIFORM_N_BINS}, got {n_values} from [{args.minnum}, {args.maxnum}]."
        )


def _uniform_bins(params: dict[str, Any]) -> np.ndarray:
    """Equal-width bars centred on integer groups (same occupancy when n % 50 == 0)."""
    minnum, maxnum = params["minnum"], params["maxnum"]
    n_values = maxnum - minnum + 1
    n_bins = min(UNIFORM_N_BINS, n_values)
    return np.linspace(minnum - 0.5, maxnum + 0.5, n_bins + 1)


def _uniform_reference(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    minnum, maxnum = params["minnum"], params["maxnum"]
    density = 1.0 / (maxnum - minnum + 1)
    return np.array([minnum - 0.5, maxnum + 0.5], dtype=float), np.array([density, density])


UNIFORM = Target(
    name="uniform",
    add_arguments=_uniform_args,
    validate=_uniform_validate,
    description=lambda args: (
        f"a uniform distribution over the integers in "
        f"{{{args.minnum}, {args.minnum + 1}, ..., {args.maxnum}}}"
    ),
    value_schema=lambda args: {
        "type": "integer",
        "minimum": args.minnum,
        "maximum": args.maxnum,
    },
    dir_name=lambda args: f"min{args.minnum}_max{args.maxnum}",
    parse_dir_name=lambda dirname: (
        {"minnum": int(m.group(1)), "maxnum": int(m.group(2))}
        if (m := re.fullmatch(r"min(-?\d+)_max(-?\d+)", dirname))
        else None
    ),
    bin_edges=_uniform_bins,
    reference=_uniform_reference,
)


# --- Gaussian ---------------------------------------------------------------


def _gaussian_args(parser: ArgumentParser) -> None:
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)


def _gaussian_validate(args: Namespace) -> None:
    if args.std <= 0:
        raise ValueError(f"--std must be > 0, got {args.std}.")


def _gaussian_bounds(mean: float, std: float, reach: float = SCHEMA_REACH) -> tuple[float, float]:
    return mean - reach * std, mean + reach * std


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


def _gaussian_bins(params: dict[str, Any]) -> np.ndarray:
    lo, hi = _gaussian_bounds(params["mean"], params["std"], PLOT_REACH)
    return np.linspace(lo, hi, 51)


def _gaussian_reference(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _gaussian_bounds(params["mean"], params["std"], PLOT_REACH)
    x = np.linspace(lo, hi, 200)
    return x, _normal_pdf(x, params["mean"], params["std"])


GAUSSIAN = Target(
    name="gaussian",
    add_arguments=_gaussian_args,
    validate=_gaussian_validate,
    description=lambda args: (
        f"a Gaussian distribution with mean {args.mean} and standard deviation {args.std}"
    ),
    value_schema=lambda args: {
        "type": "number",
        "minimum": _gaussian_bounds(args.mean, args.std)[0],
        "maximum": _gaussian_bounds(args.mean, args.std)[1],
    },
    dir_name=lambda args: f"mean{args.mean}_std{args.std}",
    parse_dir_name=lambda dirname: (
        {"mean": float(m.group(1)), "std": float(m.group(2))}
        if (m := re.fullmatch(rf"mean({NUMBER})_std({NUMBER})", dirname))
        else None
    ),
    bin_edges=_gaussian_bins,
    reference=_gaussian_reference,
)


# --- Gaussian mixture -------------------------------------------------------


def _mixture_args(parser: ArgumentParser) -> None:
    parser.add_argument("--mixture_means", type=float, nargs="+", default=[-2.0, 2.0])
    parser.add_argument("--mixture_stds", type=float, nargs="+", default=[0.5, 0.5])
    parser.add_argument("--mixture_weights", type=float, nargs="+", default=[0.5, 0.5])


def _mixture_validate(args: Namespace) -> None:
    n = len(args.mixture_means)
    if not (n == len(args.mixture_stds) == len(args.mixture_weights)):
        raise ValueError(
            "--mixture_means, --mixture_stds and --mixture_weights must have equal length; "
            f"got {n}, {len(args.mixture_stds)}, {len(args.mixture_weights)}."
        )
    if n < 2:
        raise ValueError("A mixture needs at least two components.")
    if any(s <= 0 for s in args.mixture_stds):
        raise ValueError(f"--mixture_stds must all be > 0, got {args.mixture_stds}.")
    if any(w <= 0 for w in args.mixture_weights):
        raise ValueError(f"--mixture_weights must all be > 0, got {args.mixture_weights}.")
    if not math.isclose(sum(args.mixture_weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"--mixture_weights must sum to 1, got {sum(args.mixture_weights)}.")


def _mixture_description(args: Namespace) -> str:
    components = "; ".join(
        f"with probability {w}, a Gaussian with mean {m} and standard deviation {s}"
        for w, m, s in zip(args.mixture_weights, args.mixture_means, args.mixture_stds)
    )
    return f"a mixture of {len(args.mixture_means)} Gaussian distributions ({components})"


def _mixture_bounds(params: dict[str, Any], reach: float = SCHEMA_REACH) -> tuple[float, float]:
    means, stds = params["means"], params["stds"]
    lo = min(m - reach * s for m, s in zip(means, stds))
    hi = max(m + reach * s for m, s in zip(means, stds))
    return lo, hi


def _mixture_schema(args: Namespace) -> dict[str, Any]:
    params = {"means": args.mixture_means, "stds": args.mixture_stds}
    lo, hi = _mixture_bounds(params)
    return {"type": "number", "minimum": lo, "maximum": hi}


def _mixture_bins(params: dict[str, Any]) -> np.ndarray:
    lo, hi = _mixture_bounds(params, PLOT_REACH)
    return np.linspace(lo, hi, 51)


def _mixture_reference(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _mixture_bounds(params, PLOT_REACH)
    x = np.linspace(lo, hi, 400)
    y = np.zeros_like(x)
    for w, m, s in zip(params["weights"], params["means"], params["stds"]):
        y += w * _normal_pdf(x, m, s)
    return x, y


def _mixture_parse(dirname: str) -> dict[str, Any] | None:
    m = re.fullmatch(
        rf"mixture_m({NUMBER_LIST})_s({NUMBER_LIST})_w({NUMBER_LIST})",
        dirname,
    )
    if not m:
        return None

    def _floats(text: str) -> list[float]:
        return [float(part) for part in text.split(",")]

    return {
        "means": _floats(m.group(1)),
        "stds": _floats(m.group(2)),
        "weights": _floats(m.group(3)),
    }


def _join(values) -> str:
    return ",".join(str(v) for v in values)


MIXTURE = Target(
    name="mixture",
    add_arguments=_mixture_args,
    validate=_mixture_validate,
    description=_mixture_description,
    value_schema=_mixture_schema,
    dir_name=lambda args: (
        f"mixture_m{_join(args.mixture_means)}"
        f"_s{_join(args.mixture_stds)}"
        f"_w{_join(args.mixture_weights)}"
    ),
    parse_dir_name=_mixture_parse,
    bin_edges=_mixture_bins,
    reference=_mixture_reference,
)


# --- Binomial ---------------------------------------------------------------


def _binomial_args(parser: ArgumentParser) -> None:
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--p_success", type=float, default=0.3)


def _binomial_validate(args: Namespace) -> None:
    if args.n_trials < 1:
        raise ValueError(f"--n_trials must be >= 1, got {args.n_trials}.")
    if not 0.0 < args.p_success < 1.0:
        raise ValueError(f"--p_success must be in (0, 1), got {args.p_success}.")


def _integer_bins(lo: int, hi: int) -> np.ndarray:
    """Edges centred on each integer, so one bin holds exactly one outcome."""
    return np.arange(lo - 0.5, hi + 1.5, 1.0)


def _binomial_reference(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n, p = params["n_trials"], params["p_success"]
    k = np.arange(0, n + 1)
    pmf = np.array([math.comb(n, int(i)) * p ** int(i) * (1 - p) ** (n - int(i)) for i in k])
    return k.astype(float), pmf


BINOMIAL = Target(
    name="binomial",
    add_arguments=_binomial_args,
    validate=_binomial_validate,
    description=lambda args: (
        f"a binomial distribution with {args.n_trials} trials "
        f"and success probability {args.p_success}"
    ),
    value_schema=lambda args: {"type": "integer", "minimum": 0, "maximum": args.n_trials},
    dir_name=lambda args: f"binomial_n{args.n_trials}_p{args.p_success}",
    parse_dir_name=lambda dirname: (
        {"n_trials": int(m.group(1)), "p_success": float(m.group(2))}
        if (m := re.fullmatch(rf"binomial_n(\d+)_p({NUMBER})", dirname))
        else None
    ),
    bin_edges=lambda params: _integer_bins(0, params["n_trials"]),
    reference=_binomial_reference,
)


# --- Poisson ----------------------------------------------------------------


def _poisson_args(parser: ArgumentParser) -> None:
    parser.add_argument("--rate", type=float, default=4.0)


def _poisson_validate(args: Namespace) -> None:
    if args.rate <= 0:
        raise ValueError(f"--rate must be > 0, got {args.rate}.")


def _poisson_max(rate: float, reach: float = SCHEMA_REACH) -> int:
    """Upper endpoint λ + reach*sqrt(λ), the Poisson analogue of Gaussian ±reach*σ."""
    return int(math.ceil(rate + reach * math.sqrt(rate)))


def _poisson_reference(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rate = params["rate"]
    k = np.arange(0, _poisson_max(rate, PLOT_REACH) + 1)
    pmf = np.array([math.exp(-rate) * rate ** int(i) / math.factorial(int(i)) for i in k])
    return k.astype(float), pmf


POISSON = Target(
    name="poisson",
    add_arguments=_poisson_args,
    validate=_poisson_validate,
    description=lambda args: f"a Poisson distribution with rate {args.rate}",
    value_schema=lambda args: {
        "type": "integer",
        "minimum": 0,
        "maximum": _poisson_max(args.rate),
    },
    dir_name=lambda args: f"poisson_rate{args.rate}",
    parse_dir_name=lambda dirname: (
        {"rate": float(m.group(1))}
        if (m := re.fullmatch(rf"poisson_rate({NUMBER})", dirname))
        else None
    ),
    bin_edges=lambda params: _integer_bins(0, _poisson_max(params["rate"], PLOT_REACH)),
    reference=_poisson_reference,
)


TARGETS: dict[str, Target] = {
    target.name: target for target in (UNIFORM, GAUSSIAN, MIXTURE, BINOMIAL, POISSON)
}


def get_target(name: str) -> Target:
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(f"Unknown target {name!r}; expected one of {sorted(TARGETS)}.") from None
