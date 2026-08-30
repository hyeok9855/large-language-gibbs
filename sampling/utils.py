from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def indexed_var_names(n_vars: int) -> list[str]:
    """Coordinate names X1, ..., Xn."""
    return [f"X{i}" for i in range(1, n_vars + 1)]
