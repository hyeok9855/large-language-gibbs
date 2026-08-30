"""Figures and metrics for the continuation family: ``sampling.make_plot``
pointed at the continuation results root. PATTERNS is the JSON family's list
(identical filenames), restated because it is private to that module's __main__.
"""

import argparse
import re

from sampling.make_plot import run
from sampling_continuation.utils import RESULTS_DIR

PATTERNS = [
    (re.compile(r"^independent$"), "Independent"),
    (re.compile(r"^batch(?:_nc\d+)?$"), "Batch"),
    (re.compile(r"^gibbs_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"), "Gibbs (B={b})"),
    (
        re.compile(r"^barkergibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
        "Barker-Gibbs (B={b})",
    ),
    (
        re.compile(r"^gamblinggibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
        "Gambling-Gibbs (B={b})",
    ),
]

PLOT_SUFFIXES = ("", "_reasoning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Ignore runs whose method name does not match PATTERNS.",
    )
    args = parser.parse_args()

    run(
        results_dir=RESULTS_DIR,
        patterns=PATTERNS,
        plot_suffixes=PLOT_SUFFIXES,
        ignore_unknown=args.ignore_unknown,
    )
