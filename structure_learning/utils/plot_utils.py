import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import mutual_info_score


def visualise_discrete_dataset(
    df: pd.DataFrame,
    meta_data: dict,
    figsize: tuple[float, float] = (8, 8),
    fontsize: float = 8,
    title: str = "",
) -> Figure:
    column_order = list(meta_data["features"].keys())
    var_and_order = {
        var: meta_data["features"][var]["schema"]["enum"] for var in meta_data["features"]
    }

    df = df[column_order]  # type: ignore
    for col in df.columns:
        if col in var_and_order:
            ordinal_type = pd.CategoricalDtype(categories=var_and_order[col], ordered=True)
            df[col] = df[col].astype(ordinal_type)

    cols = df.columns
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                # Diagonal: Marginal distribution (Bar chart)
                df[cols[i]].value_counts(sort=False).plot(kind="bar", ax=ax, color="gray")
                ax.set_xlabel("")
            else:
                try:
                    # Off-diagonal: Mosaic plot for joint distributions
                    mosaic(
                        df.sort_values(by=[cols[j], cols[i]]),  # type: ignore
                        [cols[j], cols[i]],
                        ax=ax,
                        labelizer=lambda k: "",
                        gap=0.05,
                    )
                except ValueError:
                    ax.text(
                        0.5,
                        0.5,
                        "const",
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        color="gray",
                        transform=ax.transAxes,
                    )

            if i == n - 1:
                ax.set_xlabel(cols[j], fontsize=fontsize)

            if j == 0:
                ax.set_ylabel(cols[i], fontsize=fontsize)

            # Clean up axes for readability
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=fontsize * 1.2)

    fig.tight_layout()
    return fig


def mi_matrix(df, columns) -> pd.DataFrame:
    n = len(columns)
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi[i, j] = mi[j, i] = mutual_info_score(df[columns[i]], df[columns[j]])
    return pd.DataFrame(mi, index=columns, columns=columns)


def compare_mi_matrices(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    meta_data: dict,
    figsize: tuple[float, float] = (10, 5),
    fontsize: float = 8,
    title: str = "",
) -> Figure:
    columns = [c for c in meta_data["features"] if c in df1.columns and c in df2.columns]
    if not columns:
        raise ValueError("No shared columns found between the two DataFrames and meta_data")

    mi1 = mi_matrix(df1, columns)
    mi2 = mi_matrix(df2, columns)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    tick_pos = range(len(columns))

    for ax, data, cmap, axtitle in [
        (ax1, mi1.values, "viridis", "MI (Dataset 1)"),
        (ax2, mi2.values, "viridis", "MI (Dataset 2)"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="equal")
        ax.set_title(axtitle, fontsize=fontsize + 1)
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(columns, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(columns, fontsize=fontsize)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=fontsize * 1.2)

    fig.tight_layout()
    return fig
