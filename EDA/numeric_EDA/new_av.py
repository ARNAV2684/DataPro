# eda_numeric.py
from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("whitegrid")          # one global seaborn style

# ────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────
IMG_DPI = 300
PALETTE = sns.color_palette("Set3")


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=IMG_DPI, bbox_inches="tight")
    print(f"✔ Saved → {path}")


# ────────────────────────────────────────────
#  main plotters
# ────────────────────────────────────────────
def violin_plots(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    num_cols = _numeric_cols(df)
    if not num_cols:
        return

    n_cols = min(4, len(num_cols))
    n_rows = -(-len(num_cols) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for ax, col in zip(axes, num_cols):
        sns.violinplot(y=df[col], ax=ax, color="lightblue", linewidth=0.8)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Values")

    for ax in axes[len(num_cols):]:
        ax.remove()

    fig.suptitle(f"Violin Plots – {name}", y=0.93, fontsize=14)
    _save(fig, out_dir / f"{name}_violin.png")
    plt.close(fig)


def kde_plots(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    num_cols = _numeric_cols(df)
    if not num_cols:
        return

    n_cols = min(4, len(num_cols))
    n_rows = -(-len(num_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for ax, col in zip(axes, num_cols):
        sns.kdeplot(df[col].dropna(), ax=ax, fill=True, color="steelblue")
        ax.set_title(f"{col} – KDE")
        ax.set_xlabel(col)

    for ax in axes[len(num_cols):]:
        ax.remove()

    fig.suptitle(f"KDE Plots – {name}", y=0.93, fontsize=14)
    _save(fig, out_dir / f"{name}_kde.png")
    plt.close(fig)


def pca_plots(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    num_cols = _numeric_cols(df)
    if len(num_cols) < 2:
        return

    clean = df[num_cols].dropna()
    if clean.empty:
        return

    X = StandardScaler().fit_transform(clean)
    pca = PCA().fit(X)
    pcs = pca.transform(X)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"PCA – {name}", y=0.95)

    # (1) scree
    axes[0, 0].plot(range(1, len(exp_var) + 1), exp_var, "-o")
    axes[0, 0].set_title("Scree")
    axes[0, 0].set_xlabel("PC")
    axes[0, 0].set_ylabel("Explained var")

    # (2) cumulative
    axes[0, 1].plot(range(1, len(cum_var) + 1), cum_var, "-o", color="crimson")
    axes[0, 1].axhline(0.8, ls="--", c="green")
    axes[0, 1].axhline(0.95, ls="--", c="orange")
    axes[0, 1].set_title("Cumulative var")
    axes[0, 1].set_xlabel("# PC")

    # (3) PC1 vs PC2
    axes[1, 0].scatter(pcs[:, 0], pcs[:, 1], s=12, alpha=0.6, color="purple")
    axes[1, 0].set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)")
    axes[1, 0].set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)")
    axes[1, 0].set_title("PC1 vs PC2")

    # (4) loadings heat-map
    load_df = pd.DataFrame(pca.components_[:2].T, columns=["PC1", "PC2"], index=num_cols)
    sns.heatmap(load_df, annot=True, cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Feature loadings")

    fig.tight_layout()
    _save(fig, out_dir / f"{name}_pca.png")
    plt.close(fig)


def categorical_plots(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    cat_cols = _categorical_cols(df)
    if not cat_cols:
        return

    n_cols = min(3, len(cat_cols))
    n_rows = -(-len(cat_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts().head(10)
        sns.barplot(x=vc.index, y=vc.values, ax=ax, color="lightcoral")
        ax.set_title(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for ax in axes[len(cat_cols):]:
        ax.remove()

    fig.suptitle(f"Categorical variables – {name}", y=0.93)
    _save(fig, out_dir / f"{name}_categorical.png")
    plt.close(fig)


def summary_dashboard(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    fig.suptitle(f"EDA summary – {name}", y=0.96, fontsize=16)

    # (1) overview text
    ax1 = fig.add_subplot(gs[0, :2])
    txt = textwrap.dedent(f"""
        Shape            : {df.shape[0]:,} rows × {df.shape[1]} cols
        Memory           : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB
        Missing cells    : {df.isna().sum().sum():,}
        Numeric columns  : {len(_numeric_cols(df))}
        Categorical cols : {len(_categorical_cols(df))}
        Duplicate rows   : {df.duplicated().sum():,}
    """)
    ax1.text(0, 0.9, txt, va="top", family="monospace")
    ax1.axis("off")

    # (2) missingness heat-map
    ax2 = fig.add_subplot(gs[0, 2:])
    if df.isna().values.any():
        sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis", ax=ax2)
    else:
        ax2.text(0.5, 0.5, "No missing values", ha="center", va="center")
    ax2.set_title("Missing pattern")

    # (3) dtype pie
    ax3 = fig.add_subplot(gs[1, :2])
    dtype_counts = df.dtypes.astype(str).value_counts()
    ax3.pie(dtype_counts, labels=dtype_counts.index, autopct="%1.0f%%", colors=PALETTE)
    ax3.set_title("Data-type share")

    # (4) numeric summary heat-map
    ax4 = fig.add_subplot(gs[1, 2:])
    num = _numeric_cols(df)
    if num:
        sns.heatmap(df[num].describe().loc[["mean", "std", "min", "max"]],
                    annot=True, fmt=".2f", cmap="coolwarm", ax=ax4, cbar=False)
    else:
        ax4.text(0.5, 0.5, "No numeric vars", ha="center", va="center")
    ax4.set_title("Numeric summary")

    _save(fig, out_dir / f"{name}_dashboard.png")
    plt.close(fig)


# ────────────────────────────────────────────
#  CLI + runner
# ────────────────────────────────────────────
def run(file_path: Path) -> None:
    df = pd.read_csv(file_path)
    out_dir = file_path.parent
    name = file_path.stem

    print(f"► Loaded {file_path.name}  ({df.shape[0]:,} × {df.shape[1]})")

    violin_plots(df, out_dir, name)
    kde_plots(df, out_dir, name)
    pca_plots(df, out_dir, name)
    categorical_plots(df, out_dir, name)
    summary_dashboard(df, out_dir, name)

    print("🏁  EDA finished.\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Quick EDA for numeric CSVs")
    parser.add_argument("csv", type=Path, help="Path to the CSV file")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    run(args.csv)


if __name__ == "__main__":
    cli()
