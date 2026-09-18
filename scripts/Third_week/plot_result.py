from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "Third_week"
    / "unified_summary.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "Third_week"
    / "figures"
)


# ============================================================
# Benchmark configuration
# ============================================================

WORKLOADS = [
    128,
    512,
    2048,
]

QUANTIZATIONS = [
    "F16",
    "Q8_0",
    "Q4_K_M",
]


# ============================================================
# Load results
# ============================================================

def load_results():

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary file not found:\n"
            f"{SUMMARY_PATH}"
        )

    df = pd.read_csv(
        SUMMARY_PATH
    )

    required_models = set(
        QUANTIZATIONS
    )

    actual_models = set(
        df["quantization"]
    )

    if not required_models.issubset(
        actual_models
    ):
        raise ValueError(
            "unified_summary.csv is missing "
            "one or more quantization results."
        )

    return df


# ============================================================
# Figure 1:
# Prompt Length -> Prefill Throughput
# ============================================================

def plot_prefill(df):

    plt.figure(
        figsize=(8, 5)
    )

    for quantization in QUANTIZATIONS:

        row = df[
            df["quantization"]
            == quantization
        ].iloc[0]

        prefill_tps = [
            row["pp128_tps"],
            row["pp512_tps"],
            row["pp2048_tps"],
        ]

        plt.plot(
            WORKLOADS,
            prefill_tps,
            marker="o",
            label=quantization,
        )

    plt.xlabel(
        "Prompt Length (tokens)"
    )

    plt.ylabel(
        "Prefill Throughput (tokens/s)"
    )

    plt.title(
        "Qwen3-1.7B Prefill Throughput"
    )

    plt.xticks(
        WORKLOADS,
        [str(x) for x in WORKLOADS],
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "prefill_throughput.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved Prefill figure:\n"
        f"{output_path}"
    )


# ============================================================
# Figure 2:
# Context Depth -> Decode Throughput
# ============================================================

def plot_decode(df):

    plt.figure(
        figsize=(8, 5)
    )

    for quantization in QUANTIZATIONS:

        row = df[
            df["quantization"]
            == quantization
        ].iloc[0]

        decode_tps = [
            row["tg128_d128_tps"],
            row["tg128_d512_tps"],
            row["tg128_d2048_tps"],
        ]

        plt.plot(
            WORKLOADS,
            decode_tps,
            marker="o",
            label=quantization,
        )

    plt.xlabel(
        "Context Depth (tokens)"
    )

    plt.ylabel(
        "Decode Throughput (tokens/s)"
    )

    plt.title(
        "Qwen3-1.7B Decode Throughput"
    )

    plt.xticks(
        WORKLOADS,
        [str(x) for x in WORKLOADS],
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "decode_throughput.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved Decode figure:\n"
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    print("=" * 80)
    print("Generating benchmark figures")
    print("=" * 80)

    plot_prefill(
        df
    )

    plot_decode(
        df
    )

    print()
    print("=" * 80)
    print("Figure generation finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()