from pathlib import Path
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "results"
    / "Third_week"
    / "benchmark_matrix.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Third_week"
    / "unified_summary.csv"
)


# ============================================================
# Existing Size / Memory benchmark results
# ============================================================

MODEL_INFO = {
    "F16": {
        "model_size_gib": 3.790,
        "peak_vram_mib": 3713,
        "peak_ram_mib": 3780,
    },

    "Q8_0": {
        "model_size_gib": 1.708,
        "peak_vram_mib": 2166,
        "peak_ram_mib": 2163,
    },

    "Q4_K_M": {
        "model_size_gib": 1.194,
        "peak_vram_mib": 1468,
        "peak_ram_mib": 1476,
    },
}


# ============================================================
# Memory benchmark metadata
# ============================================================

MEMORY_CONTEXT_SIZE = 2048
MEMORY_GENERATION_TOKENS = 128
MEMORY_GPU_LAYERS = 99


# ============================================================
# Main
# ============================================================

def main():

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark file not found:\n"
            f"{BENCHMARK_PATH}"
        )

    df = pd.read_csv(
        BENCHMARK_PATH
    )

    print("Loaded benchmark matrix:")
    print(df)
    print()

    rows = []

    for quantization, model_info in MODEL_INFO.items():

        model_df = df[
            df["quantization"] == quantization
        ].copy()

        if len(model_df) != 3:
            raise ValueError(
                f"{quantization} should contain "
                f"3 workloads, but found "
                f"{len(model_df)}."
            )

        model_df = model_df.set_index(
            "workload_tokens"
        )

        required_workloads = [
            128,
            512,
            2048,
        ]

        for workload in required_workloads:

            if workload not in model_df.index:
                raise ValueError(
                    f"{quantization} missing "
                    f"workload {workload}"
                )

        row = {
            # ----------------------------------------
            # Model
            # ----------------------------------------

            "quantization":
                quantization,

            # ----------------------------------------
            # Model Size
            # ----------------------------------------

            "model_size_gib":
                model_info[
                    "model_size_gib"
                ],

            # ----------------------------------------
            # Memory
            #
            # 注意：
            # 这些是之前固定 workload 的
            # Memory Benchmark 数据，
            # 并不是当前每个 workload 分别测出的。
            # ----------------------------------------

            "peak_vram_mib":
                model_info[
                    "peak_vram_mib"
                ],

            "peak_ram_mib":
                model_info[
                    "peak_ram_mib"
                ],

            "memory_context_size":
                MEMORY_CONTEXT_SIZE,

            "memory_generation_tokens":
                MEMORY_GENERATION_TOKENS,

            "memory_gpu_layers":
                MEMORY_GPU_LAYERS,

            # ----------------------------------------
            # Prefill Throughput
            # ----------------------------------------

            "pp128_tps":
                model_df.loc[
                    128,
                    "prefill_tps_mean"
                ],

            "pp512_tps":
                model_df.loc[
                    512,
                    "prefill_tps_mean"
                ],

            "pp2048_tps":
                model_df.loc[
                    2048,
                    "prefill_tps_mean"
                ],

            # ----------------------------------------
            # Prefill Latency
            # ----------------------------------------

            "pp128_latency_ms":
                model_df.loc[
                    128,
                    "prefill_latency_ms"
                ],

            "pp512_latency_ms":
                model_df.loc[
                    512,
                    "prefill_latency_ms"
                ],

            "pp2048_latency_ms":
                model_df.loc[
                    2048,
                    "prefill_latency_ms"
                ],

            # ----------------------------------------
            # Decode Throughput
            # ----------------------------------------

            "tg128_d128_tps":
                model_df.loc[
                    128,
                    "decode_tps_mean"
                ],

            "tg128_d512_tps":
                model_df.loc[
                    512,
                    "decode_tps_mean"
                ],

            "tg128_d2048_tps":
                model_df.loc[
                    2048,
                    "decode_tps_mean"
                ],

            # ----------------------------------------
            # TPOT
            # ----------------------------------------

            "tpot_d128_ms":
                model_df.loc[
                    128,
                    "tpot_ms"
                ],

            "tpot_d512_ms":
                model_df.loc[
                    512,
                    "tpot_ms"
                ],

            "tpot_d2048_ms":
                model_df.loc[
                    2048,
                    "tpot_ms"
                ],
        }

        rows.append(row)

    summary_df = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 80)
    print("Unified Summary")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        f"Saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()