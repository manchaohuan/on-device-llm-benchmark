from pathlib import Path
import subprocess
import re
import csv
from datetime import datetime


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

RESULT_DIR = PROJECT_ROOT / "results" / "Third_week"

RAW_DIR = RESULT_DIR / "raw"


# ============================================================
# llama.cpp
# ============================================================

LLAMA_BENCH = Path(
    r"D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64\llama-bench.exe"
)


# ============================================================
# Models
# ============================================================

MODELS = [
    {
        "name": "F16",
        "path": MODEL_DIR / "Qwen3-1.7B-f16.gguf",
    },
    {
        "name": "Q8_0",
        "path": MODEL_DIR / "Qwen3-1.7B-Q8_0.gguf",
    },
    {
        "name": "Q4_K_M",
        "path": MODEL_DIR / "Qwen3-1.7B-Q4_K_M.gguf",
    },
]


# ============================================================
# Benchmark configuration
# ============================================================

WORKLOADS = [
    128,
    512,
    2048,
]

GENERATION_TOKENS = 128

GPU_LAYERS = 99

REPETITIONS = 5


# ============================================================
# Check files
# ============================================================

def check_files():

    if not LLAMA_BENCH.exists():
        raise FileNotFoundError(
            f"llama-bench.exe not found:\n{LLAMA_BENCH}"
        )

    for model in MODELS:

        if not model["path"].exists():
            raise FileNotFoundError(
                f"Model not found:\n{model['path']}"
            )


# ============================================================
# Run command
# ============================================================

def run_command(command, title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (
        process.stdout
        + "\n"
        + process.stderr
    )

    print(output)

    if process.returncode != 0:
        raise RuntimeError(
            f"llama-bench returned "
            f"{process.returncode}"
        )

    return output


# ============================================================
# Prefill benchmark
# ============================================================

def run_prefill_benchmark(model):

    prompt_values = ",".join(
        str(x) for x in WORKLOADS
    )

    command = [
        str(LLAMA_BENCH),

        "-m",
        str(model["path"]),

        "-p",
        prompt_values,

        # Prefill only
        "-n",
        "0",

        "-ngl",
        str(GPU_LAYERS),

        "-r",
        str(REPETITIONS),
    ]

    return run_command(
        command,
        (
            f"Prefill Benchmark: "
            f"{model['name']}"
        ),
    )


# ============================================================
# Decode benchmark
# ============================================================

def run_decode_benchmark(model):

    depth_values = ",".join(
        str(x) for x in WORKLOADS
    )

    command = [
        str(LLAMA_BENCH),

        "-m",
        str(model["path"]),

        # No independent prompt benchmark
        "-p",
        "0",

        # Generate 128 tokens
        "-n",
        str(GENERATION_TOKENS),

        # Pre-fill KV Cache to different depths
        "-d",
        depth_values,

        "-ngl",
        str(GPU_LAYERS),

        "-r",
        str(REPETITIONS),
    ]

    return run_command(
        command,
        (
            f"Decode Benchmark: "
            f"{model['name']}"
        ),
    )


# ============================================================
# Parse mean ± std
# ============================================================

def parse_speed(speed_text):

    match = re.search(
        r"([\d.]+)\s*±\s*([\d.]+)",
        speed_text,
    )

    if match is None:
        return None

    return {
        "mean": float(match.group(1)),
        "std": float(match.group(2)),
    }


# ============================================================
# Parse Prefill
# ============================================================

def parse_prefill(output):

    results = {}

    for line in output.splitlines():

        if "|" not in line:
            continue

        parts = [
            part.strip()
            for part in line.strip().strip("|").split("|")
        ]

        if len(parts) < 7:
            continue

        test_name = parts[5]
        speed_text = parts[6]

        match = re.fullmatch(
            r"pp(\d+)",
            test_name,
        )

        if match is None:
            continue

        prompt_tokens = int(
            match.group(1)
        )

        speed = parse_speed(
            speed_text
        )

        if speed is None:
            continue

        results[prompt_tokens] = speed

    return results


# ============================================================
# Parse Decode
# ============================================================

def parse_decode(output):

    results = {}

    for line in output.splitlines():

        if "|" not in line:
            continue

        parts = [
            part.strip()
            for part in line.strip().strip("|").split("|")
        ]

        if len(parts) < 7:
            continue

        test_name = parts[5]
        speed_text = parts[6]

        match = re.fullmatch(
            r"tg(\d+)\s*@\s*d(\d+)",
            test_name,
        )

        if match is None:
            continue

        generation_tokens = int(
            match.group(1)
        )

        context_depth = int(
            match.group(2)
        )

        speed = parse_speed(
            speed_text
        )

        if speed is None:
            continue

        results[context_depth] = {
            "generation_tokens":
                generation_tokens,

            "mean":
                speed["mean"],

            "std":
                speed["std"],
        }

    return results


# ============================================================
# Build unified result
# ============================================================

def build_rows(
    model,
    prefill_results,
    decode_results,
):

    rows = []

    for workload in WORKLOADS:

        if workload not in prefill_results:
            raise ValueError(
                f"Missing pp{workload} "
                f"for {model['name']}"
            )

        if workload not in decode_results:
            raise ValueError(
                f"Missing tg{GENERATION_TOKENS} "
                f"@ d{workload} "
                f"for {model['name']}"
            )

        prefill = (
            prefill_results[workload]
        )

        decode = (
            decode_results[workload]
        )

        # Prefill latency:
        #
        # tokens / (tokens / second)
        #
        # Convert seconds -> milliseconds
        prefill_latency_ms = (
            workload
            / prefill["mean"]
            * 1000
        )

        # TPOT:
        #
        # 1 / decode tokens per second
        #
        # Convert seconds -> milliseconds
        tpot_ms = (
            1000
            / decode["mean"]
        )

        rows.append(
            {
                "timestamp":
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),

                "quantization":
                    model["name"],

                "model_file":
                    model["path"].name,

                "workload_tokens":
                    workload,

                "generation_tokens":
                    GENERATION_TOKENS,

                "context_depth":
                    workload,

                "gpu_layers":
                    GPU_LAYERS,

                "repetitions":
                    REPETITIONS,

                "prefill_tps_mean":
                    prefill["mean"],

                "prefill_tps_std":
                    prefill["std"],

                "prefill_latency_ms":
                    round(
                        prefill_latency_ms,
                        3
                    ),

                "decode_tps_mean":
                    decode["mean"],

                "decode_tps_std":
                    decode["std"],

                "tpot_ms":
                    round(
                        tpot_ms,
                        3
                    ),
            }
        )

    return rows


# ============================================================
# Save raw output
# ============================================================

def save_raw_output(
    model,
    prefill_output,
    decode_output,
):

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefill_path = (
        RAW_DIR
        / f"{model['name']}_prefill.txt"
    )

    decode_path = (
        RAW_DIR
        / f"{model['name']}_decode.txt"
    )

    prefill_path.write_text(
        prefill_output,
        encoding="utf-8",
    )

    decode_path.write_text(
        decode_output,
        encoding="utf-8",
    )

    print(
        f"Prefill raw result saved:\n"
        f"{prefill_path}"
    )

    print(
        f"Decode raw result saved:\n"
        f"{decode_path}"
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(rows):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULT_DIR
        / "benchmark_matrix.csv"
    )

    fieldnames = [
        "timestamp",
        "quantization",
        "model_file",
        "workload_tokens",
        "generation_tokens",
        "context_depth",
        "gpu_layers",
        "repetitions",
        "prefill_tps_mean",
        "prefill_tps_std",
        "prefill_latency_ms",
        "decode_tps_mean",
        "decode_tps_std",
        "tpot_ms",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print("=" * 80)
    print("Benchmark matrix finished.")
    print("=" * 80)

    print(
        f"Saved to:\n"
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    check_files()

    all_rows = []

    for model in MODELS:

        # --------------------------------------------
        # Experiment A:
        # Prompt Length -> Prefill
        # --------------------------------------------

        prefill_output = (
            run_prefill_benchmark(
                model
            )
        )

        # --------------------------------------------
        # Experiment B:
        # Context Depth -> Decode
        # --------------------------------------------

        decode_output = (
            run_decode_benchmark(
                model
            )
        )

        # --------------------------------------------
        # Parse
        # --------------------------------------------

        prefill_results = (
            parse_prefill(
                prefill_output
            )
        )

        decode_results = (
            parse_decode(
                decode_output
            )
        )

        # Debug information
        print()
        print(
            f"{model['name']} "
            f"parsed prefill:"
        )
        print(prefill_results)

        print(
            f"{model['name']} "
            f"parsed decode:"
        )
        print(decode_results)

        # --------------------------------------------
        # Save original llama-bench results
        # --------------------------------------------

        save_raw_output(
            model,
            prefill_output,
            decode_output,
        )

        # --------------------------------------------
        # Unified result
        # --------------------------------------------

        rows = build_rows(
            model,
            prefill_results,
            decode_results,
        )

        all_rows.extend(
            rows
        )

    save_csv(
        all_rows
    )


if __name__ == "__main__":
    main()