from pathlib import Path

import csv
import statistics
import subprocess
import time

import psutil


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

PROMPT_DIR = (
    PROJECT_ROOT
    / "data"
    / "prompts"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "Third_week"
)


# ============================================================
# llama.cpp
# ============================================================

LLAMA_CPP_DIR = Path(
    r"D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64"
)

LLAMA_CLI = (
    LLAMA_CPP_DIR
    / "llama-cli.exe"
)


# ============================================================
# Model
# ============================================================

MODEL = {
    "quantization": "Q4_K_M",
    "path": (
        MODEL_DIR
        / "Qwen3-1.7B-Q4_K_M.gguf"
    ),
}


# ============================================================
# Fixed workload
# ============================================================

PROMPT_FILE = (
    PROMPT_DIR
    / "prompt_128.txt"
)

PROMPT_TOKENS = 128

GENERATION_TOKENS = 128

GPU_LAYERS = 99


# ============================================================
# Context capacity sweep
# ============================================================

CONTEXT_SIZES = [
    512,
    1024,
    2048,
    4096,
]


# ============================================================
# Benchmark settings
# ============================================================

REPETITIONS = 3

POLL_INTERVAL = 0.05

COOLDOWN_SECONDS = 2


# ============================================================
# GPU memory
# ============================================================

def get_gpu_memory_used():

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    first_gpu = (
        result.stdout
        .strip()
        .splitlines()[0]
    )

    return float(first_gpu)


# ============================================================
# Process RAM
# ============================================================

def get_process_memory_mb(process):

    total = 0.0

    try:

        total += (
            process.memory_info().rss
            / (1024 ** 2)
        )

    except psutil.Error:
        return 0.0

    try:

        children = process.children(
            recursive=True
        )

        for child in children:

            try:

                total += (
                    child.memory_info().rss
                    / (1024 ** 2)
                )

            except psutil.Error:
                pass

    except psutil.Error:
        pass

    return total


# ============================================================
# Run one test
# ============================================================

def run_memory_test(
    model_path,
    context_size,
):

    baseline_vram = (
        get_gpu_memory_used()
    )

    command = [
        str(LLAMA_CLI),

        "-m",
        str(model_path),

        "-ngl",
        str(GPU_LAYERS),

        "-c",
        str(context_size),

        "-n",
        str(GENERATION_TOKENS),

        "-st",

        "-f",
        str(PROMPT_FILE),

        "--no-display-prompt",
    ]

    process = subprocess.Popen(
        command,
        cwd=LLAMA_CPP_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ps_process = psutil.Process(
        process.pid
    )

    peak_vram_total = baseline_vram

    peak_ram = 0.0

    while process.poll() is None:

        try:

            current_vram = (
                get_gpu_memory_used()
            )

            peak_vram_total = max(
                peak_vram_total,
                current_vram,
            )

        except Exception:
            pass

        current_ram = (
            get_process_memory_mb(
                ps_process
            )
        )

        peak_ram = max(
            peak_ram,
            current_ram,
        )

        time.sleep(
            POLL_INTERVAL
        )

    peak_vram_delta = max(
        0.0,
        peak_vram_total
        - baseline_vram,
    )

    return {
        "baseline_vram":
            baseline_vram,

        "peak_vram_total":
            peak_vram_total,

        "peak_vram_delta":
            peak_vram_delta,

        "peak_ram":
            peak_ram,
    }


# ============================================================
# Check files
# ============================================================

def check_files():

    if not LLAMA_CLI.exists():

        raise FileNotFoundError(
            f"llama-cli.exe not found:\n"
            f"{LLAMA_CLI}"
        )

    if not MODEL["path"].exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL['path']}"
        )

    if not PROMPT_FILE.exists():

        raise FileNotFoundError(
            f"Prompt file not found:\n"
            f"{PROMPT_FILE}"
        )


# ============================================================
# Run context sweep
# ============================================================

def run_context_sweep():

    results = []

    for context_size in CONTEXT_SIZES:

        print()
        print("=" * 80)

        print(
            f"Model: "
            f"{MODEL['quantization']}"
        )

        print(
            f"Prompt tokens: "
            f"{PROMPT_TOKENS}"
        )

        print(
            f"Generation tokens: "
            f"{GENERATION_TOKENS}"
        )

        print(
            f"Context capacity: "
            f"{context_size}"
        )

        print("=" * 80)

        vram_results = []

        ram_results = []

        for run in range(
            1,
            REPETITIONS + 1,
        ):

            print(
                f"Run "
                f"{run}/{REPETITIONS}...",
                end=" ",
                flush=True,
            )

            result = run_memory_test(
                MODEL["path"],
                context_size,
            )

            vram_results.append(
                result[
                    "peak_vram_delta"
                ]
            )

            ram_results.append(
                result[
                    "peak_ram"
                ]
            )

            print(
                f"VRAM: "
                f"{result['peak_vram_delta']:.0f} MiB, "
                f"RAM: "
                f"{result['peak_ram']:.0f} MiB"
            )

            time.sleep(
                COOLDOWN_SECONDS
            )

        median_vram = (
            statistics.median(
                vram_results
            )
        )

        median_ram = (
            statistics.median(
                ram_results
            )
        )

        results.append(
            {
                "quantization":
                    MODEL["quantization"],

                "prompt_tokens":
                    PROMPT_TOKENS,

                "generation_tokens":
                    GENERATION_TOKENS,

                "context_size":
                    context_size,

                "gpu_layers":
                    GPU_LAYERS,

                "repetitions":
                    REPETITIONS,

                "peak_vram_mib":
                    median_vram,

                "peak_ram_mib":
                    median_ram,
            }
        )

    return results


# ============================================================
# Save results
# ============================================================

def save_results(results):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULT_DIR
        / "context_memory_sweep_q4.csv"
    )

    fieldnames = [
        "quantization",
        "prompt_tokens",
        "generation_tokens",
        "context_size",
        "gpu_layers",
        "repetitions",
        "peak_vram_mib",
        "peak_ram_mib",
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

        writer.writerows(
            results
        )

    print()
    print("=" * 80)
    print("Context Capacity Sweep")
    print("=" * 80)

    print(
        f"{'Context':<12}"
        f"{'Peak VRAM':<18}"
        f"{'Peak RAM':<18}"
    )

    print("-" * 48)

    for result in results:

        print(
            f"{result['context_size']:<12}"
            f"{result['peak_vram_mib']:<18.0f}"
            f"{result['peak_ram_mib']:<18.0f}"
        )

    print()
    print(
        f"Results saved to:\n"
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    check_files()

    results = (
        run_context_sweep()
    )

    save_results(
        results
    )


if __name__ == "__main__":
    main()