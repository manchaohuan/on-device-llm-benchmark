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

RAW_DIR = (
    RESULT_DIR
    / "memory_raw"
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
# Current validation model
# ============================================================

MODEL = {
    "quantization": "Q4_K_M",
    "path": (
        MODEL_DIR
        / "Qwen3-1.7B-Q4_K_M.gguf"
    ),
}


# ============================================================
# Workloads
# ============================================================

WORKLOADS = [
    {
        "name": "prompt_128",
        "prompt_tokens": 128,
        "prompt_file": (
            PROMPT_DIR
            / "prompt_128.txt"
        ),
    },

    {
        "name": "prompt_512",
        "prompt_tokens": 512,
        "prompt_file": (
            PROMPT_DIR
            / "prompt_512.txt"
        ),
    },

    {
        "name": "prompt_2048",
        "prompt_tokens": 2048,
        "prompt_file": (
            PROMPT_DIR
            / "prompt_2048.txt"
        ),
    },
]


# ============================================================
# Benchmark configuration
# ============================================================

CONTEXT_SIZE = 4096

GENERATION_TOKENS = 128

GPU_LAYERS = 99

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
            process
            .memory_info()
            .rss
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
                    child
                    .memory_info()
                    .rss
                    / (1024 ** 2)
                )

            except psutil.Error:
                pass

    except psutil.Error:
        pass

    return total


# ============================================================
# Run one memory test
# ============================================================

def run_memory_test(
    model_path,
    prompt_file,
):

    # --------------------------------------------
    # GPU baseline before launching llama.cpp
    # --------------------------------------------

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
        str(CONTEXT_SIZE),

        "-n",
        str(GENERATION_TOKENS),

        # One generation turn then exit
        "-st",

        # Read fixed benchmark prompt
        # directly from file
        "-f",
        str(prompt_file),

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


    # --------------------------------------------
    # Poll while llama.cpp is alive
    # --------------------------------------------

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


    # --------------------------------------------
    # VRAM increase caused by this process
    # --------------------------------------------

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


    for workload in WORKLOADS:

        if not workload[
            "prompt_file"
        ].exists():

            raise FileNotFoundError(
                f"Prompt file not found:\n"
                f"{workload['prompt_file']}"
            )


# ============================================================
# Run workload matrix
# ============================================================

def run_workloads():

    results = []

    model_path = MODEL["path"]

    quantization = (
        MODEL["quantization"]
    )


    for workload in WORKLOADS:

        print()
        print("=" * 80)

        print(
            f"Model: {quantization}"
        )

        print(
            f"Workload: "
            f"{workload['name']}"
        )

        print(
            f"Prompt tokens: "
            f"{workload['prompt_tokens']}"
        )

        print(
            f"Context size: "
            f"{CONTEXT_SIZE}"
        )

        print(
            f"Generation: "
            f"{GENERATION_TOKENS}"
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
                model_path,
                workload[
                    "prompt_file"
                ],
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


        # --------------------------------------------
        # Median across repetitions
        # --------------------------------------------

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
                    quantization,

                "prompt_tokens":
                    workload[
                        "prompt_tokens"
                    ],

                "context_size":
                    CONTEXT_SIZE,

                "generation_tokens":
                    GENERATION_TOKENS,

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
# Save CSV
# ============================================================

def save_results(results):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        RESULT_DIR
        / "memory_workload_q4.csv"
    )


    fieldnames = [
        "quantization",
        "prompt_tokens",
        "context_size",
        "generation_tokens",
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

    print("Final Results")

    print("=" * 80)


    print(
        f"{'Prompt':<12}"
        f"{'Peak VRAM':<18}"
        f"{'Peak RAM':<18}"
    )


    print("-" * 48)


    for result in results:

        print(
            f"{result['prompt_tokens']:<12}"
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

    results = run_workloads()

    save_results(
        results
    )


if __name__ == "__main__":
    main()