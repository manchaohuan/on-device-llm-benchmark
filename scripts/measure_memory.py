from pathlib import Path
import subprocess
import time
import statistics

import psutil


PROJECT_ROOT = Path(r"D:\code\on-device-llm-banchmark")

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

LLAMA_CPP_DIR = Path(
    r"D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64"
)

LLAMA_CLI = LLAMA_CPP_DIR / "llama-cli.exe"

OUTPUT_FILE = RESULTS_DIR / "memory.md"


MODELS = [
    {
        "quantization": "F16",
        "path": MODEL_DIR / "Qwen3-1.7B-f16.gguf",
    },
    {
        "quantization": "Q8_0",
        "path": MODEL_DIR / "Qwen3-1.7B-Q8_0.gguf",
    },
    {
        "quantization": "Q4_K_M",
        "path": MODEL_DIR / "Qwen3-1.7B-Q4_K_M.gguf",
    },
]


REPETITIONS = 3
POLL_INTERVAL = 0.05


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

    first_gpu = result.stdout.strip().splitlines()[0]

    return float(first_gpu)


def get_process_memory_mb(process):
    total = 0.0

    try:
        total += process.memory_info().rss / (1024 ** 2)
    except psutil.Error:
        return 0.0

    try:
        children = process.children(recursive=True)

        for child in children:
            try:
                total += child.memory_info().rss / (1024 ** 2)
            except psutil.Error:
                pass
    except psutil.Error:
        pass

    return total


def run_memory_test(model_path):
    baseline_vram = get_gpu_memory_used()

    command = [
        str(LLAMA_CLI),
        "-m",
        str(model_path),
        "-ngl",
        "99",
        "-c",
        "2048",
        "-n",
        "128",
        "--single-turn",
        "-p",
        "Explain model quantization briefly. /no_think",
    ]

    process = subprocess.Popen(
        command,
        cwd=LLAMA_CPP_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ps_process = psutil.Process(process.pid)

    peak_vram_total = baseline_vram
    peak_ram = 0.0

    while process.poll() is None:
        try:
            current_vram = get_gpu_memory_used()
            peak_vram_total = max(
                peak_vram_total,
                current_vram,
            )
        except Exception:
            pass

        current_ram = get_process_memory_mb(ps_process)

        peak_ram = max(
            peak_ram,
            current_ram,
        )

        time.sleep(POLL_INTERVAL)

    peak_vram_delta = max(
        0.0,
        peak_vram_total - baseline_vram,
    )

    return {
        "baseline_vram": baseline_vram,
        "peak_vram_total": peak_vram_total,
        "peak_vram_delta": peak_vram_delta,
        "peak_ram": peak_ram,
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Memory Benchmark")
    print("=" * 75)

    results = []

    for model in MODELS:
        quantization = model["quantization"]
        model_path = model["path"]

        if not model_path.exists():
            print()
            print(f"Model not found: {model_path}")
            continue

        print()
        print(f"Testing {quantization}")
        print("-" * 75)

        vram_results = []
        ram_results = []

        for run in range(1, REPETITIONS + 1):
            print(
                f"Run {run}/{REPETITIONS}...",
                end=" ",
                flush=True,
            )

            result = run_memory_test(model_path)

            vram_results.append(
                result["peak_vram_delta"]
            )

            ram_results.append(
                result["peak_ram"]
            )

            print(
                f"VRAM: {result['peak_vram_delta']:.0f} MiB, "
                f"RAM: {result['peak_ram']:.0f} MiB"
            )

            time.sleep(2)

        median_vram = statistics.median(
            vram_results
        )

        median_ram = statistics.median(
            ram_results
        )

        results.append(
            {
                "quantization": quantization,
                "peak_vram": median_vram,
                "peak_ram": median_ram,
            }
        )

    print()
    print("Final Results")
    print("=" * 75)

    print(
        f"{'Quantization':<16}"
        f"{'Peak VRAM (MiB)':<20}"
        f"{'Peak RAM (MiB)':<20}"
    )

    print("-" * 75)

    for result in results:
        print(
            f"{result['quantization']:<16}"
            f"{result['peak_vram']:<20.0f}"
            f"{result['peak_ram']:<20.0f}"
        )

    markdown = [
        "# Qwen3-1.7B Quantization - Memory",
        "",
        "## Configuration",
        "",
        "- Runtime: llama.cpp",
        "- GPU offload: 99 layers",
        "- Context size: 2048",
        "- Generation length: 128 tokens",
        f"- Repetitions: {REPETITIONS}",
        "- VRAM metric: peak GPU memory increase over baseline",
        "- RAM metric: peak process working set",
        "",
        "## Results",
        "",
        "| Quantization | Peak VRAM (MiB) | Peak RAM (MiB) |",
        "|---|---:|---:|",
    ]

    for result in results:
        markdown.append(
            f"| {result['quantization']} "
            f"| {result['peak_vram']:.0f} "
            f"| {result['peak_ram']:.0f} |"
        )

    markdown.extend(
        [
            "",
            "## Notes",
            "",
            "- Each model was tested three times.",
            "- The median value is reported.",
            "- VRAM is measured relative to the GPU memory baseline before each run.",
            "- Background GPU applications may introduce measurement noise.",
            "",
        ]
    )

    OUTPUT_FILE.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()