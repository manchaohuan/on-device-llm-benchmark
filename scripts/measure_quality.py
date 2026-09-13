from pathlib import Path
import subprocess
import re


PROJECT_ROOT = Path(r"D:\code\on-device-llm-banchmark")

MODEL_DIR = PROJECT_ROOT / "models"
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "wikitext-2-raw"
    / "wiki.test.raw"
)

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "quality.md"

LLAMA_CPP_DIR = Path(
    r"D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64"
)

LLAMA_PERPLEXITY = LLAMA_CPP_DIR / "llama-perplexity.exe"


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


def run_perplexity(model_path):
    command = [
        str(LLAMA_PERPLEXITY),
        "-m",
        str(model_path),
        "-f",
        str(DATA_FILE),
        "-ngl",
        "99",
        "-c",
        "2048",
        "-b",
        "512",
    ]

    print("Running:")
    print(" ".join(command))
    print()

    process = subprocess.run(
        command,
        cwd=LLAMA_CPP_DIR,
        capture_output=True,
        text=True,
    )

    output = process.stdout + "\n" + process.stderr

    match = re.search(
        r"Final estimate:\s*PPL\s*=\s*([0-9.]+)",
        output,
    )

    if match is None:
        print(output)
        raise RuntimeError(
            f"Could not find PPL result for {model_path.name}"
        )

    return float(match.group(1))


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not LLAMA_PERPLEXITY.exists():
        raise FileNotFoundError(
            f"llama-perplexity.exe not found:\n{LLAMA_PERPLEXITY}"
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"WikiText-2 test file not found:\n{DATA_FILE}"
        )

    results = []

    print()
    print("Quality Benchmark - Perplexity")
    print("=" * 70)

    for model in MODELS:
        model_path = model["path"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        print()
        print(
            f"Testing {model['quantization']}"
        )
        print("-" * 70)

        ppl = run_perplexity(model_path)

        print(
            f"PPL: {ppl:.6f}"
        )

        results.append(
            {
                "quantization": model["quantization"],
                "ppl": ppl,
            }
        )

    f16_ppl = next(
        result["ppl"]
        for result in results
        if result["quantization"] == "F16"
    )

    for result in results:
        result["delta_ppl"] = (
            result["ppl"] - f16_ppl
        )

        result["relative_loss"] = (
            (
                result["ppl"] / f16_ppl
            )
            - 1
        ) * 100

    print()
    print("Final Results")
    print("=" * 70)

    print(
        f"{'Quantization':<16}"
        f"{'PPL':<16}"
        f"{'Delta PPL':<16}"
        f"{'Relative Loss':<16}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['quantization']:<16}"
            f"{result['ppl']:<16.6f}"
            f"{result['delta_ppl']:<16.6f}"
            f"{result['relative_loss']:<15.3f}%"
        )

    markdown = [
        "# Qwen3-1.7B Quantization - Quality",
        "",
        "## Configuration",
        "",
        "- Dataset: WikiText-2 test",
        "- Runtime: llama.cpp",
        "- Metric: Perplexity (PPL)",
        "- GPU offload: 99 layers",
        "- Context size: 2048",
        "- Batch size: 512",
        "- Baseline: F16",
        "",
        "## Results",
        "",
        "| Quantization | PPL | Delta PPL | Relative PPL Increase |",
        "|---|---:|---:|---:|",
    ]

    for result in results:
        markdown.append(
            f"| {result['quantization']} "
            f"| {result['ppl']:.6f} "
            f"| {result['delta_ppl']:.6f} "
            f"| {result['relative_loss']:.3f}% |"
        )

    markdown.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower PPL is better.",
            "- F16 is used as the reference baseline.",
            "- A small PPL increase means quantization introduces little prediction-quality loss.",
            "",
        ]
    )

    OUTPUT_FILE.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()