from pathlib import Path
import subprocess
import re
import csv
from datetime import datetime


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_DIR = PROJECT_ROOT / "data" / "prompts"

RESULT_DIR = PROJECT_ROOT / "results" / "raw"


# ============================================================
# llama.cpp configuration
# ============================================================

LLAMA_CLI = Path(
    r"D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64\llama-cli.exe"
)

MODEL_PATH = Path(
    r"D:\code\relavate_file\Qwen3-1.7B-Q4_K_M.gguf"
)


# ============================================================
# Benchmark configuration
# ============================================================

CONTEXT_SIZE = 4096
GENERATION_TOKENS = 128
GPU_LAYERS = 99


# 当前只测试一个 workload
PROMPT_CONFIGS = [
    {
        "name": "prompt_128",
        "target_tokens": 128,
        "path": PROMPT_DIR / "prompt_128.txt",
    },
]


# ============================================================
# Check files
# ============================================================

def check_files():

    if not LLAMA_CLI.exists():
        raise FileNotFoundError(
            f"llama-cli.exe not found:\n{LLAMA_CLI}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    for config in PROMPT_CONFIGS:

        if not config["path"].exists():
            raise FileNotFoundError(
                f"Prompt file not found:\n"
                f"{config['path']}"
            )


# ============================================================
# Parse llama.cpp b10892 performance output
# ============================================================

def parse_performance(output):

    # 当前 llama.cpp 输出类似：
    #
    # [ Prompt: 4063.8 t/s | Generation: 168.1 t/s ]

    pattern = re.compile(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s"
        r"\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]"
    )

    match = pattern.search(output)

    if match is None:
        return {
            "prefill_tps": None,
            "decode_tps": None,
        }

    return {
        "prefill_tps": float(match.group(1)),
        "decode_tps": float(match.group(2)),
    }


# ============================================================
# Run one workload
# ============================================================

def run_one_benchmark(config):

    prompt_text = config["path"].read_text(
        encoding="utf-8"
    )

    command = [
        str(LLAMA_CLI),

        "-m",
        str(MODEL_PATH),

        "-p",
        prompt_text,

        "-n",
        str(GENERATION_TOKENS),

        "-c",
        str(CONTEXT_SIZE),

        "-ngl",
        str(GPU_LAYERS),

        # Single turn:
        # 生成一次回答后自动退出，
        # 不继续等待下一轮用户输入。
        "-st",

        "--no-display-prompt",
    ]

    print("=" * 70)
    print(f"Running workload: {config['name']}")
    print(f"Target prompt tokens: {config['target_tokens']}")
    print(f"Generation tokens: {GENERATION_TOKENS}")
    print(f"Context size: {CONTEXT_SIZE}")
    print()
    print("Command started...")
    print()
    print("=" * 70)

    process = subprocess.Popen(
        command,

        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,

        text=True,
        encoding="utf-8",
        errors="replace",

        bufsize=1,
    )

    output_lines = []

    for line in process.stdout:

        print(
            line,
            end="",
            flush=True,
        )

        output_lines.append(line)

    process.wait()

    output = "".join(output_lines)

    if process.returncode != 0:
        raise RuntimeError(
            f"llama.cpp returned "
            f"{process.returncode}"
        )

    metrics = parse_performance(output)

    result = {

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "model":
            MODEL_PATH.name,

        "workload":
            config["name"],

        "target_prompt_tokens":
            config["target_tokens"],

        "generation_tokens":
            GENERATION_TOKENS,

        "context_size":
            CONTEXT_SIZE,

        "gpu_layers":
            GPU_LAYERS,

        "prefill_tps":
            metrics["prefill_tps"],

        "decode_tps":
            metrics["decode_tps"],
    }

    print()
    print("=" * 70)
    print("Parsed result")
    print("=" * 70)

    print(
        f"Target prompt tokens: "
        f"{result['target_prompt_tokens']}"
    )

    print(
        f"Prefill speed: "
        f"{result['prefill_tps']} tok/s"
    )

    print(
        f"Decode speed: "
        f"{result['decode_tps']} tok/s"
    )

    print("=" * 70)

    return result


# ============================================================
# Save CSV
# ============================================================

def save_results(results):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR
        / "prompt_sweep_q4_k_m.csv"
    )

    fieldnames = [
        "timestamp",
        "model",
        "workload",
        "target_prompt_tokens",
        "generation_tokens",
        "context_size",
        "gpu_layers",
        "prefill_tps",
        "decode_tps",
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

        writer.writerows(results)

    print()
    print("=" * 70)
    print("Benchmark finished.")

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    check_files()

    results = []

    for config in PROMPT_CONFIGS:

        result = run_one_benchmark(
            config
        )

        results.append(result)

    save_results(results)


if __name__ == "__main__":
    main()