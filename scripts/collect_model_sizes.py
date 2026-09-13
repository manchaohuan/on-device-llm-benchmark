from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型目录
MODEL_DIR = PROJECT_ROOT / "models"

# 结果目录
RESULTS_DIR = PROJECT_ROOT / "results"

# 输出文件
OUTPUT_FILE = RESULTS_DIR / "model_size.md"


def get_quantization(filename: str) -> str:
    """根据文件名判断量化类型"""
    name = filename.lower()

    if "f16" in name:
        return "F16"
    elif "q8_0" in name:
        return "Q8_0"
    elif "q4_k_m" in name:
        return "Q4_K_M"
    else:
        return "Unknown"


def main():
    # 找到所有 GGUF 文件
    files = list(MODEL_DIR.glob("*.gguf"))

    if not files:
        print(f"No GGUF files found in: {MODEL_DIR}")
        return

    # 找到 F16 baseline
    f16_file = next(
        (file for file in files if "f16" in file.name.lower()),
        None
    )

    if f16_file is None:
        print("F16 baseline model not found.")
        return

    # F16 文件大小（byte）
    f16_size = f16_file.stat().st_size

    results = []

    for file in files:
        size_bytes = file.stat().st_size

        # byte -> GiB
        size_gib = size_bytes / (1024 ** 3)

        # 相对于 F16 的大小
        relative = size_bytes / f16_size

        # 相对于 F16 减少多少
        reduction = (1 - relative) * 100

        quantization = get_quantization(file.name)

        results.append({
            "name": file.name,
            "quantization": quantization,
            "size_gib": size_gib,
            "relative": relative,
            "reduction": reduction,
        })

    # 固定显示顺序
    order = {
        "F16": 0,
        "Q8_0": 1,
        "Q4_K_M": 2,
        "Unknown": 99,
    }

    results.sort(
        key=lambda x: order.get(x["quantization"], 99)
    )

    # 终端输出
    print("\nModel Size Results")
    print("-" * 85)

    print(
        f"{'Quantization':<15}"
        f"{'Size (GiB)':<15}"
        f"{'Relative to F16':<20}"
        f"{'Reduction':<15}"
        f"{'Filename'}"
    )

    print("-" * 85)

    for result in results:
        print(
            f"{result['quantization']:<15}"
            f"{result['size_gib']:<15.3f}"
            f"{result['relative']:<20.3f}"
            f"{result['reduction']:<14.1f}%"
            f"{result['name']}"
        )

    # 创建 results 目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 生成 Markdown 实验表
    markdown = [
        "# Qwen3-1.7B Quantization - Model Size",
        "",
        "## Results",
        "",
        "| Quantization | File Size (GiB) | Relative to F16 | Reduction |",
        "|---|---:|---:|---:|",
    ]

    for result in results:
        markdown.append(
            f"| {result['quantization']} "
            f"| {result['size_gib']:.3f} "
            f"| {result['relative']:.3f}x "
            f"| {result['reduction']:.1f}% |"
        )

    markdown.extend([
        "",
        "## Notes",
        "",
        "- File size uses GiB (1024^3 bytes).",
        "- F16 is used as the baseline.",
        "- Reduction is calculated relative to the F16 model.",
        "",
    ])

    OUTPUT_FILE.write_text(
        "\n".join(markdown),
        encoding="utf-8"
    )

    print()
    print(f"Result saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()