from pathlib import Path
from transformers import AutoTokenizer


# ============================================================
# Project paths
# ============================================================

# 当前文件：
# on-device-llm-banchmark/scripts/Third_week/generate_prompts.py
#
# parents[0] -> Third_week
# parents[1] -> scripts
# parents[2] -> on-device-llm-banchmark
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_DIR = PROJECT_ROOT / "data" / "prompts"

BASE_PROMPT_PATH = PROMPT_DIR / "base_prompt.txt"


# ============================================================
# Tokenizer configuration
# ============================================================

TOKENIZER_NAME = "Qwen/Qwen3-1.7B"

TARGET_LENGTHS = [
    128,
    512,
    2048,
]


# ============================================================
# Load base prompt
# ============================================================

def load_base_prompt():
    if not BASE_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Base prompt not found:\n{BASE_PROMPT_PATH}"
        )

    text = BASE_PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            f"Base prompt is empty:\n{BASE_PROMPT_PATH}"
        )

    return text


# ============================================================
# Generate fixed-token prompt
# ============================================================

def generate_prompt(
    tokenizer,
    text,
    target_tokens
):
    # 将整篇基础文本转换为 token ids
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    # 基础文本必须足够长
    if len(token_ids) < target_tokens:
        raise ValueError(
            f"Base prompt only contains "
            f"{len(token_ids)} tokens, "
            f"but {target_tokens} tokens are required."
        )

    # 截取目标数量的 token
    selected_ids = token_ids[:target_tokens]

    # token ids -> text
    prompt = tokenizer.decode(
        selected_ids,
        skip_special_tokens=True
    )

    # 再次 tokenize，检查保存后的文本实际是多少 token
    actual_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    actual_tokens = len(actual_ids)

    return prompt, actual_tokens


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Qwen3 Workload Prompt Generator")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nPrompt directory:")
    print(PROMPT_DIR)

    print(f"\nBase prompt:")
    print(BASE_PROMPT_PATH)

    # 确保目录存在
    PROMPT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 加载 tokenizer
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True
    )

    print("Tokenizer loaded.")

    # 读取基础文本
    base_prompt = load_base_prompt()

    # 先确认基础文本到底有多少 tokens
    base_token_ids = tokenizer.encode(
        base_prompt,
        add_special_tokens=False
    )

    print(
        f"\nBase prompt token count: "
        f"{len(base_token_ids)}"
    )

    print("\nGenerating workload prompts...\n")

    # 生成不同长度 prompt
    for target_tokens in TARGET_LENGTHS:

        prompt, actual_tokens = generate_prompt(
            tokenizer,
            base_prompt,
            target_tokens
        )

        output_path = (
            PROMPT_DIR
            / f"prompt_{target_tokens}.txt"
        )

        output_path.write_text(
            prompt,
            encoding="utf-8"
        )

        print(
            f"Target: {target_tokens:4d} tokens | "
            f"Actual: {actual_tokens:4d} tokens"
        )

        print(
            f"Saved: {output_path}"
        )

        print("-" * 70)

    print("\nPrompt generation finished.")


if __name__ == "__main__":
    main()