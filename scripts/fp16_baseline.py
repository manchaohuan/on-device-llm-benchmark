import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = "cuda"
DTYPE = torch.float16

print("CUDA available:", torch.cuda.is_available())
print("GPU:",torch.cuda.get_device_name(0))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

PROMPT = "Explain neural network quantization in one short paragraph."

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)

print("Input IDs:", inputs["input_ids"])
print("Input token count:",inputs["input_ids"].shape[-1])

inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

import time

MAX_NEW_TOKENS = 64

print("Warming up ...")

with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
    )

torch.cuda.synchronize()

import statistics

RUNS = 5

elapsed_list = []
tps_list = []
vram_list = []

for i in range(RUNS):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    strat = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        ) 
    torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed = end - strat

    input_token_count = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_token_count:]
    output_token_count = generated_ids.shape[-1]

    tokens_per_second = output_token_count / elapsed

    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3))

    elapsed_list.append(elapsed)
    tps_list.append(tokens_per_second)
    vram_list.append(peak_vram_gb)

    print(
        f"Run {i + 1}: "
        f"{elapsed:.3f}s, "
        f"{tokens_per_second:.3f} token/s, "
        f"{peak_vram_gb:.3f} GB"
    )

avg_elapsed = statistics.mean(elapsed_list)
avg_tps = statistics.mean(tps_list)
median_tps = statistics.median(tps_list)
avg_vram = statistics.mean(vram_list)

print("\n=== FP16 Benchmark Summary ===")
print(f"Average elapsed: {avg_elapsed:.3f} s")
print(f"Average tokens/s: {avg_tps:.3f}")
print(f"Median tokens/s: {median_tps:.3f}")
print(f"Average Peak VRAM: {avg_vram:.3f} GB")

result = {
    "model": MODEL_NAME,
    "precision": "FP16",
    "gpu": torch.cuda.get_device_name(0),
    "runs": RUNS,
    "input_tokens": input_token_count,
    "output_tokens": output_token_count,
    "average_elapsed_s": round(avg_elapsed, 3),
    "average_tokens_per_second": round(avg_tps, 3),
    "median_tokens_per_second": round(median_tps, 3),
    "average_peak_vram_gb": round(avg_vram, 3),
}