# On-device LLM Benchmark

## Goal

Benchmark Qwen3-1.7B under different quantization settings.

## Hardware

- CPU: Intel Core i5-12490F
- GPU: NVIDIA GeForce RTX 4060 Ti 8GB
- RAM: 32GB
- OS: Windows x64

## Planned experiments

- FP16 / BF16 baseline
- INT8 / Q8
- INT4 / Q4
- Compare:
  - Model size
  - Peak RAM / VRAM
  - TTFT
  - TPOT
  - Token/s
  - Quality