# Qwen3-1.7B Quantization - Model Size

## Results

| Quantization | File Size (GiB) | Relative to F16 | Reduction |
|---|---:|---:|---:|
| F16 | 3.790 | 1.000x | 0.0% |
| Q8_0 | 2.016 | 0.532x | 46.8% |
| Q4_K_M | 1.194 | 0.315x | 68.5% |

## Notes

- File size uses GiB (1024^3 bytes).
- F16 is used as the baseline.
- Reduction is calculated relative to the F16 model.
