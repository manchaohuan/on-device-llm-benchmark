# Qwen3-1.7B Quantization - Quality

## Configuration

- Dataset: WikiText-2 test
- Runtime: llama.cpp
- Metric: Perplexity (PPL)
- GPU offload: 99 layers
- Context size: 2048
- Batch size: 512
- Baseline: F16

## Results

| Quantization | PPL | Delta PPL | Relative PPL Increase |
|---|---:|---:|---:|
| F16 | 14.702900 | 0.000000 | 0.000% |
| Q8_0 | 14.699600 | -0.003300 | -0.022% |
| Q4_K_M | 15.937400 | 1.234500 | 8.396% |

## Interpretation

- Lower PPL is better.
- F16 is used as the reference baseline.
- A small PPL increase means quantization introduces little prediction-quality loss.

## 分析

在 WikiText-2 困惑度（Perplexity）基准测试中，Q8_0 对 F16 模型质量的保持非常好。

与 F16 相比：

- Q8_0 的 PPL 从 14.7029 变为 14.6996，差异仅为 **-0.022%**。这个差异可以忽略不计，不应据此认为 Q8_0 在整体上优于 F16。

- Q4_K_M 的 PPL 从 14.7029 上升到 15.9374，对应 **8.396%** 的相对增幅。

结果表明，在该基准测试中，Q8_0 几乎没有引入可测量的模型质量下降；而 Q4_K_M 则通过牺牲更明显的一部分模型质量，换取了显著更低的内存需求。

结合内存基准测试结果来看，Q8_0 在困惑度几乎不变的情况下，将峰值显存占用降低了约 **41.7%**；而 Q4_K_M 将峰值显存占用降低了约 **60.5%**，但与此同时，困惑度上升了约 **8.4%**。