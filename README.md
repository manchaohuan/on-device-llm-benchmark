# on-device-llm-banchmark

一个基于 `llama.cpp` 的本地大模型推理 Benchmark 项目，用于研究不同量化方式和 Context 配置对模型质量、运行时内存以及 Prefill / Decode 性能的影响。

> 项目目录当前实际名称为 `on-device-llm-banchmark`，因此仓库中继续保留该拼写。

---

## 1. Project Goal

本项目的目标是构建一套**可复现的本地 LLM Inference Benchmark**，通过真实实验数据分析端侧大模型部署中的核心 Trade-off：

- Quantization 与模型文件大小之间的关系
- Quantization 与模型质量之间的关系
- Quantization 与运行时 RAM / VRAM 之间的关系
- Quantization 对 Prefill 与 Decode 的不同影响
- Prompt Length 对 Prefill Workload 的影响
- Context Capacity 对运行时显存的影响
- Current Context Depth 对 Decode 性能的影响

本项目不只关注：

> 哪个量化模型最快？

还希望回答：

> 为什么 Model Size、Runtime Memory、Quality 和 Performance 不能被视为同一个指标？

以及：

> 为什么 Prefill 和 Decode 应该被视为两个不同的性能问题？

最终目标是从“能够运行 llama.cpp”进一步提升到：

```text
设计 Benchmark
→ 控制变量
→ 保存原始结果
→ 重复实验
→ 分析数据
→ 形成工程结论
→ 使用 Profiler 验证机制假设
```

---

## 2. Benchmark Questions

当前项目主要围绕以下问题展开：

1. F16、Q8_0 和 Q4_K_M 能将模型文件压缩到什么程度？
2. 量化以后，WikiText-2 Perplexity 会发生怎样的变化？
3. 模型文件变小以后，运行时 Peak RAM / VRAM 是否会按照相同比例下降？
4. Quantization 对 Prefill 和 Decode 的影响是否相同？
5. Prompt Length 增加以后，Prefill Throughput 和 Prefill Latency 如何变化？
6. Current Context Depth 增加以后，Decode Throughput 和 TPOT 如何变化？
7. Context Capacity 增加以后，Peak VRAM 如何变化？
8. 哪些结论属于实验事实，哪些解释仍然需要 Profiler 证据？

---

## 3. Environment

### Hardware

```text
GPU: NVIDIA GeForce RTX 4060 Ti
VRAM: 8187 MiB
Compute Capability: 8.9
```

### Operating System

```text
Windows
```

WSL2 / Linux 环境将在后续阶段单独测试。

### Runtime

```text
Runtime: llama.cpp
Build: b10892
Git revision: e5a8d439c
```

llama.cpp 路径：

```text
D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64
```

主要使用：

```text
llama-cli.exe
llama-bench.exe
llama-perplexity.exe
```

默认 GPU Offload：

```text
-ngl 99
```

---

## 4. Models

当前比较 Qwen3-1.7B 的三种 GGUF 模型：

| Quantization | Model File |
|---|---|
| F16 | `Qwen3-1.7B-f16.gguf` |
| Q8_0 | `Qwen3-1.7B-Q8_0.gguf` |
| Q4_K_M | `Qwen3-1.7B-Q4_K_M.gguf` |

模型目录：

```text
models/
├── Qwen3-1.7B-f16.gguf
├── Qwen3-1.7B-Q8_0.gguf
└── Qwen3-1.7B-Q4_K_M.gguf
```

---

## 5. Project Structure

当前项目主要结构：

```text
on-device-llm-banchmark/
│
├── configs/
│
├── data/
│   └── prompts/
│       ├── base_prompt.txt
│       ├── prompt_128.txt
│       ├── prompt_512.txt
│       └── prompt_2048.txt
│
├── docs/
│   └── benchmark_methodology.md
│
├── models/
│   ├── Qwen3-1.7B-f16.gguf
│   ├── Qwen3-1.7B-Q8_0.gguf
│   └── Qwen3-1.7B-Q4_K_M.gguf
│
├── notes/
│
├── results/
│   └── Third_week/
│       ├── raw/
│       ├── benchmark_matrix.csv
│       ├── unified_summary.csv
│       └── analysis.md
│
├── scripts/
│   └── Third_week/
│       ├── generate_prompts.py
│       ├── run_benchmark.py
│       ├── run.matrix.py
│       ├── build_summary.py
│       ├── plot_results.py
│       ├── run_memory_matrix.py
│       └── run_context_memory_sweep.py
│
├── README.md
└── requirements.txt
```

---

## 6. Benchmark Methodology

完整实验方法见：

```text
docs/benchmark_methodology.md
```

Benchmark 遵循以下原则：

```text
控制变量
+
保存实验配置
+
保存 Raw Results
+
重复实验
+
统计实验波动
+
区分实验事实与机制假设
```

---

## 7. Important Terminology

### 7.1 Prefill

Prefill 是模型处理输入 Prompt 的阶段。

主要观察：

```text
Prefill Throughput
```

单位：

```text
tokens/s
```

当前 Prefill Latency 根据 Mean Prefill Throughput 计算：

```text
Prefill Latency (ms)
=
Prompt Length
/
Mean Prefill Throughput
×
1000
```

---

### 7.2 Decode

Decode 是模型自回归生成 Token 的阶段。

主要观察：

```text
Decode Throughput
TPOT
```

TPOT：

```text
Time Per Output Token
```

当前计算：

```text
TPOT (ms/token)
=
1000
/
Mean Decode Throughput
```

---

### 7.3 Context Capacity

例如：

```text
-c 4096
```

表示 Runtime 配置允许使用的最大 Context Capacity。

主要影响：

```text
Memory Allocation
```

---

### 7.4 Prompt Length

表示 Prefill 阶段实际输入的 Token 数量。

当前测试：

```text
128
512
2048
```

主要影响：

```text
Prefill Workload
Prefill Latency
```

---

### 7.5 Current Context Depth

表示 Decode 开始时已经存在的历史 Token 数量。

当前测试：

```text
d128
d512
d2048
```

主要影响：

```text
Attention
KV Cache Access
Decode Performance
```

因此必须区分：

```text
Prompt Length
!=
Context Capacity
!=
Current Context Depth
```

---

## 8. Statistical Methodology

### Performance Benchmark

Prefill 和 Decode 使用：

```text
Repetitions = 5
```

Benchmark 脚本将：

```text
-r 5
```

直接传递给 `llama-bench`。

数据链：

```text
run.matrix.py
    ↓
llama-bench -r 5
    ↓
5 次重复
    ↓
Mean ± Standard Deviation
    ↓
benchmark_matrix.csv
```

保存字段包括：

```text
prefill_tps_mean
prefill_tps_std

decode_tps_mean
decode_tps_std
```

最终：

```text
build_summary.py
```

主要读取 Mean 并生成：

```text
unified_summary.csv
```

因此：

```text
Performance
=
5 repetitions
→ Mean
+
Standard Deviation
```

### Memory Benchmark

Memory Benchmark 使用：

```text
Repetitions = 3
```

最终报告：

```text
Median
```

因此当前统计规则为：

| Benchmark | Repetitions | Main Statistic | Variability |
|---|---:|---|---|
| Memory | 3 | Median | - |
| Prefill | 5 | Mean | Standard Deviation |
| Decode | 5 | Mean | Standard Deviation |

---

# 9. Results

## 9.1 Model Size

| Quantization | Model Size | Reduction vs F16 |
|---|---:|---:|
| F16 | 3.790 GiB | 0% |
| Q8_0 | 1.708 GiB | 54.9% |
| Q4_K_M | 1.194 GiB | 68.5% |

### Observation

模型量化显著降低 GGUF 文件大小。

其中：

```text
Q8_0:
约减少 54.9%

Q4_K_M:
约减少 68.5%
```

但是：

```text
Model Size
!=
Runtime Memory
```

因此不能直接根据 GGUF 文件大小推断实际 RAM / VRAM。

---

## 9.2 Quality — WikiText-2 Perplexity

| Quantization | Perplexity |
|---|---:|
| F16 | 14.702900 |
| Q8_0 | 14.699600 |
| Q4_K_M | 15.937400 |

### Observation

Q8_0：

```text
14.699600
```

与 F16：

```text
14.702900
```

非常接近。

在当前 WikiText-2 Benchmark 中，Q8_0 基本保持了 F16 的 Perplexity。

Q4_K_M：

```text
15.937400
```

相比 F16 有更加明显的 Perplexity 增长。

因此当前实验表现出：

```text
Q8_0
→ 较大的文件压缩
→ 当前 PPL 基本保持

Q4_K_M
→ 更大的文件压缩
→ 当前 PPL 损失更加明显
```

需要注意：

WikiText-2 Perplexity 不能全面代表 Instruction Following、Reasoning、Coding 等真实任务质量。

---

## 9.3 Runtime Memory

配置：

```text
Context Size = 2048
Generation = 128
GPU Offload = 99
Repetitions = 3
Statistic = Median
```

结果：

| Quantization | Peak VRAM | Peak RAM |
|---|---:|---:|
| F16 | 3713 MiB | 3780 MiB |
| Q8_0 | 2166 MiB | 2163 MiB |
| Q4_K_M | 1468 MiB | 1476 MiB |

相对于 F16：

```text
Q8_0:
Peak VRAM 约下降 41.7%
Peak RAM 约下降 42.8%

Q4_K_M:
Peak VRAM 约下降 60.5%
Peak RAM 约下降 61.0%
```

### Observation

Quantization 显著降低 Runtime Memory。

但是运行时 Memory Reduction 小于 GGUF File Size Reduction。

这说明：

```text
模型权重大小
```

不是 Runtime Memory 的唯一组成部分。

运行时内存还可能包含：

```text
KV Cache
Compute Buffers
Runtime Buffers
Temporary Allocations
CUDA / Backend Allocations
```

因此：

```text
GGUF Size Reduction
!=
Runtime Memory Reduction
```

---

# 10. Prefill Performance

Prefill Prompt Length：

```text
128
512
2048
```

Performance Benchmark：

```text
Repetitions = 5
GPU Offload = 99
```

## 10.1 Prefill Throughput

单位：

```text
tokens/s
```

| Prompt Length | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 6013.39 | 7917.30 | 8821.62 |
| 512 | 12277.48 | 13937.92 | 13308.85 |
| 2048 | 12756.72 | 13742.64 | 13369.17 |

## 10.2 Prefill Latency

| Prompt Length | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 21.286 ms | 16.167 ms | 14.510 ms |
| 512 | 41.702 ms | 36.734 ms | 38.471 ms |
| 2048 | 160.543 ms | 149.025 ms | 153.188 ms |

### Key Observation

短 Prompt：

```text
Q4_K_M
```

具有最高 Prefill Throughput。

例如：

```text
pp128

F16      6013.39
Q8_0     7917.30
Q4_K_M   8821.62
```

但是在更长 Prompt 下：

```text
Q8_0
```

略高于 Q4_K_M。

例如：

```text
pp512

F16      12277.48
Q8_0     13937.92
Q4_K_M   13308.85
```

因此当前实验说明：

```text
更小的模型
!=
所有 Workload 下一定更快
```

Performance 还可能受到：

```text
Weight Memory Traffic
Dequantization
Kernel / Backend Implementation
Compute Efficiency
Memory Bandwidth
```

共同影响。

目前这些只能作为机制候选。

在没有 Profiler 证据前，不能断言具体是哪一个机制导致 Q8_0 与 Q4_K_M 的 Prefill 差异。

---

# 11. Decode Performance

Generation Length 固定：

```text
128
```

改变 Decode 开始时的 Current Context Depth：

```text
128
512
2048
```

---

## 11.1 Decode Throughput

单位：

```text
tokens/s
```

| Context Depth | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 71.52 | 124.17 | 188.68 |
| 512 | 70.41 | 120.73 | 180.55 |
| 2048 | 67.09 | 111.62 | 161.27 |

---

## 11.2 TPOT

单位：

```text
ms/token
```

| Context Depth | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 13.982 | 8.053 | 5.300 |
| 512 | 14.203 | 8.283 | 5.539 |
| 2048 | 14.905 | 8.959 | 6.201 |

### Key Observation

当前全部 Decode Workload 中：

```text
Q4_K_M > Q8_0 > F16
```

即 Q4_K_M 获得最高 Decode Throughput 和最低 TPOT。

当前实验还显示：

> Quantization 对 Decode 的加速幅度明显大于其在长 Prompt Prefill 中表现出的加速幅度。

一种合理的工程直觉是：

Decode 每一步只生成一个 Token，但每层仍需要访问大量模型权重。

因此 Decode 更容易受到：

```text
Weight Memory Traffic
Memory Bandwidth
```

影响。

低比特量化可以减少权重数据量。

不过具体瓶颈仍需要后续 Profiler 验证。

---

# 12. Context Depth → Decode Performance

当 Current Context Depth 从：

```text
128
```

增加到：

```text
2048
```

Decode Throughput：

```text
F16:
71.52 → 67.09

Q8_0:
124.17 → 111.62

Q4_K_M:
188.68 → 161.27
```

可以观察到：

```text
Context Depth ↑
→ Decode Throughput ↓
→ TPOT ↑
```

一个合理的系统解释是：

```text
Context Depth 增加
→ 历史 KV 状态增加
→ Attention / KV Cache Access 工作增加
→ Decode 成本上升
```

但是 Context Depth 从 128 增加到 2048：

```text
16 倍
```

并不会导致 Decode Time 增加 16 倍。

因为 Decode Forward 中还有大量工作不会按照 Context Depth 同比例增长，例如：

```text
Q/K/V Projection
Output Projection
FFN / MLP
LayerNorm
Runtime / Kernel Overhead
```

因此可以粗略理解：

```text
T_decode
≈
T_fixed
+
T_context
```

---

# 13. Prompt Length → Memory

该实验固定：

```text
Model = Q4_K_M
Context Capacity = 4096
Generation = 128
GPU Offload = 99
Repetitions = 3
```

改变：

```text
Prompt Length
```

结果：

| Prompt Length | Peak VRAM | Peak RAM |
|---:|---:|---:|
| 128 | 1700 MiB | 1478 MiB |
| 512 | 1697 MiB | 1482 MiB |
| 2048 | 1697 MiB | 1485 MiB |

### Observation

Prompt Length：

```text
128 → 2048
```

时 Peak VRAM 基本没有变化：

```text
1700 → 1697 MiB
```

因此当前实验说明：

> 当 Context Capacity 固定以后，实际 Prompt Length 不是 Peak Allocated VRAM 的主要决定因素。

---

# 14. Context Capacity → Memory

固定：

```text
Model = Q4_K_M
Prompt Length = 128
Generation = 128
GPU Offload = 99
Repetitions = 3
```

改变：

```text
Context Capacity
```

结果：

| Context Capacity | Peak VRAM | Peak RAM |
|---:|---:|---:|
| 512 | 1301 MiB | 1474 MiB |
| 1024 | 1357 MiB | 1474 MiB |
| 2048 | 1471 MiB | 1476 MiB |
| 4096 | 1698 MiB | 1478 MiB |

Peak VRAM：

```text
512  → 1301 MiB
1024 → 1357 MiB
2048 → 1471 MiB
4096 → 1698 MiB
```

增加量约为：

```text
512 → 1024:
+56 MiB

1024 → 2048:
+114 MiB

2048 → 4096:
+227 MiB
```

当前结果表现出近似关系：

```text
Peak VRAM
≈
Fixed Memory
+
Context-dependent GPU Memory
```

但是不能直接认为：

```text
增加的全部显存
=
KV Cache
```

因为 Context-dependent Memory 还可能包含：

```text
KV Cache
Context-related Compute Buffers
Runtime Allocations
Temporary Buffers
```

需要进一步 Profiler / Allocator 级证据才能进行准确拆分。

---

# 15. Benchmark Repeatability

Context Capacity Memory Sweep 中：

旧实验：

```text
Context = 2048
Q4_K_M Peak VRAM ≈ 1468 MiB
```

新的独立实验：

```text
Context = 2048
Q4_K_M Peak VRAM ≈ 1471 MiB
```

差异约：

```text
3 MiB
```

说明在当前测试条件下，这组 Peak VRAM 测量具有较好的重复性。

---

# 16. Key Findings

当前阶段的主要实验发现如下。

### 1. Quantization 可以显著降低模型文件大小

```text
Q8_0:
-54.9%

Q4_K_M:
-68.5%
```

---

### 2. 模型文件大小下降不等于 Runtime Memory 同比例下降

例如 Q4_K_M：

```text
Model Size:
约 -68.5%

Peak VRAM:
约 -60.5%
```

说明 Runtime Memory 不只由 Model Weights 决定。

---

### 3. Q8_0 在当前 WikiText-2 PPL 测试中基本保持 F16 质量

```text
F16:
14.702900

Q8_0:
14.699600
```

而 Q4_K_M：

```text
15.937400
```

表现出更加明显的 PPL 变化。

---

### 4. Quantization 对 Decode 的影响明显

当前所有 Decode Workload 中：

```text
Q4_K_M > Q8_0 > F16
```

---

### 5. 更小的模型并不保证所有 Prefill Workload 下都更快

短 Prompt 下：

```text
Q4_K_M
```

最快。

而较长 Prompt 下：

```text
Q8_0
```

略快于 Q4_K_M。

因此：

```text
Model Size ↓
```

不能简单推导出：

```text
Performance ↑
```

---

### 6. Context Depth 增加会降低 Decode Throughput

```text
Context Depth ↑
→ Decode Throughput ↓
→ TPOT ↑
```

但性能下降并不会和 Context Depth 等比例增长。

---

### 7. Context Capacity 对 Peak VRAM 的影响明显大于实际 Prompt Length

固定 Context Capacity 时：

```text
Prompt Length:
128 → 2048

Peak VRAM:
基本不变
```

而增加 Context Capacity：

```text
512 → 4096
```

会明显提高 Peak VRAM。

---

# 17. Experimental Facts vs Mechanism Hypotheses

本项目明确区分：

```text
Experimental Fact
```

和：

```text
Mechanism Hypothesis
```

例如：

> Q8_0 在 pp512 和 pp2048 下的 Prefill Throughput 高于 Q4_K_M。

这是 Benchmark 数据直接支持的实验事实。

但是：

> 某个具体 CUDA Kernel 导致 Q8_0 更快。

目前没有 Profiler 证据支持。

可能的性能影响因素包括：

```text
Weight Memory Traffic
Memory Bandwidth
Dequantization
Kernel Implementation
Backend Implementation
Compute Efficiency
KV Cache Access
Runtime Overhead
```

在 Nsight Systems / Nsight Compute 等 Profiling 完成以前，这些都应被视为待验证的机制假设。

---

# 18. How to Run

进入项目：

```powershell
cd D:\code\on-device-llm-banchmark
```

激活当前环境：

```powershell
conda activate pytorch_env
```

---

## Run Performance Matrix

运行 Prefill + Decode Performance Benchmark：

```powershell
python scripts/Third_week/run.matrix.py
```

脚本会比较：

```text
F16
Q8_0
Q4_K_M
```

Workload：

```text
128
512
2048
```

并将 Raw llama-bench 输出保存到：

```text
results/Third_week/raw/
```

结构化结果保存到：

```text
results/Third_week/benchmark_matrix.csv
```

---

## Build Unified Summary

运行：

```powershell
python scripts/Third_week/build_summary.py
```

生成：

```text
results/Third_week/unified_summary.csv
```

该文件将：

```text
Model Size
+
Memory
+
Prefill
+
Decode
```

整合到统一结果表。

---

## Memory Sweep

Prompt Length Memory Sweep：

```powershell
python scripts/Third_week/run_memory_matrix.py
```

Context Capacity Memory Sweep：

```powershell
python scripts/Third_week/run_context_memory_sweep.py
```

---

## Generate Plots

```powershell
python scripts/Third_week/plot_results.py
```

当前已经生成并确认过的主要性能图包括：

```text
Prefill Throughput
Decode Throughput
```

---

# 19. Raw Results and Analysis

主要结构化 Performance 数据：

```text
results/Third_week/benchmark_matrix.csv
```

项目统一结果：

```text
results/Third_week/unified_summary.csv
```

原始 llama-bench 输出：

```text
results/Third_week/raw/
```

第三周分析：

```text
results/Third_week/analysis.md
```

详细 Benchmark 方法：

```text
docs/benchmark_methodology.md
```

---

# 20. Measurement Noise

当前 Benchmark 可能受到以下因素影响：

```text
Background GPU Applications
Background CPU Activity
OS Scheduling
CUDA Runtime Initialization
GPU Power State
GPU Temperature
Memory Allocator
Runtime Initialization
Sampling Granularity
```

Memory Benchmark 中，每次运行前都会重新测量 GPU Memory Baseline。

报告：

```text
Peak VRAM Delta
=
Peak Total GPU Memory
-
Pre-run GPU Memory Baseline
```

以降低后台 GPU Memory 占用对结果的影响。

同时，由于当前 VRAM 使用外部 Polling 方式采集，非常短暂的显存峰值可能无法被完整捕获。

因此当前 Peak VRAM 应理解为：

> 当前采样方式观察到的最大 GPU Memory Usage。

而不是 Allocator 层面的理论绝对峰值。

---

# 21. Limitations

## Single GPU

当前结果来自：

```text
NVIDIA GeForce RTX 4060 Ti
```

不能直接推广到所有 GPU。

---

## Single Model Family

当前主要研究：

```text
Qwen3-1.7B
```

因此结果首先适用于：

```text
当前模型
+
当前 llama.cpp Runtime
+
当前 RTX 4060 Ti
```

组合。

---

## Runtime Specific

当前 Runtime：

```text
llama.cpp
```

其他 Runtime 可能具有不同：

```text
Kernel
Backend
Memory Allocator
Quantization Implementation
```

因此不能直接认为其他 Runtime 会表现出完全相同的结果。

---

## Limited Quality Evaluation

当前质量测试主要使用：

```text
WikiText-2 Perplexity
```

不能全面衡量：

```text
Instruction Following
Reasoning
Coding
Long-context Ability
Real-world Application Quality
```

---

## No Profiler Evidence Yet

当前可以准确回答：

```text
哪个配置更快？
```

但是对于：

```text
为什么更快？
```

目前只有系统层面的合理假设。

后续需要：

```text
Nsight Systems
Nsight Compute
perf
```

等工具进一步验证。

---

## Memory Breakdown Is Not Available Yet

当前 Peak VRAM 无法精确拆分成：

```text
Weights
KV Cache
Compute Buffers
Runtime Buffers
Temporary Allocations
```

因此不能把所有 Context-related VRAM Growth 直接归因于 KV Cache。

---

# 22. Current Status

当前 Project 1 已完成：

- Model Size Benchmark
- Quality / Perplexity Benchmark
- Peak RAM / VRAM Benchmark
- Prefill Benchmark
- Decode Benchmark
- Prompt Length Sweep
- Context Depth Sweep
- Context Capacity Memory Sweep
- Performance Mean / Std 统计
- Benchmark Methodology v1
- 第一版结果分析

当前正在进行：

```text
Project 1 v1 收口
```

包括：

```text
README
Experiment Report
Reproducibility
WSL2 / Linux llama.cpp Build
```

---

# 23. Next Steps

后续计划包括：

1. 完成 `experiment_report.md`
2. 在 WSL2 / Linux 下编译并运行 llama.cpp
3. 学习 llama.cpp Model Loading / Graph / Backend 调用链
4. 开始使用 `perf`
5. 使用 Nsight Systems 分析 CPU / GPU Timeline
6. 使用 Nsight Compute 分析关键 Kernel
7. 对当前 Q8_0 / Q4_K_M 的性能差异建立并验证机制假设
8. 完成至少一次：

```text
Hypothesis
→ Benchmark
→ Profile
→ Bottleneck
→ Change
→ Re-benchmark
```

真实性能分析 / 优化闭环。

---

# 24. Core Benchmark Principle

本项目遵循：

```text
Measure first.
Control variables.
Preserve raw data.
Repeat experiments.
Quantify variability.
Separate observation from explanation.
Use profiler evidence before claiming low-level causes.
```

即：

```text
先测量
↓
控制变量
↓
保存原始结果
↓
重复实验
↓
量化实验波动
↓
区分实验事实与机制解释
↓
使用 Profiler 证据验证底层原因
```

最终目标不是制作一个简单的性能排行榜，而是理解：

```text
Quantization
×
Quality
×
Memory
×
Prefill
×
Decode
×
Context
```

之间的真实工程 Trade-off。