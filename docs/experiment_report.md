# Qwen3-1.7B 本地推理量化与 Context Trade-off 实验报告

## 1. 实验背景

在端侧或本地设备上部署大语言模型时，通常需要同时考虑多个互相制约的指标：

Model Size、Runtime Memory、Quality、Prefill Performance、Decode Performance 和 Context Capacity。

量化能够减少模型权重的数据量，但模型文件变小并不意味着运行时内存一定按照相同比例下降，也不意味着所有推理 Workload 都一定变快。

因此，本实验基于 `llama.cpp` 和 Qwen3-1.7B，在 NVIDIA GeForce RTX 4060 Ti 上对 F16、Q8_0 和 Q4_K_M 三种模型进行系统 Benchmark，研究 Quantization 和 Context 对质量、内存与推理性能的影响。

本实验的核心目标不是简单得到一个“最快模型”的排名，而是理解：

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

之间的实际工程 Trade-off。

---

# 2. Experiment Questions

本实验主要回答以下问题。

| 编号 | 实验问题 |
|---|---|
| Q1 | Quantization 能够将模型文件压缩到什么程度？ |
| Q2 | Quantization 对 WikiText-2 Perplexity 有什么影响？ |
| Q3 | 模型文件变小以后，Runtime RAM / VRAM 是否会按照相同比例下降？ |
| Q4 | Quantization 对 Prefill 与 Decode 的性能影响是否相同？ |
| Q5 | Prompt Length 增加时，Prefill 性能如何变化？ |
| Q6 | Current Context Depth 增加时，Decode 性能如何变化？ |
| Q7 | Context Capacity 增加时，Peak VRAM 如何变化？ |
| Q8 | 当前 Benchmark 能证明哪些实验事实，哪些底层原因仍然需要 Profiler 验证？ |

---

# 3. Hypotheses

在实验开始前，建立以下可验证假设。

| 假设 | 内容 |
|---|---|
| H1 | Quantization 会显著降低模型文件大小和 Runtime Memory，但二者的下降比例不一定相同。 |
| H2 | Q8_0 的质量损失可能小于 Q4_K_M。 |
| H3 | Quantization 对 Decode 的性能影响可能比对 Prefill 更明显。 |
| H4 | Current Context Depth 增加会增加 Attention / KV 相关工作，使 Decode Throughput 下降。 |
| H5 | Context Capacity 对 Peak VRAM 的影响可能明显大于固定 Capacity 下实际 Prompt Length 的变化。 |

这些内容在实验开始阶段只是 Hypothesis。

实验结果只能验证可直接观察到的现象。

涉及具体 Kernel、Memory Transaction、Dequantization 或 Backend 实现的底层原因，需要进一步 Profiling 才能确认。

---

# 4. Experimental Environment

## 4.1 Hardware

```text
GPU: NVIDIA GeForce RTX 4060 Ti
VRAM: 8187 MiB
Compute Capability: 8.9
```

## 4.2 Runtime

```text
Runtime: llama.cpp
Build: b10892
Git Revision: e5a8d439c
```

主要程序：

```text
llama-cli.exe
llama-bench.exe
llama-perplexity.exe
```

GPU Offload：

```text
-ngl 99
```

## 4.3 Models

| Quantization | Model |
|---|---|
| F16 | Qwen3-1.7B-f16.gguf |
| Q8_0 | Qwen3-1.7B-Q8_0.gguf |
| Q4_K_M | Qwen3-1.7B-Q4_K_M.gguf |

---

# 5. Experimental Method

详细 Benchmark 方法参见：

```text
docs/benchmark_methodology.md
```

本实验遵循控制变量原则。

Quantization Comparison 中主要改变 Quantization，而保持模型系列、Runtime、Hardware、Workload 和 GPU Offload 等配置一致。

Performance Benchmark 使用：

```text
llama-bench -r 5
```

对每个配置进行 5 次重复，并保存 `llama-bench` 输出的 Mean 和 Standard Deviation。

主要性能结果使用 Mean。

Memory Benchmark 每个配置重复 3 次，最终使用 Median。

---

# 6. Benchmark Terminology

本项目明确区分以下概念。

## 6.1 Prefill

Prefill 是模型处理输入 Prompt 的阶段。

主要指标：

```text
Prefill Throughput
```

单位：

```text
tokens/s
```

当前 Prefill Latency 由 Mean Throughput 派生：

```text
Prefill Latency (ms)
=
Prompt Length
/
Mean Prefill Throughput
×
1000
```

## 6.2 Decode

Decode 是模型进行自回归 Token Generation 的阶段。

主要指标：

```text
Decode Throughput
TPOT
```

其中：

```text
TPOT (ms/token)
=
1000
/
Mean Decode Throughput
```

## 6.3 Context Capacity

Runtime 配置允许使用的最大 Context 容量。

例如：

```text
-c 4096
```

## 6.4 Prompt Length

Prefill 阶段实际输入的 Token 数量。

## 6.5 Current Context Depth

Decode 开始时已经存在的历史 Token 数量。

因此：

```text
Prompt Length
!=
Context Capacity
!=
Current Context Depth
```

---

# 7. Experiment 1 — Model Size

## 7.1 Results

| Quantization | Model Size | Reduction vs F16 |
|---|---:|---:|
| F16 | 3.790 GiB | 0% |
| Q8_0 | 1.708 GiB | 54.9% |
| Q4_K_M | 1.194 GiB | 68.5% |

## 7.2 Analysis

Quantization 显著减少 GGUF 模型文件大小。

Q8_0 相比 F16 减少约：

```text
54.9%
```

Q4_K_M 相比 F16 减少约：

```text
68.5%
```

Q4_K_M 的模型文件只有 F16 的约三分之一。

但是这一结果只说明磁盘上的 Model Weight Representation 更小，不能直接推导运行时 RAM / VRAM 会按照相同比例减少。

因此需要进一步进行 Runtime Memory Benchmark。

---

# 8. Experiment 2 — Quality

Quality Benchmark 使用 WikiText-2 Perplexity。

## 8.1 Results

| Quantization | Perplexity |
|---|---:|
| F16 | 14.702900 |
| Q8_0 | 14.699600 |
| Q4_K_M | 15.937400 |

## 8.2 Analysis

Q8_0：

```text
14.699600
```

与 F16：

```text
14.702900
```

几乎相同。

二者差值约：

```text
-0.0033
```

相对变化约：

```text
-0.022%
```

在当前 WikiText-2 PPL Benchmark 中，没有观察到 Q8_0 相比 F16 的明显质量下降。

Q4_K_M：

```text
15.937400
```

相比 F16 增加：

```text
1.2345
```

相对增加约：

```text
8.396%
```

因此当前实验表现出明显 Trade-off：

```text
Q8_0
→ 较高压缩率
→ 当前 PPL 基本保持

Q4_K_M
→ 更高压缩率
→ 当前 PPL 变化更加明显
```

需要注意，Perplexity 只能表示当前数据集上的语言建模指标。

它不能完整代表：

```text
Instruction Following
Reasoning
Coding
Long-context Ability
Real-world Application Quality
```

因此本文只将其作为 Quantization Quality Trade-off 的基础指标。

---

# 9. Experiment 3 — Runtime Memory

## 9.1 Configuration

```text
Context Size = 2048
Generation = 128
GPU Offload = 99
Repetitions = 3
Statistic = Median
```

## 9.2 Results

| Quantization | Peak VRAM | Peak RAM |
|---|---:|---:|
| F16 | 3713 MiB | 3780 MiB |
| Q8_0 | 2166 MiB | 2163 MiB |
| Q4_K_M | 1468 MiB | 1476 MiB |

相对于 F16：

| Quantization | VRAM Reduction | RAM Reduction |
|---|---:|---:|
| Q8_0 | ≈ 41.7% | ≈ 42.8% |
| Q4_K_M | ≈ 60.5% | ≈ 61.0% |

## 9.3 Analysis

Quantization 同样能够显著降低 Runtime Memory。

但是可以看到：

```text
Q8_0

Model Size Reduction:
54.9%

Peak VRAM Reduction:
约 41.7%
```

以及：

```text
Q4_K_M

Model Size Reduction:
68.5%

Peak VRAM Reduction:
约 60.5%
```

说明：

```text
Model File Size Reduction
!=
Runtime Memory Reduction
```

运行时显存除模型权重外，还可能包含：

```text
KV Cache
Compute Buffers
Runtime Buffers
CUDA / Backend Allocations
Temporary Allocations
```

因此模型压缩比例不能直接等价为 Runtime Memory 节省比例。

---

# 10. Experiment 4 — Prefill Performance

## 10.1 Workload

Prompt Length：

```text
128
512
2048
```

GPU Offload：

```text
99
```

Repetitions：

```text
5
```

## 10.2 Prefill Throughput

单位：

```text
tokens/s
```

| Prompt Length | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 6013.39 | 7917.30 | 8821.62 |
| 512 | 12277.48 | 13937.92 | 13308.85 |
| 2048 | 12756.72 | 13742.64 | 13369.17 |

## 10.3 Prefill Latency

| Prompt Length | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 21.286 ms | 16.167 ms | 14.510 ms |
| 512 | 41.702 ms | 36.734 ms | 38.471 ms |
| 2048 | 160.543 ms | 149.025 ms | 153.188 ms |

## 10.4 Analysis

在短 Prompt：

```text
pp128
```

下：

```text
Q4_K_M > Q8_0 > F16
```

Q4_K_M 获得最高 Prefill Throughput。

但是 Prompt Length 增加以后，结果发生变化。

在：

```text
pp512
pp2048
```

中，Q8_0 均略高于 Q4_K_M。

例如：

```text
pp512

F16      12277.48
Q8_0     13937.92
Q4_K_M   13308.85
```

这说明：

```text
模型更小
!=
所有 Workload 下一定更快
```

Quantization 对性能的影响不仅与权重大小有关，还可能受到：

```text
Weight Memory Traffic
Dequantization Cost
Compute Efficiency
Kernel Implementation
Backend Implementation
Memory Bandwidth
```

影响。

当前 Benchmark 能证明 Q8_0 和 Q4_K_M 在不同 Prompt Workload 下表现不同。

但是尚不能证明导致这些差异的具体底层原因。

---

# 11. Experiment 5 — Decode Performance

## 11.1 Workload

Generation：

```text
128
```

改变 Decode 开始时的 Current Context Depth：

```text
128
512
2048
```

Repetitions：

```text
5
```

## 11.2 Decode Throughput

单位：

```text
tokens/s
```

| Context Depth | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 71.52 | 124.17 | 188.68 |
| 512 | 70.41 | 120.73 | 180.55 |
| 2048 | 67.09 | 111.62 | 161.27 |

## 11.3 TPOT

| Context Depth | F16 | Q8_0 | Q4_K_M |
|---:|---:|---:|---:|
| 128 | 13.982 ms/token | 8.053 ms/token | 5.300 ms/token |
| 512 | 14.203 ms/token | 8.283 ms/token | 5.539 ms/token |
| 2048 | 14.905 ms/token | 8.959 ms/token | 6.201 ms/token |

## 11.4 Analysis

所有当前 Decode Workload 中均观察到：

```text
Q4_K_M > Q8_0 > F16
```

Quantization 对 Decode 的性能影响非常明显。

以 Context Depth = 128 为例：

```text
F16      71.52 tok/s
Q8_0    124.17 tok/s
Q4_K_M  188.68 tok/s
```

与 Prefill 相比，Decode 中低比特 Quantization 的性能优势更加明显。

一个合理的机制假设是：

Decode 每一步生成一个 Token，但每一步 Forward 仍然需要访问大量模型权重。

因此 Decode 可能更加容易受到：

```text
Weight Memory Traffic
Memory Bandwidth
```

影响。

低比特 Quantization 减少模型权重的数据量，因此可能减少每个 Decode Step 的 Weight Traffic。

但是：

> 当前 Benchmark 尚不能证明 Decode 的性能差异完全由 Memory Bandwidth 或 Weight Traffic 导致。

需要后续 Profiler 进一步验证。

---

# 12. Experiment 6 — Context Depth vs Decode

将 Current Context Depth：

```text
128
```

增加到：

```text
2048
```

后，Decode Throughput 变化如下：

```text
F16:
71.52 → 67.09

Q8_0:
124.17 → 111.62

Q4_K_M:
188.68 → 161.27
```

可以明确观察：

```text
Context Depth ↑
→ Decode Throughput ↓
→ TPOT ↑
```

随着历史 Token 数增加，需要访问的历史 KV 状态增加。

因此 Attention / KV Cache 相关工作量增加。

但是 Context Depth：

```text
128 → 2048
```

增加了：

```text
16 倍
```

Decode Time 并没有增加 16 倍。

这是因为一次 Decode Forward 中还存在大量并不会按照 Context Depth 同比例增长的工作，例如：

```text
Q/K/V Projection
Output Projection
FFN / MLP
LayerNorm
其他 Kernel / Runtime 工作
```

因此可以建立一个简单系统模型：

```text
T_decode
≈
T_fixed
+
T_context
```

其中：

```text
T_fixed
```

代表不会随着 Context Depth 同比例增长的部分，

而：

```text
T_context
```

代表 Attention / KV Access 等 Context-dependent Work。

---

# 13. Experiment 7 — Prompt Length vs Memory

## 13.1 Configuration

固定：

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

## 13.2 Results

| Prompt Length | Peak VRAM | Peak RAM |
|---:|---:|---:|
| 128 | 1700 MiB | 1478 MiB |
| 512 | 1697 MiB | 1482 MiB |
| 2048 | 1697 MiB | 1485 MiB |

## 13.3 Analysis

Prompt Length 从：

```text
128
```

增加到：

```text
2048
```

以后：

```text
Peak VRAM:
1700 → 1697 MiB
```

基本没有变化。

这说明在当前 Runtime 配置中，当 Context Capacity 固定时：

> 实际 Prompt Length 并不是 Peak Allocated VRAM 的主要决定因素。

需要注意：

这一结论描述的是当前 `llama.cpp + Qwen3-1.7B + RTX 4060 Ti` 配置下观察到的 Peak VRAM 行为。

不能直接推广到所有 Runtime。

---

# 14. Experiment 8 — Context Capacity vs Memory

## 14.1 Configuration

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

## 14.2 Results

| Context Capacity | Peak VRAM | Peak RAM |
|---:|---:|---:|
| 512 | 1301 MiB | 1474 MiB |
| 1024 | 1357 MiB | 1474 MiB |
| 2048 | 1471 MiB | 1476 MiB |
| 4096 | 1698 MiB | 1478 MiB |

Peak VRAM 增量：

```text
512 → 1024:
+56 MiB

1024 → 2048:
+114 MiB

2048 → 4096:
+227 MiB
```

## 14.3 Analysis

可以观察到 Context Capacity 增长时，Peak VRAM 明显增加。

当前结果近似表现为：

```text
Peak VRAM
≈
Fixed Memory
+
Context-dependent GPU Memory
```

但是必须注意：

```text
Context-dependent GPU Memory
!=
KV Cache only
```

增加的显存还可能包含：

```text
KV Cache
Context-related Compute Buffers
Runtime Allocations
Temporary Buffers
```

因此当前实验只能证明：

> Context Capacity 增加与 Peak VRAM 增长存在明确关系。

不能仅根据当前数据判断增加的每一 MiB 显存具体属于哪种 Runtime Buffer。

---

# 15. Benchmark Repeatability

旧的 Q4_K_M Memory Benchmark：

```text
Context = 2048
Peak VRAM ≈ 1468 MiB
```

后续 Context Capacity Sweep：

```text
Context = 2048
Peak VRAM ≈ 1471 MiB
```

两个独立实验只相差约：

```text
3 MiB
```

说明当前 Peak VRAM Benchmark 在相同配置附近具有较好的重复性。

这也表明当前 Baseline Subtraction + Repetition + Median 方法能够在一定程度上控制 Memory Measurement Noise。

---

# 16. Cross-Metric Trade-off Analysis

将多个 Benchmark 放在一起以后，可以观察到 Quantization 并不是单维度优化。

## F16

```text
Model Size:
3.790 GiB

PPL:
14.702900

Peak VRAM:
3713 MiB

Decode d128:
71.52 tok/s
```

F16 可以作为当前项目的高精度 Baseline，但模型文件和 Runtime Memory 最大，Decode Throughput 也最低。

## Q8_0

```text
Model Size:
1.708 GiB

PPL:
14.699600

Peak VRAM:
2166 MiB

Decode d128:
124.17 tok/s
```

当前实验中，Q8_0 获得显著的模型压缩和 Runtime Memory Reduction，同时 WikiText-2 PPL 与 F16 基本相同。

Prefill 长 Prompt Workload 中，Q8_0 还略快于 Q4_K_M。

## Q4_K_M

```text
Model Size:
1.194 GiB

PPL:
15.937400

Peak VRAM:
1468 MiB

Decode d128:
188.68 tok/s
```

Q4_K_M 获得最大的 Model Size / Runtime Memory Reduction，并且当前全部 Decode Workload 中具有最高 Throughput。

代价是在当前 WikiText-2 Benchmark 中出现更加明显的 Perplexity 增长。

因此当前实验显示的是一个典型的 Systems Trade-off：

```text
更低 Precision
        ↓
更小 Weight Representation
        ↓
更低 Runtime Memory
+
更高 Decode Throughput
        ↓
但可能带来更明显 Quality Loss
```

与此同时：

```text
更低 Precision
```

也并不保证：

```text
所有 Prefill Workload 都获得最高性能
```

---

# 17. Experimental Facts

根据当前 Benchmark，可以直接得到以下实验事实。

| Experimental Fact | Evidence |
|---|---|
| Q8_0 与 Q4_K_M 的 GGUF 文件明显小于 F16 | Model Size Benchmark |
| Q8_0 的当前 WikiText-2 PPL 与 F16 非常接近 | Quality Benchmark |
| Q4_K_M 的当前 WikiText-2 PPL 高于 F16 / Q8_0 | Quality Benchmark |
| Quantization 显著降低 Peak RAM / VRAM | Memory Benchmark |
| Q4_K_M 在当前所有 Decode Workload 中最快 | Decode Benchmark |
| Q8_0 在 pp512 / pp2048 中略快于 Q4_K_M | Prefill Benchmark |
| Context Depth 增长时 Decode Throughput 下降 | Decode Context Sweep |
| 固定 Context Capacity 后，Prompt Length 对 Peak VRAM 影响很小 | Prompt Memory Sweep |
| Context Capacity 增长时 Peak VRAM 明显增加 | Context Capacity Sweep |

---

# 18. Mechanism Hypotheses

下面这些内容目前属于机制假设，而不是已经证明的实验结论。

Q4_K_M Decode 性能更高，可能与以下因素有关：

```text
Lower Weight Traffic
Memory Bandwidth
Quantized Kernel Behavior
Backend Implementation
```

Q8_0 在长 Prompt Prefill 中超过 Q4_K_M，可能与：

```text
Dequantization Cost
Compute Efficiency
Kernel Implementation
Arithmetic Characteristics
```

等因素有关。

Context Depth 增长后 Decode 性能下降，很可能与：

```text
Attention
KV Cache Access
```

增加有关。

但是，要将这些机制从：

```text
Hypothesis
```

升级为：

```text
Profiler-supported Conclusion
```

还需要使用：

```text
Nsight Systems
Nsight Compute
perf
```

获得更直接的证据。

---

# 19. Engineering Interpretation

当前实验说明，端侧 LLM 部署不能只根据模型文件大小选择 Quantization。

实际部署决策至少需要同时考虑：

```text
Quality
Runtime Memory
Prefill
Decode
Context Capacity
Target Workload
```

如果应用主要关注 Runtime Memory 和 Decode Throughput，Q4_K_M 在当前测试环境中表现出更明显的资源与速度优势，但同时伴随更高的 WikiText-2 Perplexity。

如果应用对当前 PPL 指标更加敏感，Q8_0 在当前实验中表现出接近 F16 的 PPL，同时仍能显著降低模型文件大小和 Runtime Memory。

因此：

```text
不存在只根据“模型越小越好”
就能够完成的部署选择。
```

实际 Quantization 选择应该根据：

```text
Quality Requirement
Memory Budget
Prompt Workload
Generation Workload
Context Requirement
Hardware
Runtime Backend
```

共同决定。

---

# 20. Limitations

当前实验存在以下限制。

| Limitation | 影响 |
|---|---|
| 单 GPU | 当前数据只来自 RTX 4060 Ti，不能直接推广到其他 GPU |
| 单模型系列 | 当前主要测试 Qwen3-1.7B |
| 单 Runtime | 当前主要使用 llama.cpp |
| Quality Metric 有限 | 当前主要使用 WikiText-2 Perplexity |
| 缺少 Profiler | 尚不能证明 Q4 / Q8 性能差异的底层机制 |
| VRAM 为外部采样 | 极短暂 Memory Peak 可能无法被捕获 |
| 无精细 Memory Breakdown | 当前不能准确拆分 Weight / KV Cache / Compute Buffer |
| Prefill Latency 为派生指标 | 当前通过 Throughput 计算，而非独立 Timer |
| TPOT 为派生指标 | 当前通过 Decode Throughput 计算 |

因此本文中的低层机制解释应被视为后续 Profiling 的研究方向，而不是已经完成证明的结论。

---

# 21. Conclusion

本实验基于 Qwen3-1.7B、llama.cpp 和 RTX 4060 Ti，系统比较了 F16、Q8_0 和 Q4_K_M 在：

```text
Model Size
Quality
Runtime Memory
Prefill
Decode
Context
```

上的差异。

当前实验说明：

```text
Model Size
!=
Runtime Memory
!=
Quality
!=
Performance
```

同时：

```text
Prefill
!=
Decode
```

并且：

```text
Prompt Length
!=
Context Capacity
!=
Current Context Depth
```

Quantization 能显著降低模型文件大小与运行时内存，并对 Decode 性能产生明显影响。

但是不同 Quantization 在不同 Prefill Workload 下的性能关系并不完全一致。

Context Depth 增长会降低 Decode Throughput，而 Context Capacity 增长会明显提高 Peak VRAM。

这些实验进一步说明：

> LLM Inference Optimization 是 Model、Runtime、Workload 和 Hardware 共同作用的系统问题，而不是单纯比较模型文件大小的问题。

当前阶段已经建立 Benchmark 层面的实验事实。

下一阶段需要通过 Profiler 将：

```text
Performance Difference
```

进一步推进到：

```text
Performance Bottleneck
```

最终形成：

```text
Benchmark
→ Profile
→ Bottleneck
→ Optimization
→ Re-benchmark
```

的完整 AI Inference Systems 性能分析闭环。