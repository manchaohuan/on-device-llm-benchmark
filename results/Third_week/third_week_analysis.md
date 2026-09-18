# 第三周实验分析：Workload、Prefill、Decode 与 Memory

## 1. 实验目标

本周主要研究以下问题：

1. Prompt Length 增长时，Prefill 性能如何变化。
2. Context Depth 增长时，Decode 性能如何变化。
3. F16、Q8_0、Q4_K_M 三种量化方式对 Prefill / Decode 性能的影响。
4. 实际 Prompt Length 与配置的 Context Capacity 对 Peak RAM / VRAM 的影响是否相同。
5. 建立统一的 `Model Size / Memory / Prefill / Decode` 实验结果，为后续 README、profiling 和性能优化提供基础数据。

## 2. 实验环境

- Runtime：llama.cpp
- llama.cpp build：b10892
- Git revision：e5a8d439c
- GPU：NVIDIA GeForce RTX 4060 Ti
- GPU VRAM：8187 MiB
- CUDA Compute Capability：8.9
- GPU offload：99 layers
- 模型：Qwen3-1.7B
- 量化版本：F16、Q8_0、Q4_K_M

## 3. Model Size 与基础 Memory Benchmark

### 3.1 模型大小

| Quantization | Model Size (GiB) |
|---|---:|
| F16 | 3.790 |
| Q8_0 | 1.708 |
| Q4_K_M | 1.194 |

相对于 F16，Q8_0 模型文件缩小约 54.9%，Q4_K_M 缩小约 68.5%。

### 3.2 固定配置 Memory Benchmark

配置：Context size = 2048，Generation length = 128，GPU offload = 99，Repetitions = 3。VRAM 指标为相对于运行前 baseline 的峰值增量，RAM 指标为进程 Peak Working Set，最终使用 median。

| Quantization | Peak VRAM (MiB) | Peak RAM (MiB) |
|---|---:|---:|
| F16 | 3713 | 3780 |
| Q8_0 | 2166 | 2163 |
| Q4_K_M | 1468 | 1476 |

量化显著降低运行时内存，但运行时内存下降幅度并不完全等于 GGUF 文件大小下降幅度，因为运行时还包含 KV Cache、Compute Buffers、CUDA Runtime Buffers 和 Temporary Buffers 等额外内存。

## 4. Prefill Benchmark

### 4.1 实验设计

只改变 Prompt Length：128、512、2048 tokens，每项重复 5 次。Prefill 和 Decode 分开测试，避免把两个阶段混成一个总速度指标。

### 4.2 Prefill Throughput

| Prompt Length | F16 (tok/s) | Q8_0 (tok/s) | Q4_K_M (tok/s) |
|---:|---:|---:|---:|
| 128 | 6013.39 | 7917.30 | 8821.62 |
| 512 | 12277.48 | 13937.92 | 13308.85 |
| 2048 | 12756.72 | 13742.64 | 13369.17 |

### 4.3 Prefill Latency

计算公式：

```text
Prefill Latency = Prompt Tokens / Prefill Throughput
```

| Prompt Length | F16 (ms) | Q8_0 (ms) | Q4_K_M (ms) |
|---:|---:|---:|---:|
| 128 | 21.286 | 16.167 | 14.510 |
| 512 | 41.702 | 36.734 | 38.471 |
| 2048 | 160.543 | 149.025 | 153.188 |

### 4.4 结果分析

随着 Prompt Length 从 128 增长到 512，三种模型的 Prefill throughput 都明显提高。这并不意味着 Prompt 越长总延迟越低，而是因为较大的 Prefill workload 更容易摊薄固定开销，并充分利用 GPU 并行计算能力。

例如 Q4_K_M 从 128 tokens 的 8821.62 tok/s 上升到 2048 tokens 的 13369.17 tok/s，但 Prefill latency 仍从 14.51 ms 增长到 153.19 ms，因为实际需要处理的 token 数量增加了 16 倍。

Q8_0 在 pp512 和 pp2048 中略快于 Q4_K_M，说明模型大小不是 Prefill 性能的唯一决定因素。实际性能还会受到 Weight memory traffic、Dequantization overhead、Quantized kernel efficiency、Backend implementation、GPU compute efficiency 和 Memory bandwidth 等因素共同影响。

Q8_0 可能在“权重体积下降”和“反量化 / kernel 开销”之间取得了更好的平衡，因此在较长 Prompt 的 Prefill 阶段略快于 Q4_K_M。需要注意，这只是机制上合理的解释，目前还没有 profiler 数据证明具体瓶颈来自哪一个 kernel。

## 5. Decode Benchmark

### 5.1 实验设计

固定 Generation length = 128 tokens，改变 Decode 开始时的 Context Depth：d128、d512、d2048，每项重复 5 次。

### 5.2 Decode Throughput

| Context Depth | F16 (tok/s) | Q8_0 (tok/s) | Q4_K_M (tok/s) |
|---:|---:|---:|---:|
| 128 | 71.52 | 124.17 | 188.68 |
| 512 | 70.41 | 120.73 | 180.55 |
| 2048 | 67.09 | 111.62 | 161.27 |

### 5.3 TPOT

TPOT（Time Per Output Token）近似：

```text
TPOT ≈ 1000 / Decode Throughput
```

| Context Depth | F16 (ms/token) | Q8_0 (ms/token) | Q4_K_M (ms/token) |
|---:|---:|---:|---:|
| 128 | 13.982 | 8.053 | 5.300 |
| 512 | 14.203 | 8.283 | 5.539 |
| 2048 | 14.905 | 8.959 | 6.201 |

### 5.4 结果分析

Decode 阶段的量化收益明显高于 Prefill。在 context depth = 128 时，F16 为 71.52 tok/s，Q8_0 为 124.17 tok/s，Q4_K_M 为 188.68 tok/s，Q4_K_M 约为 F16 的 2.64 倍。

Decode 每一步只生成一个新 token，但每一层仍需要读取大量模型权重，因此 Decode 更容易受到 Memory Bandwidth 和 Weight Memory Traffic 的影响。量化降低了权重体积，从而减少每个 Decode step 中需要搬运的权重数据量，因此 Q4_K_M 的 Decode 优势十分明显。

## 6. Context Depth 对 Decode 的影响

随着 Context Depth 增长，三种量化模型的 Decode throughput 都下降：

```text
F16:    71.52 → 67.09 tok/s，下降约 6.2%
Q8_0:  124.17 → 111.62 tok/s，下降约 10.1%
Q4_K_M:188.68 → 161.27 tok/s，下降约 14.5%
```

原因是 Decode 时 Attention 需要访问更长的历史 K/V：

```text
Context Depth ↑
        ↓
历史 KV 数量 ↑
        ↓
Attention / KV Cache 访问成本 ↑
        ↓
Decode Throughput ↓
```

但是 Context Depth 增长 16 倍并不会让 Decode 速度下降 16 倍，因为一次 Decode forward 中不仅包含 Attention，还包括 Q/K/V projection、Output projection、FFN / MLP、LayerNorm 以及其他 kernel 和 runtime 开销，其中大量工作不会随着 Context Depth 同比例增长。

可以粗略表示为：

```text
T_decode ≈ T_fixed + T_context
```

只有 `T_context` 部分会随着 Context Depth 显著增加。

## 7. Prompt Length 对 Peak Memory 的影响

### 7.1 实验设计

固定 Model = Q4_K_M、Context Capacity = 4096、Generation = 128、GPU offload = 99、Repetitions = 3，只改变实际 Prompt Length：128、512、2048。

### 7.2 结果

| Prompt Length | Peak VRAM (MiB) | Peak RAM (MiB) |
|---:|---:|---:|
| 128 | 1700 | 1478 |
| 512 | 1697 | 1482 |
| 2048 | 1697 | 1485 |

### 7.3 结果分析

Prompt Length 从 128 增长到 2048，增加了 16 倍，但 Peak VRAM 基本没有变化：1700 → 1697 → 1697 MiB。RAM 也只增加约 7 MiB：1478 → 1482 → 1485 MiB。

因此在固定 `context_size=4096` 的情况下，实际 Prompt Length 并不是 Peak allocated VRAM 的主要决定因素。

一个合理解释是，llama.cpp 在创建 context 时已经按照配置的 Context Capacity 为 KV Cache / Compute Buffers 等预留或申请了较大的内存区域。因此 Prompt Length 增长虽然会增加实际有效 KV 的数量，但不一定继续增加已经分配的 Peak VRAM。

需要区分：

```text
Memory Allocated
```

和：

```text
Memory Actually Occupied by Valid KV Entries
```

当前通过 `nvidia-smi memory.used` 测量的更接近前者。

## 8. Context Capacity 对 Peak Memory 的影响

### 8.1 实验设计

固定 Model = Q4_K_M、Prompt Length = 128、Generation = 128、GPU offload = 99、Repetitions = 3，只改变 `context_size`：512、1024、2048、4096。

### 8.2 结果

| Context Capacity | Peak VRAM (MiB) | Peak RAM (MiB) |
|---:|---:|---:|
| 512 | 1301 | 1474 |
| 1024 | 1357 | 1474 |
| 2048 | 1471 | 1476 |
| 4096 | 1698 | 1478 |

### 8.3 结果分析

Peak VRAM 随 Context Capacity 明显增加：

```text
512  → 1301 MiB
1024 → 1357 MiB
2048 → 1471 MiB
4096 → 1698 MiB
```

每次 Context Capacity 翻倍时，VRAM 增量约为：

```text
512 → 1024:  +56 MiB
1024 → 2048: +114 MiB
2048 → 4096: +227 MiB
```

增长幅度接近倍增，说明在当前范围内存在明显的、近似与 Context Capacity 成比例的 GPU Memory allocation。

可以粗略写成：

```text
Peak VRAM ≈ Fixed Memory + Context-dependent Memory
```

其中 Context-dependent 部分可能包括 KV Cache、Context-related Compute Buffers、Runtime temporary buffers，以及其他随 context capacity 增长的 GPU allocation。当前实验无法证明增加的显存全部属于 KV Cache，因此不应直接将 VRAM 增量等同于 KV Cache 大小。

## 9. Prompt Length、Context Capacity 与 Context Depth 的区别

### 9.1 Context Capacity

例如：

```text
-c 4096
```

表示 llama.cpp 为 context 配置的最大容量。当前实验显示：

```text
Context Capacity ↑ → Peak VRAM ↑
```

说明它会直接影响 GPU Memory allocation。

### 9.2 Prompt Length

表示当前 Prefill 实际输入的 token 数量，例如 128、512、2048。它主要影响 Prefill workload、Prefill latency 和 Prefill throughput。在 Context Capacity 固定时，对 Peak VRAM 的影响很小。

### 9.3 Current Context Depth

表示 Decode 开始时已经存在的历史 token 数量，例如 d128、d512、d2048。它主要影响 Attention 需要访问的历史 K/V 数量，以及 Decode latency / throughput。

因此可以总结为：

```text
Context Capacity
    → 更影响 Memory Allocation

Prompt Length
    → 更影响 Prefill Workload

Current Context Depth
    → 更影响 Decode Attention / KV Access Cost
```

三者不能混为一谈。

## 10. Benchmark 稳定性验证

Context Capacity Sweep 中，Context = 2048 时得到 Peak VRAM = 1471 MiB、Peak RAM = 1476 MiB；之前固定配置 Memory Benchmark 中对应结果为 Peak VRAM = 1468 MiB、Peak RAM = 1476 MiB。

两次实验仅相差约 3 MiB VRAM，RAM 完全一致。Context = 4096 时，不同实验得到约 1697～1700 MiB，也具有较好一致性。

这说明当前使用的 baseline subtraction、`nvidia-smi` polling、process RSS、3 repetitions、median 这一套 Memory Benchmark 方法在当前机器上具有较好的重复性。

## 11. 本周核心工程结论

### 11.1 Quantization

量化显著降低 Model Size、Peak RAM、Peak VRAM，并显著提升 Decode throughput。其中 Q4_K_M 在 Decode 阶段优势最明显。

### 11.2 Prefill

Prefill 性能并不只由模型大小决定。Q8_0 在 pp512 和 pp2048 中略快于 Q4_K_M，说明：

```text
Performance ≠ Model Size only
```

实际还受到 Dequantization、Kernel implementation、Compute efficiency 和 Memory bandwidth 等因素共同影响。

### 11.3 Decode

Decode 对量化更加敏感。当前数据表现为：

```text
Q4_K_M > Q8_0 > F16
```

说明低比特量化减少 Weight Memory Traffic 后，可以显著提高单 token Decode throughput。

### 11.4 Context Depth

Context Depth 增长会降低 Decode throughput，但 Decode 中还有大量不会随着 Context Depth 同比例增长的计算，因此吞吐下降幅度远小于 Context Depth 增长幅度。

### 11.5 Memory

实际 Prompt Length 从 128 增长到 2048，对固定 `context_size=4096` 下的 Peak VRAM 几乎没有影响。相比之下，Context Capacity 从 512 增长到 4096 时，Peak VRAM 从 1301 MiB 增长到 1698 MiB。

因此在当前 llama.cpp 配置下，Peak VRAM 更敏感于 configured context capacity，而不是当前实际输入的 Prompt Length。

## 12. 当前阶段还不能下定论的部分

以下内容目前只能作为假设，不能写成已经证明的事实：

1. Q8_0 在长 Prompt Prefill 中比 Q4_K_M 更快，是否主要由 dequantization kernel 导致。
2. Context Capacity 增长时增加的 GPU Memory 中，究竟多少来自 KV Cache。
3. Q4_K_M 随 Context Depth 增长时 Decode throughput 相对下降更多的具体原因。
4. 当前 Prefill / Decode 是否分别属于 Compute-bound 或 Memory-bound。

这些问题需要后续通过 profiler，例如 Nsight Systems / Nsight Compute，进一步观察 GPU utilization、Memory bandwidth、Kernel execution time、Kernel launch、Dequantization kernel、Memory stalls 和 Compute utilization，再进行判断。

## 13. 第三周阶段性总结

本周已经从单一量化实验扩展到完整的 workload 分析，完成：

```text
3 Quantizations
×
3 Workloads
×
Prefill / Decode
```

并进一步完成：

```text
Prompt Length × Peak Memory
```

以及：

```text
Context Capacity × Peak Memory
```

通过这些实验，已经能够从数据角度解释：

- 为什么 Prompt Length 和 Context Capacity 不是同一个概念；
- 为什么 Prefill 和 Decode 不能混成一个速度指标；
- 为什么量化对 Prefill 和 Decode 的收益不同；
- 为什么 Model Size 更小不代表所有 workload 都更快；
- 为什么 Context Depth 增长会降低 Decode throughput；
- 为什么固定 Context Capacity 时，Prompt Length 增长不一定增加 Peak allocated VRAM。

这些结果已经构成后续进行 GPU profiling、KV Cache 分析和推理优化实验的基础。
