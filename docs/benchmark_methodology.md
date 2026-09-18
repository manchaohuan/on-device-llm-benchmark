# Benchmark 方法说明

## 1. 实验目的

本文档用于说明 `on-device-llm-banchmark` 项目采用的 Benchmark 方法。

本项目主要研究以下几个因素之间的关系：

* 模型大小（Model Size）
* 模型量化（Quantization）
* 模型质量（Quality）
* 运行时内存（Runtime Memory）
* Prefill 性能
* Decode 性能
* Context 长度

本项目的目标不仅是比较“哪个模型更快”或“哪个模型更小”，更重要的是建立一套**可复现的端侧大模型推理 Benchmark 方法**，用于分析不同量化配置下 LLM 的实际推理行为。

当前主要测试对象为：

```text
Qwen3-1.7B
```

推理 Runtime：

```text
llama.cpp
```

测试 GPU：

```text
NVIDIA GeForce RTX 4060 Ti
```

---

# 2. Benchmark 环境

## 2.1 硬件环境

GPU：

```text
NVIDIA GeForce RTX 4060 Ti
```

可用显存：

```text
8187 MiB
```

Compute Capability：

```text
8.9
```

当前实验均在同一台机器上执行，以尽量减少硬件差异带来的变量。

---

## 2.2 操作系统

当前主要 Benchmark 环境：

```text
Windows
```

后续会单独开展：

```text
WSL2 / Linux
```

环境实验。

Windows 与 Linux 环境下采集的结果不能在没有记录 Runtime 和系统配置的情况下直接混合比较。

---

## 2.3 推理 Runtime

当前推理 Runtime：

```text
llama.cpp
```

llama.cpp build：

```text
b10892
```

Git revision：

```text
e5a8d439c
```

当前主要使用的程序：

```text
llama-cli.exe
llama-bench.exe
llama-perplexity.exe
```

llama.cpp 程序目录：

```text
D:\code\relavate_file\llama-b10892-bin-win-cuda-13.3-x64
```

除非某个实验特别说明，否则 GPU Offload 设置为：

```text
-ngl 99
```

这样做的目的是尽可能让主要模型计算运行在 GPU 上，同时在不同量化模型之间保持相同的 GPU Offload 策略。

---

# 3. 模型

当前 Benchmark 比较 Qwen3-1.7B 的三种 GGUF 模型：

| 模型文件                   | 量化类型   |
| ---------------------- | ------ |
| Qwen3-1.7B-f16.gguf    | F16    |
| Qwen3-1.7B-Q8_0.gguf   | Q8_0   |
| Qwen3-1.7B-Q4_K_M.gguf | Q4_K_M |

三个模型属于同一模型系列，用来研究不同权重精度对以下指标的影响：

* 模型文件大小
* 运行时内存
* Perplexity
* Prefill 性能
* Decode 性能

在本项目中，以下四个指标必须明确区分：

```text
Model Size
!=
Runtime Memory
!=
Quality
!=
Performance
```

即：

```text
模型文件大小
!=
运行时内存
!=
模型质量
!=
推理性能
```

它们表示的是完全不同的系统属性，不能混为一谈。

---

# 4. Benchmark 主要维度

当前项目主要测试四类指标：

1. Model Size
2. Quality
3. Runtime Memory
4. Inference Performance

分别对应：

1. 模型文件大小
2. 模型质量
3. 运行时内存
4. 推理性能

这些指标描述的是模型和 Runtime 的不同性质。

---

# 5. Model Size Benchmark

Model Size 用于测量 GGUF 模型文件在磁盘上的大小。

当前测试结果：

| 量化类型   | GGUF 文件大小 |
| ------ | --------: |
| F16    | 3.790 GiB |
| Q8_0   | 1.708 GiB |
| Q4_K_M | 1.194 GiB |

模型文件大小不能直接理解成模型运行时需要的 RAM 或 VRAM。

实际推理过程中还可能存在：

* 模型权重（Weights）
* KV Cache
* Compute Buffers
* Runtime Buffers
* Backend Allocations
* Temporary Buffers

因此：

```text
GGUF 文件大小 != 推理运行时内存
```

即使模型文件压缩比例很大，也不能直接认为运行时显存会按照相同比例减少。

---

# 6. Quality Benchmark

当前使用：

```text
WikiText-2
```

数据集上的：

```text
Perplexity
```

作为模型质量指标。

使用程序：

```text
llama-perplexity.exe
```

数据集：

```text
WikiText-2
wiki.test.raw
```

不同量化模型进行比较时，应保持：

* 数据集一致
* Runtime 一致
* Context 配置一致
* GPU Offload 一致

当前实验结果：

| 量化类型   | Perplexity |
| ------ | ---------: |
| F16    |  14.702900 |
| Q8_0   |  14.699600 |
| Q4_K_M |  15.937400 |

Perplexity 属于：

```text
模型质量指标
```

而不是：

```text
推理性能指标
```

通常情况下，在相同数据集和测试方法下，较低的 Perplexity 表示模型对该数据集的预测能力更好。

但是 WikiText-2 Perplexity 并不能完整代表模型在真实应用中的所有能力，例如：

* Instruction Following
* Reasoning
* Coding
* Long Context
* 对话能力

因此当前 PPL 主要用于观察**量化带来的相对质量变化**。

---

# 7. LLM 推理阶段

本项目将 LLM 推理性能明确拆分为：

```text
Prefill
```

和：

```text
Decode
```

两个阶段。

由于这两个阶段的计算特征不同，因此不能简单合并成一个“总推理速度”。

---

## 7.1 Prefill

Prefill 是模型处理用户输入 Prompt 的阶段。

例如 Prompt Length 为：

```text
512 tokens
```

那么 Prefill 阶段需要处理这 512 个输入 Token，并建立后续 Decode 所需要的内部状态。

当前主要指标为：

```text
Prefill Throughput
```

单位：

```text
tokens / second
```

即：

```text
tok/s
```

Prefill Latency 可以近似表示为：

```text
Prefill Latency
≈
Prompt Tokens / Prefill Throughput
```

例如：

```text
Prompt Length = 512
Prefill Throughput = 10000 tok/s
```

则 Prefill Latency 大约为：

```text
512 / 10000
≈
0.0512 s
≈
51.2 ms
```

因此：

```text
Prefill Throughput 上升
```

并不意味着：

```text
Prompt Length 增加以后总延迟不会增加
```

分析 Prefill 时，必须同时考虑：

* Throughput
* Prompt Length
* Latency

---

## 7.2 Decode

Decode 是模型进行自回归 Token Generation 的阶段。

Decode 过程中：

```text
Token 1
→ Token 2
→ Token 3
→ ...
```

Token 通常按顺序逐个生成。

当前主要指标包括：

```text
Decode Throughput
```

单位：

```text
tokens / second
```

以及：

```text
TPOT
```

TPOT 全称：

```text
Time Per Output Token
```

即：

```text
每生成一个输出 Token 所需要的时间
```

如果 Decode Throughput 使用 `tokens/s` 表示，则近似有：

```text
TPOT ≈ 1 / Decode Throughput
```

例如：

```text
Decode Throughput = 200 tok/s
```

那么：

```text
TPOT
≈ 1 / 200 s
≈ 5 ms/token
```

Decode 性能必须与 Prefill 性能分开分析。

---

# 8. Context 相关概念

当前项目中必须严格区分以下三个概念：

```text
Context Capacity
Prompt Length
Current Context Depth
```

它们不是同一个东西。

---

## 8.1 Context Capacity

Context Capacity 表示 Runtime 配置允许使用的最大 Context 容量。

例如：

```text
-c 4096
```

表示 Runtime 被配置为支持大约：

```text
4096 tokens
```

的 Context Capacity。

Context Capacity 主要影响：

```text
Memory Allocation
```

以及与 Context 相关的 Runtime Buffer。

但是：

```text
-c 4096
```

并不表示当前 Prompt 一定包含 4096 个 Token。

---

## 8.2 Prompt Length

Prompt Length 表示 Prefill 阶段实际输入模型的 Token 数量。

当前测试使用：

```text
128
512
2048
```

例如：

```text
Prompt Length = 2048
```

表示 Prefill 实际需要处理约 2048 个输入 Token。

Prompt Length 主要决定：

```text
Prefill workload
```

以及：

```text
Prefill latency
```

---

## 8.3 Current Context Depth

Current Context Depth 表示开始进行 Decode 时，模型当前已经存在多少历史 Token。

当前 Decode 实验使用：

```text
d128
d512
d2048
```

例如：

```text
d2048
```

表示 Decode 开始时已经具有约：

```text
2048 个历史 Token
```

Context Depth 增加以后，需要访问的历史 KV 状态会增加。

因此它主要影响：

* Attention
* KV Cache Access
* Decode Performance

---

## 8.4 三者必须严格区分

因此：

```text
Prompt Length
!=
Context Capacity
!=
Current Context Depth
```

即：

```text
实际 Prompt 长度
!=
最大 Context 容量
!=
Decode 当前历史长度
```

Benchmark 配置和结果文件中必须分别记录。

---

# 9. Performance Benchmark Workload

## 9.1 量化配置

当前性能实验比较：

```text
F16
Q8_0
Q4_K_M
```

不同量化模型之间进行比较时，应保持 Workload 配置一致。

---

## 9.2 Prefill Workload

当前 Prompt Length：

```text
128
512
2048
```

Generation Length：

```text
128
```

GPU Offload：

```text
99 layers
```

Performance Benchmark 重复：

```text
5 次
```

主要观察：

```text
Prefill Throughput
Prefill Latency
```

当研究：

```text
Quantization
```

对性能的影响时，应该将其他变量保持一致。

即：

```text
只有量化方式发生变化
```

而 Prompt、Runtime、GPU、Generation 等变量保持不变。

---

## 9.3 Decode Workload

Decode Benchmark 使用以下 Context Depth：

```text
128
512
2048
```

Generation Length：

```text
128
```

GPU Offload：

```text
99 layers
```

Performance Benchmark 重复：

```text
5 次
```

主要指标：

```text
Decode Throughput
TPOT
```

该实验用于研究：

```text
Current Context Depth
```

增加以后 Decode 性能如何变化。

---

# 10. Memory Benchmark

Memory Benchmark 当前主要测量：

```text
Peak VRAM
Peak RAM
```

Memory Benchmark 与模型磁盘文件大小属于不同实验维度。

---

## 10.1 VRAM 测量方法

每次实验开始之前，首先记录：

```text
GPU Memory Baseline
```

即当前 GPU 已经被系统或其他程序占用的显存。

模型运行过程中持续记录：

```text
nvidia-smi memory.used
```

观察到的最大值记为：

```text
Peak Total GPU Memory
```

最终报告：

```text
Peak VRAM Delta
=
Peak Total GPU Memory
-
Pre-run GPU Memory Baseline
```

也就是：

```text
Benchmark Peak VRAM
=
运行期间最大显存
-
运行前显存基线
```

进行 Baseline Subtraction 的主要目的是降低后台 GPU 程序对测试结果的影响。

因此当前的 Peak VRAM 更准确地理解为：

> 在当前测量方法下，Benchmark 运行导致的最大可观测 GPU 显存增加量。

---

## 10.2 RAM 测量方法

CPU 侧内存主要测量：

```text
Benchmark 进程及其子进程的 Peak Resident Memory
```

在 Windows 当前测量实现中，对应进程：

```text
RSS / Working Set
```

相关指标。

因此这里的 Peak RAM 表示：

```text
运行过程中观察到的最大进程驻留内存
```

而不是模型文件大小。

---

# 11. Memory Benchmark 重复实验

当前 Memory Benchmark 每组配置执行：

```text
3 次
```

最终结果使用：

```text
median
```

即：

```text
中位数
```

作为最终报告结果。

原因是 Peak Memory 测量可能受到：

* 系统后台进程
* Runtime 短时分配
* GPU 背景程序
* Memory Allocator
* OS Scheduling

等因素影响。

Median 相比 Mean：

```text
对偶发异常值更加稳定
```

因此当前 Memory Benchmark 使用：

```text
3 repetitions
+
median
```

作为统计方法。

---

# 12. Memory Sweep 实验设计

为了区分：

```text
Prompt Length
```

和：

```text
Context Capacity
```

对内存的影响，目前设计了两个独立实验。

---

## 12.1 Prompt Length → Memory

固定变量：

```text
Model = Q4_K_M
Context Capacity = 4096
Generation = 128
GPU offload = 99
Repetitions = 3
```

改变变量：

```text
Prompt Length
```

测试：

```text
128
512
2048
```

该实验主要回答：

> 在最大 Context Capacity 固定的情况下，实际 Prompt Length 增加是否会显著改变 Peak Memory？

这里只改变 Prompt Length，而其他变量保持不变。

---

## 12.2 Context Capacity → Memory

固定：

```text
Model = Q4_K_M
Prompt Length = 128
Generation = 128
GPU offload = 99
Repetitions = 3
```

改变：

```text
Context Capacity
```

测试：

```text
512
1024
2048
4096
```

该实验用于研究：

> Runtime 配置的 Context Capacity 增加以后，运行时内存如何变化？

但是：

```text
Context Capacity 增加带来的显存增长
```

不能直接全部认为是：

```text
KV Cache
```

Context 相关显存可能包含：

* KV Cache
* Context-related Compute Buffers
* Runtime Allocations
* Temporary Buffers

在没有进一步 profiler 或 allocator 证据之前，不能定量断言这些显存分别来自哪里。

---

# 13. 统计方法

Benchmark 不能只运行一次。

重复实验中主要关注三个统计概念：

```text
Mean
Median
Standard Deviation
```

即：

```text
均值
中位数
标准差
```

---

## 13.1 Mean —— 均值

Mean 的计算公式：

```text
Mean
=
所有结果之和 / 实验次数
```

即：

```text
mean = sum(x_i) / N
```

例如：

```text
180
182
181
179
183
```

Mean 为：

```text
181
```

Mean 用于表示多次实验的整体平均性能水平。

如果实验噪声较小，而且不存在明显异常值，Mean 可以很好地描述整体性能。

---

## 13.2 Median —— 中位数

将实验结果从小到大排列以后：

```text
位于中间的结果
```

就是 Median。

例如：

```text
1468
1470
1600
```

Median 为：

```text
1470
```

而 Mean 为：

```text
1512.7
```

可以看到，1600 这个异常高值明显改变了 Mean。

Median 对异常值不那么敏感。

因此当前 Memory Benchmark 使用：

```text
Median
```

作为主要统计值。

---

## 13.3 Standard Deviation —— 标准差

Standard Deviation：

```text
std
```

用于描述多次测试结果的波动程度。

例如：

```text
A:
179 180 181 180 180
```

与：

```text
B:
150 210 170 200 170
```

两组数据的 Mean 都可能接近：

```text
180
```

但是 A 明显比 B 稳定。

因此 Std 的作用主要是判断：

```text
实验重复性
```

以及：

```text
Measurement Noise
```

而不是判断模型“快不快”。

---

## 13.4 Performance Benchmark 统计方法

当前 Prefill 和 Decode Performance Benchmark 的重复次数统一设置为：

```text
Repetitions = 5
```

Benchmark 脚本通过：

```text
-r 5
```

参数将重复次数直接传递给 `llama-bench`。

因此当前 Performance Benchmark 的统计流程为：

```text
run.matrix.py
        ↓
llama-bench -r 5
        ↓
5 次重复测试
        ↓
llama-bench 输出 Mean ± Standard Deviation
        ↓
run.matrix.py 解析 Mean 和 Std
        ↓
benchmark_matrix.csv
```

也就是说，当前 Python Benchmark 脚本并不自行对 5 次结果重新计算 Mean 和 Standard Deviation，而是解析并保存 `llama-bench` 已经汇总输出的统计结果。

对于 Prefill，保存：

```text
prefill_tps_mean
prefill_tps_std
```

对于 Decode，保存：

```text
decode_tps_mean
decode_tps_std
```

这些数据写入：

```text
results/Third_week/benchmark_matrix.csv
```

因此当前 Performance Benchmark 的主要性能指标采用：

```text
5 repetitions
+
Mean
```

同时使用：

```text
Standard Deviation
```

观察多次运行之间的波动和实验重复性。

需要注意：

```text
results/Third_week/unified_summary.csv
```

当前主要用于生成项目级统一结果表。

该文件通过 `build_summary.py` 从 `benchmark_matrix.csv` 中读取：

```text
prefill_tps_mean
decode_tps_mean
```

因此最终统一结果表主要展示 Mean，而没有将 Std 同时带入最终汇总表。

Std 仍然保存在：

```text
benchmark_matrix.csv
```

中，可用于分析 Measurement Noise 和实验稳定性。

### Prefill Latency

当前 Prefill Latency 不是独立计时得到的指标，而是根据 Mean Prefill Throughput 计算得到：

```text
Prefill Latency (ms)
=
Prompt Length
/
Mean Prefill Throughput
×
1000
```

因此 Prefill Latency 是由 Prefill Throughput 派生得到的指标。

### TPOT

当前 TPOT 同样由 Mean Decode Throughput 计算得到：

```text
TPOT (ms/token)
=
1000
/
Mean Decode Throughput
```

因此 TPOT 是 Decode Throughput 的派生指标。

当前 Performance Benchmark 的统计关系可以表示为：

```text
llama-bench -r 5
        ↓
Mean ± Std
        │
        ├── Mean Prefill Throughput
        │       ↓
        │   Prefill Latency
        │
        └── Mean Decode Throughput
                ↓
               TPOT
```

这种设计既保留了主要性能值 Mean，也保留了 Standard Deviation 用于判断 Benchmark 的重复性。

---

# 14. 控制变量

所有 Benchmark 都遵循：

```text
控制变量
```

原则。

研究某一个因素时，应该尽可能保持其他变量不变。

---

## 14.1 Quantization 实验

自变量：

```text
F16
Q8_0
Q4_K_M
```

控制变量包括：

```text
Model Family
Runtime Build
Hardware
Prompt Length
Context Configuration
Generation Length
GPU Offload
Benchmark Procedure
```

也就是说：

```text
主要只改变 Quantization
```

这样才能合理判断量化与性能变化之间的关系。

---

## 14.2 Prompt Length 实验

自变量：

```text
Prompt Length
```

控制：

```text
Model
Context Capacity
Generation Length
GPU Offload
Hardware
Runtime
```

这样才能研究 Prompt Length 对：

```text
Prefill / Memory
```

等指标的影响。

---

## 14.3 Context Capacity Memory 实验

自变量：

```text
Context Capacity
```

控制：

```text
Model
Prompt Length
Generation Length
GPU Offload
Hardware
Runtime
```

这样可以避免把其他因素导致的内存变化错误归因到 Context Capacity。

---

# 15. Measurement Noise

Benchmark 结果可能受到 Measurement Noise 的影响。

主要来源包括：

* 后台 GPU 程序
* 后台 CPU 程序
* 操作系统调度
* CUDA Runtime 初始化
* Memory Allocator
* GPU 温度变化
* GPU Power State
* Runtime Cache
* 初始化开销
* 外部监控工具的 Sampling Granularity

尤其是在 VRAM 测量中：

```text
后台程序
```

可能会改变当前 GPU Baseline。

因此 Memory Benchmark 采用：

1. 每次运行前重新测 Baseline
2. 重复实验
3. 保存 Raw Results
4. 对重复结果进行统计
5. 遇到异常结果优先调查原因

此外，当前 VRAM 测量依赖：

```text
外部轮询
```

因此如果某次显存分配持续时间非常短，采样工具可能无法完全捕获该瞬时峰值。

所以当前 Peak VRAM 应理解为：

> 当前采样方法实际观察到的最大显存值。

而不是：

> GPU Allocator 在每一个瞬间产生过的理论绝对最高值。

---

# 16. 实验事实与机制假设

本项目必须严格区分：

```text
实验事实
```

和：

```text
机制假设
```

---

## 16.1 实验事实

Experimental Fact 指的是：

```text
可以直接从 Benchmark 数据得到的现象
```

例如：

> 在当前测试的长 Prompt Workload 下，Q8_0 的 Prefill Throughput 高于 Q4_K_M。

这是：

```text
实验数据直接支持的事实
```

---

## 16.2 机制假设

Mechanism Hypothesis 用于解释：

```text
为什么产生这个结果
```

可能的影响因素包括：

* Weight Memory Traffic
* Memory Bandwidth
* Dequantization
* Compute Efficiency
* Kernel Implementation
* Backend
* KV Cache Access
* Runtime Overhead

例如实验发现：

```text
Q8_0 在某个 Workload 下比 Q4_K_M 快
```

不能直接得出：

```text
某一个 CUDA Kernel 导致了这个结果
```

因为 Benchmark 数据只能证明：

```text
存在性能差异
```

但不能单独证明：

```text
底层差异的具体原因
```

未来需要通过：

```text
Nsight Systems
Nsight Compute
perf
```

等 profiler 获取进一步证据。

---

# 17. 可复现性要求

每一个 Benchmark 至少应该记录以下信息：

```text
Model
Quantization
Runtime
Runtime Version / Commit
Hardware
Prompt Length
Context Capacity
Current Context Depth（如适用）
Generation Length
GPU Offload
Repetitions
Measured Metrics
Statistical Aggregation Method
```

在生成 Summary 之前，应尽可能保留：

```text
Raw Benchmark Output
```

推荐项目结构：

```text
configs/
scripts/
results/
analysis/
docs/
```

Benchmark 的实验记录不能只存在于：

```text
PowerShell / Terminal History
```

中。

最终应该形成：

```text
代码
+
配置
+
Raw Results
+
Summary
+
Analysis
+
Documentation
```

的完整实验链路。

---

# 18. 当前实验限制

## 18.1 单 GPU

当前主要测试 GPU 为：

```text
NVIDIA GeForce RTX 4060 Ti
```

因此当前结果不能直接推广到所有 GPU。

不同 GPU 的：

* Architecture
* Memory Bandwidth
* VRAM
* Compute Capability
* Kernel Implementation

可能导致不同结果。

---

## 18.2 单模型系列

当前主要研究：

```text
Qwen3-1.7B
```

因此关于 Quantization 的结论首先应该理解为：

> 当前模型 + 当前 Runtime + 当前 GPU 条件下的实验结果。

而不是所有 LLM 都必然具有完全相同的行为。

---

## 18.3 Runtime 依赖

当前 Benchmark 使用：

```text
llama.cpp
```

不同 Runtime 可能具有不同的：

* Kernel
* Quantization Implementation
* Memory Allocator
* Backend Strategy

例如未来使用其他 Runtime 时，性能结果可能不同。

因此：

```text
llama.cpp 上的结果
```

不能在没有验证的情况下直接推广到所有推理框架。

---

## 18.4 当前缺少 Profiler 证据

目前 Benchmark 已经可以回答：

```text
A 是否比 B 快
```

以及：

```text
Context 增加后性能是否下降
```

但是暂时不能准确证明：

```text
为什么出现这种差异
```

特别是涉及：

* CUDA Kernel
* Dequantization
* Memory Transactions
* Backend Implementation

等底层原因时。

在 Profiler 实验完成之前，这些解释应该标记为：

```text
Mechanism Hypothesis
```

而不是实验事实。

---

## 18.5 Memory 测量限制

当前 Peak VRAM 使用：

```text
外部 GPU Memory Sampling
+
Baseline Subtraction
```

进行测量。

这种方式适合研究：

```text
整体显存变化趋势
```

但是不能直接把显存拆分成：

```text
Weights
KV Cache
Compute Buffers
Runtime Buffers
Temporary Allocations
```

因此：

```text
Peak VRAM 增加
```

不能直接写成：

```text
KV Cache 增加
```

除非后续获得更多证据。

---

## 18.6 Quality Benchmark 限制

当前 Quality Benchmark 只使用：

```text
WikiText-2 Perplexity
```

它适合用于观察 Quantization 的基础质量变化。

但不能全面表示：

* Instruction Following
* Reasoning
* Long Context
* Coding
* Real-world Application Quality

未来如果项目继续扩展，可以增加更多 Quality Benchmark。

---

# 19. Benchmark 核心原则

本项目遵循以下 Benchmark 原则：

```text
先测量
↓
控制变量
↓
保留原始数据
↓
重复实验
↓
量化实验波动
↓
区分实验事实与机制解释
↓
在声称底层原因之前获得 Profiler 证据
```

即：

```text
Measure first.
Control variables.
Preserve raw data.
Repeat experiments.
Quantify variability.
Separate observation from explanation.
Use profiler evidence before claiming low-level causes.
```

本项目最终目标并不是简单得到：

```text
哪个量化最快
```

这样的排序。

而是建立对：

```text
Quantization
Quality
Memory
Prefill
Decode
Context
```

之间 Trade-off 的系统理解，并确保所有结论都有明确的实验配置和可复现证据支持。
