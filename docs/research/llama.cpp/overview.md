# llama.cpp 参考总览

`3rdparty/llama.cpp`（[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，浅克隆 `--depth 1`，HEAD `7a2db1a`）。

引入目的：lake 计划在 **CPU 上也做正确性验证**，需要一份"生产级 CPU attention"的参照来对照 lake 的纯 torch SDPA 路径（`CpuAttentionBackend.forward_varlen`，方案 A）。llama.cpp 是最成熟的 CPU 推理引擎，其 attention 是 ggml 库里手写的 C 实现，正好回答"CPU 上有没有 FA2 等价物"。

> 本项**只作 CPU attention 算子设计参考**，不作存储/调度参考（llama.cpp 是单进程引擎，无存算分离、无分布式池）。lake 的存储/调度参考见其他 submodule。

## 1. attention 用什么：ggml `GGML_OP_FLASH_ATTN_EXT`

llama.cpp 的注意力是 **ggml 张量库里的一个 op**，不是 Dao-AILab `flash-attn`（那是 CUDA kernel），也不是它的移植。是 ggml 自己写的 CPU "flash-like" 实现——借鉴 FlashAttention 的 **tiling 思路**（分块累加 softmax，不物化完整 QK^T→softmax 矩阵），但完全是 CPU SIMD 的 C 代码。

op 注册：`ggml/src/ggml.c::ggml_flash_attn_ext`（L5402），签名：

```c
struct ggml_tensor * ggml_flash_attn_ext(
    ctx, q, k, v, mask, scale, max_bias, logit_softcap);
```

- `q/k/v`：ggml 4D 张量，`permute(0,2,1,3)` 后 result 形状 `{v->ne[0], q->ne[2], q->ne[1], q->ne[3]}`（即 `[DV, H, N, B]`）。
- `mask`：可选 F16 causal/padding mask（`q->ne[2] % mask->ne[2] == 0` 支持广播）。
- `scale`：softmax scale；`max_bias`（ALiBi）、`logit_softcap`（Gemma 风格 softcap）作为参数传入。
- KV cache 由调用方作为 `k/v` 张量传入（llama.cpp 在 graph 里把 KV cache 接成 `k/v`），op 本身不管理 cache 生命周期。

## 2. CPU kernel 结构（`ggml/src/ggml-cpu/ops.cpp`）

CPU 后端实现在 `ggml/src/ggml-cpu/ops.cpp`，分两条路径，由 `ggml_compute_forward_flash_attn_ext_f16`（L9066）按形状分派：

### 2a. split-KV 路径（decode：`neq1==1`，单 query token）

`use_split_kv_path`（L9115）条件：单 query token（`neq1==1 && neq3==1`）+ KV 是 F32/F16 + `nek1 >= 512`。

- 把 KV 维（`nek1`）按线程切 chunk（`chunk_size = (nek1 + nth - 1) / nth`，L9118），每线程算一段 KV 的 partial（`[M, S, VKQ]`，L9121）。
- barrier 后 `ggml_flash_attn_ext_reduce_partials`（L8996）归并所有线程的 partial → 最终输出。
- 这是 **split-KV**（FlashAttention 论文里的 split-KV / split-KV reduce），让 decode 单 token 也能吃满多核。

### 2b. tiled 路径（prefill / extend：`neq1>1`）

`ggml_compute_forward_flash_attn_ext_tiled`（L8706）+ `ggml_compute_forward_flash_attn_ext_f16_one_chunk`（L8468）。

- 按 query 行（`nr = neq1*neq2*neq3`）切 chunk，每线程 4x chunk（`nth_scaled = nth*4`，L9155）做负载均衡。
- 每 chunk 内按 FlashAttention tiling：分块算 QK、online-softmax 累加、再乘 V 累加，全程不物化 `[N, N]` softmax 矩阵。
- `use_ref`（L9112）强制走 vec-only 参考实现（无 tiling、无 KV-chunking），用于校验。

### 2c. SIMD 分派

`one_chunk` 内按编译目标 ISA 分派（`ops.cpp` L10290/L10314/L10502/L10526）：

- `__AVX__` 且非 `__AVX512F__`：AVX/AVX2 路径
- `__AVX512F__`：AVX-512 路径（含 FP16 支持）
- `__ARM_NEON` + `__aarch64__`：NEON 路径

> AMX（`ggml/src/ggml-cpu/amx/`）目前只用于 **matmul（mmq）**，不用于 attention。CPU attention 的 SIMD 全在 `ops.cpp`。其它后端（Metal `ggml-metal-ops.cpp`、CUDA `ggml-cuda/fattn*.cu`、Vulkan、SYCL `ggml-sycl/fattn*`、CANN/Ascend `ggml-cann`、Hexagon HTP）各有自己的 `flash_attn_ext` 实现，本项不参考（lake 生产目标是 GPU/NPU，CPU 仅验证）。

## 3. 对 lake 的含义

### 借鉴点

| llama.cpp | lake 对应 | 说明 |
|---|---|---|
| **split-KV（decode）** | （未来 CPU 性能路径参考） | decode 单 token 用 split-KV 吃满多核；lake 当前 CPU 是验证路径，不做，但若将来要生产 CPU 性能可照此 |
| **tiled prefill（不物化 softmax）** | `CpuAttentionBackend.forward_varlen`（方案 A，torch SDPA） | 同一定位：避免物化完整 attention 矩阵。lake 走 torch SDPA（math 后端）即可，不必自己写 tiling |
| **`use_ref` 参考实现** | lake `forward_varlen` 即"参考实现" | llama.cpp 留一条无优化参考路径用于校验；lake 的纯 torch SDPA 路径正是同款"校验参照" |
| **op 不管理 KV cache 生命周期** | lake KV cache 归存储池 | 一致：attention op 只读 KV cache 张量句柄，不拥有。llama.cpp 调用方接 KV cache 进 graph，lake 由池 L0 arena 提供句柄 |

### 关键差异

- **llama.cpp 是单进程引擎**：无存算分离、无分布式 KV 池、无 PD 分离/混部/D-direct。lake 的核心架构（存储池统一管 L0-L3、Router 逐请求选路）在 llama.cpp 里完全没有对应物。本项**只参考 CPU attention 算子**这一层。
- **ggml 是 C 库内嵌实现，非可组合算子库**：`ggml_flash_attn_ext` 绑在 ggml 计算图里，不是 pip 可装的独立 attention 算子包。lake 不能"装 llama.cpp 当 CPU 后端"——要包它得写 C 扩展 + Python binding，重，且只为 CPU 验证不值。
- **lake CPU 定位是 dev/test 验证**，非生产。llama.cpp 的 SIMD-tiled flash-attn 是**生产 CPU 性能**方案；lake CPU 验证用 `CpuAttentionBackend.forward_varlen`（纯 torch SDPA）即可拿到正确结果，不需要 llama.cpp 的性能优化。

### 结论

- **CPU 上没有 FA2 等价的可 pip 安装算子包**。llama.cpp 的 `ggml_flash_attn_ext` 是"CPU 上的 flash-like 实现"，但它是 ggml C 库内嵌、和引擎绑死，不是独立算子库。
- **lake 的 CPU 验证路径已就绪**：`CpuAttentionBackend.forward_varlen`（纯 torch SDPA，方案 A）在 CPU 上跑通，和 SGLang `torch_native` 同款，是 CPU 上的合理"参照实现"。
- **若将来 lake 要做生产 CPU 推理**（当前不在路线图），再参考 llama.cpp `ggml_flash_attn_ext` 的 split-KV / tiled / SIMD 分派设计。当前阶段不实现。

## 代码索引

| 概念 | 位置（符号） |
|---|---|
| op 注册 + 参数（scale/max_bias/logit_softcap） | `ggml/src/ggml.c::ggml_flash_attn_ext` (L5402) |
| CPU 后端总分派（split-KV vs tiled） | `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_flash_attn_ext_f16` (L9066) |
| split-KV decode kernel | `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_flash_attn_ext_f16_one_chunk` (L8468) + `ggml_flash_attn_ext_reduce_partials` (L8996) |
| tiled prefill kernel | `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_flash_attn_ext_tiled` (L8706) |
| SIMD 分派（AVX / AVX-512 / NEON） | `ggml/src/ggml-cpu/ops.cpp` L10290 / L10314 / L10502 / L10526 |
| AMX（仅 matmul，非 attention） | `ggml/src/ggml-cpu/amx/mmq.cpp` |
| 其它后端 attention（不参考） | `ggml/src/ggml-cuda/fattn*.cu`、`ggml/src/ggml-metal/ggml-metal-ops.cpp`、`ggml/src/ggml-sycl/fattn*`、`ggml/src/ggml-cann/` |
