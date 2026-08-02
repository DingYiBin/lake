# Attention 后端对照（vLLM × SGLang）

> 源码锚点（行号会漂移，以符号为准）：
> - vLLM：`vllm/v1/attention/backend.py::{AttentionBackend,AttentionMetadata,CommonAttentionMetadata,AttentionMetadataBuilder,AttentionImpl}`；`vllm/v1/attention/backends/{flash_attn,cpu_attn,triton_attn}.py`；`vllm/v1/attention/backends/fa_utils.py`；`vllm/v1/worker/{gpu_model_runner.py,gpu/model_runner.py,cpu_model_runner.py,cpu/model_runner.py}`；`csrc/cpu/cpu_attn.cpp::cpu_attention_with_kv_cache`；`cmake/external_projects/vllm_flash_attn.cmake`
> - SGLang：`python/sglang/srt/layers/attention/{base_attn_backend,attention_registry,torch_native_backend,triton_backend,flashinfer_backend,intel_amx_backend}.py`；`python/sglang/srt/model_executor/{model_runner,forward_batch_info}.py`；`python/sglang/srt/managers/{schedule_batch.py::mix_with_running,scheduler.py::is_mixed_chunk}`；`python/sglang/srt/server_args.py::_handle_cpu_backends`
> 相关：[`vllm/compute.md`](vllm/compute.md)、[`sglang/model-runner.md`](sglang/model-runner.md)、lake [`../architecture/compute-layer.md`](../architecture/compute-layer.md)「计算引擎结构」/ D4 / Q1 / Q2。

本文梳理 **attention 后端的基类/分派/metadata 形态、各平台 kernel 来源、CPU 纯 torch 路径、model runner 是否按平台子类化、混部（MIXED）与并行度的关系**——供 lake 定 attention backend 路线、model runner 形态、MIXED 语义时对照。

---

## 1. 基类与分派形态

### vLLM：单一 `forward` + `AttentionImpl`，metadata 由 builder 构造

`vllm/v1/attention/backend.py`：

- `AttentionBackend`（ABC，注册表式选择）——后端元信息（支持的 dtype/head_size/block_size/sliding_window/...）+ `get_impl_cls` / `get_builder_cls`。
- `AttentionMetadata`（基类，空）+ `CommonAttentionMetadata`（`backend.py:402`）——批级共享字段：`query_start_loc [B+1]`、`seq_lens [B]`、`num_actual_tokens`、`max_query_len`、`max_seq_len`、`block_table_tensor`、`slot_mapping`、`causal`、`positions`、`is_prefilling`、`rswa_prefix_lens` 等。
- `AttentionMetadataBuilder`（`backend.py:573`）——从 `CommonAttentionMetadata` 构造 per-layer `AttentionMetadata`（各后端子类，如 `FlashAttentionMetadata`）。
- `AttentionImpl`（`backend.py:820`）——消费 paged KV 做前向，单一 `forward(layer, query, key, value, kv_cache, attn_metadata, output)`，内部靠 metadata 判 mode（无显式 `forward_decode`/`forward_extend` 分派）。

形态：**metadata 是独立 dataclass**，由 builder 从 `CommonAttentionMetadata` 构造，传给 `AttentionImpl.forward`。mode 判断在 impl 内部（如 `is_prefilling` 字段）。

### SGLang：按 `forward_mode` 分派 + `ForwardBatch` 即 metadata + backend 自建 metadata

`python/sglang/srt/layers/attention/base_attn_backend.py:18`：

- `AttentionBackend(ABC)`——基类 `forward`（`base_attn_backend.py:160`）按 `forward_batch.forward_mode` 分派到 `forward_decode` / `forward_extend` / `forward_mixed`（子类实现）。
- **metadata 生命周期按 graph capture 拆三段**（`base_attn_backend.py:47-89`）：
  - `init_forward_metadata(fb)` — eager 入口（默认 = out_graph + in_graph）
  - `init_forward_metadata_out_graph(fb, in_capture=False)` — 每步 metadata 准备，**在 graph capture 之外**（host op、动态 shape、`.cpu()`/`.tolist()` 都在这）
  - `init_forward_metadata_in_graph(fb)` — **可被 graph 录制的静态 shape GPU op**，capture 时录制、replay 时自动重放
- **`ForwardBatch` 本身就是 metadata 载体**（`forward_batch_info.py:353`）：`forward_mode` / `req_pool_indices` / `seq_lens` / `out_cache_loc` / `extend_seq_lens` / `extend_prefix_lens` / `seq_lens_cpu`。backend 在 `init_forward_metadata(fb)` 里从 `ForwardBatch` 算出 kernel 要的形态（indptr、indices 等）存进 `self.forward_metadata`。

形态：**无独立 `AttentionMetadata` dataclass**，`ForwardBatch` 即 metadata；backend 自己建 per-layer metadata。mode 分派在基类，子类按 mode 实现不同方法。

### 对照

| | vLLM | SGLang |
|---|---|---|
| metadata 载体 | 独立 `AttentionMetadata` dataclass | `ForwardBatch` 本身 |
| metadata 构造 | `AttentionMetadataBuilder` 从 `CommonAttentionMetadata` 构造 | backend 自己 `init_forward_metadata(fb)` 算 |
| forward 分派 | 单一 `AttentionImpl.forward`，内部靠 metadata 判 mode | 基类按 `forward_mode` 分派 `forward_decode`/`forward_extend`/`forward_mixed` |
| paged 寻址粒度 | `block_table`（block 级）+ `slot_mapping`（token 级写） | `req_to_token`（token 级，更细） |
| graph 支持 | `AttentionCGSupport` 标记 | `init_forward_metadata_out_graph`/`_in_graph` 拆分 |

**对 lake**：lake 已选 vLLM 形态——独立 `AttentionMetadata`（`attention_metadata.py:16`，字段与 `CommonAttentionMetadata` 镜像）+ `build_attn_metadata` 构造 + `Qwen3PagedAttention.forward` 单一入口。保持现状即可；graph 支持的 out/in 拆分留 C12+ 后续生产化时参考 SGLang。

---

## 2. 各平台 kernel 来源

### vLLM（`fa_utils.py:21-62` 按平台分派）

| 平台 | kernel 来源 | 形态 |
|---|---|---|
| **CUDA** | `vllm.vllm_flash_attn` → `_vllm_fa2_C`/`_vllm_fa3_C` | **vLLM 自己的 fork**（`vllm-project/flash-attention`），编译成 C++ 扩展；`flash_attn_varlen_func` 签名带 `block_table`/`seqused_k`/`scheduler_metadata`/`dynamic_causal`/`s_aux` |
| **ROCm** | `from flash_attn import flash_attn_varlen_func` | **stock 上游** `flash-attn` 包（Dao-AILab） |
| **XPU** | `vllm._xpu_ops.xpu_ops.flash_attn_varlen_func` | vLLM 自写 XPU kernel |
| **CPU** | `ops.cpu_attention_with_kv_cache`（`csrc/cpu/cpu_attn.cpp:169`） | **完全独立的 C++ kernel**，按 ISA 分派（AMX/AVX-512/NEON/RVV/VXE/VSX），与 flash-attn 无关 |

**关键**：vLLM CUDA 用的不是 stock `flash-attn`，是 `vllm-project/flash-attention` fork（`cmake/external_projects/vllm_flash_attn.cmake:39-46` 构建时 `FetchContent` 拉取，pinned commit `caaa4eb...`）。stock `flash_attn_varlen_func` **不接 `block_table`**，做不了 paged attention。要拿到 vLLM 那套签名（`block_table`+`seqused_k`+`scheduler_metadata`），只有装 vLLM 自己或独立构建其 fork。

### SGLang

| 平台 | backend | kernel 来源 |
|---|---|---|
| **CUDA（首选）** | `flashinfer` | `flashinfer` PyPI 包（`BatchPrefillWithRaggedKVCacheWrapper`/`BatchDecodeWithPagedKVCacheWrapper`，plan+run） |
| **CUDA（备选）** | `triton` | `sgl_kernel` 包（SGLang 自写 Triton kernel 集：`extend_attention_fwd`/`decode_attention_fwd`） |
| **CUDA（FA3/4）** | `fa3`/`fa4` | 同 vLLM 的 `vllm-project/flash-attention` fork |
| **CPU（ARM）** | `torch_native` | 纯 `torch.nn.functional.scaled_dot_product_attention` + per-request 循环 |
| **CPU（x86）** | `intel_amx` | Intel IPEX AMX kernel |
| **NPU（Ascend）** | `ascend` | Ascend 自写 kernel（含 `forward_mixed`） |
| **XPU** | `intel_xpu` | Intel XPU kernel |

CPU 默认选择在 `server_args.py:3415-3421`：ARM → `torch_native`，x86 → `intel_amx`。

---

## 3. CPU 纯 torch 路径：vLLM 废弃 vs SGLang 活路径

### vLLM：SDPA 路径已废弃

`vllm/v1/attention/backends/cpu_attn.py:125-129` 的 `CPUAttentionMetadata` 有死字段：

```python
# can be removed after deprecate sdpa
use_sdpa_prefill: bool = False
sdpa_attn_masks: list[torch.Tensor | None] | None = None
sdpa_start_loc: torch.Tensor | None = None
```

全仓 grep 这三个字段**除定义处外无任何赋值或读取**——死代码。vLLM CPU 现走 `ops.cpu_attention_with_kv_cache`（C++/ISA kernel），SDPA 路径走过又放弃（追求 x86 CPU 生产性能，改写 C++ AMX）。

### SGLang：`torch_native` 是 ARM CPU 默认生产路径

`torch_native_backend.py:61` 的 `_run_sdpa_forward_extend`：per-request Python 循环 + gather paged KV（`k_cache[req_to_token[req_pool_idx, start:end]]`）+ pad query 到 full seq_len + SDPA `is_causal` + slice 出新 query 段。注释只写"待优化"，没写"要废弃"——是 ARM CPU 的默认生产路径（`server_args.py:3418`）。

### 对 lake 的启示（方案 A 验证）

SGLang `torch_native` 正是 lake `CpuAttentionBackend` 应走的**方案 A**（per-request 循环 + gather paged + SDPA），且是**活路径**。vLLM 的 SDPA 路径死了不代表方案 A 不可行——vLLM 死掉是因为它要追求 x86 CPU 生产性能（改写 C++ AMX），不是方案 A 本身有问题。lake 的 CPU 是 dev/test 路径，不是生产，方案 A 完全合理。

**方案 A 步骤**（对齐 SGLang `torch_native`）：
1. 按 `query_start_loc` 切请求
2. 每请求 gather paged KV：`k_cache[block_table[i] → slots]` → contiguous
3. extend：pad query 到 full seq_len（前缀 KV 全可见 + 新 query 段内部 causal），调 SDPA `is_causal=True`，slice 出新 query 段
4. decode（q_len=1）：不 pad，直接调 SDPA `is_causal=False`（单 token 对全部 KV full-visible）

---

## 4. Model runner 形态：子类化 vs 单类

### vLLM：CPU runner 是 GPU runner 的子类（历史包袱）

两个正交维度：

**V1 vs V2（代际重构，与平台无关）**：
- `vllm/v1/worker/gpu_model_runner.py` — V1 单体大文件
- `vllm/v1/worker/gpu/model_runner.py` — V2 模块化（`gpu/{states,input_batch,block_table,spec_decode,sample,kv_connector}`）
- 由 `vllm_config.use_v2_model_runner` 选（`gpu_worker.py:402-416`），**两个都是 GPU runner**

**CPU runner = GPU runner 子类**：
- `vllm/v1/worker/cpu_model_runner.py`（V1 时代，~240 行）— `class CPUModelRunner(GPUModelRunner)`，补丁：`CpuGpuBuffer.gpu=.cpu`、`use_cuda_graph=False`、`cascade_attn_enabled=False`、Triton 后处理、torch accelerator→noop、CUDA wrapper context
- `vllm/v1/worker/cpu/model_runner.py`（V2 时代，17 行）— `class CPUModelRunner(GPUModelRunner)`，只覆盖 `warming_up_model`

**结论**：vLLM 的 CPU runner 继承 GPU runner 全部逻辑，只覆盖设备相关状态。真正平台差异（attention kernel）在 backend 注册表，不在 runner。子类化是历史原因（V1 单体，CPU 后加，子类化最小侵入）；V2 时代 CPU runner 缩到 17 行，正在向单类收敛。

### SGLang：单一 `ModelRunner` + device 字段，无子类化

`model_executor/model_runner.py:235` 的 `ModelRunner` 单类，`self.device = server_args.device`（`"cpu"`/`"cuda"`/`"xpu"`/`"hpu"`/`"npu"`）参数化。平台差异走：
1. **attention backend 注册表**（`attention_registry.py`）：`torch_native`/`intel_amx`/`triton`/`flashinfer`/...，由 `server_args.attention_backend` 选
2. **类内 device 条件分支**：`if self.device == "cpu": self.init_threads_binding()` 等

**无 per-platform 子类**。这是 vLLM V2 时代正在收敛到的形态。

### 对 lake

lake 的 `ModelRunner`（`model_runner.py:52`）已是 SGLang 形态：单一类 + `model_backend` 字符串 + `build_attn_backend(name)` 注册表 + 无子类化。**不需要引入 vLLM 式的 CPU/GPU runner 分裂**。平台差异全进 attention backend；runner 内必要时加 `if device == "cpu"` 分支（如 warmup 跳过 graph capture），但不子类化。

---

## 5. 混部（MIXED）：同节点 prefill + decode

### SGLang 两种粒度

**A. Chunked prefill（默认）——同节点、不同 step**

`chunked_prefill_size` 设了但 `enable_mixed_chunk=False`（`scheduler.py:1003-1006`）。调度器**不同 step 交替**跑 prefill 和 decode：一步切一段长 prompt，下一步跑所有 running 请求的 decode。节点同时承担两角色，但每次 forward 要么全 prefill 要么全 decode（`forward_mode` = EXTEND 或 DECODE，不混）。这是 SGLang/vLLM 的默认混部形态。

**B. Mixed chunk（`enable_mixed_chunk=True`）——同 batch、同 forward**

调度器把 running decode 请求和新 prefill 请求**合并进同一个 batch**，一次 forward 同时算。入口 `scheduler.py:3056-3074`；合并逻辑 `schedule_batch.py:2534` 的 `mix_with_running`：

```python
def mix_with_running(self, running_batch: ScheduleBatch):
    self.forward_mode = ForwardMode.MIXED
    for req in running_batch.reqs:
        full_len = len(req.full_untruncated_fill_ids)
        req.set_extend_range(full_len - 1, full_len)   # query 段 = 最后 1 token
    ...
    self.extend_lens = self.extend_lens + [1] * running_bs   # decode → extend_len=1
```

**关键手法**：每个 running decode 请求被改造成 **"长度为 1 的 extend"**——`set_extend_range(full_len - 1, full_len)`（query 段 = 最后 1 个 token），`extend_lens += [1]`。新 prefill 请求保持自己的 extend 长度。合并后 `forward_mode = MIXED`，但本质是"一批 extend，部分 extend_len=1（原 decode）、部分 extend_len=N（原 prefill）"。

### attention backend 怎么处理 MIXED

**CUDA 后端：MIXED 走 `forward_extend`，不另开 `forward_mixed`**。

基类 `base_attn_backend.py:184` 的分派：只有 **NPU**（`is_npu()`）才走 `forward_mixed`（`ascend_backend.py:2776`）。CUDA 的 flashattention / triton backend 都让 MIXED 落进 `forward_extend`——因为 `is_extend_or_draft_extend_or_mixed()` 对 MIXED 返回 True，metadata 初始化和 forward 路径都把 MIXED 当 extend 处理。

`cu_seqlens_k`（每请求 KV 长度前缀和）+ 每请求不同的 `extend_lens`（1 或 N）喂给 varlen kernel——**varlen 天然支持同批不同 q_len**，所以 MIXED 不需要特殊 kernel，只是 extend 的一个特例（q_len 集合里混了 1 和 N）。

NPU 例外是因为 Ascend 的 attention kernel 对 q_len=1 和 q_len>1 走不同优化路径，混批效率差，所以单独写 `forward_mixed`。

### 对 lake

lake 的 `ForwardMode` 已定义 `MIXED`（`compute-layer.md` D1 节），`SchedulerOutput.forward_mode` 会标 MIXED。SGLang 的做法给出两条路：

1. **如果 lake 的 attention backend 用 varlen kernel（Triton/flash-attn fork）**：MIXED 不需要单独 `forward_mixed`——调度侧把 decode 请求的 `num_scheduled_tokens` 设为 1、`query_start/query_end` 设成最后 1 个 token，runner 侧 `forward_extend` 的 varlen 路径自然处理。这与 lake 的 `prepare_inputs`（`model_runner.py:182`）现有逻辑一致——它已按 `num_scheduled_tokens` 切 query 区间，decode 是 n=1 的特例。
2. **如果 lake 的 `CpuAttentionBackend`（纯 torch SDPA）**：per-request 循环（方案 A）也天然支持 MIXED——每请求独立调 SDPA，q_len=1 的走 decode 形态（不 pad）、q_len>1 的走 extend 形态（pad 到 seq_len）。和 SGLang `torch_native` 的 `forward_extend` 处理 MIXED 同款。

**结论**：lake 走 varlen 路径（Triton 或纯 torch 方案 A）后，MIXED 是"调度侧把 decode 标成 extend_len=1 + runner 侧 varlen 自然吃"的结果，不需要单独的 `forward_mixed`。这与 lake 现有的 `prepare_inputs`（按 `num_scheduled_tokens` 切 query）已经对齐，落地时只需确认 `forward_mode=MIXED` 时 runner 不走特殊分支即可。

---

## 6. MIXED 与并行度的关系

### 核心约束：并行度是进程级配置，不是 per-step/per-request 属性

SGLang 的并行度（TP/PP/DP/EP）在进程启动时 `init_distributed` 建好，rank 分配、NCCL communicator 都固化。`attn_tp_size` / `tensor_model_parallel_size` / `pipeline_model_parallel_size` / `moe_ep_size` 都是启动参数，**不能 per-step 或 per-request 切**。

MIXED 把 decode 伪装成 extend_len=1 的 extend，和 prefill 合进同一个 batch 走同一次 `forward_extend`。这次 forward 里所有请求过**同一套 layer 模块**、同一个 TP group all-reduce、同一组 EP group all-to-all。请求 A（prefill）和请求 B（decode）在同一次 forward 里**共享这些 communicator**——不存在"请求 A 用 TP=4、请求 B 用 TP=1"的可能。

### SGLang 真正支持"prefill/decode 不同并行度"的机制：PD 分离（disaggregation）

`disaggregation/utils.py` 的 `DisaggregationMode.PREFILL` / `DisaggregationMode.DECODE`——**完全独立的进程/实例**，各自有自己的启动配置：
- prefill 实例可以 `tp_size=8`（算力密集，要大 TP）
- decode 实例可以 `tp_size=1, dp_size=8`（访存密集，DP 更优）

两者经 KV 传输（Mooncake/NIXL/memfabric）交接。**但这不是 MIXED**——是两个独立进程，不是同节点同 batch。

### 对 lake

这正好对应 lake 已定的三模式（`CLAUDE.md` 执行模式节）：

| lake 模式 | prefill/decode 并行度能否不同 | 对应 SGLang 机制 |
|---|---|---|
| **混部**（同节点同 batch） | **不能**——同进程同 forward 同 communicator | SGLang `enable_mixed_chunk`（MIXED） |
| **PD 分离**（不同节点） | **能**——P 和 D 是不同进程，各自并行配置 | SGLang `disaggregation_mode=prefill/decode` |
| **D-direct**（前缀 KV 已在执行节点 HBM） | decode 侧单一并行配置（D 节点） | SGLang 无直接对应（lake 增量） |

**关键结论**：lake 的"混部"模式（同节点同 batch）和"PD 分离"模式（不同节点不同并行配置）是**互斥的并行度策略**——混部要共享并行度，PD 分离才能差异化并行度。这正是 lake 设计成三种独立模式、由 Router 逐请求选路的原因：要 prefill 大 TP + decode DP 的差异化并行，**必须走 PD 分离**，不能在混部里做。

lake 的 `compute-layer.md` D1 节 `ForwardMode.MIXED` 的定义（"同批两种几何"）已经隐含了这一点——MIXED 是几何混合（q_len 不同），不是并行度混合。

---

## 7. 对 lake 的总结决策

| 议题 | 决策 | 依据 |
|---|---|---|
| attention backend 路线 | GPU 生产走 FA2 paged varlen（`FlashAttn2Backend`，上游 `flash-attn` 包，非入图）；CPU/dev/test 走纯 torch SDPA 方案 A（`CpuAttentionBackend.forward_varlen`，per-request 循环 + gather paged + SDPA） | SGLang `torch_native` 验证方案 A 可持续；vLLM CPU 改 C++ 是为 x86 生产，lake CPU 非生产 |
| 是否装 vllm | **不装**——vllm 是单体 serving 系统非 library，import 副作用重、要编译 C++、与 lake Q1/Q2 架构冲突 | `fa_utils.py` 显示 vLLM CUDA 用自家 fork，CPU 用独立 C++ kernel，无纯 torch 路径可借 |
| GPU kernel 是否借 flash-attn | 优先上游 `flash-attn` 包（FA2 varlen 已带 `block_table`）；上游若不支持小 `block_size`（`with_kvcache` 要求 256 倍数）或入图 `out=`，再回退 `vllm-project/flash-attention` fork | 上游 `flash_attn_varlen_func` 签名带 `block_table`；fork 优势在小 block_size + `out=` + `seqused_k` |
| metadata 形态 | 保持 vLLM 形态（独立 `AttentionMetadata` + `build_attn_metadata`），不学 SGLang 把 metadata 并进 ForwardBatch | lake `AttentionMetadata` 字段已与 `CommonAttentionMetadata` 镜像、与 varlen kernel 入参对齐 |
| model runner 形态 | 保持 SGLang 形态（单一 `ModelRunner` + `model_backend` 字段 + 注册表），**不**按平台子类化 | vLLM V2 时代 CPU runner 缩到 17 行，正在向单类收敛 |
| 后端注入（送达层） | **构造期注入**：runner 经 `build_attn_backend(name)` 建实例，沿模型树构造期传到 `Qwen3PagedAttention.backend`；模型层不 import 任何具体后端 | vLLM 每层 `Attention.__init__` 调 `get_impl_cls()` 实例化自己的 impl + metadata/kv_cache 经 forward context；SGLang runner 持单实例、层经 `get_attn_backend()` 取（forward context）。lake 取折中：runner 持单实例（SGLang 式，后端无状态）+ 构造期注入（vLLM 式，免 forward context 改造） |
| MIXED 处理 | 不单独 `forward_mixed`——调度侧把 decode 标成 extend_len=1，runner 侧 varlen 自然吃 | SGLang CUDA 后端 MIXED 走 `forward_extend`，varlen 天然支持同批不同 q_len |
| 混部 vs PD 分离并行度 | 混部共享进程级并行度；prefill/decode 差异化并行**必须走 PD 分离**（Router 选路），不能在混部里做 | 并行度是进程级配置，MIXED 同 forward 同 communicator |

### 参考实现与关键差异

- **参考 vLLM**：`CommonAttentionMetadata`（`backend.py:402`）字段形态；`FlashAttentionImpl.forward`（`flash_attn.py:811`）varlen+paged 入参契约；`CPUAttentionBackendImpl.forward`（`cpu_attn.py:316`）+ `csrc/cpu/cpu_attn.cpp:169` CPU C++ kernel；废弃 SDPA 字段（`cpu_attn.py:125-129`）反证纯 torch SDPA 非生产路径。
- **参考 SGLang**：`base_attn_backend.py` 按 `forward_mode` 分派 + `init_forward_metadata_out_graph`/`_in_graph` 拆分；`torch_native_backend.py:61` 方案 A 活参考；`schedule_batch.py:2534` `mix_with_running` decode→extend_len=1；`disaggregation/utils.py` PD 分离机制。
- **关键差异**：vLLM/SGLang 引擎既拥有 KV 又发起传输；lake engine 不知地址、不组装 block table（Q1/Q2）。vLLM CPU 用 C++/ISA kernel；lake CPU 走纯 torch 方案 A（dev/test，非生产）。vLLM runner 按平台子类化（历史）；lake/SGLang 单类 + device 字段。MIXED 在 vLLM/SGLang 是几何混合，并行度差异化靠 PD 分离——lake 三模式与之同构。
