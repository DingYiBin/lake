# InputBatch / InputBuffers Device 化

> 目标：把本步执行热路径从 Python list/CPU 迁到**固定地址 device tensor**，对齐 vLLM V2，为 FA2 / 未来 CUDA graph 铺路。  
> 权威请求状态仍在 host `Req`（`node_scheduler`）；device 只持本步镜像。  
> 关联：[`compute-layer.md`](compute-layer.md) C11、[`../research/vllm/compute.md`](../research/vllm/compute.md)、[`../research/scheduler-worker-interface.md`](../research/scheduler-worker-interface.md)。

## 参考实现

| 概念 | 路径:符号 |
|------|-----------|
| 固定 device buffer | `vllm/v1/worker/gpu/input_batch.py::InputBuffers` |
| 混合 InputBatch | 同文件 `InputBatch`（device 切片 + numpy/CPU 控制字段 + host `req_ids`） |
| H2D | `vllm/v1/worker/gpu/buffer_utils.py::async_copy_to_gpu` / `UvaBuffer` |
| Prefill pack | 同 `input_batch.py` Triton `_prepare_prefill_inputs_kernel`（依赖 device request state） |

**关键差异**：vLLM worker 自持跨步 `RequestState` + block table；lake 是 host `Req` + **agent 出表**，引擎只读固定地址视图。device 化不引入 runner 跨步状态机。

## 目标形态

```
SchedulerOutput + host Req
        │  几何权威仍在 host
        ▼
InputBatch（host 轻量：req_ids / query 区间 / phase；token 暂 host）
        │  materialize
        ▼
InputBuffers（预分配 device tensor；CUDA 可经 pin staging H2D）
        │
        ▼
AttentionMetadata（device 视图：qsl / seq_lens / slot_mapping / block_table_2d）
        │
        ▼
model forward / FA2 /（未来 graph）
```

## 字段可行性

| 字段 | 迁 device？ | 可行性 | 说明 |
|------|-------------|--------|------|
| `InputBuffers.input_ids/positions/query_start_loc/seq_lens/slot_mapping/is_padding` | 要 | **高** | P0 |
| `block_table` 2D pad + lens | 要 | **中高** | P1；内容由 agent 表填充 |
| `AttentionMetadata` 热路径张量 | 要 | **高** | P1；去掉 FA2 每步 `torch.tensor(list)` |
| `InputBatch.token_ids` dict | 暂不整迁 | **低→中** | 无 device RequestState 前 host pack |
| `req_ids` / `is_prompt_phase` / 调度计数 | 留 host | **保持** | 无 kernel 语义 |
| device Triton pack / UVA | 后置 | **中低** | P3；pin+async 足够先上 |

## 阶段

### P0 — `InputBuffers` device 化（本轮）

- 预分配 `torch` tensor；构造参数 `device`（默认 `cpu` 保单测，生产 `cuda`）
- CUDA：CPU pin staging → `non_blocking` `copy_`
- `materialize` 仍从 host `InputBatch` 填本步切片

### P1 — `AttentionMetadata` device 视图（本轮）

- `query_start_loc` / `seq_lens_ordered` / `slot_mapping` / `positions` → tensor（多来自 buffer 切片）
- `block_table_tensor` → 固定 `[max_reqs, max_blocks]` int32 + `block_table_lens`
- FA2 / CpuAttention 读 tensor，不再现场从 Python list 建临时 tensor

### P2 — 压缩 host `InputBatch`（本轮）

- dict-by-req_id → **行稠密 list**（平行于 `req_ids`，row = 在 `req_ids` 中的位置），对齐 vLLM V2 `InputBatch` 行索引形态
- `InputBatch.add_request(...)` 按行追加；`index_of(req_id)` 按需反查
- `InputBuffers.materialize` 直接按行迭代（不再 `dict[req_id]` 查找）
- `prepare_attn` 由行稠密 batch 构一过性 req_id-keyed dict 视图喂 `build_attn_metadata`（builder 仍取 dict 几何权威，亦可来自 scheduler）

### P3 — device pack（后续，可选）

> 单独设计节：vLLM/SGLang 的 device pack 依赖 runner 跨步 device token 历史，lake 无此状态，需厘清 P3 对 lake 的真实含义。

**参考实现**

| 概念 | 路径:符号 |
|------|-----------|
| prefill token scatter kernel | `vllm/v1/worker/gpu/input_batch.py::_prepare_prefill_inputs_kernel`（经 `idx_mapping` 读 device 常驻 `all_token_ids`，scatter 进静态 `input_ids`） |
| positions/seq_lens kernel | 同文件 `_prepare_pos_seq_lens_kernel` / `prepare_pos_seq_lens`（device 上算 `positions`/`seq_lens`/`query_start_loc`） |
| SGLang device token | `ScheduleBatch`/`ForwardBatch`：prefill `prefill_input_ids_cpu`(pinned)→`input_ids` GPU；decode 从 FutureMap device relay gather |

**张力（为什么 P3 不能照搬 vLLM）**

vLLM/SGLang 的 device pack 价值 = **省 token H2D**——token 历史已 device 常驻（vLLM runner 跨步 `RequestState` 的 `all_token_ids [num_reqs_state, max_model_len]`；SGLang `ForwardBatch` GPU `input_ids` + FutureMap relay），kernel 直接读 device tensor scatter。lake 的 runner **无跨步状态**（`Req` 含 token 历史在 host `node_scheduler`，引擎无跨步请求表——比 SGLang 更彻底），token 必须从 host H2D，**无法从 device tensor 读**。因此"省 token H2D"这条收益在 lake 不成立。

**P3 对 lake 的真实含义**

P3-for-lake ≠ 省 token H2D，而是 **把几何 packing（`positions`/`seq_lens`/`query_start_loc`/`slot_mapping` scatter）从 host Python 循环迁到 device Triton kernel**，token 字节仍 H2D 进静态 `input_ids` buffer。收益有二：
1. 去 host 热路径 Python 循环（`materialize` 现为 per-token Python loop）。
2. **使能 CUDA graph capture**——graph replay 要求 packing 全为 device kernel（host Python loop 不可 capture）。

**与「runner 无跨步状态」的兼容**

几何 packing kernel 只读本步 `SchedulerOutput` 派生的少量标量（per-req `num_computed`/`query_start`/`query_end`/`q_len`）+ agent 的 `slot_mapping`/`block_table`，**不读跨步 token 历史**。token 字节由 host staging H2D 进静态 buffer（capture 时该 H2D 在 graph 外，或经 async copy stream）。故 P3-for-lake 不引入 runner 跨步 `RequestState`，与 Q1/Q2 契约兼容。

**结论**

- vLLM 式 device pack（省 token H2D）**不适用** lake（token host by design）。
- P3-for-lake = 几何 device kernel → **graph-capture 使能器**。而 CUDA graph capture 本身标后置（D6），故 P3 是**前置使能、非独立热路径收益**。
- 取舍：现在实现 = 前瞻投资（kernel 不上当前热路径，等 graph capture 落地才生效）；defer = 等 graph capture 提上日程时一并做（kernel 与 capture 形状/约束同设计更稳）。

### P3.1 — 固定 shape padding（graph capture 地基，本轮做）

> graph capture 的核心约束是**输入 shape 固定**：把 `num_reqs` / `num_tokens` pad 到配置的固定值，forward 永远看到同一形状，靠 `is_padding` mask 区分真实/pad。**本轮只做固定 shape 的 padding 基础设施，不实现 graph capture 本身**（graph 仍后置；本节是它的地基，也使 P3 几何 kernel 的形状约束提前钉死）。

**参考实现**

| 概念 | 路径:符号 |
|------|-----------|
| `num_reqs_after_padding` / `num_tokens_after_padding` | `vllm/v1/worker/gpu/input_batch.py:41,56`（InputBatch 字段） |
| `is_padding` pad 段标记 | `vllm/v1/worker/gpu/model_runner.py:870-873`（`[:num_tokens]=False`、`[num_tokens:padded]=True`） |
| `query_start_loc` 非递减 pad | `vllm/v1/worker/gpu/model_runner.py:924-929`（pad req 零长 query，尾段填 `num_tokens`） |
| `num_reqs_padded` 驱动 forward shape | `vllm/v1/worker/gpu/model_states/default.py:148-149`（FULL graph 用 padded） |

**lake 落点**

- `RoleConfig`：加 `pad_num_reqs` / `pad_num_tokens`（0=变长/关；>0=pad 到固定值）+ env `LAKE_PAD_NUM_REQS` / `LAKE_PAD_NUM_TOKENS`；校验 ≤ `InputBuffers` max。
- `InputBuffers`：`materialize` 已把 pad 区置好（`is_padding=True`、`slot_mapping=-1`、`query_start_loc` 非递减、pad 行 `seq_lens`/`block_table_lens`=0）。加 `set_padding(padded_num_reqs, padded_num_tokens)` + `effective_num_reqs` / `effective_num_tokens`（= padded 或 actual）。
- `AttentionMetadata`：加 `padded_num_reqs` / `padded_num_tokens` / `is_padding`；`build_attn_metadata` 切到 **effective（padded）shape**——`num_reqs` / `num_actual_tokens` = padded（forward 形状），另加 `actual_num_reqs` / `actual_num_tokens`（sampler/host 用真实计数）。padding 关时 effective=actual，行为不变。
- `ModelRunner`：构造期按 `RoleConfig` 调 `InputBuffers.set_padding`；`_forward_model` / `sample_tokens` 仍按 `batch.req_ids`（真实）迭代——pad 不产 logits；`is_padding` 供未来真实 forward 的 logits 收集用。
- `CpuAttentionBackend.forward_varlen`：已跳过 `q_len<=0` 的 pad req（line 120-121），固定 shape 天然兼容。

**不做**

- 不实现 CUDA graph capture / replay（仍后置）。
- 不做 device Triton pack kernel（P3 主体，待 graph capture 提日程）。
- 不改 dummy/scheduler 路径的默认行为（padding 默认 off）。

## 不做

- 把 host `Req` / 调度字典搬上 GPU  
- 为迁 tensor 在 runner 重建跨步 RequestState  
- 本轮不上 UVA

## 状态

| 阶段 | 状态 |
|------|------|
| 计划文档 | **done** |
| P0 InputBuffers | **done**（`device` + pin staging H2D；默认 cpu） |
| P1 AttentionMetadata | **done**（热路径 tensor + 2D `block_table`；FA2/Cpu 消费） |
| P2 压缩 host `InputBatch` | **done**（行稠密 list 平行于 `req_ids`；`add_request` / `index_of`；materialize 按行迭代） |
| P3 device pack | **设计 done**（vLLM 式省 H2D 不适用；lake 价值=几何 device kernel→graph-capture 使能器；graph capture 后置→P3 实现待 graph capture 提日程时一并做） |
| P3.1 固定 shape padding | **done**（`RoleConfig.pad_num_reqs/tokens` + env；`InputBuffers.set_padding`/`effective_*`；`AttentionMetadata` padded shape + `is_padding` + `actual_*`；CPU 后端跳 pad req；默认 off，行为不变） |
