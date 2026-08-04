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

### P2 — 压缩 host `InputBatch`（后续）

- dict → 按 batch 行稠密；或 `prepare_inputs` 直写 staging

### P3 — device pack（后续，可选）

- token 镜像归属 + Triton pack；与「runner 无跨步状态」需单独设计

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
| P2 / P3 | pending |
