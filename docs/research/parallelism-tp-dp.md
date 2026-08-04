# TP / DP 并行：通信模块与 Layer 对照（vLLM × SGLang）

> 目标：为 lake 增加 **TP（张量并行）+ DP（数据并行）** 前，先钉清两边「通信栈」与「层内如何切分/汇合」。  
> **本稿范围**：TP + DP 的模块清单、**通信域构造**、vLLM 请求→DP 选路，与数据面/控制面边界。PP / EP / CP / DCP 仅作组拓扑旁注，不展开。  
> 控制面宏观对照（谁 schedule、谁扇出）已写在 [`sglang/model-runner.md`](sglang/model-runner.md)「Data Parallel」「Tensor / Pipeline Parallel」与 [`vllm/compute.md`](vllm/compute.md) 同名节；本文补 **通信原语 + layer 调用点 + mesh/选路细节**。

## 术语（两边同名异义处）

| 词 | vLLM | SGLang | lake 讨论时建议用法 |
|----|------|--------|---------------------|
| **TP** | 权重按列/行切到 `tp_size` 卡；层内 NCCL collectives | 同（Megatron 风格） | **计算面锁步**；KV 若按头切，池侧需记 `(tp_rank)` shard |
| **DP（副本）** | `--data-parallel-size`：每 rank 独立 `EngineCore`+Scheduler+KV | `--dp`：独立 Scheduler 树；引擎内另有 LB | **副本级水平扩展**；选路归 Router，不归 worker 内 LB |
| **dp-attn** | 无同名一等特性 | `--enable-dp-attention`：在同一 TP world 内按 attn_dp **切请求**，MLP/MoE 仍跨卡同步 | **另一类 DP**（请求分片），与副本 DP 勿混；引入则必须显式 IDLE/陪跑 |
| **EP / MoE** | `get_ep_group()`，常跨 `DP×TP` | `_MOE_TP` / `_MOE_EP` / `_MOE_DP` | 本稿不展开；若日后上 MoE，IDLE/dummy 才变硬需求 |

---

## 1. 通信模块（谁提供 collectives）

两边都是：**`GroupCoordinator` 持一组 ranks → 设备 communicator 真正做 AR/AG → `communication_op` 给 layer 薄封装**。

### 1.1 公共形态

```
layer.forward
  → communication_op.tensor_model_parallel_all_reduce / all_gather / …
       → GroupCoordinator.all_reduce / all_gather / …
            → device communicator（Custom AR / PyNCCL / torch.distributed / …）
```

| 能力 | 典型用途（TP） |
|------|----------------|
| `all_reduce` | `RowParallelLinear` 输出汇合；`VocabParallelEmbedding` 分片求和 |
| `all_gather` / `gather` | `ColumnParallelLinear(gather_output=True)`；logits 拼回完整 vocab |
| `reduce_scatter` | 部分 FFN/量化路径（非每层必经） |
| `broadcast` / `send`/`recv` | PP 激活（本稿不展开）；对象广播多走 CPU/gloo |
| `all_to_all` | MoE dispatch/combine（本稿不展开） |

### 1.2 vLLM

| 角色 | 路径:符号 |
|------|-----------|
| 组与 mesh 初始化 | `vllm/distributed/parallel_state.py`::`GroupCoordinator` / `initialize_model_parallel` / `get_tp_group` / `get_dp_group` / `get_pp_group` / `get_ep_group` |
| Layer 薄封装 | `vllm/distributed/communication_op.py`::`tensor_model_parallel_all_reduce` / `_all_gather` / `_reduce_scatter` / `_gather` / `broadcast_tensor_dict` |
| CUDA 侧 AR 分派 | `vllm/distributed/device_communicators/cuda_communicator.py`::`CudaCommunicator.all_reduce` |
| 自定义 AR（同机 P2P） | `…/custom_all_reduce.py`::`CustomAllreduce` |
| PyNCCL | `…/pynccl.py`::`PyNcclCommunicator` |
| Worker 入口 | `vllm/v1/worker/gpu_worker.py`::`init_worker_distributed_environment` |

**组拓扑（单注释约定）**：`ExternalDP × DP × PP × PCP × TP`（`initialize_model_parallel` 内 reshape）。  
**TP 热路径常用**：`get_tp_group()`。  
**DP 组用途**：跨副本共识（如 `ParallelConfig.sync_dp_state`）、MoE/EP 对齐时的全局未完成探测——**不是**做权重 all_reduce（权重在副本间通常独立）。

**AR 后端优先级（CUDA，简化）**：symm-mem → Quick/FlashInfer/AITER（平台）→ **CustomAllreduce** → Torch SymmMem → **PyNccl** → `torch.distributed.all_reduce`。  
开关：`parallel_config.disable_custom_all_reduce` → `set_custom_all_reduce`。

### 1.3 SGLang

| 角色 | 路径:符号 |
|------|-----------|
| 组与 mesh | `sglang/srt/distributed/parallel_state.py`::`GroupCoordinator` / `initialize_model_parallel` / `get_tp_group` / `get_attn_tp_group` / `get_pp_group` / `get_moe_*_group` |
| Bootstrap | `sglang/srt/distributed/bootstrap.py`::`_init_parallel_groups`（`world ≈ tp×pp`，再 `initialize_dp_attention`） |
| Layer 薄封装 | `sglang/srt/distributed/communication_op.py`::`tensor_model_parallel_all_reduce` / `_all_gather`；另有 `attention_tensor_model_parallel_*`、`moe_*_all_reduce` |
| Attn/MoE 与 DP 拼缝 | `sglang/srt/layers/communicator.py`（attn_tp AR、`dp_gather_*` 等） |
| dp-attn 初始化 | `sglang/srt/layers/dp_attention.py`::`initialize_dp_attention` |
| Communicator 挂点 | AR/AG 实现挂在 `GroupCoordinator` 上（pynccl / custom_all_reduce(_v2) / mscclpp / torch_symm_mem / …） |

**与 vLLM 的关键差别**（细节见下节「通信域构造」）：

1. **副本 DP 常在 `tp×pp` world 之外**（多进程树）；**dp-attn** 才把「请求维 DP」折进同一 TP world，并拆出 `_ATTN_TP`（`attn_tp_size = tp / (attn_dp · attn_cp)`）。  
2. Layer 在 dp-attn 下可能对 **attn_tp 组**做 AR，而对 MLP 仍走完整 TP/MoE 组——同一 forward 里多组并存。

### 1.4 通信域构造差异（process group / mesh）

两端的「TP 小域」长得很像（相邻 rank 成组 + `GroupCoordinator` + NCCL/gloo），**大域怎么切**不同——差在 **DP 进不进同一个 torch world**。

| | vLLM | SGLang（默认副本 DP） |
|--|------|------------------------|
| **torch world** | 常含 DP：`world ≈ DP × PP × PCP × TP`（init 时可用 `dp_rank * world_size + local` 抬成跨 DP 全局 rank，见 `parallel_state` 内 DP 相关逻辑） | **不含**副本 DP：硬约束 `world_size == tp × pp`（`initialize_model_parallel` 内 assert） |
| **DP 组** | 一等公民：`get_dp_group()`，从同一 `all_ranks` 张量切出 | 副本 DP **不建**进这套 PG；N 份进程树各自一个小 world |
| **EP 与 DP** | EP 常跨 `DP × PCP × TP`，和 DP 绑在同一 mesh | MoE 的 `_MOE_EP` / `_MOE_DP` 在 **tp×pp world 内**再切 |
| **dp-attn** | 无同构一等特性 | 另开一岔：在已有 TP world 里再拆 `_ATTN_TP`（`compute_dp_attention_world_info`：`attn_tp = tp/(dp·cp)`，布局 `(dp, cp, tp)`） |

**vLLM 构造方式**——一次 reshape 切出所有组（`initialize_model_parallel`）：

```text
# 注释约定：ExternalDP × DP × PP × PCP × TP
all_ranks = arange(world).reshape(-1, DP, PP, PCP, TP)
→ TP / DCP / PCP / PP / DP / EP / EPLB …
```

源码注释写明：同一 DP 组必须一起 `generate`，否则会死锁——因为 DP/EP 集体通信挂在这张网上。

**SGLang 构造方式**——先钉死小 world，再按相邻 rank 建 TP/PP，再按需加子组：

```text
assert world == tp * pp
TP: [0..tp), [tp..2tp), …
PP: 跨 TP 条带
(+ attn_tp / moe_ep / moe_dp / dcp …)
```

副本 DP 的 N 份进程各自跑一遍上面的 init；彼此默认 **没有** 共享的 `dp_group` NCCL 域。

**同构部分**（别被上面带偏）：都有 `GroupCoordinator` + `init_model_parallel_group`；TP「相邻成组」、PP「跨 TP 条带」；layer 侧都走 `tensor_model_parallel_*` 薄封装；Custom AR / PyNCCL 是 **组内后端**，不是「域怎么切」的差别。

#### 跨 DP 的 linear TP → 更贴 vLLM mesh

若模型脚本优化会对 **linear 层做跨 DP 的 TP/EP**（典型：attention 侧按 DP 副本、expert/MLP 侧组大小约 `DP×TP`，对齐 DeepSeek MLA + MoE 叙事），则：

- 需要 **同一 torch world 内**能切出跨 DP 的通信域（`get_dp_group` / `get_ep_group`），forward 锁步 + dummy 陪跑才有落点；
- **vLLM 的「DP 进 mesh」更合适**；
- SGLang 默认「副本 DP 在 mesh 外」要另建跨进程组，或改走其 MoE/dp-attn 子组路径，不能当「独立副本」假设用。

官方口径见 vLLM `docs/serving/data_parallel_deployment.md`：MoE/MLA 下 DP ranks **不完全独立**，expert 层跨 rank 同步；默认 expert 形成大小 `DP×TP` 的 TP 组（`--enable-expert-parallel` 则改 EP）。

---

## 2. Layer 层：TP 切分与汇合点

两边线性层几乎同构（Megatron 风格），文件均在各自 `layers/linear.py`。

### 2.1 权重切分模式

```
          列并行 Column                    行并行 Row
        ┌──────────────┐                ┌──────────────┐
X ───►  │ W[:, shard]  │ ──► Y_partial  │ W[shard, :]  │ ──► 局部和
        └──────────────┘                └──────┬───────┘
              │                                 │
     (可选) all_gather                   all_reduce → Y
```

| 层类型 | vLLM | SGLang | forward 集体通信 |
|--------|------|--------|------------------|
| `ColumnParallelLinear` | `linear.py::ColumnParallelLinear` | 同名 | 可选 `all_gather`（`gather_output`） |
| `MergedColumnParallelLinear` | 同文件（FFN gate/up 融合列切） | 同 | 同 Column |
| `QKVParallelLinear` | `QKVParallelLinear`（继承 Column） | 同 | 线性层内通常 **无** collective；切的是 Q/K/V 头维 |
| `RowParallelLinear` | `RowParallelLinear` | 同 | **`all_reduce`**（`reduce_results`；SGLang 可 `skip_all_reduce` / 改走 `attn_tp`） |
| `ReplicatedLinear` | 全复制 | 同 | 无 |
| `VocabParallelEmbedding` | `vocab_parallel_embedding.py::VocabParallelEmbedding` | 同名 | 本地 vocab 掩码 → embed → **`all_reduce`** |
| `ParallelLMHead` | 同文件 / logits 侧 | 同 | 权重分片；完整 logits 多在 `LogitsProcessor` **`all_gather`/`gather`** |

### 2.2 Attention 头切分（MHA / GQA）

两边 Llama 系模型同一套算术（例：`models/llama.py`）：

- `num_heads = total_num_heads // tp_size`（必须整除）
- KV：`total_kv_heads >= tp` → 按头切；否则 **复制** KV 头（`num_kv_head_replicas = tp // total_kv`）
- `QKVParallelLinear(total_q, total_kv, …)` → 本地 `Attention(num_heads, num_kv_heads, …)`

含义：

- **计算**：每 TP rank 只算自己的头子集。  
- **KV 布局**：MHA/GQA 下 KV cache **按头（或复制）落在各 rank**——引擎侧通常每 rank 私有一块 HBM KV。  
- **lake**：池拥有 L0；TP>1 时字节块仍可不透明，但放置/传输视图需能区分 **`tp_rank`（或 shard id）**，否则补拉/D-direct 会对错分片。

**SGLang dp-attn**：有效 attention TP 变为 `attn_tp_size`（常见 `tp==dp` 时为 1）——每 attn_dp rank 可对**本地请求子集**持「满头」attention，同时 MLP 仍跨大 TP 组同步。

### 2.3 一层 Transformer 内的典型通信节奏

```
embed (vocab AR)
  → N × {
        attn: QKV 列切 → 本地 attention → o_proj 行切 + AR
        mlp:  gate/up 列切 → act → down 行切 + AR
     }
  → norm → lm_head / logits gather
```

单层最少 **2 次** TP `all_reduce`（attn out + MLP down）；无 fuse/skip 时更多。这是 TP 延迟税的主因，也是 Custom AR 存在的理由。

---

## 3. DP：控制面与「何时需要通信」

### 3.1 副本 DP（两边默认含义）

| | vLLM | SGLang |
|--|------|--------|
| 拓扑 | 每 DP rank 一个 `EngineCore`（自有 Scheduler + Executor + KV） | 每 DP rank 一套 Scheduler 进程树（自有 KV） |
| 前端选路 | `DPLBAsyncMPClient` / External LB；`DPCoordinator` 管 wave/统计 | `DataParallelController`（`ROUND_ROBIN` / `TOTAL_*` / …）或外部 gateway |
| 副本间模型通信 | **默认无**（独立副本） | **默认无** |
| TP 叠在上面 | 每 EngineCore 内再挂 `TP×PP` worker | 每副本 `tp×pp` world |

锚点：

- vLLM：`vllm/v1/engine/coordinator.py::DPCoordinator`；`core_client.py::DPLBAsyncMPClient.get_core_engine_for_request`；官方 `docs/serving/data_parallel_deployment.md`
- SGLang：`srt/managers/data_parallel_controller.py::DataParallelController`；`launch_dp_schedulers`

**副本 DP 本身不强制 NCCL 热路径**；需要通信的是「叠了 MoE/EP / dp-attn / 全局 idle 共识」时。

### 3.2 空闲陪跑（与 DP 强相关）

| | vLLM | SGLang |
|--|------|--------|
| 触发 | 部分 DP rank 有活请求且集体通信（尤其 MoE）必须对齐 | `enable_dp_attention` / `require_mlp_sync`：每步 token 数 `all_gather` |
| 机制 | 无活 rank：`execute_dummy_batch` → `_dummy_run` / `execute_model(dummy_run=True)` | 无活 rank：`ForwardMode.IDLE` + 仍 `run_batch` |
| 全局空 | `sync_dp_state` 类共识后 pause | `on_idle`（无 IDLE forward） |

锚点：vLLM `EngineCore.execute_dummy_batch` / `Worker.execute_dummy_batch`；SGLang `prepare_mlp_sync_batch_raw` / `ScheduleBatch.prepare_for_idle`。

### 3.3 TP 控制面扇出（和 DP 正交，但实现时绑在一起）

| | vLLM | SGLang |
|--|------|--------|
| 调度权威 | **每 DP 副本 1 个** Scheduler | **每 GPU 1 个** Scheduler |
| 同副本 TP 如何拿到同一步 | `Executor.collective_rpc` 扇出同一 `SchedulerOutput`（SHM MQ / Ray） | leader ZMQ + gloo `broadcast_pyobj`，各 rank 本地重建 batch |
| 层内通信 | NCCL TP 组 | NCCL TP / attn_tp / moe_* 组 |

lake 倾向（已有文档口径）：副本内 **一份调度决策 + 多卡执行**（偏 vLLM）；副本间选路偏 **Router + 池位置视图**，不把引擎内 DP LB 当权威。

### 3.4 vLLM：一次多 DP 启动时能否指定打到哪个 DP

**能。** 官方三种部署形态见 `docs/serving/data_parallel_deployment.md`；请求级控 DP 如下。

| 部署形态 | 如何控「打到哪个 DP」 |
|----------|----------------------|
| **Internal LB**（`--data-parallel-size=N`，单 HTTP 入口） | ① HTTP 头 **`X-data-parallel-rank: <i>`** → `_get_data_parallel_rank` → `EngineCoreRequest.data_parallel_rank` → `DPLBAsyncMPClient.get_core_engine_for_request` **直达**该 engine（engines 按 rank 序）；② **不带 header** 时 API server 内置 LB：`score = waiting*4 + running`，选最闲（文档写明暂非 KV-aware；可叠 `--api-server-count` 扩前端） |
| **Hybrid LB**（`--data-parallel-hybrid-lb`） | 每节点自有 API，只排队到本机 DP engines；上游 ingress 选节点，节点内再 LB |
| **External LB** | 每 DP rank 独立 `vllm serve` / 独立 port（或独立 pod）；外部路由器选 endpoint 即选 DP。MoE 仍需 `--data-parallel-size` + `--data-parallel-rank` 等把 ranks 织进同一协调网 |

锚点：

- Header：`vllm/entrypoints/generate/base/serving.py`::`_get_data_parallel_rank`（读 `X-data-parallel-rank`；chat/completion serving 注释写明 *router can inject it*）
- 选路：`vllm/v1/engine/core_client.py`::`DPLBAsyncMPClient.get_core_engine_for_request`
- 协议字段：`EngineCoreRequest.data_parallel_rank`（经 `engine/protocol.py` 等传入）

**重要区分**：「控发到哪个 DP」≠「其它 DP 完全空转」。MoE / 跨 DP expert 集体通信场景下，即使用户请求只进一个 rank，coordinator 仍可能拉其它 rank 做 **`execute_dummy_batch` / dummy forward** 对齐——这正是 §1.4「DP 进 mesh」的代价。

---

## 4. 对照总表（实现清单）

### 4.1 通信栈要落地的模块（TP 最小集）

| 模块 | 职责 | 可先简后繁 |
|------|------|------------|
| `parallel_state` | rank/world、`tp_group`（及日后 `dp_group`）初始化 | 先 `torch.distributed` + 单后端 NCCL |
| `communication_op` | `all_reduce` / `all_gather` / `gather` 稳定 API | 与 layer 解耦，便于换 Custom AR |
| device communicator | 真正 AR 实现 | 初版 PyNCCL 或 `torch.distributed` 即可 |
| Worker bootstrap | 每卡 init 组、对齐 `tp_rank` | 与 `RoleConfig` / 启动脚本挂钩 |

### 4.2 Layer 最小替换集（相对现单卡 Qwen3）

| 现单卡 | TP 版 |
|--------|--------|
| `nn.Linear`（qkv / o / gate_up / down） | **已换**：`ColumnParallel`（qkv/gate_up 占位）/ `RowParallel`（o/down）；待 `QKVParallelLinear` / `MergedColumnParallelLinear` |
| `nn.Embedding` / tied lm_head | embed 仍 `nn.Embedding`；未 tie 的 `lm_head` 暂 `ColumnParallel`（待 `ParallelLMHead`） |
| Attention 构造 | **已**：`num_heads`/`num_kv_heads` 按 `tp_size` 除；KV arena 按本地头数编址 |
| Runner | 不按平台子类化；持 `tp_size`/`tp_rank`；构造期注入与 attn backend 同级 |

### 4.3 DP 最小集（副本）

| 模块 | 职责 | lake 边界 |
|------|------|-----------|
| 多副本进程/节点 | 每副本独立 Python worker(+其 TP 组) | 计算节点可毁可拉 |
| 选路 | 按负载 / 前缀本地命中 | **Go Router**；worker 只上报 capacity |
| 池视图 | 每副本 HBM 是池的 L0 载体 | 放置仍归存储池；D-direct 读位置视图 |
| IDLE/dummy | 仅当引入跨副本集体通信时 | **不要**为纯副本 DP 预埋隐式 all_gather |

---

## 5. 对 lake 的含义（讨论用，未定案）

1. **TP 是计算面问题**：通信模块 + parallel linear/embedding 落在 Python 计算层；控制面「一步决策」宜仍一份，再扇出到 TP workers（对齐 vLLM Executor，而非 SGLang 每 GPU 全量 Scheduler）。  
2. **通信域选型倾向 vLLM mesh**：若模型脚本会对 linear 做 **跨 DP 的 TP/EP**，应采用「DP 进同一 torch world」构造（§1.4），而非 SGLang 默认「副本 DP 在 mesh 外」。纯独立副本、层内无跨 DP collective 时，仍可退回每副本小 world。  
3. **请求→DP 选路**：对照 vLLM Internal LB 的 `X-data-parallel-rank` / External endpoint；lake 侧权威宜在 **Go Router**（可读池位置视图做 D-direct），worker 只执行与上报——不必把引擎内 `waiting*4+running` 当最终策略。  
4. **不要照搬**：引擎私有 APC 当 SSOT；SGLang dp-attn 的每步 mlp `all_gather`（除非明确做同构特性）；把过载 shedding 做进 DP LB。  
5. **池 × TP**：KV 不透明字节仍成立，但元数据/传输必须认识 **头维分片**（或 MLA 共享键的「多 rank 同一份字节」特例——见 HiCache MLA 去重讨论）；跨 DP expert 时还需分清「请求落在哪一 DP」与「哪几 rank 必须陪跑 collective」。  
6. **与现有原则**：失败不设 mode fallback 链；过载归 gateway；worker 上报信号——DP 扩展时同样适用。

### 5.1 已落代码（骨架）

| 落点 | 路径:符号 | 状态 |
|------|-----------|------|
| 配置 | `python/lake/engine/config/parallel.py::ParallelConfig`（`RoleConfig` 同包） | `tp/pp/dp`、`world_size(_across_dp)`、env `LAKE_*` |
| mesh 切组 | `python/lake/engine/distributed/parallel_state.py::{tp,pp,dp}_group_ranks` / `initialize_model_parallel` | 与 vLLM reshape 同序（PCP=1） |
| 组对象 | `…/parallel_state.py::GroupCoordinator` | ranks + `all_reduce`/`all_gather`（有 `device_group` 时）；custom AR 未挂 |
| 通讯封装 | `…/communication_op.py` | 可传 `group`；默认 TP |
| 并行 Linear | `python/lake/engine/model_executor/layers/linear/`（一类一文件） | `Column`/`Row` 可 `pg=`（默认 TP）；`Replicated` 全复制、无集体通信 |
| 进程挂点 | `RoleConfig.parallel`；`WorkerEngine.start` → `ensure_model_parallel_initialized` | 单卡跳过 dist；多卡需先 `init_distributed_environment` |

**与 vLLM 差异（linear）**：vLLM 的 Column/Row **硬编码** `get_tp_group()`（经 `communication_op`）；lake 允许 `pg: GroupCoordinator | None`。`ReplicatedLinear` 与 vLLM 同：不使用通讯域。包路径按类拆分（vLLM 仍单文件 `linear.py`）。

**未做**：NCCL/custom AR、QKV/MergedColumn、多进程 launcher、Router↔DP rank 头。

---

## 6. 代码索引

### vLLM

| 概念 | 文件:符号 |
|------|-----------|
| 组协调器 / 初始化 | `vllm/distributed/parallel_state.py`::`GroupCoordinator` / `initialize_model_parallel` / `get_tp_group` / `get_dp_group` |
| TP 封装 | `vllm/distributed/communication_op.py`::`tensor_model_parallel_all_reduce` / `tensor_model_parallel_all_gather` |
| Custom AR / PyNCCL | `device_communicators/custom_all_reduce.py`::`CustomAllreduce`；`pynccl.py`::`PyNcclCommunicator` |
| CUDA AR 分派 | `device_communicators/cuda_communicator.py`::`CudaCommunicator.all_reduce` |
| Column / Row / QKV | `vllm/model_executor/layers/linear.py`::`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` |
| Vocab | `vllm/model_executor/layers/vocab_parallel_embedding.py`::`VocabParallelEmbedding` |
| Worker 分布式 init | `vllm/v1/worker/gpu_worker.py`::`init_worker_distributed_environment` |
| DP 协调 / LB | `vllm/v1/engine/coordinator.py`::`DPCoordinator`；`core_client.py`::`DPLBAsyncMPClient.get_core_engine_for_request` |
| 指定 DP rank（HTTP） | `entrypoints/generate/base/serving.py`::`_get_data_parallel_rank`（头 `X-data-parallel-rank`） |
| Dummy 陪跑 | `vllm/v1/worker/gpu/model_runner.py`::`GPUModelRunner._dummy_run`；`gpu_worker.py`::`execute_dummy_batch` |
| DP idle 共识 | `vllm/config/parallel.py`::`ParallelConfig.sync_dp_state` |
| 官方 DP 部署 | `docs/serving/data_parallel_deployment.md`（Internal / Hybrid / External LB；MoE 下 `DP×TP` expert 组） |

### SGLang

| 概念 | 文件:符号 |
|------|-----------|
| 组协调器 / 初始化 | `srt/distributed/parallel_state.py`::`GroupCoordinator` / `initialize_model_parallel` / `get_tp_group` / `get_attn_tp_group` |
| Bootstrap | `srt/distributed/bootstrap.py`::`_init_parallel_groups` |
| TP 封装 | `srt/distributed/communication_op.py`::`tensor_model_parallel_all_reduce` |
| Column / Row / QKV | `srt/layers/linear.py`::`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` |
| Vocab | `srt/layers/vocab_parallel_embedding.py`::`VocabParallelEmbedding` |
| dp-attn | `srt/layers/dp_attention.py`::`initialize_dp_attention` / `compute_dp_attention_world_info` |
| 层间 communicator | `srt/layers/communicator.py` |
| 副本 DP 控制 | `srt/managers/data_parallel_controller.py`::`DataParallelController` |
| TP worker 入口 | `srt/managers/tp_worker.py`::`TpModelWorker.forward_batch_generation` |

### 相关本仓库文档

| 文档 | 内容 |
|------|------|
| [`sglang/model-runner.md`](sglang/model-runner.md) | DP/TP/PP **控制面**对照、IDLE vs dummy |
| [`vllm/compute.md`](vllm/compute.md) | EngineCore / Executor / DPCoordinator |
| [`attention-backends.md`](attention-backends.md) | Attention 后端与 runner 形态（与 TP 头切正交） |
| [`../architecture/compute-layer.md`](../architecture/compute-layer.md) | lake 计算层定案（并行扩展时回写） |
