# schedule.json 契约(h3-schedule-v2)

> 状态:P0 契约,2026-08-14 修订。
> v1 的数字槽位端口和固定 52 分片验收作废。

## 1. 顶层结构

```json
{
  "format": "h3-schedule-v2",
  "model": "minimax_h3_fl2va_pruned_fp8_scaled_fp32_sharded",
  "precision": {
    "activation": "fp32",
    "attention_gemm": "fp16",
    "mlp_compute": "fp32",
    "mlp_storage": "fp8_e4m3fn_scaled"
  },
  "resources": {
    "activation_peak_slots": 5,
    "session_slot_hints": {"vram_4gib": 1, "vram_24gib": 8}
  },
  "shards": [],
  "steps": []
}
```

必需字段:

| 字段 | 语义 |
| --- | --- |
| `format` | 固定 `h3-schedule-v2` |
| `model` | 与 manifest 模型标识一致 |
| `precision` | 存储、边界与计算精度事实 |
| `resources` | 激活峰值与 session 驻留建议,两者不得混用 |
| `shards` | L1/L2/L3 统一分片清单 |
| `steps` | preamble 与 denoise 的静态执行序 |

## 2. shard

```json
{
  "id": "shard_002",
  "file": "shard_002.onnx",
  "storage_bytes": 865056000,
  "materialized_weight_bytes": 1096000000,
  "session_peak_bytes": 1370000000,
  "resident": false,
  "graphs": [
    "main_block_00_attention_qkv"
  ]
}
```

- `storage_bytes`:打包后 ONNX + external data 的预计/实测字节。
- `materialized_weight_bytes`:fp8 反量化物化后的权重估计。
- `session_peak_bytes`:构建 session 的安全估计;实测后可回写 manifest trace。
- `resident`:常驻片标记。preamble、embeddings、conditioning、head 所在前缀片为 true。
- 每个 source graph 必须是独立 shard。ORT 1.23 CUDA 在 sibling `If` 依次处理
  高幅 activation 时可能复用错误的分支状态。
- 分片器按执行序确定性贪心,但同时受 storage 和 materialized 两个上限约束。

## 3. 命名绑定

graph step 不再使用没有端口语义的 `input_slots/output_slots`。每个真实 ONNX
端口都映射为 binding:

```json
{
  "id": 12,
  "phase": "denoise",
  "kind": "graph",
  "shard": "shard_002",
  "graph": "main_block_00_attention_qkv",
  "inputs": {
    "hidden_states": {"source": "buffer", "name": "hidden"},
    "timestep_embedding": {"source": "const", "name": "timestep_embedding"},
    "modulation_ids": {"source": "const", "name": "modulation_ids"},
    "rotary_table": {"source": "const", "name": "rotary_table"}
  },
  "outputs": {
    "hidden_states_out": {"target": "buffer", "name": "qkv_packed"}
  },
  "release": [],
  "barrier_after": false
}
```

binding 的 `source/target` 仅允许:

- `external`:调用输入或最终返回值。
- `const`:preamble 或当前 denoise step 内复用的值。
- `buffer`:有静态生命周期的激活。
- `discard`:仅 output target 可用,表示图输出不进入张量仓库。

schema validator 必须逐端口对照 ONNX graph metadata。端口缺失、多余、重名或绑定
方向错误均为构建失败。

## 4. phase 与最小 op 集

### preamble

1. `preamble_inputs`:为 embeddings 生成空 video/audio patch。
2. embeddings graph:取 `text_embeddings`。
3. 两个 refiner block 的 attention/MLP。
4. refiner norm:写入 const `text_states`。

### denoise

1. `denoise_inputs`:patchify video、pack audio,生成 times/modulation/position。
2. embeddings graph:产出 video/audio embeddings。
3. `concat_hidden`:按 text、audio、video 顺序构造 hidden。
4. conditioning graph:生成 timestep/rotary,Turbo 额外生成 silu timestep。
5. 50 x `[qkv -> sdpa -> attention_output -> mlp]`。
6. `split_hidden`:取 audio/video token 区间。
7. `select_head_timestep`:选择 video/audio head 时间条件。
8. head graph:同时产出 video/audio patches。
9. `unpack_velocity`:生成最终两个 velocity external 输出。

v2 op 注册表:

| op | 用途 |
| --- | --- |
| `preamble_inputs` | 构造 token refiner embeddings 输入 |
| `denoise_inputs` | patch/pack 与时间、位置、modulation 生成 |
| `concat_hidden` | text/audio/video 拼接 |
| `sdpa` | scaled dot product attention |
| `split_hidden` | head 前拆分 audio/video hidden |
| `select_head_timestep` | 选择双时钟 timestep 条件 |
| `unpack_velocity` | 反 patch 并取负得到 velocity |

op step 使用与 graph step 相同的命名 binding 结构,但端口由 op 注册表验证。

## 5. 激活生命周期

- schedule 编译器按 buffer 名称做 last-read 扫描。
- `release` 列出该步完成后失效的 buffer 名。
- `resources.activation_slots` 可记录 buffer 到物理槽的静态映射。
- session 驻留数不由 activation slot 数推导。
- correctness runtime 可按名称持有数组;GPU runtime 必须遵守相同生命周期。

## 6. If 容器

- 每个 shard 一个 session且只含一个源图;当前保留单分支 If 作为统一包装。
- shard 只暴露一个唯一命名的 `int64 selector`;父图用 `Equal` 生成各 If 条件。
- 源图 initializer 提升到 shard 父图并使用 graph 名作命名空间;If then branch
  不得持有权重 initializer。
- main graph input 是片内输入并集;未参与当前分支的输入由解释器提供同 dtype、
  合法动态形状的零占位。
- 解释器只请求选中分支的 `{graph}/{output}` 输出。
- qkv 与 attention_output 分属相邻 shard;SDPA 是两者之间的 op step。
- CUDA session 必须禁用 memory pattern,并使用 `kSameAsRequested` arena 扩展策略。

## 7. 精度与尺寸

- 图边界激活 fp32。
- attention GEMM 内部 fp16,输出恢复 fp32。
- MLP GEMM fp32。
- 未合并 LoRA 的 MLP 权重以源 fp8 + scale 保存,图内反量化到 fp16 存储值。
- `storage_bytes` 不能用作 `materialized_weight_bytes`。分片规划默认:
  `materialized = storage + fp8_weight_bytes`,即 fp8 权重按 2 倍物化。

## 8. P0 验收

1. 源目录构建前后哈希一致。
2. 真实 158 图全部且仅出现一次。
3. graph step 端口与 ONNX metadata 逐项一致。
4. qkv/attention_output 屏障、双尺寸上限、确定性通过。
5. 每个 If 分支与源图数值对拍。
6. 50 block 串联 smoke 无 inf/NaN,重点记录 block 4。
7. staging 中断可续跑,最终 manifest/schedule 原子发布。

## 9. P1 correctness 验收

- 产品主模型只走 schedule runtime;缺失或不支持的 schedule 直接报错。
- reference 细图执行器仅放在测试代码中。
- preamble 与完整 denoise 输出分别对拍。
- 所有 external 输出、const 和 buffer 在执行结束时满足生命周期验证。
