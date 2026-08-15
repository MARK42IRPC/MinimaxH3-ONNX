# 推理运行时重构规划: schedule v2 唯一路径

> 状态:P0/P1 correctness 收尾,2026-08-14 修订。
> 本文档定义重构推进顺序;字段级契约以 `docs/SCHEDULE_CONTRACT.md` 为准。

## 1. 已确认事实

- 主模型 158 张细粒度 ONNX 基线已经完整导出,约 37.93 GiB,单图对拍通过。
- fp16 激活链在 block 4 溢出,因此采用 fp32 激活边界、fp16 注意力内部 GEMM、
  fp32 MLP 计算。bf16 暂不作为统一底座。
- MLP fp8 只负责 L3/L2 存储压缩。ORT 初始化时会把反量化链物化为 fp16 权重,
  因此磁盘大小不能直接作为 L1 显存大小。
- 原 158 图自适应批处理运行时不再保留。schedule runtime 是主模型唯一执行路径。

## 2. 目标架构

导出物由不可变基线图构建到独立 staging 目录,最终原子发布:

- `shard_NNN.onnx` 与 `shard_NNN.onnx.data`: If 容器分片。
- `schedule.json`: `h3-schedule-v2` 唯一执行表。
- `manifest.json`:包含 schedule 指针、产物哈希、精度和完整性状态。
- `build-state.json`:仅 staging 期间存在,用于断点续跑。

运行时严格解释 schedule:

1. preamble 将 Qwen 状态编码成常量区 `text_states`。
2. denoise op 生成 patch、位置、时间和 modulation 输入。
3. graph/op 步按表执行,不做动态重排。
4. L1 session 驻留与激活缓冲是两套独立资源池。
5. 运行时只返回 video/audio velocity;采样器的 Euler 更新在 GPU op 阶段接入。

## 3. 资源口径

每个 shard 同时记录:

- `storage_bytes`:L3 文件和 L2 pinned 窗口占用。
- `materialized_weight_bytes`:fp8 展开后预计权重占用。
- `session_peak_bytes`:session 构建时的估计或实测峰值。

分片规划同时满足存储上限与 L1 物化上限。分片数量由 requant 后真实清单决定,
不再把 52 写死为验收条件。

当前主模型计划为 158 个单图 shard。单图隔离是 ORT 1.23 CUDA correctness
约束,不能用容量装箱重新合并;旧运行时仍不保留,执行只由 schedule 驱动。

资源命名:

- `activation_slot`:张量缓冲槽,由 schedule 的生命周期静态计算。
- `session_slot`:驻留 shard session 的槽,由 VRAM 探测决定。
- `const`:跨多个 graph 步复用的小型或常驻张量。
- `external`:一次调用的输入或最终输出。

## 4. 推进阶段

### P0-A 契约与保护

- 基线目录只读使用,构建输出进入独立 staging。
- schedule 使用命名端口绑定,每个 ONNX 输入/输出都有明确来源/目标。
- schema validator 对照真实 ONNX metadata 拒绝缺口、错名和 dtype 不一致。

验收:真实 158 图可生成 schedule,所有 graph step 的端口完整匹配。

### P0-B 事务式打包

- requant 和 shard 保存均使用临时文件后原子替换。
- 每张 MLP 和每个 shard 有完成记录与哈希,中断后可继续。
- manifest 最后发布;没有完整 manifest 的 staging 目录不得被运行时发现。

验收:主动中断后可续跑;源目录字节不变;最终图覆盖恰好一次。

### P0-C 全链路验证

- 每个 If 分支与对应源图对拍。
- 50 block 串联 smoke 使用前一 block 输出作为后一 block 输入。
- 覆盖 block 4、动态序列、双 head、finite guard。

验收:数值门限通过,无 inf/NaN,manifest 写入验证摘要和哈希。

### P1-A correctness runtime

- 实现 schedule parser、命名张量仓库、单 selector feed 和 placeholder 生成。
- CPU/普通 `session.run` 先完成语义闭环。
- `H3MainRuntime` 只接受带 `h3-schedule-v2` 的模型目录。

验收:相同输入下,shard runtime 与基线 158 图 reference runtime 的完整 denoise
输出在既定门限内一致。reference 只存在于测试工具,不进入产品路径。

### P1-B GPU residency

- session slot 池、IOBinding 激活池、pinned L2、静态 K 前瞻。
- OOM 仅缩小 K/session 驻留深度,不修改 schedule 顺序。

验收:4 GiB 与 24 GiB trace 均通过,无非预期 D2H,共享显存不作为容量来源。

### P2 Qwen

主模型 P1 稳定后,Qwen 再使用相同容器、schedule 和驻留机制。现有 persistent
原型作为测量样本,不与 P0/P1 正确性提交混合。

## 5. 发布门禁

- P0 未通过前不得删除基线图。
- P1 correctness 未通过前不得开始 GPU 零拷贝优化。
- 所有生成器必须可恢复、可校验、源目录不可变。
- 文档中的性能结论必须附真实 trace,不能以单测或磁盘体积代替。

性能预算:相对旧 158 细图路径,收益来自约 29% 的存储/传输缩减、静态生命周期
和预取,而非 session 数减少。在 360p/17F 实测完成前,整任务只按 1.05-1.3x
加速规划;更高倍率需要无 If 的 fused block 或 TensorRT 路径。

## 6. 2026-08-14 发布记录

- 158/158 单图 shard CUDA 对拍通过:`max_abs=0`,`relative_l2=0`。
- 50 block 链式 smoke finite;block 4 与最终 hidden 幅值已写入验证报告。
- 产品目录不再依赖 source 细图;启动只校验 schedule、manifest 与 shard 文件。
- 旧基线与 requant source 细图已归档到
  `D:\ckpt\onnx_models(炉渣)\archive`，manifest 保留准确路径。

一步主模型 smoke（1 个零文本 token、360p/17F latent、chunk 128、RTX 3050
Laptop 4 GiB）:runtime 初始化 0.248 秒,preamble 3.173 秒,denoise 128.497 秒,
合计 131.919 秒。视频 velocity finite;一步 `sigma=1` 的音频 velocity 非 finite,
标准采样器会记录 audio fallback 并保留原音频 latent。该 smoke 不包含 Qwen/VAE,
不能替代真实 prompt 的端到端 A/B。

### 2026-08-14 cf4 主模型性能复测

在 schedule runtime 中启用 block hidden 的 CUDA `OrtValue` 常驻链路，覆盖
attention-output、MLP 和下一 block QKV；同时移除每个 shard/SDPA 后的热路径全量
GC。新增 `h3-main-benchmark` 持久化入口，逐图写 JSONL，逐步原子保存 latent
checkpoint，失败记录 traceback、最后活动图和累计 runtime metrics。

复测参数取历史任务 `cf4c178c1a4e` 的主模型几何：352x360、17 请求帧、22 原生
帧、1190 sequence tokens、4 步、seed 1、query chunk 256；为隔离主模型，使用
192x5120 零文本状态，不包含 Qwen、VAE 和 MP4。

- 前三步 checkpoint finite，耗时分别为 133.702、144.743、130.658 秒。
- 第四步完成 `main_head` 后，video velocity 出现 88704 个非 finite 值；累计
  536.445 秒，正确性门禁按预期终止任务。
- 618 次 graph 中 596 次走设备 IOBinding；峰值显存 3807 MiB，平均显存
  3189 MiB，峰值 GPU 利用率 100%，平均 GPU 利用率仍只有 11.95%。
- session 构建 253.159 秒，graph 执行 185.082 秒，其余 SDPA/调度/检查点
  98.204 秒。进程平均 CPU 为 531.86%（psutil 多核口径）。
- 历史 cf4 Turbo 主模型 sampling span 为 720.062 秒；本次即使计入第四步末端
  finite 失败，墙钟时间减少 25.5%，约 1.342x。两次文本状态和主模型变体不同，
  此数据只作为运行时性能参考，不构成数值 A/B。

下一性能优先级：先减少每步 153 个非驻留 session 的构建次数，再将 QKV 切片、
归一化和 streamed SDPA 改为 CUDA `OrtValue` 闭环。共享显存分页不作为性能方案。

### 2026-08-14 scaled-FP16 与低 CPU 复测

50 个主模型 MLP 已导出为 scaled-FP16 直出图，未改动的分片在独立 hybrid 目录
中通过硬链接复用。候选权重占 21.70 GiB，C 盘剩余 130.26 GiB；批处理在每块
开始前执行 90 GiB 硬保护。50 块独立误差最大 `5.342e-4`，均通过 CUDA、finite、
3 Gemm/4 Cast 和 FP16 大权重门禁。

完整一步 cf4 velocity A/B：video relative L2 为 `2.497e-3`，audio 为
`1.590e-3`，均 finite；累计链路按 `5e-3` 门槛通过。产品与 hybrid 内部 elapsed
分别为 122.966 和 107.091 秒；graph time 从 42.182 降至 18.361 秒，但 session
build 从 59.755 增至 70.053 秒，说明 MLP kernel 已不再是首要瓶颈。

四步 hybrid 在 104.944、206.841、310.024 秒保存前三步 finite checkpoint；第
四步仍在 `main_head` 后出现与产品完全相同的 88704 个 non-finite video velocity，
累计 412.782 秒。该结果把故障范围排除到 scaled-FP16 MLP 之外，但在根因修复前
hybrid 仍保持 `validation_passed=false`。

随后 CUDA ORT 会话默认改为单 CPU operator 线程并关闭 intra/inter-op spinning。
相同一步 hybrid 输出逐元素不变，平均 CPU 从 626.87% 降至 66.68%，峰值从
657.4% 降至 79.2%；内部 elapsed 从 107.091 降至 91.271 秒，session build 从
70.053 降至 57.934 秒。下一阶段按顺序验证：会话构建缓存/预优化模型、外部权重
解析与上传、QKV/SDPA 的 CUDA `OrtValue` 闭环。所有方案必须同时降低 CPU 核时且
不增加墙钟，不能以限制线程换取更慢的任务。

### 2026-08-14 第四步溢出修复

逐图诊断从第三步 checkpoint 直接恢复，首次 non-finite 出现在
`main_block_44_attention_output`，而此前 SDPA finite。attention-output 原生 FP16
投影在残差流约 350 万时发生中间溢出；缩放 FP16 投影在真实 block 44 输入上
relative L2 为 `1.410e-4`，同时保持约 24.1 ms warm time。

50 个 attention-output 块统一重导后，独立误差范围为 `2.521e-4` 到
`3.301e-4`。完整 cf4 四步分别在 90.141、180.501、268.332、356.788 秒完成，
video/audio 全部 finite。平均 CPU 66.93%，平均 GPU 19.51%，平均显存 2224.6
MiB；session build 224.828 秒，占内部总耗时 63.0%，仍是下一主瓶颈。

算子融合按 correctness-first 推进：先在直出图上对拍 ORT offline optimized
model，再尝试 RMSNorm+AdaLN、QKV+QK norm+RoPE、SwiGLU 和 attention
output+gate+residual。当前运行时显式使用 `ORT_DISABLE_ALL`，不能直接全局开启
高级优化；每种融合必须保留 attention-output 的 FP16 缩放保护，并独立记录首次
session build、warm graph time、GPU 利用率和数值误差。

### 2026-08-14 持久拓扑与权重预读

QKV、scaled attention-output 和 scaled MLP 现各复用一个运行时拓扑，块权重作为
输入流入。外部数据由单线程顺序预读，默认保留 3 图有限窗口，图完成后立即释放
宿主缓冲。mmap-only 原型虽把 session 数从 159 降到 61，但一步耗时从 91.271
回退到 92.354 秒且 RSS 达 23.31 GiB，已废弃。顺序预读后一步为 59.540 秒，
RSS 2.134 GiB，输出逐元素一致。

完整四步在 59.572、116.600、173.600 和 230.913 秒落盘并全部 finite；旧链路为
356.791 秒，墙钟减少 35.28%。session build 从 618 次/224.828 秒降到 21 次/
3.595 秒；最终 video/audio 与旧链路逐元素一致。峰值显存 2808 MiB，峰值 RSS
2.166 GiB，平均 GPU 从 19.51% 升到 22.13%。剩余主瓶颈是四步 156.24 GB 权重
流带来的 81.156 秒前台等待，下一阶段再独立验证算子融合与上传流水并行。

### 2026-08-14 锁页权重池

运行时现按 QKV、attention-output、MLP 各复用一个 CUDA Portable 锁页缓冲，
预读线程直接把外部数据读入锁页内存，块执行完成后同类下一块才能复用。默认从
项目依赖 `nvidia-cuda-runtime-cu12` 加载 cudart；缺失或分配失败会自动回退到
可复用 pageable 缓冲，并在 runtime metrics 中记录原因。

一步 cf4 从 59.540 降到 44.726 秒；完整四步在 45.425、88.459、132.120 和
176.272 秒完成，最终 video/audio 逐元素一致。相对最初 356.791 秒缩短 50.60%。
graph time 为 61.329 秒，权重前台等待为 57.969 秒；宿主池仅分配 3 次。峰值
RSS 2.418 GiB、峰值显存 2936 MiB、平均 GPU 29.88%、峰值 GPU 95%。

### 2026-08-14 双路权重预读

QKV、attention-output、MLP 各自已有独立锁页缓冲，因此可在不增加缓冲数量的
前提下并行读取。三 worker 的纯读取吞吐虽从 0.755 提升到 1.067 GiB/s，但真实
推理争用偏高；两 worker 综合结果最佳，现作为默认值，可用
`H3_WEIGHT_PREFETCH_WORKERS=1..3` 覆盖。

两次一步验证为 37.975 和 44.003 秒，均快于单 worker 的 44.726 和 45.719 秒，
且输出逐元素一致。最终四步在 37.859、73.117、108.708、144.496 秒落盘，总计
144.500 秒；相比锁页单 worker 的 176.272 秒缩短 18.02%，相比最初 356.791 秒
缩短 59.50%。前台权重等待从 57.969 降到 12.620 秒（-78.23%）。代价是 graph
time 从 61.329 增至 68.467 秒、进程平均 CPU 从 116.05% 增至 160.44%；峰值
RSS 2.420 GiB、峰值显存 2935 MiB，最终 video/audio 仍逐元素一致。

### 2026-08-14 设备 SDPA 与显存感知上传试验

设备 SDPA 将 QKV 与 attended 保持为 CUDA `OrtValue`，50 次一步 SDPA 合计降至
1.52-1.69 秒，一步为 36.142/39.149 秒；但四步相对接受版的 video/audio
relative L2 累积到 5.04%/6.47%，未通过严格数值门禁。因此
`H3_DEVICE_SDPA` 默认关闭，仅保留实验入口。

下一图权重上传实现了实时 CUDA free、384 MiB 余量、in-flight bytes 和单设备槽
联合准入，自动模式还要求总显存至少 6 GiB。ORT allocator 版本把 4 GiB 显存推到
3913 MiB 并回退至 51.709 秒；改用可立即 `cudaFree` 的直接 CUDA 缓冲后峰值降至
3554 MiB，但仅命中 6/150 图，仍回退至 40.239 秒。本设备自动关闭该路径，
`H3_DEVICE_WEIGHT_PREFETCH=1` 仅供显式试验。

设备 SDPA 四步期间 SSD 状态明显变差：156.24 GB 权重的累计 load/wait 为
406.95/148.00 秒，接受版为 224.06/12.62 秒，总墙钟因此达到 235.87 秒，不能
作为计算性能对比。固定三锁页缓冲使 RSS 仍只有 2.223 GiB；SSD 高而内存低属于
有界流式读取的预期表现。

ORT Basic/Extended/All 离线优化也完成了交错复测。只有 MLP 融合出 QuickGelu，
节点从 36 降到 33，但 24 次样本的四档中位数均在 116-117 ms，差异不足 1%，
因此不发布 ORT optimized 候选；后续融合需要针对剩余实际 CUDA kernel 边界。

selector-free QKV 也完成了独立和四步复测。单图中位数从 147.41 ms 降至
130.46 ms，四步 graph time 从 61.329 降至 58.191 秒，输出逐元素一致；但额外
磁盘等待把总墙钟从 176.272 拉高到 179.363 秒。因此保留提取算法和结构测试供
后续融合或权重驻留使用，默认发布仍采用已验证的 `Equal + If` QKV wrapper。

最终验证目录发布为
`exported/minimax_h3_fl2va_scaled_fp16_tensor_core_v3`。旧产品和可重建候选均已
删除，D 盘未保留旧产品副本；源 checkpoint 保持不变。清理后 C 盘可用空间从
125.69 GiB 回升到 145.02 GiB，释放约 19.33 GiB，为 QwenVL 编码器留出空间。

### 2026-08-14 Qwen INT8 虚拟切片

Qwen 源改用工作区中的 `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`。
线性权重为逐输出通道 `I8 + FP32 scale`，embedding/norm 为 BF16。读取器已改为
解析 safetensors 头并按绝对偏移读取，避免 27 GB 整文件映射在 Windows 上触发
1455 页面文件错误。

新产品不复制约 23 GB 文本权重，只发布 Attention/融合 MLP 两个持久拓扑和
虚拟分片 manifest；大矩阵直接 memmap 原 checkpoint，小型 scale/norm 首次使用
时转换并缓存。Embedding 在 CPU 按 token 行读取 BF16，取消整张词表上传。当前
产品目录约 167 KiB，符合后续 WebUI“下载原模型、客户端本地切片”的规划。

准确版使用 `INT8 + FP32 scale -> DequantizeLinear(FP32) -> Cast(FP16)`。流式
Attention/MLP 相对 FP16 权重输入分别为 1.391x/1.249x；block 0/24/49 最坏
relative L2 为 `7.20e-4`/`4.27e-4`，通过 `1e-3` 门禁。50 层四 token 完整运行
为 51.042 秒且 finite，相对历史 67.2 秒缩短 24.0%。FP16-scale 快速候选虽有
约 1.6x 单层收益，但全链与准确版相差 1.5036% relative L2，已淘汰。

D 盘旧产品对照在确认机械盘后取消，运行到第 24 层的 567.7 秒不得用于性能结论。
Qwen 全链仍主要受 C 盘权重流限制；32 GiB 主机不允许常驻钉死全部文本权重，
后续只做基于实时可用内存的有限窗口预取。

### 2026-08-14 Video VAE 持久块切片

Video VAE 的 36 个 decoder block 复用一个 27,942-byte 无权重拓扑，每块 12 个
FP16 initializer（134,287,360 bytes）继续留在原 ONNX 中，由单 worker 预取下一
块。hidden/rotary tile 在 block 链中保持 CUDA OrtValue，仅进入 head 前回主机。

C 盘单块实测：原 session build 为 163.561 ms、warm 65.709 ms；持久路径权重
H2D 为 28.917 ms、warm 59.503 ms，输出逐元素一致。单块 host load 仍需
209.127 ms，所以单 tile 串行场景不承诺收益；360p 常见 4 tile 可用约 238 ms
当前块计算覆盖下一块读取。batch-2 每 tile 为 71.760 ms，慢于 batch-1 常驻的
69.436 ms，已淘汰。manifest 未通过 ready-check 时保持旧路径，可用
`H3_VIDEO_VAE_PERSISTENT=0` 显式关闭新路径。
