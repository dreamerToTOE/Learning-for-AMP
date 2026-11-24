# R1 路线纠偏与训练方案复审

日期：2026-11-24（Asia/Seoul）  
基线提交：`79c9a7d` 及之前的 R1 诊断结果

## 纠偏结论

R1 的主要目标重新定义为官方 AMP 配置复现和训练趋势验证。Humanoid28 的质量、惯量、COM、principal axes、接触参数、USD drive 和 actuator 映射诊断全部保留为背景证据，暂停扩大底层参数搜索。

## reference replay 语义修正

官方 Walk motion 为 60 FPS，环境控制步长为 `1/30 s`，Walk 文件有 154 帧。`start_frame=30`、`frame_stride=2` 时，实际目标帧为 `32...153`，共 62 个控制步。

旧脚本会在 frame153 之后继续把目标帧截断为 153。此前的 63 步结果包含一次额外的末帧保持，因此不能继续解释为参考轨迹本身在第 62 步失败。

`scripts/r1_reference_replay.py` 现在提供：

- `--hold-final-frame`，显式选择是否在参考轨迹结束后保持末帧
- `motion_frame_raw` 和 `motion_frame_clamped`
- `reference_end_step` 和 `reference_end_truncated`

修正后的 CUDA 回放结果为：

```text
steps_requested: 77
steps_completed: 62
reference_end_step: 61
reference_end_truncated: true
terminated_step: null
all_finite: true
```

这次回放验证了参考轨迹末尾的有限状态，不能替代策略闭环稳定性测试。参考位置直接回放没有根据实际机器人状态调整动作，也不能单独证明 AMP 无法学习稳定运动。

## 官方 AMP 配置审计

Isaac Lab v2.1.1 官方 `Isaac-Humanoid-AMP-Walk-Direct-v0` 配置包含：

| 配置项 | 官方值 |
|---|---:|
| 默认并行环境 | 4096 |
| simulation dt | `1/60 s` |
| decimation | 2 |
| episode length | 10 s |
| AMP trainer timesteps | 80000 |
| rollout length | 16 |
| learning epochs / mini-batches | 6 / 2 |
| learning rate | `5e-5` |
| policy/value/discriminator MLP | `1024, 512` |
| task reward weight | 0 |
| style reward weight | 1 |
| discriminator reward scale | 2 |
| discriminator batch size | 4096 |
| motion dataset memory | 200000 |
| reply buffer | 1000000 |
| action clipping | false |

官方 checkpoint 路径检查和本机 probe 均未找到可用的 Humanoid AMP Walk checkpoint。Nucleus 返回 checkpoint unavailable，不能用 checkpoint play 作为当前环境验证。

## 既有训练结果重新标记

此前正式组使用 `--max_iterations 500`。Isaac Lab skrl train script 将其转换为：

```text
trainer timesteps = 500 × rollouts16 = 8000
```

在 2 个并行环境下约为 `16000` 个环境交互步。官方 AMP 配置是 `80000` trainer timesteps，因此此前预算约为官方 trainer timesteps 的十分之一。

此前的 task-only 与 task+style 结果重新标记为：

- AMP 训练链路和奖励消融 smoke
- 资源、日志和 checkpoint 流程验证
- 不能作为 AMP 与普通 PPO 行走能力的最终比较

官方 Walk 每步任务奖励固定为 1，且没有明确前进速度目标。只用这个任务奖励训练 PPO，不能把结果解释成速度控制行走策略。

## 单种子训练方案

当前 RTX 3070 只有 8 GiB 显存。官方 4096 环境配置不能直接用于本机，之前 2 环境运行峰值已经接近整卡容量。后续保持官方网络、AMP 超参数和奖励定义，明确记录并行环境数的本机偏差。

| 阶段 | 并行环境 | max_iterations | trainer timesteps | 环境交互步 | 目的 |
|---|---:|---:|---:|---:|---|
| 已完成 pilot | 2 | 500 | 8000 | 16000 | 验证链路和日志 |
| 单种子趋势 A | 2 | 2000 | 32000 | 64000 | 检查 reward、判别器和动作统计趋势 |
| 单种子趋势 B | 2 | 5000 | 80000 | 160000 | 对齐官方 trainer timesteps |
| 三种子对照 | 2 | 5000 | 80000 | 160000/seed | 仅在单种子出现稳定学习趋势后进行 |

每个单种子阶段保存中间 checkpoint，在 `500/1000/2000/5000` 更新点检查：任务奖励、风格奖励、判别器损失、episode length、动作范围、目标关节合法性、NaN/Inf 和 GPU 显存。单种子趋势没有出现前，不启动三种子奖励对照。

按已有 500 更新运行时间估算，2000 更新约需 10 分钟，5000 更新约需 25 分钟，前提是 GPU 上没有其他 Isaac Sim 进程。显存仍按 8 GiB 上限安排，运行前检查 GPU 占用。

## G1 路线并行审查

当前不直接把 R1 环境切换为 G1。候选路线的版本和资产闭环如下：

- `legged_lab` 有 `LeggedLab-Isaac-AMP-G1-v0` 和 29 DoF G1 配置，是最佳代码候选。它要求 Isaac Lab 2.3.2、通常配合 Isaac Sim 5.1、forked `rsl_rl` AMP 分支和 Git LFS 资产。
- `AMP_Running_baseline` 包含 G1 AMP 任务、checkpoint 和演示视频，是最佳 checkpoint 验证候选。motion、USD、checkpoint 依赖 Git LFS，后续仿真 checkpoint 的真机验证范围需要区分。
- `unitree_rl_lab` 支持 G1-29DoF，但当前没有 AMP 任务，适合资产和常规任务接口核对。
- `GMR` 支持 G1 29 DoF retarget，可作为专家动作工具。AMASS、LAFAN1 等原始数据授权需要单独闭合。

切换门槛是独立安装 Isaac Sim 5.1 / Isaac Lab 2.3.2、安装 Git LFS、记录 LFS object hash、USD/motion/checkpoint hash 和许可证，然后在本机完成官方 checkpoint import/play smoke。README、演示视频和 checkpoint 文件名不能替代本项目的本机证据。

## 当前回答

1. 距离成功复现 AMP 还需要完成官方配置下的单种子趋势训练、checkpoint 评估和训练指标记录。底层 PhysX 扫描不再是前置工作。
2. 目前没有证据证明官方 Humanoid28 物理环境无法支持 AMP。修正后的 reference replay 能走完真实参考轨迹，旧的 63 步终止包含末帧保持。
3. 既有 16000 环境交互步不足以判定 AMP 学不会。单种子阶段先扩展到 64000，再对齐 160000 环境交互步。
4. 先继续官方 Humanoid28 做单种子 AMP 趋势验证，同时准备 G1 2.3.2/Sim5.1 隔离环境。单种子趋势长期无效且 G1 checkpoint smoke 通过后，再切换主路线。
5. 本机单种子 2 环境、5000 更新预计约 25 分钟，峰值显存接近 8 GiB。官方 4096 环境需要更大 GPU 资源，不能在当前 RTX 3070 上直接复现。

AMP 长训尚未启动。本报告将 reference replay、官方 checkpoint 不可用和 G1 切换门槛作为当前 R1 的真实边界。
