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

## 单种子趋势 A 实测结果

2026-11-24 在 GPU 空闲后按官方 AMP 配置完成了 seed 42 的 2000 次更新。训练命令使用 `--num_envs 2`、`--max_iterations 2000`、`rollouts=16`、`task_reward_weight=0` 和 `style_reward_weight=1`。因此 skrl trainer 完成 32000 个 trainer timesteps，环境共交互 64000 步。网络、任务、动作文件和奖励定义均沿用官方配置，没有启动 PPO 或三种子实验。

训练进程正常退出，未出现 OOM、NaN 或 Inf，并保存了 `agent_3200.pt` 到 `agent_32000.pt`、`best_agent.pt` 和 TensorBoard 事件文件。训练事件共记录 100 个标量点：

| 指标 | 首点（320） | 末点（32000） | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| `Reward / Total reward (mean)` | 22.289 | 20.449 | 11.636 | 24.473 |
| `Loss / Discriminator loss` | 1.985 | 1.724 | 1.379 | 1.985 |
| `Loss / Policy loss` | -0.078 | -0.104 | -0.114 | -0.046 |
| `Loss / Value loss` | 0.887 | 0.440 | 0.041 | 1.306 |
| `Policy / Standard deviation` | 0.0550 | 0.0550 | 0.0550 | 0.0550 |

总奖励均值和判别器损失都在区间内振荡，没有形成单调改善或稳定平台。由于官方 Walk 的每步任务奖励固定为 1，`Total reward` 在这里主要反映 episode 长度，不能单独解释为行走质量。

GPU 监控记录了 471 个样本。整卡显存峰值为 4507 MiB，最低空闲显存为 3334 MiB。最终 checkpoint `agent_32000.pt` 的 SHA256 为 `a632d268ec20caca75d4e126527a055bec34b133e4a701f0e2daf8beff947534`。原始日志、事件文件、checkpoint 和显存 CSV 保存在 `runs/r1_single_seed_official/20261124T113000+0900_seed42_env2_updates2000/` 与 `logs/skrl/r1_single_seed_official_20261124/`，可复核摘要见 `reports/r1_single_seed_official_20261124.json`。

对最终 checkpoint 使用同一官方环境做了 10 个独立回合评估。10/10 回合以跌倒结束，没有 timeout。平均存活 25.7 个控制步、0.857 秒，平均前进距离 0.902 m，平均净前向速度 1.118 m/s，参考速度分布 W1 为 0.553 m/s。这个结果说明策略已经产生了有限的前向位移，但尚未形成稳定步态，不能作为 AMP 复现成功证据。

单步动作探针保持有限值，但最终 checkpoint 的原始动作范围为 `[-0.544, 1.078]`，目标关节范围为 `[-2.915, 3.057]` rad，最大目标变化为 `3.250` rad。动作上界仍有越过 `1` 的样本，动作接口和闭环稳定性需要继续作为诊断项记录。

本阶段的结论是训练链路和资源预算可以支持 2000 更新的官方 AMP 运行，但当前单种子没有显示清晰学习趋势，三种子正式对照暂缓。若继续 Humanoid28，应先把本次结果作为趋势基线，再决定是否执行对齐官方 trainer timesteps 的 5000 更新阶段。底层 PhysX 参数搜索仍不恢复。

## 单种子趋势 B 实测结果

2026-11-24 在相同独立环境中完成了 5000 次更新。训练保持 seed 42、`--num_envs 2`、`rollouts=16`、`task_reward_weight=0`、`style_reward_weight=1`、官方网络和动作文件不变。skrl trainer 完成 80000 timesteps，环境共交互 160000 步，达到官方 trainer timesteps 配置。

训练进程正常退出，未出现 OOM、NaN 或 Inf。TensorBoard 记录了 100 个标量点，选定更新点如下：

| trainer timesteps | Total reward mean | Discriminator loss | Policy loss | Value loss |
|---:|---:|---:|---:|---:|
| 800 | 22.482 | 1.662 | -0.065 | 0.406 |
| 16000 | 18.077 | 1.596 | -0.102 | 0.118 |
| 32000 | 18.455 | 1.716 | -0.101 | 0.507 |
| 48000 | 13.661 | 1.730 | -0.100 | 0.175 |
| 64000 | 17.440 | 1.750 | -0.097 | 0.431 |
| 80000 | 15.714 | 1.930 | -0.092 | 0.309 |

整个训练过程的总奖励均值范围为 13.147–22.482，判别器损失范围为 1.435–1.986。两者没有形成持续改善，训练后段的总奖励低于早期。官方 Walk 每步任务奖励固定为 1，因此总奖励主要表示 episode 长度。

GPU 监控记录了 1041 个样本。整卡显存峰值为 4203 MiB，最低空闲显存为 3637 MiB。最终 checkpoint `agent_80000.pt` 的 SHA256 为 `f648fc4264ca09710e3a0b6d4a4ccbd3c3f99df85a4b2fd431bb40ae4588a6a2`。原始日志、事件文件、checkpoint 和显存 CSV 保存在 `runs/r1_single_seed_official/20261124T132500+0900_seed42_env2_updates5000/` 与 `logs/skrl/r1_single_seed_official_20261124_budget5000/`，可复核摘要见 `reports/r1_single_seed_official_20261124_budget5000.json`。

最终 checkpoint 的单步动作探针保持有限值，但原始动作范围为 `[-0.544, 1.446]`，目标关节范围为 `[-2.916, 4.500]` rad，最大目标变化为 `4.693` rad。相比 2000 更新 checkpoint，动作上界和目标变化进一步增大。

使用最终 checkpoint 做同一套 10 回合评估，10/10 回合以跌倒结束，没有 timeout。平均存活 24.0 个控制步、0.800 秒，平均前进距离 0.847 m，平均净前向速度 1.077 m/s，参考速度分布 W1 为 0.525 m/s。5000 更新阶段没有形成稳定步态，不能作为 AMP 复现成功证据。

本阶段完成了官方 trainer timesteps 的预算验证，但结果没有显示可持续学习趋势。三种子正式对照暂缓，底层 PhysX 参数搜索不恢复。后续应把 Humanoid28 的结果作为 R1 失败边界和训练链路证据，并继续 G1 版本、资产和专家数据闭环审查。

## G1 路线并行审查

当前不直接把 R1 环境切换为 G1。候选路线的版本和资产闭环如下：

- `legged_lab` 有 `LeggedLab-Isaac-AMP-G1-v0` 和 29 DoF G1 配置，是最佳代码候选。它要求 Isaac Lab 2.3.2、通常配合 Isaac Sim 5.1、forked `rsl_rl` AMP 分支和 Git LFS 资产。
- `AMP_Running_baseline` 包含 G1 AMP 任务、checkpoint 和演示视频，是最佳 checkpoint 验证候选。motion、USD、checkpoint 依赖 Git LFS，后续仿真 checkpoint 的真机验证范围需要区分。
- `unitree_rl_lab` 支持 G1-29DoF，但当前没有 AMP 任务，适合资产和常规任务接口核对。
- `GMR` 支持 G1 29 DoF retarget，可作为专家动作工具。AMASS、LAFAN1 等原始数据授权需要单独闭合。

切换门槛是独立安装 Isaac Sim 5.1 / Isaac Lab 2.3.2、安装 Git LFS、记录 LFS object hash、USD/motion/checkpoint hash 和许可证，然后在本机完成官方 checkpoint import/play smoke。README、演示视频和 checkpoint 文件名不能替代本项目的本机证据。

## 当前回答

1. 官方配置下的单种子 2000 和 5000 更新、checkpoint 评估和训练指标记录已经完成。5000 更新达到官方 trainer timesteps，但没有形成稳定学习趋势，三种子对照不启动，底层 PhysX 扫描不恢复。
2. 目前没有证据证明官方 Humanoid28 物理环境无法支持 AMP。修正后的 reference replay 能走完真实参考轨迹，旧的 63 步终止包含末帧保持。
3. 160000 环境交互步的单种子预算验证仍未显示稳定学习。现有证据足以暂缓三种子对照。
4. 同时准备 G1 2.3.2/Sim5.1 隔离环境，优先闭合 USD、专家动作、LFS 对象和许可证，再决定是否切换主路线。
5. 本机单种子 2 环境、5000 更新预计约 25 分钟，峰值显存接近 8 GiB。官方 4096 环境需要更大 GPU 资源，不能在当前 RTX 3070 上直接复现。

AMP 的 2000 和 5000 更新单种子趋势已完成，但尚未进行三种子对照。本报告将 reference replay、官方 checkpoint 不可用和 G1 切换门槛作为当前 R1 的真实边界。
