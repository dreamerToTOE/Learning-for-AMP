# R1 速度跟踪 PPO/AMP 公平对照协议

日期：2026-11-25

## 对照目标

在 Isaac Lab v2.1.1 官方 `Isaac-Humanoid-AMP-Walk-Direct-v0` 和 Humanoid28 上，比较同一明确前向速度任务下的 PPO 与 AMP。历史无速度目标的 PPO smoke、task-only 训练和 20.48M/40.96M 官方 AMP 训练继续作为独立证据，不并入本对照结论。

## 锁定配置

- Isaac Sim 4.5.0、Isaac Lab v2.1.1、skrl 1.4.3、CUDA 环境和独立 venv 保持不变。
- 世界坐标 +X 方向目标速度固定为 `target_vx = 1.5 m/s`。官方 `humanoid_walk.npz` 的 torso `body_linear_velocities[..., 0]` 均值为 `1.5064869 m/s`，因此目标在观察训练结果前冻结为 1.5。
- 速度尺度固定为 `sigma_vx = 0.5 m/s`。
- 两组使用完全相同的任务奖励

  `r_task = alive * (1 + exp(-0.5 * ((torso_vx - 1.5) / 0.5)^2))`

  `alive` 在躯干高度低于 0.5 m 时为 0。该奖励保留官方每步存活项，同时加入明确的前向速度跟踪项。速度奖励由外层运行时 patch 注入，不修改 Isaac Lab checkout。
- 观测 81 维、动作 28 维、动作目标映射、机器人资产、接触和动力学参数完全一致。
- Policy 与 value 网络均为 `[1024, 512]`，激活函数为 ReLU。两组共享 `rollouts=16`、`learning_epochs=6`、`mini_batches=2`、`learning_rate=5e-5`、`gamma=0.99`、`lambda=0.95`、PPO clip=0.2、value loss scale=2.5、RunningStandardScaler 和固定动作方差配置。
- PPO 使用 `r_task`，不包含判别器和风格奖励。
- AMP 使用相同 `r_task`，并设置 `task_reward_weight=1.0`、`style_reward_weight=1.0`。官方判别器、专家动作数据和 AMP 超参数保持不变。AMP 额外报告风格奖励和判别器开销。

## 预算和 seed

### Smoke

- 训练 seed：`42`
- 并行环境：`2`
- 更新次数：`50`
- 每次更新采样：`2 * 16 = 32` 个环境交互步
- 每组总环境交互步：`1,600`
- PPO 与 AMP 顺序使用相同环境和 GPU 配置。smoke 只验证奖励注入、入口、日志、checkpoint 和有限性，不用于判断算法优劣。

### 正式趋势阶段

- 训练 seeds：`42, 43, 44`
- 并行环境：`256`。该规模已通过 RTX 3070 容量测试。
- 每个 seed、每个算法：`256` 次更新，即 `256 * 256 * 16 = 1,048,576` 个环境交互步。
- 检查点：64、128、256 次更新。
- 正式训练在 smoke 通过并确认速度奖励日志后启动。三 seed 结果按配对差值汇总。

## 评估协议

- Smoke 使用固定评估 seeds `42000–42009`，每个 checkpoint 10 回合，仅用于确认推理和评估入口。
- 正式评估使用此前固定协议的 `42000–42099`，每个最终 checkpoint 100 回合。两组使用相同 seed、相同初始状态分布和 10 秒窗口，并记录初始观测 hash。
- 每回合记录完整运行率、跌倒率、存活时间、+X 净位移、实际 torso `vx`、目标速度 MAE、目标速度偏差、`|vx-target| < 0.2 m/s` 比例、参考速度 W1、动作有限性、目标角超限比例、关节跟踪误差和失败原因。
- 每个算法保存成功与失败视频。任务速度 MAE 和参考动作 W1 分开报告。判别器分数只作为 AMP 辅助诊断。
- Smoke 和正式对照均不改变 PhysX 质量、惯量、接触或 USD drive 参数。
