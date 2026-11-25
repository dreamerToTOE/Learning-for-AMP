# R1 速度跟踪 PPO/AMP smoke 结果

日期：2026-11-25

本 smoke 使用预注册的速度任务奖励和相同的 2 环境、seed42、50 次更新预算。它验证入口、奖励注入、日志和 checkpoint，不能用于宣称 PPO 或 AMP 的最终性能优劣。

## 固定配置

- 目标速度为世界坐标 +X 的 1.5 m/s，sigma 为 0.5 m/s。奖励公式为 `alive * (1 + exp(-0.5 * ((torso_vx - 1.5) / 0.5)^2))`。
- 两组均使用 Humanoid28、81 维观测、28 维动作、`[1024, 512]` Actor/Critic、rollouts=16、6 epochs、2 mini-batches、学习率 5e-5、gamma 0.99 和 GAE lambda 0.95。
- AMP 使用 task weight 1 和 style weight 1。PPO 使用相同 task reward，风格奖励为 0。

## Smoke 结果

| 组别 | 交互步 | 训练任务奖励均值 | 训练 torso vx 均值 | 训练速度 MAE | 评估完整率 | 评估目标速度 MAE | 参考速度 W1 | 评估前进距离 | 峰值显存 | 最终 checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AMP | 1600 | 1.504 | 1.004 | 0.639 | 0% (0/10) | 0.525 | 0.475 | 0.498 m | 7212 MiB | `56cfccfcfdf8…` |
| PPO | 1600 | 1.419 | 0.777 | 0.811 | 0% (0/10) | 0.412 | 0.311 | 1.000 m | 6440 MiB | `c9bbb57503a0…` |

两组训练均退出码 0，任务奖励窗口和 TensorBoard loss 均可读取。AMP 的 smoke 记录了判别器 loss，PPO 没有判别器 loss。两组 10 回合 smoke 评估都在短预算下发生跌倒，因此结果只证明链路可运行。

## 原始证据

- [预注册协议](R1_speed_tracking_comparison_protocol.md)
- [机器可读汇总](r1_speed_tracking_smoke_summary.json)
- [AMP 训练速度指标](../runs/r1_speed_tracking_smoke/amp_seed42_env2_updates50/speed_metrics.jsonl)
- [PPO 训练速度指标](../runs/r1_speed_tracking_smoke/ppo_seed42_env2_updates50/speed_metrics.jsonl)
- [AMP 评估](r1_speed_tracking_smoke_eval/amp/episodes.json)
- [PPO 评估](r1_speed_tracking_smoke_eval/ppo/episodes.json)
- [AMP 失败案例视频](r1_speed_tracking_smoke_eval/amp/episode_001.mp4)
- [PPO 失败案例视频](r1_speed_tracking_smoke_eval/ppo/episode_001.mp4)
- [可复现 smoke 脚本](../scripts/r1_speed_tracking_smoke.sh)
