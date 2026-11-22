# Project Status

- **状态：PARTIAL**
- **日期：2026-11-23**
- **当前阶段为 R1，独立评估 smoke 和等预算奖励资源 pilot 已完成**
- **运行前项目 SHA 为 `65f7a8d9baa1d9b55e66a8941b50ddfc5807d78b`**
- **独立评估为 AMP 和同环境 PPO 各 10 个回合，10 对初始 policy observation 哈希全部一致**
- **本轮训练为同一 AMP 实现下的任务奖励组和任务加风格组，各 2 环境、50 次更新、1,600 个环境交互步**
- **正式多 seed 等预算对照尚未执行**

## 本轮完成

- 克隆并核对 `dreamerToTOE/Learning-for-AMP`，读取项目规划、实验协议、移交说明和参考资料。
- 实测系统、GPU、显存、驱动、CUDA runtime、Python、Conda 环境、Isaac Sim、Isaac Lab、磁盘和内存。
- 确认当前主机是 RTX 3070 8 GiB，和项目记忆中的 RTX 4070 12 GiB 不一致。
- 发现现有 `/home/ubuntu2004/GRScenes/a/IsaacLab` 是 v1.2.0 且工作树有大量未提交修改，没有官方 Humanoid AMP 任务，最小空场景导入失败。
- 静态核对官方 Isaac Lab v2.1.1 AMP 任务、skrl 入口、网络配置和三份 `.npz` 动作数据。
- 使用 Isaac Sim 4.5.0 Kit Python、Isaac Lab v2.1.1 和 `skrl==1.4.3` 完成官方 Walk AMP 最小 smoke。policy、value、discriminator 初始化成功，16/16 rollout 完成，退出码为 0。
- 核对 `legged_lab`、`AMP_Running_baseline`、`unitree_rl_lab`、`unitree_model` 和 `GMR` 的版本、许可证、G1 任务、动作数据和 LFS 状态。
- 记录 R1 独立版本方案和 R2 的 G1 资产、授权、版本兼容性阻塞项。
- 新建带 Git metadata 的独立 Isaac Lab v2.1.1 checkout，锁定 Isaac Sim 4.5.0、Kit Python、PyTorch、skrl、Gymnasium、Hydra 和 Walk 动作 hash。
- 在同一个官方 `Isaac-Humanoid-AMP-Walk-Direct-v0` 环境中完成 AMP 与临时 PPO 配置的最小 smoke，固定 seed 42、4 环境、16 steps/env。
- 记录两组 GPU before/after/0.5 秒采样、stdout、Hydra 配置和 checkpoint hash。
- 保留一次因 sparse checkout 缺少 `isaaclab_mimic` 导致的失败日志，并扩大独立 checkout 的 source 范围后重跑成功。
- 创建真正独立的 Python 3.10 venv，清除旧 PYTHONPATH，验证运行时不含 anaconda、用户或 ROS site-packages。
- 在该 venv 中锁定并重跑 AMP/PPO，补齐 `flatdict==4.0.1` 和 `h5py==3.13.0` 后两组均退出码为 0。
- 在同一 venv 中顺序完成两环境短训，固定 seed 42，逐次导出损失并保存最终 checkpoint。RTX 3070 整卡显存采样峰值为 AMP 4393 MiB、PPO 3607 MiB。
- 新增独立评估入口，记录跌倒、超时、存活时间、位移、速度和参考速度分布距离，保存 20 个回合的数据与两个骨架视频。
- 预注册正式奖励消融，验证两组任务奖励权重一致，仅风格权重不同。资源 pilot 两组均完成 50 次更新。

## 下一步

1. 根据资源 pilot 预先写定正式训练步数，运行已预注册的多 seed 奖励消融和独立评估。
2. R2 先确认 G1 USD、专家动作、retarget 数据和许可证，再决定是否使用 `legged_lab` 的 Isaac Lab 2.3.x 路线。未满足前不启动 G1 AMP 训练。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | PARTIAL | [R1 报告](reports/R1_amp_baseline.md)、[配对回合数据](reports/r1_eval/)、[正式对照协议](docs/EXPERIMENT_PROTOCOL.md#6-r1-walk-正式对照预注册) |
| R2 G1 单技能 | BLOCKED | G1 动作/USD 对象需要 Git LFS，数据来源和授权尚未闭合，候选版本面向 Isaac Lab 2.3.x / Isaac Sim 5.1 |
| R3 多技能诊断与消融 | NOT_STARTED | — |
| R4 报告与求职材料 | NOT_STARTED | — |

## R1 状态字段

```text
date: 2026-11-23
stage: R1 — 独立评估 smoke 与奖励消融资源 pilot
status: PARTIAL
code_sha: 65f7a8d9baa1d9b55e66a8941b50ddfc5807d78b
environment: Ubuntu 22.04.5; RTX 3070 8 GiB; driver 580.178.04; Isaac Sim 4.5.0; official Isaac Lab v2.1.1; Kit Python 3.10.15; torch 2.5.1+cu118; isolated venv skrl 1.4.3; gymnasium 0.29.1; hydra-core 1.3.2; numpy 1.26.0; tensorboard 2.19.0; flatdict 4.0.1; h5py 3.13.0
commands_run: scripts/r1_eval_walk.py with --episodes 10 --seed 1000 --video; AMP style-weight 0.0/1.0 pilot with --num_envs 2 --max_iterations 50 --seed 42
evidence_paths: reports/R1_amp_baseline.md; reports/r1_eval/; runs/r1_eval/20261123T004510+0900_seed1000_n10/; runs/r1_reward_pilot/20261123T004402+0900_seed42_env2_steps800/
key_metrics: 10/10 paired initial observation matches; AMP 9 falls and 1 timeout, PPO 10 falls; pilot 1600 env transitions per arm, 4274/4290 MiB whole-GPU peaks
blockers: formal multi-seed baseline still pending; G1 assets/data require LFS and separate license/hash verification
next_action: freeze formal training budget from resource pilot and execute the paired protocol
```
