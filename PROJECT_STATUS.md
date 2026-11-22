# Project Status

- **状态：PARTIAL**
- **日期：2026-11-23**
- **当前阶段为 R1，官方 Humanoid AMP/PPO 两环境短训 smoke 已完成**
- **运行前项目 SHA 为 `9f38f2037274561391eba02db45f57057f7e99d5`**
- **本轮训练为 AMP 和同环境 PPO 各 2 环境、5 次更新、160 个环境交互步**
- **真实评估：0 次**
- **验证结果为两组均退出码 0、五次更新的损失和最终模型参数均为有限值。独立评估尚未开始。**

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

## 下一步

1. 建立独立评估入口，固定初态、seed、episode 数和姿态/接触指标。
2. R2 先确认 G1 USD、专家动作、retarget 数据和许可证，再决定是否使用 `legged_lab` 的 Isaac Lab 2.3.x 路线。未满足前不启动 G1 AMP 训练。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | PARTIAL | [R1 短训报告](reports/R1_amp_baseline.md#6-两环境短训-smoke2026-11-23)、`configs/r1/requirements-venv.lock`、本机 `runs/r1_short_train/` |
| R2 G1 单技能 | BLOCKED | G1 动作/USD 对象需要 Git LFS，数据来源和授权尚未闭合，候选版本面向 Isaac Lab 2.3.x / Isaac Sim 5.1 |
| R3 多技能诊断与消融 | NOT_STARTED | — |
| R4 报告与求职材料 | NOT_STARTED | — |

## R1 状态字段

```text
date: 2026-11-23
stage: R1 — 官方 Humanoid AMP/PPO 两环境短训 smoke
status: PARTIAL
code_sha: 9f38f2037274561391eba02db45f57057f7e99d5
environment: Ubuntu 22.04.5; RTX 3070 8 GiB; driver 580.178.04; Isaac Sim 4.5.0; official Isaac Lab v2.1.1; Kit Python 3.10.15; torch 2.5.1+cu118; isolated venv skrl 1.4.3; gymnasium 0.29.1; hydra-core 1.3.2; numpy 1.26.0; tensorboard 2.19.0; flatdict 4.0.1; h5py 3.13.0
commands_run: isolated runtime probe; AMP Walk and same-env PPO with --num_envs 2 --max_iterations 5 --seed 42 --headless --device cuda:0; nvidia-smi 500 ms sampling; TensorBoard and checkpoint validation
evidence_paths: reports/R1_amp_baseline.md; runs/r1_short_train/20261123T002939+0900_seed42_env2_steps80/
key_metrics: both exit code 0, 80 steps/env and 5 finite loss records; AMP/PPO whole-GPU peaks 4393/3607 MiB; independent eval episodes 0
blockers: R1 PASS still requires independent evaluation and a comparable baseline under the experimental protocol; G1 assets/data require LFS and separate license/hash verification
next_action: build the independent evaluation entry point and freeze its protocol
```
