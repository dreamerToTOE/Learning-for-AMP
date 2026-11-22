# Project Status

- **状态：PARTIAL**
- **日期：2026-11-23**
- **当前阶段：R1 — 官方 Humanoid AMP/PPO 最小 smoke（已完成，未进入长训）**
- **R1 基线 code_sha：`a290114737a346b32e02fc9ae17d8038c69934cf`**
- **真实训练：2 次最小 smoke（AMP 与同环境 PPO），各 4 环境、1 iteration、16 steps/env，不能作为收敛结果**
- **真实评估：0 次**
- **支持结论：官方 Humanoid AMP 与同环境临时 PPO 入口均已实测启动、完成一次更新并保存 checkpoint。没有 AMP/PPO 性能、独立评估、G1 训练或真机验证结论。**

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

## 下一步

1. 在释放外部 Isaac Sim 进程或降低并行环境后，固定当前版本开始短训练 smoke。
2. 建立独立评估入口，固定初态、seed、episode 数和速度/姿态/接触指标。
3. R2 先确认 G1 USD、专家动作、retarget 数据和许可证，再决定是否使用 `legged_lab` 的 Isaac Lab 2.3.x 路线。未满足前不启动 G1 AMP 训练。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | PARTIAL | `reports/R1_amp_baseline.md`、`configs/r1/skrl_ppo_amp_smoke.yaml`、`scripts/r1_ppo_amp_smoke.py`、`runs/r1_smoke/` |
| R2 G1 单技能 | BLOCKED | G1 动作/USD 对象需要 Git LFS，数据来源和授权尚未闭合，候选版本面向 Isaac Lab 2.3.x / Isaac Sim 5.1 |
| R3 多技能诊断与消融 | NOT_STARTED | — |
| R4 报告与求职材料 | NOT_STARTED | — |

## R1 状态字段

```text
date: 2026-11-23
stage: R1 — 官方 Humanoid AMP/PPO 最小 smoke
status: PARTIAL
code_sha: a290114737a346b32e02fc9ae17d8038c69934cf
environment: Ubuntu 22.04.5; RTX 3070 8 GiB; driver 580.178.04; Isaac Sim 4.5.0; official Isaac Lab v2.1.1; Kit Python 3.10.15; torch 2.5.1+cu118; skrl 1.4.3; gymnasium 0.29.1; hydra-core 1.3.2
commands_run: fresh v2.1.1 checkout; AMP Walk --algorithm AMP --num_envs 4 --max_iterations 1 --seed 42; same-env PPO wrapper --algorithm PPO --num_envs 4 --max_iterations 1 --seed 42; 0.5s nvidia-smi sampling
evidence_paths: reports/R1_amp_baseline.md; configs/r1/skrl_ppo_amp_smoke.yaml; scripts/r1_ppo_amp_smoke.py; runs/r1_smoke/
key_metrics: AMP and PPO exit code 0; both 16/16 rollout steps; AMP peak monitor 7203 MiB; PPO peak monitor 6475 MiB; no independent evaluation
blockers: RTX 3070 8 GiB with external Isaac Sim occupancy leaves little headroom; R1 full baseline still needs short training and independent evaluation; G1 assets/data require LFS and separate license/hash verification
next_action: run short AMP/PPO training smoke only after controlling external GPU occupancy, then add independent evaluation
```
