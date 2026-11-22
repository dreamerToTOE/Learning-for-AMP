# Project Status

- **状态：PARTIAL**
- **日期：2026-11-22**
- **当前阶段：R0 — 环境与上游审计（已完成审计，G1 资产链路仍阻塞）**
- **审计基线 code_sha：`803f84f55af0044a8649d0fad5c0c4205ca406ca`**
- **真实训练：1 次最小 AMP smoke，4 环境、1 iteration、16 rollout steps，不能作为收敛结果**
- **真实评估：0 次**
- **支持结论：官方 Humanoid AMP 启动和一次更新链路已实测通过。没有 AMP/PPO 性能、G1 训练或真机验证结论。**

## 本轮完成

- 克隆并核对 `dreamerToTOE/Learning-for-AMP`，读取项目规划、实验协议、移交说明和参考资料。
- 实测系统、GPU、显存、驱动、CUDA runtime、Python、Conda 环境、Isaac Sim、Isaac Lab、磁盘和内存。
- 确认当前主机是 RTX 3070 8 GiB，和项目记忆中的 RTX 4070 12 GiB 不一致。
- 发现现有 `/home/ubuntu2004/GRScenes/a/IsaacLab` 是 v1.2.0 且工作树有大量未提交修改，没有官方 Humanoid AMP 任务，最小空场景导入失败。
- 静态核对官方 Isaac Lab v2.1.1 AMP 任务、skrl 入口、网络配置和三份 `.npz` 动作数据。
- 使用 Isaac Sim 4.5.0 Kit Python、Isaac Lab v2.1.1 和 `skrl==1.4.3` 完成官方 Walk AMP 最小 smoke。policy、value、discriminator 初始化成功，16/16 rollout 完成，退出码为 0。
- 核对 `legged_lab`、`AMP_Running_baseline`、`unitree_rl_lab`、`unitree_model` 和 `GMR` 的版本、许可证、G1 任务、动作数据和 LFS 状态。
- 记录 R1 独立版本方案和 R2 的 G1 资产、授权、版本兼容性阻塞项。

## 下一步

1. 在独立的 Isaac Lab v2.1.1 checkout 中锁定 Isaac Sim 4.5.0、Kit Python 和 `skrl==1.4.3`，保存依赖清单和动作文件 hash。
2. 以 4 环境复跑官方 Walk AMP 与同环境、同预算的 PPO smoke，再逐级测试 8、16、32 环境的峰值显存。
3. 固定 R1 的 seed、总环境步数、评估条件和日志字段后，才开始短训练与独立评估。
4. R2 先确认 G1 USD、专家动作、retarget 数据和许可证，再决定是否使用 `legged_lab` 的 Isaac Lab 2.3.x 路线。未满足前不启动 G1 AMP 训练。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | NOT_STARTED | — |
| R2 G1 单技能 | BLOCKED | G1 动作/USD 对象需要 Git LFS，数据来源和授权尚未闭合，候选版本面向 Isaac Lab 2.3.x / Isaac Sim 5.1 |
| R3 多技能诊断与消融 | NOT_STARTED | — |
| R4 报告与求职材料 | NOT_STARTED | — |

## R0 状态字段

```text
date: 2026-11-22
stage: R0 — 环境与上游审计
status: PARTIAL
code_sha: 803f84f55af0044a8649d0fad5c0c4205ca406ca
environment: Ubuntu 22.04.5; RTX 3070 8 GiB; driver 580.178.04; Isaac Sim 4.5.0; official Isaac Lab v2.1.1; Kit Python 3.10.15; torch 2.5.1+cu118; skrl 1.4.3
commands_run: hardware/version probes; official AMP Walk smoke with --num_envs 4 --max_iterations 1 --seed 42; current IsaacLab v1.2.0 empty-scene import check; upstream git ls-remote
evidence_paths: reports/R0_environment_audit.md; ../r0_smoke_output.txt; logs/skrl/humanoid_amp_walk/2026-11-22_23-43-34_amp_torch/
key_metrics: 4 envs; 16/16 rollout steps; exit code 0; humanoid_walk.npz 154 frames at 60 fps; no independent evaluation
blockers: actual GPU is 8 GiB; existing IsaacLab checkout is dirty v1.2.0; G1 assets/data require LFS and separate license/hash verification; G1 candidates target newer Isaac versions
next_action: R1 short AMP/PPO smoke in isolated v2.1.1 environment, then freeze evaluation protocol
```
