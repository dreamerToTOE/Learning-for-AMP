# Project Status

- **状态：PARTIAL**
- **日期：2026-11-23**
- **当前阶段为 R1，正式三 seed 奖励对照、独立评估和全跌倒诊断已完成**
- **最新提交为 `51319a3b7bc8d3a62ad15d1fba9dc1bf14727291`**
- **运行前项目 SHA 为 `7bcd195fe83589a6b4d3d5836f9cfdaea765df38`**
- **正式训练为两种风格权重各 3 个 seed，每组 2 环境、500 次更新、16,000 个环境交互步**
- **每个最终 checkpoint 使用同一组 100 个评估 seed，六组共 600 个评估回合**
- **验证结果为六组 100% 跌倒，不能宣称 Walk 已学成，R1 仍为 PARTIAL**

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
- 完成正式三 seed 训练和同一 100 seed 评估，记录 paired deltas、参考速度 W1、视频和所有 checkpoint hash。
- 对六组 100% 跌倒做了可复现诊断。16,000 交互步/组仍不足以形成稳定步态，最终策略动作从 agent800 到 agent8000 明显放大。
- 单步动力学探针发现 agent 配置未裁剪动作，官方环境用完整关节范围作为 action scale。最终 checkpoint 首步目标关节范围达到 -5.49–7.96 rad，CPU PhysX 报告 spherical joint 目标角超出 [-pi, pi]。
- 同一 checkpoint 的临时动作裁剪和半范围探针将首步最大目标偏差从 7.83 rad 降至 4.97/3.01 rad，但 60 步探针仍跌倒，说明动作接口是首要问题但不是唯一原因。
- AMP 风格奖励在 task+style seed42 最终 checkpoint 的同一初态约为 4.43，而任务奖励固定为 1，风格项可能主导早期优势估计。正式三 seed 尚不能把该尺度单独归因为全跌倒。
- 核对 reset 代码和初态范围，躯干 z 相比参考动作稳定多出官方明确的 0.15 m，x 范围一致，暂未发现初始化错位或独立接触求解异常。
- 完成动作范围修正的 2 环境、50 更新短训和同一 10 初态评估。修正组无 PhysX 目标角错误，平均存活时间从 0.563 s 提高到 0.747 s，但跌倒率仍为 10/10。
- 在修正动作接口下完成 style weight 1、`discriminator_reward_scale=0.5` 的同预算短训。10/10 跌倒，平均存活时间 0.753 s，与修正 task-only 的 0.747 s 无实质差异。
- 新增参考动作直接回放。参考目标动作没有 PhysX 角度越界，但原始、去掉 reset 抬升、保持初始姿态和显式 stiffness 5000/damping 500 探针均在约 0.6 s 跌倒。
- 运行时 actuator tensor 中 `right_elbow`、`left_elbow`、`right_knee`、`left_knee` 的 stiffness、damping 和 effort limit 为 0，但同一环境加载的 USD joint prim 分别有 elbow 150/15、knee 300/30 的 drive 属性。显式填充增益后膝关节跟踪误差仍约 2.76/2.22 rad，说明存在 USD 到 PhysX tensor 的解析不一致和更深的接触动力学问题。
- 关闭重力的参考回放完成 77 步而有重力在第 17 步跌倒，失稳触发点位于重力接触承载阶段。
- solver position/velocity 单因素探针显示默认 `4/0` 完成 18 步，`16/4` 完成 49 步，`32/4` 反而完成 17 步。position 12、14、16、18、20、24 的窄 sweep 为 23、37、49、26、18、16 步。`16/4` 加显式 stiffness 5000/damping 500 后完成 47 步。接触 solver 对承载时间有明显但非单调影响，仍未完成 77 步参考回放。
- 新增项目侧 USD drive override，按 joint prim gains 转换为运行时单位。gain scale `0.5/1/2/4` 在 solver `16/4` 下完成 49/49/50/50 步，仍未完成 77 步。直接 IdealPD 探针出现 NaN，暂不进入训练。
- 新增足底接触力观测。contact-force view 能读到约 640/1405 N 的初始足底力，但会把同一回放从 49 步改变为 23 步，因此仅作为独立观测证据。
- 地面摩擦 sweep `0.1/0.5/1/2/5` 的完成步数为 `25/58/49/23/16`，起始帧 `0/30/60/90/120` 在摩擦 0.5 下完成 `58/63/22/25/22` 步，仍未完成 77 步。
- 已改用不创建 contact-force view 的载荷 proxy。它基于躯干 link 速度有限差分、整机质量和重力项，phase 30、摩擦 0.5 仍完成 63 步，和无 proxy 基线一致。proxy 最大绝对载荷约 893.9/119.3/3122.5 N。

## 下一步

1. 保留摩擦 0.5、起始帧 30 作为诊断条件，拆分接触观测扰动和真实物理参数，参考回放稳定后才延长训练预算。
2. R2 先确认 G1 USD、专家动作、retarget 数据和许可证，再决定是否使用 `legged_lab` 的 Isaac Lab 2.3.x 路线。未满足前不启动 G1 AMP 训练。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | PARTIAL | [R1 报告](reports/R1_amp_baseline.md)、[正式评估汇总](reports/r1_formal_eval/summary.csv)、[正式对照协议](docs/EXPERIMENT_PROTOCOL.md#6-r1-walk-正式对照预注册) |
| R2 G1 单技能 | BLOCKED | G1 动作/USD 对象需要 Git LFS，数据来源和授权尚未闭合，候选版本面向 Isaac Lab 2.3.x / Isaac Sim 5.1 |
| R3 多技能诊断与消融 | NOT_STARTED | — |
| R4 报告与求职材料 | NOT_STARTED | — |

## R1 状态字段

```text
date: 2026-11-23
stage: R1 — 正式三 seed 奖励对照与独立评估
status: PARTIAL
code_sha: 7bcd195fe83589a6b4d3d5836f9cfdaea765df38
environment: Ubuntu 22.04.5; RTX 3070 8 GiB; driver 580.178.04; Isaac Sim 4.5.0; official Isaac Lab v2.1.1; Kit Python 3.10.15; torch 2.5.1+cu118; isolated venv skrl 1.4.3; gymnasium 0.29.1; hydra-core 1.3.2; numpy 1.26.0; tensorboard 2.19.0; flatdict 4.0.1; h5py 3.13.0
commands_run: six AMP training runs with style-weight 0.0/1.0, seeds 42/43/44, --num_envs 2 --max_iterations 500; six 100-episode paired evaluations with seeds 42000–42099; 500 ms GPU sampling
evidence_paths: reports/R1_amp_baseline.md; reports/r1_formal_eval/; runs/r1_formal/; runs/r1_formal_eval/20261123T021500+0900_seed42000-42099/; scripts/r1_diagnose_dynamics.py; runs/r1_diagnostics/
key_metrics: six formal evaluation groups all fell; action correction smoke 10/10 fell but survival mean improved 0.563 to 0.747 s; corrected style scale 0.5 remained 10/10 fall with survival 0.753 s; direct reference replay fell at step 17 with zero target-angle errors; USD elbow/knee drive attrs disagree with runtime actuator tensor; no-gravity replay completed 77 steps; solver 16/4 completed 49 steps; drive override scales 0.5/1/2/4 completed 49/49/50/50 steps; friction and phase sweeps still completed at most 63 steps; non-invasive load proxy preserved phase 30 result at 63 steps
blockers: reference motion is not dynamically trackable under gravity in the current Humanoid28 USD/PhysX actuator setup, so longer AMP budgets are not interpretable yet; G1 assets/data require LFS and separate license/hash verification
next_action: use friction 0.5 and phase 30 for a focused gravity/contact diagnosis while isolating contact-force instrumentation; do not claim R1 PASS
```
