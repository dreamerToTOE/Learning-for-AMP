# Project Status

- **状态：PARTIAL**
- **日期：2026-11-23**
- **当前阶段为 R1，正式三 seed 奖励对照、独立评估和全跌倒诊断已完成**
- **最新提交为 `a63db00`**
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
- 围绕摩擦 0.4–0.55、起始帧 26–34 的细网格仍以摩擦 0.5、起始帧 30 的 63 步最好，未找到 77 步稳定窗口。
- 目标帧 offset `0/1/2/3/4` 完成 `32/32/63/63/38` 步，参考关节速度目标仍为 63 步。restitution sweep 最好为 `0.12`、68 步，仍未达到 77 步。
- 结构审计显示 reset 后 body position 最大误差 `3.8e-7 m`、joint position 误差 0、动作无 soft-limit 越界，名称映射和坐标系基本正确。foot body velocity 误差最高约 `0.63 m/s`，NPZ 存在重复位置帧与非零速度不一致。有限差分速度回放 phase30 仅 30 步，未形成修复。
- reset velocity frame offset `0/1/2/3` 在 phase30 分别完成 `63/27/37/31` 步，原始速度相位反而最好，简单换下一帧或有限差分不能修复结构不匹配。
- unique 30 Hz 位置重建速度并用于 reset/velocity target 仅完成 29 步，每步直接写回参考 joint state 仅完成 28 步，未恢复 77 步。速度字段问题与执行器跟踪都不是唯一失稳来源。
- 去重为 30 Hz、77 帧的 motion file，有限差分速度回放完成 29 步，保留原始速度完成 63 步。IdealPD 低增益探针产生 NaN，未形成稳定修复。
- Humanoid28 runtime 总质量约 35.73 kg，body mass/inertia 有限且左右对称。USD elbow/knee axis 与 radian soft limits 一致，未发现轴反向或单位错误。foot collision geometry 位于引用 payload，已完成 payload、实例原型和预物理修改路径审计。
- 已完成 Humanoid28 USD payload 接触几何审计。左右脚 Cube 来自 `Props/instanceable_meshes.usd`，extent 为 `±(0.0885, 0.045, 0.0275) m`，局部平移 z 为 `-0.0225 m`，frame30 参考姿态下底部约为 `-0.0295/-0.0641 m`。运行时路径是 instance proxy，直接改 prototype 不可用。
- 已验证展平 USD、解除 instanceability、保留 root→mesh 分层复制和预物理 payload 修改四条路径。原始远程资产 CUDA 回放为 63 步且全状态有限，展平资产为 24 步，分层 mesh 几何偏移在 47–57 步，预物理远程 payload 偏移在 23–52 步。没有几何偏移达到 77 步。
- `scripts/r1_reference_replay.py` 已加入 root、feet、joint state 的逐步有限性检查，记录 `nonfinite_step` 和 `all_finite`，旧的 NaN 假成功结果不再作为证据。详细结果见 `reports/R1_humanoid28_contact_geometry.md`。
- 沿 CUDA、摩擦 0.5、起始帧 30、solver 16/4 完成接触参数诊断。默认 contact offset 为 0.00545 m、rest offset 为 0 m。contact offset sweep 为 46–56 步，正 rest offset 为 47 步，friction combine multiply 为 20 步，restitution 0.12 加 average combine 最好为 68 步，全部状态有限。
- 完成质量/惯量来源诊断。runtime mass 与碰撞体积乘 1000 kg/m³ 精确对应。显式锁定当前 runtime mass/diagonal inertia 仍为 63 步，质量缩放 0.5/2.0 降至 41/32 步，说明质量会影响稳定性，但当前推断值不是唯一失稳原因。
- 进一步提取并覆盖 runtime COM、完整 inertia tensor 和 principal axes 后，回放为 38 步且全有限，说明显式质量属性的坐标语义或 PhysX 资产重建路径会改变动力学。完整覆盖暂不作为修复方案。
- 保持 body-frame diagonal inertia、写入 COM 并使用 identity principal axes 的变体为 49 步，仍低于原始几何推断资产的 63 步。
- 完成 USD drive 到 actuator tensor 映射诊断。四个 elbow/knee 在原始 tensor 中 stiffness、damping、effort 为 0，repair/override 后计算 torque 变为非零，但三条完整轨迹逐步一致并均为 63 步。ImplicitActuator 将位置/速度目标直接交给 PhysX，实际轨迹仍由 USD implicit drive 主导。显式 torque PD 探针在低增益下仅 13/21 步。
- 验证了本地 root USD 的实际 drive 覆盖工具。将四个关节从原始 150/15、300/30 设为 300/30 后，CUDA 回放仍为 63/77 且全状态有限，表明简单 USD 增益覆盖也不是当前失稳的单一来源。
- 将本地 USD 四个关节 drive 直接提高到约 8594/859，并关闭 Isaac Lab override 后，CUDA 回放仍为 63/77，进一步排除了简单关节增益不足作为单一原因。
- 在 restitution `0.12` average combine 下细扫起始帧和 restitution 参数，起始帧 28/30/32 为 26/68/27 步，restitution 0.1175–0.1205 为 68/68/68/67 步，确认当前接触最佳结果稳定在 68 步附近。
- 完成 R1 路线纠偏。reference replay 现在默认在真实 motion 末帧停止，明确区分末帧保持。CUDA phase30 exact replay 为 62 个真实参考控制步，`terminated_step=null`、`all_finite=true`，旧的 63 步终止包含一次额外末帧保持。
- 审计官方 AMP 配置。默认 4096 环境、80000 trainer timesteps、rollouts16、task reward 0、style reward 1、官方网络 1024/512。官方 Humanoid AMP Walk checkpoint probe 返回 unavailable，本机没有可用预训练 checkpoint。
- 既有每组 16000 环境交互步重新标记为奖励消融和训练链路 smoke，暂不作为 AMP 与普通 PPO 行走性能的最终证据。新的单种子预算计划为 64000 环境步趋势阶段，再对齐 160000 环境步阶段。
- 完成 G1 路线复审。`legged_lab` 是最佳代码候选，`AMP_Running_baseline` 是最佳 checkpoint 验证候选，但都要求新的 Isaac Sim 5.1 / Isaac Lab 2.3.2 环境、Git LFS 和资产/许可证闭环。
- 诊断结果记录在 `reports/R1_humanoid28_gravity_diagnosis.md`，并新增 `scripts/r1_probe_actuator_mapping.py`、`scripts/r1_make_humanoid28_mass_override.py`。

## 下一步

1. 暂停底层物理参数搜索，按官方 AMP 配置在当前 RTX 3070 上进行单种子趋势训练。先完成 64000 环境交互步，再决定是否对齐 160000 环境步并进入三种子对照。
2. R2 先确认 G1 USD、专家动作、retarget 数据和许可证，再决定是否使用 `legged_lab` 的 Isaac Lab 2.3.x 路线。未满足前不启动 G1 AMP 训练。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | PARTIAL | [R1 报告](reports/R1_amp_baseline.md)、[路线纠偏](reports/R1_reassessment.md)、[正式评估汇总](reports/r1_formal_eval/summary.csv)、[正式对照协议](docs/EXPERIMENT_PROTOCOL.md#6-r1-walk-正式对照预注册) |
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
key_metrics: six formal evaluation groups all fell; action correction smoke 10/10 fell but survival mean improved 0.563 to 0.747 s; corrected style scale 0.5 remained 10/10 fall with survival 0.753 s; direct reference replay fell at step 17 with zero target-angle errors; USD elbow/knee drive attrs disagree with runtime actuator tensor; no-gravity replay completed 77 steps; CUDA gravity baseline with friction 0.5, phase 30 and solver 16/4 completed 63 finite steps; restitution 0.12 average combine reached 68 finite steps; contact offset/rest offset sweeps reached at most 63/47 steps; mass scale 0.5/2.0 reached 41/32 steps; actuator gain repair changed logged torque but left trajectories bit-identical at 63 steps; payload collision geometry audit found foot-Cube bottoms around -0.0295/-0.0641 m at frame 30; finite-value gating now rejects NaN runs
blockers: reference motion is not dynamically trackable under gravity in the current Humanoid28 USD/PhysX actuator setup, so longer AMP budgets are not interpretable yet; G1 assets/data require LFS and separate license/hash verification
next_action: run a single-seed official AMP trend stage with 64000 environment transitions under the current 2-env memory constraint; reclassify prior 16000-step results as smoke/ablation; keep G1 2.3.2/Sim5.1 and asset verification in parallel; do not claim R1 PASS
```
