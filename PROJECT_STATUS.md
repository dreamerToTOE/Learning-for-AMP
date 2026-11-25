# Project Status

- **当前阶段 R3 / IN_PROGRESS，更新于 2026-11-25。** R3-0 数据与冻结 R2 的漂移诊断已完成。R3-1 原始混合 AMP 以 `PARTIAL` 归档。seed42 累计 8.389M 步，末档 30/30 步行初态走满 10 秒；30 个慢跑初态中 21 个减速后走满、9 个跌倒，持续慢跑 0/30。下一项为条件 Actor 与普通判别器。 [基线报告](reports/R3_walk_jog_baseline.md)
- **R2 / COMPLETE_WITH_LIMITATIONS。** 单技能 AMP 仿真行走通过基础验收。原始专家片段的来源和许可待核验，R3 可以使用来源和许可独立核实的数据。
- **G1 纯风格 AMP 的 4,194,304 步 checkpoint 已冻结。** seed42 在独立 MimicKit / Newton 环境完成连续 1,024 次更新。100 个新初态均直立行走至 10 秒，完整率 100%（Wilson 95% 区间 96.3%–100%），无跌倒，平均前进 9.21 m、横向位移 −2.51 m。 [独立评估](reports/r2_g1_single_skill/amp_4m_heldout_100.json)
- **固定初态下，2.052M 档 0/5 完整，3.281M 档首次达到 5/5。** RTX 3070 实测 128 环境总耗时 22 分 10 秒，整卡显存峰值 5,277 MiB。 [阶段证据与录像说明](reports/R2_g1_single_skill.md)
- **R1 / COMPLETE_WITH_LIMITATIONS。** 训练、推理与独立评估链路已复现。R1 对照训练、PPO 奖励修改和 seed43、44 复验已取消。
- **官方纯风格 AMP 行走已通过基础验收。40.96M checkpoint 已冻结，100 个独立初态中有 97 个跑满约 10 秒。**
- **1.5 m/s 速度任务的 seed42 对照已完成。** 两组各采样 40,009,728 步。约 40M 的评估 checkpoint 上，PPO 持续前进 82/100，AMP 为 0/100。实景回放显示 PPO 代表回合采用低姿爬行。两组直立行走均为 0/100。 [seed42 对照报告](reports/R1_speed_tracking_seed42.md)
- **R1 结论限于已完成的实验。** 现有结果不足以判定 AMP 在公平的直立行走任务中全面优于 PPO。这个限制不影响开展 G1 单技能 AMP。 [奖励漏洞诊断与冻结证据](reports/R1_reward_hacking_audit.md)

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
- 按官方 `Isaac-Humanoid-AMP-Walk-Direct-v0` 配置完成 seed 42、2 环境、2000 更新的单种子 AMP 趋势训练。skrl trainer 完成 32000 timesteps，环境共交互 64000 步，训练正常退出并保存了中间和最终 checkpoint。
- 记录了 100 个 TensorBoard 标量点、471 个 GPU 显存样本和最终动作探针。总奖励均值从 22.289 波动到 20.449，判别器损失从 1.985 波动到 1.724，均未形成清晰收敛趋势。整卡显存峰值 4507 MiB，最低空闲 3334 MiB。
- 用最终 checkpoint 完成 10 回合独立评估，10/10 跌倒，平均存活 25.7 控制步、0.857 秒，平均前进距离 0.902 m。该结果不支持宣称稳定行走或 AMP 复现成功。
- 保存单种子可复核摘要 `reports/r1_single_seed_official_20261124.json`，原始日志和 checkpoint 保留在被忽略的 `runs/` 与 `logs/` 目录。
- 按同一官方配置完成 5000 更新单种子预算验证。skrl trainer 完成 80000 timesteps，环境共交互 160000 步，训练正常退出并保存了 `agent_80000.pt` 和中间 checkpoint。
- 5000 更新阶段记录 100 个 TensorBoard 标量点和 1041 个 GPU 显存样本。总奖励均值为 22.482（800 timesteps）到 15.714（80000 timesteps），判别器损失为 1.662 到 1.930，未形成持续改善。整卡显存峰值 4203 MiB，最低空闲 3637 MiB。
- 用最终 checkpoint 完成同一 10 回合评估，10/10 跌倒，平均存活 24.0 控制步、0.800 秒，平均前进距离 0.847 m。5000 更新结果不支持宣称稳定行走或 AMP 复现成功。
- 保存单种子预算验证摘要 `reports/r1_single_seed_official_20261124_budget5000.json`。
- 完成 2、8、16、32、64、128、256 并行环境各 50 次更新的容量测试。所有组退出码为 0，未出现 OOM、NaN 或 Inf。256 环境达到 6606.45 环境步/s，峰值显存 4859 MiB，最低空闲 2974 MiB。
- 核对 skrl AMP 的 rollout、GAE、优势归一化、motion dataset、reply buffer 和判别器采样。2 环境每次更新只有 32 个 PPO rollout 样本，每个 mini-batch 16 个样本，判别器每个 mini-batch 实际使用 16 个策略、16 个回放和 16 个专家状态。
- 保存容量与小样本审计报告 `reports/R1_capacity_and_small_batch.md`、机器可读摘要 `reports/r1_capacity_sweep_20261124.json` 和可复现实验脚本 `scripts/r1_capacity_sweep.sh`。
- 补测 512、1024、2048 并行环境各 50 次更新。三组均退出码为 0，未出现 OOM、NaN 或 Inf。2048 环境吞吐约 34,860 环境步/s，峰值显存 5823 MiB，最低空闲 2009 MiB。
- 新增 `scripts/r1_amp_train_monitored.py`，在 AMP 更新边界记录实际风格奖励、策略/专家判别器评分、动作范围、目标角超限比例、目标位置误差、存活时间和跌倒率。2 环境监控 smoke 正常退出并写入逐更新 JSONL。 之后补充逐关节 `target_angle_violation_by_joint` 字段，2 环境 1 更新 smoke 记录了全部 28 个关节。
- 使用 2048 环境、seed42 连续完成 625 次更新和 20,480,000 个环境交互步，主训练峰值显存 6194 MiB，最低空闲 1638 MiB，退出码为 0。
- 独立评估四个阶段 checkpoint，各使用固定评估 seed 42000–42009。约 20.48M 步 checkpoint 的 10 个回合中 6 个完整运行 9.97 秒，4 个跌倒，平均前进距离 7.07 m，参考速度 W1 为 0.392 m/s。
- 对 20.48M checkpoint 重复运行 42000–42009，并新增 43000–43019、44000–44019 两组独立评估。重复组的初始观测哈希和回合结果完全一致，三组共 50 回合中 41 回合跑满 9.97 秒，9 回合跌倒。
- 保存固定初态与独立 seed 复评报告 `reports/R1_repeatability_eval_env2048.md` 和汇总 `reports/r1_repeatability_eval_env2048/summary.json`。
- 保存高并行容量摘要 `reports/r1_capacity_sweep_high_20261124.json`、正式训练报告 `reports/R1_formal_seed42_env2048.md` 和阶段评估汇总 `reports/r1_formal_eval_env2048/summary.json`。
- 审计 20.48M checkpoint 的恢复内容，确认 policy、value、discriminator、optimizer 和三个 running preprocessor 可恢复，motion dataset 与 replay buffer 未写入 checkpoint。
- 从 20.48M checkpoint 在 2048 环境、seed42 下完成 625 次恢复更新，新增 20.48M 环境交互步，随后用官方 skrl 入口完成最后 5 次更新，累计达到 40.96M。该过程记录为有边界恢复，不称为严格无缝连续训练。
- 恢复段峰值显存 7832 MiB，采样到的最低空闲显存 0 MiB，GPU 利用率最高 98%，无 OOM、NaN 或 Inf。为维持训练路径稳定，额外 AMP 监控降为每 31 次更新一次，抽样 128 个状态。
- 对 40.96M checkpoint 按同一协议完成固定基准 10 回合和新增独立 100 回合评估。新增组 97/100 完整运行约 10 秒，3 回合因躯干高度低于 0.5 m 跌倒，完整率 97.0%，Wilson 95% 区间 91.5%–99.0%。
- 保存 40.96M 评估原始数据、成功/失败视频、恢复审计和阶段报告。当前稳定行走验收达标，不推进 81.92M。
- 冻结明确速度跟踪 PPO/AMP 对照协议。目标速度取官方 walk 专家 torso 前向速度均值约 1.506 m/s，预注册为 1.5 m/s，任务奖励为 `alive * (1 + exp(-0.5 * ((torso_vx - 1.5) / 0.5)^2))`。
- 新增不修改 Isaac Lab checkout 的速度奖励训练入口、同网络 PPO 配置和目标速度评估字段。
- 完成 seed42、2 环境、50 次更新的 PPO/AMP smoke，每组 1,600 环境交互步，均退出码 0 并保存 checkpoint。AMP 训练任务奖励均值 1.504，PPO 为 1.419；AMP 记录判别器 loss，PPO 记录 policy/value loss。
- 两组 smoke checkpoint 用相同 42000–42009 评估 seed 各运行 10 回合，均在短预算下全回合跌倒。AMP 平均存活 0.547 s，PPO 0.807 s。该结果仅确认速度奖励、训练入口、评估入口和日志链路可运行，不作为算法性能结论。
- Smoke AMP 峰值显存 7212 MiB、最低空闲 620 MiB；PPO 峰值显存 6440 MiB、最低空闲 1392 MiB。两组均无 OOM、NaN、Inf 或非有限动作。
- 速度入口的 AMP 容量测试覆盖 256、512、1024、2048 环境。先前 smoke 有约 2.6 GiB 的额外整卡基线占用，排查见到并行 Isaac Sim 进程约占 2671 MiB。2048 环境容量测试吞吐约 22,835 步/s，整卡峰值 5781 MiB。
- 使用 2048 环境、seed42 从随机初始化各训练 1221 次更新，AMP 与 PPO 环境交互预算相同。实测训练耗时为 AMP 843.53 s、PPO 660.83 s。AMP 长跑曾采到 0 MiB 空闲显存，两组正常退出。
- 两组在约 1M、4M、10M、20M、40M 的 checkpoint 上各完成 100 个固定初态评估。约 40M 步时，PPO 跑满 91 回合、持续前进 82 回合、平均前进 13.48 m。AMP 跑满 71 回合、持续前进 0 回合、平均前进 1.24 m。五档评估合计 1,000 回合。所有跑满回合的终止躯干高度低于 0.66 m，均低于直立判据。 [独立评价数据](reports/r1_speed_tracking_seed42/summary.json)
- 在同初态回放中实测地面法向力，确认 PPO 爬行和速度 AMP 低姿支撑存在非足端接触。诊断原型在另一组 20 个初态保留了官方 AMP 的完整回放，并提前终止两个低姿策略。短训试跑因显存余量不足中止，等待重启的任务随 R1 收尾取消。 [冻结模型和诊断数据](reports/R1_reward_hacking_audit.md)

## 下一步

1. R3-2 为 G1 Actor 加入速度指令并保留普通判别器。沿用 HOLD 数据和 G1 环境，先做含直立与接触约束的速度任务 smoke，再评估不同初态的指令跟踪和走跑切换。
2. R3-3 开展一次条件判别器对照。R3-4 做轻量对称性尝试，R3-5/6 按实际问题选做。每项一轮诊断、一轮修正。R3 正式训练建议累计 2–3 小时，预计超出此前 4–5 小时授权预算先汇报。
3. R4 收尾 AMP 报告、视频、简历与面试材料，保留负结果与限制。
4. R5 优先进入 G1 Motion Tracking / BeyondMimic 的数据重定向和 RL 跟踪部分。R6 聚焦独立推理、Sim2Sim、DR 与扰动及部署基础。均尚未开始，真机另需硬件和授权。

## 阶段记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 环境与上游审计 | PARTIAL | `reports/R0_environment_audit.md`、项目外部 `../r0_smoke_output.txt`、官方 v2.1.1 Walk smoke 日志 |
| R1 官方 AMP baseline | COMPLETE_WITH_LIMITATIONS | [R1 报告](reports/R1_amp_baseline.md)、[路线纠偏](reports/R1_reassessment.md)、[容量与小样本审计](reports/R1_capacity_and_small_batch.md)、[高并行容量补测](reports/r1_capacity_sweep_high_20261124.json)、[20.48M 冻结评估](reports/R1_eval_freeze_20m.md)、[20.48M 到 40.96M 恢复报告](reports/R1_resume_to_40m.md)、[40.96M 冻结评估](reports/R1_eval_freeze_40m.md)、[速度跟踪 PPO/AMP 协议](reports/R1_speed_tracking_comparison_protocol.md)、[速度跟踪 smoke](reports/R1_speed_tracking_smoke.md)、[seed42 速度对照](reports/R1_speed_tracking_seed42.md)、[奖励漏洞诊断](reports/R1_reward_hacking_audit.md) |
| R2 G1 单技能 | COMPLETE_WITH_LIMITATIONS | 仿真步行验收通过，原始数据来源与许可待核验。 [冻结评估](reports/R2_g1_single_skill.md) |
| R3 多技能诊断与消融 | IN_PROGRESS | R3-0 完成，R3-1 `PARTIAL` 已冻结。 [混合 AMP 独立评估](reports/R3_walk_jog_baseline.md) · [阶段路线图](docs/ROADMAP.md) |
| R4 AMP 归档与求职材料 | NOT_STARTED | — |
| R5 Motion Tracking / BeyondMimic | NOT_STARTED | [路线图](docs/ROADMAP.md) |
| R6 鲁棒运控与部署 | NOT_STARTED | [路线图](docs/ROADMAP.md) |

## R1 状态字段

```text
date: 2026-11-25
stage: R1 — 官方 AMP 40.96M 基线与 seed42 速度跟踪 PPO/AMP 对照
status: COMPLETE_WITH_LIMITATIONS
training_code_sha: d853659
environment: Ubuntu 22.04.5; RTX 3070 8 GiB; driver 580.178.04; Isaac Sim 4.5.0; official Isaac Lab v2.1.1; Kit Python 3.10.15; torch 2.5.1+cu118; isolated venv skrl 1.4.3; gymnasium 0.29.1; hydra-core 1.3.2; numpy 1.26.0; tensorboard 2.19.0; flatdict 4.0.1; h5py 3.13.0
commands_run: r1_speed_tracking_capacity.sh; r1_speed_tracking_formal_seed42.sh; r1_speed_tracking_stage_eval.sh; r1_summarize_speed_tracking.py
evidence_paths: reports/R1_eval_freeze_20m.md; reports/r1_eval_freeze_20m/summary.json; reports/r1_resume_checkpoint_audit_20m.json; reports/R1_resume_to_40m.md; reports/r1_resume_to_40m_summary.json; reports/R1_eval_freeze_40m.md; reports/r1_eval_freeze_40m/summary.json; reports/r1_monitor_joint_smoke_20261125.json; reports/R1_speed_tracking_comparison_protocol.md; reports/R1_speed_tracking_smoke.md; reports/r1_speed_tracking_smoke_summary.json; reports/R1_speed_tracking_smoke_eval/amp/episodes.json; reports/R1_speed_tracking_smoke_eval/ppo/episodes.json; scripts/r1_speed_tracking_train.py; scripts/r1_speed_tracking_smoke.sh; configs/r1/skrl_ppo_speed_tracking.yaml; runs/r1_resume_seed42_env2048_to4096_updates625/; logs/skrl/r1_amp_resume_seed42_20251125/; logs/skrl/r1_amp_resume_seed42_20251125_final5/; final_checkpoint: logs/skrl/r1_amp_resume_seed42_20251125_final5/2026-11-25_02-40-09_amp_torch_resume40m_final5/checkpoints/agent_80.pt; final_checkpoint_sha256: 22f389039512e97a542403702e391b3d7efde619a78008026d849144dadd860c
speed_smoke_metrics: seed42; 2 environments; 50 updates; 1,600 transitions per group; AMP task reward mean 1.504, training torso vx 1.004 m/s, target MAE 0.639 m/s; PPO task reward mean 1.419, training torso vx 0.777 m/s, target MAE 0.811 m/s; smoke eval 0/10 complete for both groups; AMP peak 7212 MiB and PPO peak 6440 MiB; no OOM/NaN/Inf
key_metrics: 40.96M cumulative environment transitions; new independent evaluation 97/100 complete, 3 falls, 97.0% complete rate, Wilson 95% CI 91.5%–99.0%; pooled baseline plus new evaluation 106/110 complete, 96.4%, Wilson 95% CI 91.0%–98.6%; new-set mean survival 9.71 s, forward distance 11.26 m, mean speed 1.13 m/s, reference-speed W1 0.354 m/s; mean target-angle violation 0.777%, maximum episode 2.029%; all actions finite and failures were torso-height termination; resumed segment peak 7832 MiB, sampled minimum free 0 MiB, GPU utilization max 98%, no OOM/NaN/Inf
resume_boundary: policy, value, discriminator, optimizer and running preprocessors restored; motion dataset and reply buffer were absent from checkpoint and reinitialized; final 5 updates used the official entrypoint without extra monitor forwards; this is bounded resume training, not strict seamless continuity
speed_comparison: seed42, 2048 envs, 1221 updates, 40009728 transitions per algorithm; 100 paired initial states per checkpoint; final evaluated checkpoint 39976960 transitions; PPO 82/100 sustained forward, 9/100 height terminations, fixed-horizon MAE 0.358 m/s; AMP 0/100 sustained forward, 29/100 height terminations, fixed-horizon MAE 1.387 m/s; zero upright walking episodes in both groups
speed_comparison_evidence: reports/R1_speed_tracking_seed42.md; reports/r1_speed_tracking_seed42/summary.json
limitations: single training seed for the speed comparison; zero matched upright walking episodes; no evidence for general AMP superiority on a fair upright task
closed: additional R1 reward changes and comparisons cancelled; seed43 and seed44 replication cancelled; artifacts retained; R2 allowed to proceed
```
