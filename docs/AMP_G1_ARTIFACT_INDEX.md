# AMP-G1 工件与复现索引

2026-11-26 · 第一轮冻结快照。本文路径从 `work/Learning-for-AMP` 仓库根起算，`../r1`、`../r2`、`../r3` 是同级的本机工作目录。原始报告中有更早的绝对路径，迁移时以经过此次核验的相对路径为准。

## 完整清单和副本口径

[机器可读清单](../reports/amp_g1_archive/local_artifacts.json) 对每个找到的文件保存 `path`、`kind`、`bytes`、`sha256`、`github_status`，并在匹配到时记录对应的 GitHub 路径。覆盖本机训练、评估、诊断、冻结目录，含权重、TensorBoard、stdout、GPU 采样、配置、MP4、派生 HOLD 素材与用到的 G1 MJCF/网格。环境由依赖锁和版本记录还原。按本地路径共 3,585 项、21,263,576,303 B，其中 1,134 项 checkpoint、77 项视频、959 项日志、1,415 项其他实验记录。目录拷贝形成的同字节文件分别记录。

核对远程基线为项目 GitHub `refs/heads/main` 的 `b00128eead251a69bf5d8cc713fe062d4fe481fb`。该次核查没有发现 Releases 或其他远端分支与 tag。以该树中的文件对象核对同字节内容，以解压后的 SHA-256 核对 gzip 对应源。归档提交发布的是报告和清单，扫描时的二进制副本状态没有因文档归档而改变。 [GitHub 核对基线](https://github.com/dreamerToTOE/Learning-for-AMP/tree/b00128eead251a69bf5d8cc713fe062d4fe481fb)

| 清单状态 | 含义 |
| --- | --- |
| `exact` | GitHub main 基线有同字节文件，可能是不同路径 |
| `gzip_payload` | GitHub 的 gzip 解压后与本地文件字节相同 |
| `no_exact_copy` | 当前核对范围内没有对应字节副本。报告或曲线可能保留摘要，上游文件可能另有获取入口 |

七份主权重和全部扫描到的 checkpoint 都是本机保存。GitHub 已有 17 个不同 MP4 路径，包括 R1 速度对照 4 段、R3 混合基线 3 段、R3 速度 A/B 10 段。完整训练日志和轨迹按清单逐条区分，远程摘要只能恢复已记录的指标。HOLD、镜像专家文件及其权重来源需按各自授权处理。

## 冻结权重

表中路径均已读取并计算 SHA-256，对应原始运行或原冻结清单。七份都没有本项目 GitHub 二进制副本。R2 最佳步行模型以粗体标记。

| 用途及独立证据 | 本地文件 | 字节 | SHA-256 |
| --- | --- | ---: | --- |
| R1 官方纯风格 AMP · 40,960,000 步，97/100 跑满。 [归档](../reports/R1_eval_freeze_40m.md) | `artifacts/r1_reward_hacking/frozen_models/official_amp_40_96m/checkpoints/agent_80.pt` | 23,124,598 | `22f389039512e97a542403702e391b3d7efde619a78008026d849144dadd860c` |
| R1 速度 PPO · 39,976,960 步末档，爬行反例。 [对照](../reports/R1_speed_tracking_seed42.md) | `artifacts/r1_reward_hacking/frozen_models/speed_ppo_40m/checkpoints/agent_19520.pt` | 14,807,261 | `c44a7e64054b9780ca6a9ad5ffb3bf0eb0e39b2ec379565488a9dc7b89b1bb8a` |
| R1 速度 AMP · 39,976,960 步末档，低姿支撑反例。 [诊断](../reports/R1_reward_hacking_audit.md) | `artifacts/r1_reward_hacking/frozen_models/speed_amp_40m/checkpoints/agent_19520.pt` | 23,124,600 | `3ec7e37f62c917eaa06a62bd97e1a5ce525dd8e3f1ca2d9e44f907ab4e9bbd21` |
| **R2 G1 单步行 · 4,194,304 步，100/100 持续步行。** [独立评估](../reports/r2_g1_single_skill/amp_4m_heldout_100.json) | `artifacts/r2_g1_single_skill/frozen_walk_seed42_4194304/model_0000001023.pt` | 18,136,129 | `81e0d7de6f62fd38eba67cf94357a2744bf42b2197e04ef8adce756dae973bc1` |
| R3 原始混合 AMP · 累计 8,388,608 步，慢跑 0/30。 [数据](../reports/r3_walk_jog/warm_4m.json) | `artifacts/r3_walk_jog/frozen_baseline_seed42_8388608/model_0000001023.pt` | 18,136,129 | `fa8004ee65152e0c1c31d5d686641358773b71a27ab54b598c8940a34a319b4a` |
| R3 条件 Actor + 普通 D · 4,194,304 步，速度控制基准。 [数据](../reports/r3_speed_amp/ordinary_held_out.json) | `artifacts/r3_speed_amp/frozen_seed42_4194304/ordinary/model_0000001023.pt` | 18,144,449 | `7e5e4deedf99116b752604c27522dbfde4c7cdee26bfb052ab347b4520a03314` |
| R3 条件 Actor + 条件 D · 4,194,304 步，对照。 [数据](../reports/r3_speed_amp/conditional_held_out.json) | `artifacts/r3_speed_amp/frozen_seed42_4194304/conditional/model_0000001023.pt` | 18,148,545 | `fb67f453264ecab452ae21555bd180d48d66a0ea7efa4dc5664c6ef7765ed8c8` |

另外几份本机恢复入口如下。项目 GitHub 均只保存记录。

| 文件及用途 | 字节 | SHA-256 |
| --- | ---: | --- |
| R1 20.48M 对照权重 `logs/skrl/r1_amp_formal_seed42_20261124_final5_official/2026-11-25_00-23-05_amp_torch_resume_agent9920_updates5/checkpoints/agent_80.pt` | 23,124,598 | `01ff324c5454c157f743b3bab8f3fba6cdea978781bafa46686cd8d6291b356` |
| R1 两组速度对照证据包 `artifacts/r1_reward_hacking/seed42_evidence_2998bda.tar.gz`，包含原报告目录中的数据、曲线及录像 | 8,496,861 | `57e16bbb3a09b280a5f8066ac95bac2372ef96fc539a2e68c8b963c6607738af` |
| MimicKit 镜像 `../r2/assets/MimicKit/deepmimic_g1_walk_model.pt`，显式参考跟踪的历史推理正例。 [镜像版本](https://huggingface.co/datasets/blanchon/MimicKit_Data/tree/3ab706e7fd951478745ea78e71435d5721de12c3) | 11,236,396 | `5adec8cdbd90389f4aac981307a676a452e5cb478cab1ddddb27869e4cdaab4d` |
| `../r2/assets/AMP_Running_baseline/model_6200.pt`，候选上游 BASE，已核对 LFS 哈希和 CPU 结构，尚未独立运动验证 | 19,815,093 | `5de7628fe92bd74118016c592b95eb17183547a5289708622c80a7815da025ac` |

原分阶段清单也保留在 GitHub。R1 记录训练原档及三个模型的配置文件。R3 的冻结记录另含派生动作、逐步轨迹和运行日志的哈希。 [R1 本机原档映射](../reports/r1_reward_hacking/frozen_models.json) · [R3 混合](../reports/r3_walk_jog/frozen_artifacts.sha256) · [R3 A/B](../reports/r3_speed_amp/frozen_artifacts.sha256)

## R0

| 类别 | 从哪里获取及怎么核对 |
| --- | --- |
| 原报告、配置和版本 | [环境与实际 smoke 命令](../reports/R0_environment_audit.md#43-已执行-smoke) · [R0 提交 `5d22605`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/5d22605) · [随后依赖隔离 `9f38f20`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/9f38f20) · [venv 锁](../configs/r1/requirements-venv.lock) |
| 本机原件 | `../r0_smoke_output.txt`，9,986 B，SHA-256 `1a84a21634bc672256026a30b361497109463461e7ad6f27e01e483efb6e6cee`。`../../logs/skrl/humanoid_amp_walk/2026-11-22_23-43-34_amp_torch/` 存原日志和 checkpoint。两者都没有 GitHub 二进制副本 |
| 运行环境 | Isaac Sim 4.5.0 standalone 和 Kit 3.10.15，Isaac Lab v2.1.1 `90b79bb2d44feb8d833f260f2bf37da3487180ba`，skrl 1.4.3、torch 2.5.1+cu118。初次 smoke 与后来独立版本建立时的 PYTHONPATH 有差异 |
| 最小核验 | 从仓库根执行 `git -C ../r1/IsaacLab-v2.1.1 rev-parse HEAD && sha256sum ../r0_smoke_output.txt`。真实启动的历史命令见报告，隔离路径见下节 |

## R1

| 问题/产物 | 确认到的本地位置及远程证据 |
| --- | --- |
| 隔离、最初短训和 16K 全跌倒 | 原件在 `../r1_runs/`、`runs/r1_{smoke,venv_smoke,short_train,formal,formal_eval}/`，骨架视频 `../../outputs/R1-*.mp4`。GitHub 有 600 回合摘要和逐回合数据。 [锁定入口](../scripts/r1_python.sh) · [基线报告](../reports/R1_amp_baseline.md) · [评估数据](../reports/r1_formal_eval/summary.json) |
| 动作、参考动力学与错计末帧 | `runs/r1_{actionrange_smoke,actuator,contact_probe,diagnostics,reference_replay,structure,usd_inspect}/`。确切 62 步结果在 `runs/r1_structure/replay_exact_reference_phase30_cuda.json`。 [诊断脚本](../scripts/r1_diagnose_dynamics.py) · [回放入口](../scripts/r1_reference_replay.py) · [结构审计](../reports/R1_humanoid28_contact_geometry.md) · [纠偏 `20792b0`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/20792b0) |
| 两环境机制及高并行容量 | `runs/r1_capacity`、`runs/r1_single_seed_official/` 和 `logs/skrl/r1_capacity_sweep_*`，可找每组 `command.txt`、stdout、GPU CSV 和事件。 [小样本 `2bad715`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/2bad715) · [高并行容量](../reports/r1_capacity_sweep_high_20261124.json) · [高并行训练](../reports/R1_formal_seed42_env2048.md) |
| 20.48M 到 40.96M | `runs/r1_formal_seed42_env2048_*`、`runs/r1_resume_seed42_env2048_*`、`logs/skrl/r1_amp_{formal,resume}_*`。TensorBoard/逐次 JSONL 与最终 checkpoint 留本机。 [20.48M 逐回合](../reports/r1_eval_freeze_20m/new_independent_45000/episodes.json) · [40.96M 恢复提交 `1528ba9`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/1528ba9) · [训练恢复记录](../reports/R1_resume_to_40m.md) · [40.96M 逐回合](../reports/r1_eval_freeze_40m/new_independent_45000/episodes.json) |
| 同任务对照与新规则中止 | 训练 `runs/r1_speed_tracking_formal_seed42/{ppo,amp}/` 及对应 `logs/skrl/r1_speed_tracking_formal_*`。独立评估 `runs/r1_speed_tracking_seed42_eval/`。触地、门限和未完成短试跑 `runs/r1_reward_hacking/`。远端保存上千回合、曲线、GPU 记录及回放。 [速度对照 `2998bda`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/2998bda) · [速度奖励核心](../scripts/r1_speed_task.py) · [规则审计](../reports/R1_reward_hacking_audit.md) |

纯风格 20.48M 和 40.96M 成功、失败 MP4 均在 `artifacts/r1_reward_hacking/frozen_videos/r1_eval_freeze_{20m,40m}/` 下，逐文件哈希见清单。这四个冻结视频为本机保存。两组速度对照可在 GitHub 查四段 viewport 录像，包括 PPO 爬行、AMP 低姿支撑和各自提前终止。 [速度对照录像表](../reports/R1_speed_tracking_seed42.md#实景回放)

下表定位本机原始诊断文件。JSON 记录运行条件和逐步数值，末端报告或脚本说明如何调用。文件大小、SHA-256 和远端状态可按该路径在总清单检索。

| 问题史条目 | 本机最小证据 | 复现入口及提交 |
| --- | --- | --- |
| R1.2–3 动作超限、奖励与重置 | `runs/r1_diagnostics/task_seed42_{probe_cpu,raw_60,clip_60,clip_half_60}.json`、`task_style_seed42_probe_cpu.json`。两套修正短训的 `runs/r1_actionrange_smoke/20261123T103000+0900_seed42_env2_steps800/{corrected,corrected_style_scale05}/command.txt` | [首步探针 `51319a3`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/51319a3) · [短训命令](../reports/R1_amp_baseline.md#11-动作范围修正短对照2026-11-23) |
| R1.4–5 solver 及 USD drive | `runs/r1_reference_replay/reference_no_lift_pos{12,14,16,18,20,24,32}vel4.json`、`runs/r1_actuator/probe_{base,override,repair}.json`、`replay_local_drive_high.json` | [驱动解析 `6abc5e2`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/6abc5e2) · [探针](../scripts/r1_probe_actuator_mapping.py) |
| R1.6 接触与相位 | `runs/r1_contact_probe/phase30_{contact,estimated_load}.json`、`grid/f0p5_p30.json`、`runs/r1_structure/replay_cuda_combo_rest012_p{28,30,32}.json` | [相位摩擦 `1c5ec6b`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/1c5ec6b) · [参数记录](../reports/R1_humanoid28_gravity_diagnosis.md#接触参数) |
| R1.7 重建速度 | `runs/r1_structure/frame0_nolift.json`、`reconstructed_phase15.json`、`reconstructed_rawvel_phase15.json`、`runs/r1_contact_probe/velocity_phase/voff{0,1,2,3}.json` | [重建试验 `dd7b987`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/dd7b987) · [重建数据入口](../scripts/r1_reconstruct_motion.py) |
| R1.8 几何、质量与惯量 | `runs/r1_structure/humanoid28_{physics_audit,payload_layers,runtime_mass_properties,full_mass_override_audit}.json`、`mass_inertia_probe_summary.json`、`replay_cuda_baseline_finite.json` | [接触几何 `3a41ee5`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/3a41ee5) · [质量坐标 `3419846`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/3419846) · [抽取代码](../scripts/r1_extract_humanoid28_mass_properties.py) |

在已装好的独立环境中做冻结权重的只读复评。下面输出到新的临时目录，`--episodes 100` 可复跑完整同口径独立组。两个速度任务的同初态奖励和接触测量用原有回放入口。 [回放脚本](../scripts/r1_reward_hacking_replay.sh) · [原诊断条件](../reports/R1_reward_hacking_audit.md#同初态诊断)

```bash
out=$(mktemp -d)
frozen=artifacts/r1_reward_hacking/frozen_models/official_amp_40_96m/checkpoints/agent_80.pt
sha256sum "$frozen"
bash scripts/r1_python.sh -u scripts/r1_eval_walk.py \
  --checkpoint "$frozen" --output "$out/official" --seed 45000 --episodes 10 --headless --device cuda:0
```

历史结构诊断的最小入口。用于恢复原探针的证据口径，指定起点、摩擦和求解迭代。回放到真实参考末帧结束。

```bash
out=$(mktemp -d)
bash scripts/r1_python.sh -u scripts/r1_reference_replay.py --output "$out/replay.json" \
  --headless --device cuda:0 --seed 1000 --steps 77 --start-frame 30 --frame-stride 2 \
  --ground-friction 0.5 --solver-position-iterations 16 --solver-velocity-iterations 4 \
  --no-reset-lift --override-usd-drives
```

## R2

- **环境、代码与资产。** `../r2/mimickit-venv` 和 `../r2/upstream/MimicKit`，上游 `a2c41528c2a34b638611f6c618e4a59f0e151179`，Python 3.12.14、torch 2.8.0+cu128、Newton 1.6.1、Warp 1.17.0、MuJoCo-Warp 3.12.0。G1 29 DoF 使用 `data/assets/g1/g1.xml` 与网格。两项容量补丁及冻结配置要一并复核。 [版本与命令](../reports/R2_g1_single_skill.md#复现命令) · [依赖锁](../reports/r2_g1_single_skill/requirements.lock.txt) · [资产及哈希](../reports/r2_g1_single_skill/asset_manifest.json) · [容量补丁](../configs/r2/)
- **专家来源。** `../r2/upstream/MimicKit/data/motions/g1/g1_walk.pkl` 通过固定镜像取得，原始动捕和再分发许可未闭合。实际加载的 G1 模型目录有 BSD-3-Clause，和动作、权重的权利分开核查。 [来源审计](../reports/R2_g1_single_skill.md#资产与动作)
- **原训练与录像。** `../r2/runs/amp_walk_smoke_seed42*`、`amp_walk_pilot_seed42_env128*`、`amp_walk_seed42_env128_to1m`、`amp_walk_seed42_env128_to4m`。最后一目录的 `videos/fall_2m.mp4`（77,240 B）及 `videos/walk_4m.mp4`（435,433 B）均为本机原件，精确哈希见总清单。 [训练过程 `0f515b1`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/0f515b1) · [冻结 `0d7eca4`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/0d7eca4) · [评估数据与视频校验记录](../reports/r2_g1_single_skill/amp_4m_heldout_100.json)

冻结模型走路评估示例。使用 `--seeds $(seq 43100 43199)` 可按原协议复跑 100 相位。

```bash
project="$PWD"
r2=$(realpath ../r2)
out=$(mktemp -d)
frozen="$project/artifacts/r2_g1_single_skill/frozen_walk_seed42_4194304"
(cd "$frozen" && sha256sum -c model.sha256)
env -u PYTHONPATH -u PYTHONHOME XDG_CACHE_HOME="$out/cache" TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" scripts/r2_eval_mimickit_walk.py --mimickit "$r2/upstream/MimicKit" \
  --checkpoint "$frozen/model_0000001023.pt" --seeds 43000 43001 43002 43003 43004 \
  --output "$out/r2_fixed.json"
```

## R3

| 阶段 | 本地原件 | 公共证据及运行入口 |
| --- | --- | --- |
| R3-0 HOLD 与 R2 漂移 | `../r3/assets/hold/` 原 H5，`../r3/assets/walk_jog/` 派生动作、命令元数据、预览与配置，`../r3/runs/r3_0_r2_drift/` 原始漂移记录 | [来源 `d51afd1`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/d51afd1) · [审计和裁剪清单](../configs/r3/hold_walk_jog.json) · [转换](../scripts/r3_prepare_hold.py) · [预览](../scripts/r3_preview_hold.py) · [漂移](../reports/r3_walk_jog/r2_frozen_drift.json) |
| R3-1 混合基线 | `../r3/runs/r3_1_walk_jog_{smoke,seed42_4m,seed42_warm_4m}/`，包括 GPU、权重、stdout、评估 JSON/NPZ。冻结目录有 3 段对应视频副本 | [冻结 `a9ecf38`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/a9ecf38) · [训练与数据命令](../reports/R3_walk_jog_baseline.md#复现入口) · [已上传 3 段视频](../reports/R3_walk_jog_baseline.md#冻结与下一步) · [评估器](../scripts/r3_eval_amp_modes.py) |
| R3-2 初版规则、修订与普通 D | `../r3/runs/r3_2_speed_seed42_{smoke,4m,pelvis_smoke,pelvis_4m}/`。前两目录保留原躯干终止试跑。后两目录是修订后的 smoke 及正式训练 | [门限修订 `c2e093a`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/c2e093a) · [普通 D `e135245`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/e135245) · [锁定任务和评估](../configs/r3/speed_amp.json) · [核心环境](../scripts/r3_speed_amp.py) |
| R3-3 条件 D | `../r3/runs/r3_3_speed_seed42_conditional_{smoke,4m}/`。两组冻结目录另存确切的 `speed_amp.json`、运行代码 SHA、模型和评估 JSON/NPZ | [同预算对照 `b00128e`](https://github.com/dreamerToTOE/Learning-for-AMP/commit/b00128e) · [末档汇总](../reports/r3_speed_amp/comparison.json) · [训练/评估命令](../reports/R3_speed_amp.md#复现) · [已上传 10 段配对视频](../reports/r3_speed_amp/media/) |

HOLD 源版本 `f80b04d384d4ec8922c88deada878c88597a3cf9`。派生文件由配置、转换审计记录各片段的来源、许可、帧区间与关节约定。H5、PKL 与原预览的实际存在性和哈希可在总清单按路径查询。 [上游数据集](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9) · [核定来源及数据设置](../reports/R3_walk_jog_baseline.md#r3-0-专家数据)

在 R2 原依赖、上游及容量补丁就绪，且 HOLD 已按冻结配置准备好的情况下，以下命令从仓库根复扫末档。输出写到临时目录，原始模型和冻结评估文件保持可校验。

```bash
project="$PWD"
r2=$(realpath ../r2)
r3=$(realpath ../r3)
upstream="$r2/upstream/MimicKit"
out=$(mktemp -d)

env -u PYTHONPATH -u PYTHONHOME XDG_CACHE_HOME="$out/cache" TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" scripts/r3_eval_amp_modes.py --upstream "$upstream" \
  --data "$r3/assets/walk_jog" --protocol configs/r3/walk_jog_eval.json --split held_out \
  --checkpoint "$project/artifacts/r3_walk_jog/frozen_baseline_seed42_8388608/model_0000001023.pt" \
  --output "$out/mixed.json"

(
  cd "$upstream"
  for variant in ordinary conditional; do
    frozen="$project/artifacts/r3_speed_amp/frozen_seed42_4194304/$variant"
    env -u PYTHONHOME PYTHONPATH="$upstream/mimickit" XDG_CACHE_HOME="$out/cache" TORCH_COMPILE_DISABLE=1 \
      "$r2/mimickit-venv/bin/python" "$project/scripts/r3_eval_speed_amp.py" \
      --data "$r3/assets/walk_jog" --protocol "$frozen/speed_amp.json" \
      --gait "$project/configs/r3/walk_jog_eval.json" --variant "$variant" --split held_out \
      --checkpoint "$frozen/model_0000001023.pt" --output "$out/$variant.json"
  done
)
```

## 迁移与查验

从 GitHub 可以恢复协议及已有的公开证据。需要在另一主机重跑时，先从本项目原机按对应权限恢复 checkpoint、资产和所需原始文件，保持相对路径或更新配置路径，再校验 SHA-256。以文件身份和许可为依据恢复，旧 run 同名文件也要验哈希。R7 恢复和新实验的初始状态协议见后续计划。 [二轮恢复入口](AMP_G1_FUTURE_PLAN.md#二轮恢复入口)
