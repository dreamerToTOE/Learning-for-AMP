# R3-2 / R3-3 · G1 前向速度指令与条件判别器

## 预注册

本阶段从随机权重开始。使用 R3-1 已验证的 G1 29 DoF、Newton、240 Hz 仿真、30 Hz 控制、HOLD 四段动作及每段 25% 的专家采样。R3-1 冻结为纯风格参考。两组的 Actor 与 Critic 接收当前物理观测和一个以 m/s 表示的速度指令。参考动作从原分布独立采样用于重置，参考类别及相位均不进入 Actor。 [数据与基线](R3_walk_jog_baseline.md)

指令为世界 +X 方向的 1.0 或 1.9 m/s。重置和每隔 3–5 秒均以相等概率重新抽取指令。沿用 MimicKit 的速度和朝向奖励，分项权重为 0.7 / 0.3。速度项指数系数用上游另一 steering 示例的 2.0。实速度差 0.9 m/s 时，这个速度项约为匹配值的 20%。保留 G1 AMP 网络、风格奖励函数和训练超参数，任务与风格权重各为 0.5。每步奖励使用生成该次动作时的速度指令。 [MimicKit 速度任务](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/mimickit/envs/task_steering_env.py) · [速度项系数来源](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/data/envs/amp_steering_humanoid_sword_shield_env.yaml)

骨盆低于 0.66 m、骨盆倾角超过 50° 或非脚底连杆实际触地力超过 5 N 时，任务奖励归零并以失败终止。使用同一 G1 MJCF 核对，HOLD 四段的最低骨盆高度为 0.751 m，骨盆最大倾角为 39.7°。脚底限于 G1 足底碰撞球所在的左右踝滚转连杆。

R3-2 使用普通判别器。R3-3 使用同一套 AMP 更新，只给判别器增加一个速度条件。专家条件按 MotionLib 实际取到的文件对应其技能。策略和 replay 条件是产生转移的指令，历史运动窗口在切向时可包含上一指令下的运动。判别器梯度惩罚作用于运动特征。采用同预算比较，不变更任务、初始化分布及采样机制。

128 环境、32 步 rollout、seed42。每组 smoke 为 8 次更新、32,768 步。正式预算为每组 1,024 次更新、4,194,304 步，每 250 次更新保存，同时保留末档。观察到晚期持续慢跑开始形成、结果仍在改善时，才考虑两组同预算追加。 [固定配置](../configs/r3/speed_amp.json)

## 独立评估

- 固定组每种参考技能 6 个初态，末档独立组各 16 个。每个物理初态运行四条指令序列，分别为 1.0、1.9 m/s 持续 10 秒，以及两种在第 5 秒切换的方向。两算法和各 checkpoint 共享这些相位，每段 HOLD 等量抽样。
- 从 R3-1 的 1.5 秒窗口判据出发，保留速度、交替支撑、姿态和非足端接触门限。慢跑要求双脚同时无支撑且两个足底最低点都高于地面 1 cm，窗口内至少两次腾空，腾空帧占比至少 7%。步行窗口最多允许一个这样的离地帧。
- 恒定指令成功要求跑满 10 秒，且末尾两个窗口均符合目标步态及 0.35 m/s 的世界前向速度 MAE 门限。切换成功要求跑满，切换前后各自的最后两个窗口均达标。另报切换后目标步态首次达标的时间，从切换起每 0.5 秒移动一次 1.5 秒判定窗口。
- 跌倒后按零速度补至固定时长，记录全时域和存活期速度误差、跌倒原因、净位移。足端接触、实际腾空、关节角和角速度用于独立风格评价。报告与对应 HOLD 技能的分布距离时披露参与对比的回合数。

## R3-2 Smoke

最初使用 50° 躯干倾角做直立门限。该版 smoke 正常退出，Actor、Critic 和判别器参数均更新。Actor 输入为 237 个原始观测加 1 个速度值，普通判别器仍接收 2,390 个运动特征。强制切换探针核对了当步奖励使用旧指令而下一次动作使用新指令。

初版 smoke 权重的 48 个指令与初态组合全部提前跌倒，确认独立评估能够识别失败、截去终止后轨迹并按固定时域计入误差。同一物理初态的四次回合使用相同观测散列。足底几何在 48 条 smoke 轨迹上的计算与 MJCF 正向运动学最大差值约 1.7 μm。合成窗口验证中，仅有零接触力且无足底离地的快速动作未被归为慢跑。

进一步核对专家姿态发现，`u4_jog_002` 的躯干倾角最高达 68.9°，其中 32% 的帧超过原 50° 门限。主动停止初版预跑，最后完整记录 388 次更新、1,589,248 步，耗时 582.67 秒，退出码 143。用骨盆倾角取代躯干倾角后重新 smoke。正式 A/B 都从随机参数开始，使用修订后的同一任务。旧权重及数据单独保留于本地 `../r3/runs/r3_2_speed_seed42_4m`。

修订版普通判别器 smoke 完成 32,768 步并正常退出，耗时 25.46 秒，整卡显存峰值 4,516 MiB。在相同 128 环境首步探针中，触发终止的环境从 11 减至 0，切换时序和三个网络更新检查均通过。

## R3-2 普通判别器

取得可控步行和对速度指令的部分响应，阶段记为 `PARTIAL`。固定组在 2.052M 档仍全部跌倒，3.076M 档开始在步行指令下有 9/12 回合达标，4.100M 和 4.194M 档分别为 10/12 和 9/12。三个后期检查点的 1.9 m/s 指令下均无慢跑达标回合。 [固定初态与训练记录](r3_speed_amp/ordinary_summary.json)

4,194,304 步末档在 32 个新初态上逐一运行四条指令序列，共 128 回合。误差为世界前向速度的绝对误差均值，跌倒后以 0 m/s 补齐到 10 秒。

| 物理初态 | 恒定指令 | 完整 10 s | 目标步态成功 | 跌倒 | 速度 MAE (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 步行 | 1.0 m/s | 16/16 | 16/16 | 0/16 | 0.171 |
| 步行 | 1.9 m/s | 16/16 | 0/16 | 0/16 | 0.543 |
| 慢跑 | 1.0 m/s | 9/16 | 9/16 | 7/16 | 0.570 |
| 慢跑 | 1.9 m/s | 9/16 | 0/16 | 7/16 | 1.083 |

完整回合的末段局部前速由 1.0 指令下的平均 1.07 m/s 提高到 1.9 指令下的 1.36 m/s。1.9 指令下的 50 个末段窗口没有达到 1 cm 的双脚腾空判据，50 个窗口中有 49 个未分类、1 个为步行。完整回合末段的速度分布与对应 HOLD 的 W1 差距为步行 0.126 m/s、慢跑目标 0.569 m/s，关节角速度平均 W1 分别为 0.730 和 0.822 rad/s。两项风格估计均来自每档完整的 25 个回合。

两种切换均有 25/32 回合活到切换点并完成 10 秒。升至 1.9 m/s 后慢跑达标为 0/25。降至 1.0 m/s 后有 25/25 在第一个 1.5 秒窗口达到步行判据，切换前均无持续慢跑，因此两个方向的完整技能切换均为 0/32。 [末档逐回合结果](r3_speed_amp/ordinary_held_out.json)

正式训练耗时 1,558.95 秒，约 2,690 环境步/s，整卡峰值 7,035 MiB。峰值包含并行的另一 Isaac Sim 进程，当时观察到对方占约 2.4 GiB。训练正常退出，日志无 OOM 或非有限指标。末档本地冻结在 `artifacts/r3_speed_amp/frozen_seed42_4194304/ordinary/`，SHA-256 为 `7e5e4deedf99116b752604c27522dbfde4c7cdee26bfb052ab347b4520a03314`。

### R3-2 录像

四条指令序列从能完成该回合的样本中选取接近该组中位速度误差的案例，加一例慢跑指令下的典型跌倒。配对实验沿用相同初态和命令。画面显示实际速度、指令和左右脚接触，`air` 代表双脚无支撑并且足底离地。 [带时间标注的代表帧](r3_speed_amp/ordinary_frames.png) · [视频校验值](r3_speed_amp/media/video.sha256)

| 行为 | 评估 seed | 视频 |
| --- | ---: | --- |
| 慢跑初态在 1.0 m/s 指令下稳定步行 | 47101 | [下载 MP4](https://raw.githubusercontent.com/dreamerToTOE/Learning-for-AMP/main/reports/r3_speed_amp/media/ordinary_47101_walk.mp4) |
| 步行初态收到 1.9 m/s 后加快步行 | 47012 | [下载 MP4](https://raw.githubusercontent.com/dreamerToTOE/Learning-for-AMP/main/reports/r3_speed_amp/media/ordinary_47012_jog.mp4) |
| 5 秒时从 1.0 切到 1.9 m/s | 47000 | [下载 MP4](https://raw.githubusercontent.com/dreamerToTOE/Learning-for-AMP/main/reports/r3_speed_amp/media/ordinary_47000_walk_to_jog.mp4) |
| 5 秒时从 1.9 切到 1.0 m/s | 47110 | [下载 MP4](https://raw.githubusercontent.com/dreamerToTOE/Learning-for-AMP/main/reports/r3_speed_amp/media/ordinary_47110_jog_to_walk.mp4) |
| 慢跑初态在 1.9 m/s 指令下提前跌倒 | 47106 | [下载 MP4](https://raw.githubusercontent.com/dreamerToTOE/Learning-for-AMP/main/reports/r3_speed_amp/media/ordinary_47106_jog.mp4) |

## R3-3 条件判别器

Smoke 正常完成 32,768 步，耗时 24.85 秒，GPU 整卡峰值 4,415 MiB。Actor 观测仍为 238 维，判别器从 2,390 增为 2,391 维。两组的 Actor/Critic 初始权重散列相同。逐条验证了 rollout 指令与生成动作的指令一致，四段专家标签与源文件一致，replay 样本同时含两个速度标签。末次 rollout 的两档标签数为 2,013 和 2,083，随机抽取 2,048 个 replay 样本的计数为 1,068 和 980。 [Smoke 检查](r3_speed_amp/conditional_smoke.json)

## 复现

先按 R3-1 报告准备 MimicKit 环境与 HOLD 数据。从仓库根目录运行以下命令，使用各自的新目录。 [环境与数据准备](R3_walk_jog_baseline.md#复现入口)

```bash
ordinary=../r3/runs/r3_2_speed_seed42_pelvis_4m
conditional=../r3/runs/r3_3_speed_seed42_conditional_4m
bash scripts/r3_train_speed_amp.sh ordinary smoke ../r3/runs/r3_2_speed_seed42_pelvis_smoke
bash scripts/r3_train_speed_amp.sh ordinary formal "$ordinary"
for index in 0000000000 0000000250 0000000500 0000000750 0000001000 0000001023; do
  bash scripts/r3_eval_speed_amp.sh ordinary fixed "$ordinary" "$index"
done
bash scripts/r3_eval_speed_amp.sh ordinary held_out "$ordinary" 0000001023
bash scripts/r3_train_speed_amp.sh conditional smoke ../r3/runs/r3_3_speed_seed42_conditional_smoke
bash scripts/r3_train_speed_amp.sh conditional formal "$conditional"
for index in 0000000000 0000000250 0000000500 0000000750 0000001000 0000001023; do
  bash scripts/r3_eval_speed_amp.sh conditional fixed "$conditional" "$index"
done
bash scripts/r3_eval_speed_amp.sh conditional held_out "$conditional" 0000001023
```
