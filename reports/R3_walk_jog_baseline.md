# R3-0 / R3-1 · G1 步行与慢跑 AMP

更新于 2026-11-25。冻结的 R2 模型是单技能对照，R3-1 从随机权重开始。

## 复用范围

| 参考 | 本阶段用途 |
| --- | --- |
| [MimicKit `a2c4152`](https://github.com/xbpeng/MimicKit/tree/a2c41528c2a34b638611f6c618e4a59f0e151179) | 直接使用上游 G1 AMP 环境、Actor、Critic、单判别器、replay、MotionLib 和训练入口。数据转换及独立评价由本项目实现。 |
| [原始 AMP](https://xbpeng.github.io/projects/AMP/index.html) · [Multiple AMP](https://arxiv.org/abs/2203.14912) | 以无配对的多轨迹专家分布学习风格，后续参考已有技能条件和多风格方案。R3-1 使用 MimicKit 原实现作数据混合实验。 |
| [AMP Running Baseline](https://github.com/Jiarui-Xie/AMP_Running_baseline/tree/4a4dff791ec9ba376a39894d60832754124247de) · [legged_lab](https://github.com/zitongbai/legged_lab/tree/69b7f73d912a9d5695a60b5ed41d0785c8b302e) · [RSL-RL AMP fork](https://github.com/zitongbai/rsl_rl/tree/6ddda9bace990a7e2cb9c317b94f783259f6436b) | 供后续速度奖励、对称增强及判别器实现审查。沿用本地 MimicKit / Newton 训练环境。 |
| [State-Dependent AMP §III](https://arxiv.org/html/2605.18611v1#S3) | 供后续简化的速度条件判别器与恢复门控参考。原文同时报告 `abs(g_z + 1) > 0.6` 和约 37°。若 `g_z = -cos(倾角)` 为单位重力投影，公式对应约 66.4°。实现前还需核对坐标定义和作者代码。 |

本阶段保留 R2 的 G1 29DoF MJCF 和 240 Hz 仿真 / 30 Hz 控制。训练使用已锁定的相同 Python 3.12、PyTorch 2.8.0+cu128、Newton 1.6.1、Warp 1.17.0、MuJoCo-Warp 3.12.0。上轮已实测的接触及求解缓冲区容量为 512 / 512。 [R2 环境与容量证据](R2_g1_single_skill.md)

## R3-0 专家数据

采用 RAI Institute 与 Boston Dynamics 发布的 **HOLD**，固定 `f80b04d384d4ec8922c88deada878c88597a3cf9`。发布者为该数据提供 CC BY 4.0，`data/g1` 已重定向。候选库同时有走、慢跑、冲刺、碎步和侧移。首轮训练使用两种前行技能的原始样本。再次访问 MimicKit 作者资产包仍返回 HTTP 401，原 R2 的 `g1_walk.pkl` 来源和许可继续待核实。 [发布说明](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9) · [数据许可](https://github.com/rai-opensource/hold/blob/f80b04d384d4ec8922c88deada878c88597a3cf9/LICENSE) · [重定向说明](https://github.com/rai-opensource/hold/blob/f80b04d384d4ec8922c88deada878c88597a3cf9/docs/g1.md)

| 技能 | HOLD 源文件及裁剪帧号 `[start, stop)` | 时长 | 局部前速均值 | 横速均值 | yaw 率均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 行走 | `S0_normal_walk_0.h5` · 1432–1849 | 3.10 s | 0.934 m/s | 0.004 m/s | −0.002 rad/s |
| 行走 | `S1_normal_walk_0.h5` · 3136–3553 | 3.10 s | 1.075 m/s | 0.077 m/s | 0.001 rad/s |
| 慢跑 | `s3_jog_001.h5` · 11584–11907 | 2.40 s | 2.003 m/s | 0.008 m/s | 0.018 rad/s |
| 慢跑 | `u4_jog_002.h5` · 100–369 | 2.00 s | 1.901 m/s | −0.008 m/s | 0.011 rad/s |

- 选择前速分布可区分、整体轨迹近直且绕竖直轴净转动小的段落。片段保留真实的摆臂、腿部摆动、向前倾和短时侧速。使用原版轨迹，暂留镜像和转向素材用于后续实验。
- 根据 G1 重定向 H5 的时间戳，四段真实采样频率均为约 134.174 Hz。HOLD 对 120 Hz 的说明对应原始动捕。H5 的根位置是米、四元数为 `wxyz`、关节角为弧度。输入 29 个关节标签与现用 MJCF 顺序逐一一致，所有裁剪关节角都在模型限位内。
- 转换只截取帧、将起点移到原点的 XY 位置，并以同一个 yaw 刚体旋转对齐前向位移及根朝向；保留 Z、动作时间和关节角。以 `CLAMP` 保存，避免把不同相位的首尾强行接成循环。MimicKit 按时间戳重建 m/s 与 rad/s。审计保留前向、横向和 yaw 率分布，以及所有关节范围和峰值速度。 [源文件与裁剪配置](../configs/r3/hold_walk_jog.json) · [动作审计](r3_walk_jog/hold_audit.json)
- 用现有 G1 网格做逐帧运动学预览。四段都能看到对应步态，慢跑有更明显的腾空和躯干前倾。足底高度代理只反映当前 MJCF 运动学，裁剪段保留了源轨迹少量离地与穿透。 [运动学预览](r3_walk_jog/hold_preview.png)

预览来源为 HOLD（RAI Institute 与 Boston Dynamics，CC BY 4.0），画面为本项目进行裁剪、坐标对齐和 G1 网格渲染后的运动学结果。 [HOLD 许可与署名](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9#research-using-hold)

### 冻结 R2 的漂移初查

原权重用 `43000–43004` 复跑，五个初态哈希与 R2 一致，均完成 10 秒。把每控制步的世界横向位移投影到机器人当前航向，平均世界横移 −2.52 m 中，约 −1.35 m 来自前行方向偏离世界 +X，约 −1.17 m 来自机器人局部横移。局部横速均值为 −0.118 m/s，航向相对起点平均变化 −2.47°。

基于 5 N 接触阈值和足端连杆原点，左、右脚落地步长均值为 0.390 / 0.235 m，支撑时长均值为 0.401 / 0.424 s。左、右膝关节的 5–95 百分位活动范围为 1.32 / 1.21 rad。漂移与航向、局部横移和可见的左右步态差异同时出现。镜像能否缩小差异留给后续轻量对照。 [逐步航向与横速](r3_walk_jog/r2_frozen_drift.png) · [逐回合与双腿数据](r3_walk_jog/r2_frozen_drift.json)

## R3-1 原始双技能基线

上游 `MotionLib` 使用四个权重为 1 的文件，每个采样概率为 25%。seed42 的 4,096 次本地采样计数依次为 1,046、1,005、1,053、992，返回的各维度和速度为有限值。专家转移和参考初态都沿用上游采样入口。组间条件、状态门控、样本分组和速度奖励均未加入本轮训练。

保持任务奖励权重 0、风格权重 1。128 环境、32 步 rollout、seed42 的 smoke 正常完成 8 次更新、32,768 步，耗时 25.04 s，整卡峰值 4,758 MiB。已比较首末权重，Actor、Critic 和判别器都发生更新。输出周期设为 250 次更新，以保存约 1.03M、2.05M、4.10M 及 4.19M 权重。MimicKit 在每个输出档进行内部评估并重置环境。 [Smoke 摘要](r3_walk_jog/smoke.json)

### 独立评估协议

每档固定 5 个步行参考初态、5 个慢跑参考初态，两组按相同规则抽取连续相位。初态标签用于分组。Actor 输入不含此标签或外部命令。剔除开头 1 s，按 1.5 s 窗口依据局部前向速度、实际足端接触、交替支撑、骨盆高度和倾角分类。快走与慢跑之间保留速度空档，慢跑要求观测到腾空。完整回合长 10 s，持续技能要求至少两个连续窗口通过同一技能判据。记录提前跌倒前的已识别窗口，单列回合级持续结果。 [冻结阈值与 seed](../configs/r3/walk_jog_eval.json) · [独立评估入口](../scripts/r3_eval_amp_modes.py)

### 4.194M 从随机初始化训练

128 环境完成 1,024 次更新、4,194,304 交互步。终档 checkpoint 能正常读取和运行。训练记录中未见 NaN、OOM 或容量溢出。首条显存快照到终档落盘约 24 分 50 秒。启动训练的外层计时和 GPU 采样在任务切换时结束，保留了前 442.21 s 的 885 个样本，期间整卡已用峰值 7,770 MiB，最低空闲 48 MiB，包含同时运行的另一 Isaac Sim 进程。正式训练的全部更新日志仍在，进程退出码和后半段连续显存采样缺失。一次与其他进程并发的评估启动因显存不足退出，后续固定评估均成功。 [完整的阶段汇总和逐回合记录](r3_walk_jog/formal_4m.json)

| 交互步 | 步行初态平均存活 | 慢跑初态平均存活 | 完整回合 | 已识别步行 / 慢跑窗口 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.52 s | 0.45 s | 0/10 | 0 / 0 |
| 1,028,096 | 0.59 s | 0.51 s | 0/10 | 0 / 0 |
| 2,052,096 | 1.39 s | 0.54 s | 0/10 | 0 / 0 |
| 3,076,096 | 1.49 s | 0.55 s | 0/10 | 0 / 0 |
| 4,100,096 | 3.81 s | 1.01 s | 1/10 | 4 / 0 |
| 4,194,304 | 2.93 s | 1.00 s | 0/10 | 3 / 0 |

4.10M 档出现一个完整且满足持续步行判据的回合。4.19M 档只有三个步行窗口，最好的步行初态存活 5.37 秒。慢跑初态通常在不足 1 秒时跌倒，早于稳定步态判断所需的 2.5 秒。现有行为显示明显的模式覆盖不足，暂不足以用模式坍塌解释训练失败。

冻结的 R2 单技能模型也在这十个 HOLD 初态上运行，初始观测哈希逐一相同。2/10 跑满十秒，两例都被识别为步行，其中一例从慢跑初态转入。R2 在自己的参考初态已通过 100/100 步行验收。这个对照表明两套参考初态的难度不同，解释混合训练结果时需保留这一差别。 [R2 同初态回放](r3_walk_jog/r2_frozen_hold_resets.json)

后段步行存活和步行窗口均增加，因此追加一轮相同数据、算法和环境的 4,194,304 步训练。初始权重固定为首轮 4.194M 终档。上游加载恢复 Actor、Critic、判别器和模型归一化器，优化器、replay buffer、随机数状态和环境重置为新进程状态。追加档的累计交互量以前后两段相加，实验保留两段各自的记录。

## 复现入口

```bash
uv venv --python 3.12.14 ../r3/data-venv
uv pip install --python ../r3/data-venv/bin/python -r reports/r3_walk_jog/data_requirements.lock.txt
../r3/data-venv/bin/python scripts/r3_prepare_hold.py --manifest configs/r3/hold_walk_jog.json \
  --upstream ../r2/upstream/MimicKit --cache ../r3/assets/hold --output ../r3/assets/walk_jog
MUJOCO_GL=egl ../r3/data-venv/bin/python scripts/r3_preview_hold.py \
  --xml ../r2/upstream/MimicKit/data/assets/g1/g1.xml \
  --output ../r3/assets/walk_jog/previews ../r3/assets/walk_jog/*.pkl
mkdir -p ../r3/runs
bash scripts/r3_train_amp_walk_jog.sh ../r3/assets/walk_jog ../r3/runs/r3_1_walk_jog_smoke 32768
bash scripts/r3_train_amp_walk_jog.sh ../r3/assets/walk_jog ../r3/runs/r3_1_walk_jog_seed42_4m 4194304
bash scripts/r3_train_amp_walk_jog.sh ../r3/assets/walk_jog ../r3/runs/r3_1_walk_jog_seed42_warm_4m \
  4194304 ../r3/runs/r3_1_walk_jog_seed42_4m/int_models/model_0000001023.pt
```
