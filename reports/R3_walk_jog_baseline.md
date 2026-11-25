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

## R3-1 原始双技能基线

上游 `MotionLib` 使用四个权重为 1 的文件，每个采样概率为 25%。seed42 的 4,096 次本地采样计数依次为 1,046、1,005、1,053、992，返回的各维度和速度为有限值。专家转移和参考初态都沿用上游采样入口。组间条件、状态门控、样本分组和速度奖励均未加入本轮训练。

保持任务奖励权重 0、风格权重 1。128 环境、32 步 rollout、seed42 的 smoke 正常完成 8 次更新、32,768 步，耗时 25.04 s，整卡峰值 4,758 MiB。已比较首末权重，Actor、Critic 和判别器都发生更新。输出周期设为 250 次更新，以保存约 1.03M、2.05M、4.10M 及 4.19M 权重。MimicKit 在每个输出档进行内部评估并重置环境。 [Smoke 摘要](r3_walk_jog/smoke.json)

### 独立评估协议

每档固定 5 个步行参考初态、5 个慢跑参考初态，两组按相同规则抽取连续相位。初态标签用于分组。Actor 输入不含此标签或外部命令。剔除开头 1 s，按 1.5 s 窗口依据局部前向速度、实际足端接触、交替支撑、骨盆高度和倾角分类。快走与慢跑之间保留速度空档，慢跑要求观测到腾空。完整回合长 10 s，持续技能要求至少两个连续窗口通过同一技能判据。记录提前跌倒前的已识别窗口，单列回合级持续结果。 [冻结阈值与 seed](../configs/r3/walk_jog_eval.json) · [独立评估入口](../scripts/r3_eval_amp_modes.py)

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
```
