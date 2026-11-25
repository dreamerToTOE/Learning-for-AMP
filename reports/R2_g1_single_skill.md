# R2 · 宇树 G1 单技能 AMP

日期 2026-11-25 · R2 / IN_PROGRESS

## 路线选择

选用 AMP 原作者维护的 **MimicKit**，固定在 `a2c41528c2a34b638611f6c618e4a59f0e151179`。该版本提供 G1 的 AMP 环境和独立的 G1 Agent 配置，默认只读取 `g1_walk.pkl`。Actor 无外部速度指令和参考相位输入，任务奖励权重为 0，判别器风格奖励权重为 1。目标是复现平地向前行走。代码使用 Apache-2.0。 [G1 环境](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/data/envs/amp_g1_env.yaml) · [AMP 参数](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/data/agents/amp_g1_agent.yaml) · [代码许可证](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/LICENSE)

本机 RTX 3070 有 8192 MiB 显存，驱动为 580.178.04。MimicKit 当前版本测试过 Newton 1.6.1，可在独立 venv 中复用已定义的 G1 模型、控制器和 AMP 训练入口。原论文实验使用 Isaac Gym，展示了 G1 仿真技能。其表 1 报告的是 Humanoid 角色的指标。公开镜像中可见 G1 Walk 的 DeepMimic 权重和训练日志。 [MimicKit 安装说明](https://github.com/xbpeng/MimicKit/tree/a2c41528c2a34b638611f6c618e4a59f0e151179#installation) · [原论文实验](https://arxiv.org/html/2510.13794v2#S7) · [镜像权重清单](https://huggingface.co/datasets/blanchon/MimicKit_Data/tree/3ab706e7fd951478745ea78e71435d5721de12c3/models)

### 对照候选

| 工程及已核版本 | 机器人与数据 | 权重、运行证据和本机取舍 |
| --- | --- | --- |
| [legged_lab `69b7f73`](https://github.com/zitongbai/legged_lab/tree/69b7f73d912a9d5695a60b5ed41d0785c8b302e) | G1 29 DoF USD 与步行、跑步、转向、侧移 LFS 片段。仓库 `LICENCE` 为 Apache-2.0，动作原始来源与单独许可待核 | 仓库有上游演示和 AMP 入口，当前树未见 G1 AMP checkpoint。Isaac Sim 5.1 / Isaac Lab 2.3.2 / Python 3.11，加 [RSL-RL AMP fork `6ddda9b`](https://github.com/zitongbai/rsl_rl/tree/6ddda9bace990a7e2cb9c317b94f783259f6436b) |
| [AMP_Running_baseline `4a4dff7`](https://github.com/Jiarui-Xie/AMP_Running_baseline/tree/4a4dff791ec9ba376a39894d60832754124247de) | 同类 G1 LFS USD，BASE 配置含两个 LaFAN1 主片段和九个辅助片段。代码 Apache-2.0。辅助动作来源和权利未完整注明 | 上游展示 BASE checkpoint 的真机跑步和另一个权重的仿真。BASE 的 LFS 权重已在本机通过 SHA-256 校验。Isaac Sim 5.1 / Isaac Lab 2.3.1 / Python 3.11。同一 fork 训练，权重维度由其控制器定义 |
| [AMP_mjlab `6c7a294`](https://github.com/ccrpRepo/AMP_mjlab/tree/6c7a2947fccc973e4af8e6d90e550400f1b6fcfc) | 29 和 23 DoF G1 MJCF，带走跑、转向、恢复片段。内置 RSL-RL 修改版，并提供 mjlab 观测顺序补丁 | `mjlab==1.2.0`、`mujoco-warp==3.8.1`、`warp-lang==1.12.0`。上游提供训练截图，当前树未见权重和根目录许可证。默认策略含走跑和恢复 |
| [basic_loco_lib `e28e6eb`](https://github.com/PoplarCrystal/basic_loco_lib/tree/e28e6eb04a99bef770f274327da14006fe913cc3) | G1 29 DoF Unitree USD、URDF 和 MJCF，训练支持筛选单个动作目录。现成轨迹按 Newton 顺序排列，步态源为 BONES-SEED，恢复源为 LaFAN1。代码 Apache-2.0 | 当前树未见匹配的 AMP 权重。主线为 Isaac Lab 3.0.0 / Isaac Sim 6.2 / Python 3.12；支持导出 ONNX 和 MuJoCo 验证。 [数据来源及许可](https://github.com/PoplarCrystal/basic_loco_lib/blob/e28e6eb04a99bef770f274327da14006fe913cc3/data/amp/README.md) · [G1 资产来源](https://github.com/PoplarCrystal/basic_loco_lib/blob/e28e6eb04a99bef770f274327da14006fe913cc3/source/basic_loco_lib/basic_loco_lib/assets/unitree_g1/readme.md) |

优先验证 MimicKit 的单技能配置。Isaac Sim 5.1 文档列出的最低显存为 16 GB，因此暂缓独立安装 5.1 验证赛事权重。已从 GitHub 的 LFS 对象地址取回 `model_6200.pt`，大小 19,815,093 B，SHA-256 为 `5de7628fe92bd74118016c592b95eb17183547a5289708622c80a7815da025ac`，与仓库指针相同。CPU 读取显示训练迭代 6200，含 Actor、Critic、优化器、判别器及其归一化器。Actor 实际为 495 维输入、29 维输出，说明文档写的是 384 维输入。兼容性仍需在原任务中确认。BASE 播放配置默认将全部速度指令置零。 [显存要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) · [BASE 权重及部署接口](https://github.com/Jiarui-Xie/AMP_Running_baseline/blob/4a4dff791ec9ba376a39894d60832754124247de/g1-base-amp.md) · [BASE 环境](https://github.com/Jiarui-Xie/AMP_Running_baseline/blob/4a4dff791ec9ba376a39894d60832754124247de/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_base_env_cfg.py)

## 资产与动作

作者 README 链接的 SharePoint 资产包在本机匿名 GET 和下载请求均返回 HTTP 401。用于本地验证的副本来自 `blanchon/MimicKit_Data` 的固定修订 `3ab706e7fd951478745ea78e71435d5721de12c3`。已核对 G1 模型目录和步行片段共 66 个文件、46,498,592 B 的对象哈希。模型目录含 Unitree Robotics 的 BSD-3-Clause 许可。镜像未给出步行动作的原始动捕、重定向过程或单独许可，也无法与作者侧资产包逐文件比对。公开仓库记录来源、校验值和实测日志，第三方二进制保存在本地。LaFAN1 官方源标注 CC BY-NC-ND 4.0，许可证独立于候选工程代码。 [镜像 G1 资产](https://huggingface.co/datasets/blanchon/MimicKit_Data/tree/3ab706e7fd951478745ea78e71435d5721de12c3/assets/g1) · [镜像 G1 动作](https://huggingface.co/datasets/blanchon/MimicKit_Data/tree/3ab706e7fd951478745ea78e71435d5721de12c3/motions/g1) · [LaFAN1 原始许可](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)

| 检查项 | 本地结果 |
| --- | --- |
| G1 模型 | `g1_29dof` MJCF。29 个受控铰链关节及 29 个执行器，36 个引用网格全部存在。MuJoCo 3.12 实际编译得到 nq=36、nv=35、nu=29、32 个刚体。XML SHA-256 `adab92061ba6d5a7f6739bc1a86090ed6139955d63ecca3e5cba95c7988ef8e0` |
| 步行文件 | `g1_walk.pkl` 的 SHA-256 为 `0030f5ba1db9497e7c9b511c674aefb820b946416e52ba26faf4813b0e998286`。125 帧、120 Hz，循环周期 1.033 s |
| 字段与关节映射 | 每帧 35 个有限浮点数，依次为根位置（m）、根旋转指数映射（rad）及 29 个关节角（rad）。关节列按 MJCF 深度优先顺序，先左腿、右腿、腰、左臂、右臂。按该顺序对照 MJCF，3,625 个关节角没有超出关节范围 |
| 运动参考 | 根节点平均前向速度 1.002 m/s，横向约 0.002 m/s，骨盆高度 0.754–0.790 m。为动作文件的运动学统计 |
| 控制 | Newton 配置为 240 Hz 物理、30 Hz 控制，每个动作保持 8 个物理步。Actor 的 29 维动作反归一化为位置目标，环境在送给位置控制器前裁剪；PD 参数和力矩限位来自上游 MJCF |
| 预训练文件 | 本地校验 `deepmimic_g1_walk_model.pt` 的 SHA-256 为 `5adec8cdbd90389f4aac981307a676a452e5cb478cab1ddddb27869e4cdaab4d`，文件对应 DeepMimic。镜像未找到与单技能 AMP 配置对应的预训练权重 |

## 单技能验证

独立环境位于本项目 checkout 旁的 `../r2/mimickit-venv`，使用 Python 3.12.14、PyTorch 2.8.0+cu128、Newton 1.6.1、Warp 1.17.0、MuJoCo/MuJoCo-Warp 3.12.0。venv 的 `include-system-site-packages` 为 `false`。依赖用 `uv pip check` 核验，执行前清除 `PYTHONPATH` 与 `PYTHONHOME`。 [依赖锁定](r2_g1_single_skill/requirements.lock.txt) · [资产校验表](r2_g1_single_skill/asset_manifest.json)

使用上游环境和 AMP Agent 配置，seed42、16 个环境、32 步 rollout，4 次更新共 2,048 交互步。首轮按上游接触缓冲区 `nconmax=200` 跑完并保存 checkpoint，出现 2,016 条 broadphase 溢出消息，峰值建议值为 285。该轮只证明训练链路可执行。复测将容量改为 512，保留上游机器人、奖励、执行器和求解器其他设置。 [最小容量补丁](../configs/r2/mimickit_contact_capacity.patch) · [MuJoCo-Warp 缓冲区说明](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#overflow-detection)

扩大到 128 环境后，首次 pilot 在 122,880 个完整训练步后因 `NEFC` 约束缓冲区溢出退出。上游 `njmax=256`，日志中最高需求为 258。将每环境的约束缓冲区调为 512，然后从 seed42 重新开始。两项补丁只提高预分配容量。 [约束容量补丁](../configs/r2/mimickit_constraint_capacity.patch)

| 运行 | 环境数 | 完成交互步 | 进程耗时 | 整卡显存峰值 / 最低空闲 | 退出结果 |
| --- | ---: | ---: | ---: | --- | --- |
| 原始 smoke | 16 | 2,048 | 429.40 s | 4,959 / 2,859 MiB | 退出 0，碰撞候选溢出 |
| 扩大碰撞候选缓冲区 | 16 | 2,048 | 17.99 s | 4,248 / 3,571 MiB | 退出 0，无溢出 |
| 首次 pilot | 128 | 122,880 | 51.29 s | 4,805 / 3,014 MiB | 退出 1，约束缓冲区溢出 |
| 扩大约束缓冲区后的 pilot | 128 | 262,144 | 99.02 s | 4,818 / 3,000 MiB | 退出 0，无溢出 |
| 约 1M 步趋势验证 | 128 | 1,048,576 | 342.43 s | 4,666 / 3,152 MiB | 退出 0，无溢出 |

首轮包含 Warp 冷编译，后续复用独立缓存。整卡显存采样间隔 0.5 s，含系统及并行进程。128 环境 pilot 完成 64 次更新，训练器报告 3,049 步/s。最后一次更新的 Actor / Critic / 判别器损失为 −0.0336 / 0.7693 / 2.0365，实际风格奖励均值 0.574。环境原始任务奖励为 0。所有已完成更新的记录均为有限值。 [训练及显存汇总](r2_g1_single_skill/amp_training_runs.json) · [pilot 逐更新指标](r2_g1_single_skill/amp_pilot_updates.csv)

约 1M 步从随机初始化连续完成 256 次更新，训练器报告 3,177 步/s。GPU 利用率均值 78.0%，峰值 100%。训练日志无 OOM 或非有限损失。最后一次更新的 Actor / Critic / 判别器损失为 −0.0492 / 2.7904 / 0.8391，实际风格奖励均值 0.681。 [1M 逐更新指标](r2_g1_single_skill/amp_1m_updates.csv)

### 独立仿真评估

固定评估 seed `43000–43004`，从同一参考片段的不同相位初始化，每个回合最多 300 个控制步（10 s）。从 Newton 地面接触传感器读取每个控制步的末端数据，使用 5 N 作为接触阈值。上游终止条件允许膝盖接触，评估中把膝部计入非足端。步行初筛看是否跑满、前进至少 5 m、骨盆高度第 5 百分位至少 0.68 m、左右独立足端支撑切换至少 8 次，且无超过阈值的非足端接触。这些阈值用于筛除原地站立和低姿支撑。接触采样可能漏掉控制周期内的瞬时接触。 [评估入口](../scripts/r2_eval_mimickit_walk.py)

| 权重 | 完整行走 | 平均时长 | 平均前进 | 行为 |
| --- | ---: | ---: | ---: | --- |
| G1 AMP smoke · 2,048 步 | 0/5 | 0.86 s | 0.87 m | 五回合均触发跌倒终止 |
| G1 AMP pilot · 4,096 步 | 0/5 | 0.83 s | 0.85 m | 五回合均触发跌倒终止 |
| G1 AMP pilot · 262,144 步 | 0/5 | 1.17 s | 0.95 m | 五回合均触发跌倒终止 |
| 上游 DeepMimic G1 Walk 权重 | 5/5 | 10.00 s | 10.13 m | 每回合 19 次左右足交替支撑，非足端接触步数为 0 |

DeepMimic Actor 使用未来参考状态。该推理结果确认这套 G1 资产和 Newton 控制链路能执行稳定的已学步态。AMP 的两个 pilot checkpoint 在五个初态的观测哈希逐个相同。CPU 校验显示两档之间 Actor、Critic、Discriminator 的权重均发生更新，所有保存张量均为有限值。 [AMP smoke 逐回合](r2_g1_single_skill/amp_smoke_eval.json) · [pilot 首档](r2_g1_single_skill/amp_pilot_initial_eval.json) · [pilot 末档](r2_g1_single_skill/amp_pilot_final_eval.json) · [DeepMimic 逐回合](r2_g1_single_skill/deepmimic_reference_eval.json)

约 1M 步运行的四个 checkpoint 也使用同一批初态，20 回合的初始观测哈希按 seed 一致。最终档平均前进距离增加，平均存活在约 82 万步时最好。最终档每回合有 1–8 次足端支撑切换，仍都发生跌倒；R2 稳定步态验收尚未通过。 [四档逐回合数据](r2_g1_single_skill/amp_1m_checkpoint_evals.json)

| AMP 训练步数 | 完整行走 | 跌倒 | 平均存活 | 平均前进 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0/5 | 5/5 | 0.83 s | 0.85 m |
| 413,696 | 0/5 | 5/5 | 1.13 s | 0.64 m |
| 823,296 | 0/5 | 5/5 | 1.86 s | 1.03 m |
| 1,048,576 | 0/5 | 5/5 | 1.60 s | 1.30 m |

### 复现命令

在独立 checkout 固定 MimicKit 提交后，将 G1 MJCF、网格和动作放入表中的 `data/` 路径并核对资产清单。从本项目根目录执行以下命令，输出和缓存位于本项目的独立目录。16 环境复测只应用接触容量补丁。

```bash
project="$PWD"
r2=$(realpath ../r2)
upstream="$r2/upstream/MimicKit"
uv venv --python 3.12.14 "$r2/mimickit-venv"
uv pip install --python "$r2/mimickit-venv/bin/python" \
  --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r "$project/reports/r2_g1_single_skill/requirements.lock.txt"
git -C "$upstream" apply "$project/configs/r2/mimickit_contact_capacity.patch"
cd "$upstream"
env -u PYTHONPATH -u PYTHONHOME XDG_CACHE_HOME="$r2/cache" TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" mimickit/run.py --mode train --arg_file args/amp_g1_args.txt \
  --engine_config data/engines/newton_engine.yaml --devices cuda:0 --visualize false --video false \
  --num_envs 16 --rand_seed 42 --max_samples 2048 --out_dir "$r2/runs/amp_walk_smoke_seed42_capacity512"

env -u PYTHONPATH -u PYTHONHOME XDG_CACHE_HOME="$r2/cache" TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" "$project/scripts/r2_eval_mimickit_walk.py" --mimickit "$upstream" \
  --checkpoint "$r2/runs/amp_walk_smoke_seed42_capacity512/model.pt" \
  --seeds 43000 43001 43002 43003 43004 --output "$r2/runs/amp_walk_smoke_seed42_capacity512/eval.json"
```

128 环境训练在上述独立 checkout 上再应用 `git -C "$upstream" apply "$project/configs/r2/mimickit_constraint_capacity.patch"`。把命令中的环境数改为 128，`--max_samples` 设为 262144 或 1048576，并加入 `--save_int_models true`。输出写进新的目录。MimicKit 保存的是模型和归一化器状态，训练优化器、replay buffer 和环境状态需要另外记录才可严格续训。

## 后续基线

采用当前 128 环境、seed42、单步行片段和上游纯风格配置，优先验证约 2M 和 4M 的连续训练档位。按本轮吞吐，4.19M 步约需 22 分钟训练时间。沿用 5 个初态作趋势比较，出现可重复的完整步行后扩大独立评估并保存运动视频。原始动作来源和许可仍待核对。
