# R0 环境与上游审计

审计日期：2026-11-22（Asia/Seoul）  
审计范围：仓库接手、Isaac/AMP 工具链核验、G1 上游与动作数据可用性核验。  
审计基线：`803f84f55af0044a8649d0fad5c0c4205ca406ca`。  
本轮没有启动长时间训练、没有修改既有 Isaac Lab 工作树、没有操作真实机器人。

## 结论

- **官方 AMP 基线可运行。** 已使用 Isaac Sim 4.5.0 的 Kit Python 和 Isaac Lab v2.1.1 源码完成 `Isaac-Humanoid-AMP-Walk-Direct-v0` 的 4 环境、1 iteration headless smoke。policy、value、discriminator 均初始化成功，Walk 专家数据成功加载，16/16 rollout 完成，进程退出码为 0。
- **本机资源与项目记忆不一致。** 当前可见 GPU 是 RTX 3070 8 GiB，不是预期的 RTX 4070 12 GiB。R1 必须按 8 GiB 预算规划，并在关闭其他 Isaac/桌面进程后重新测峰值显存。
- **现有 Isaac Lab 工作树不能直接作为 R1 基线。** `/home/ubuntu2004/GRScenes/a/IsaacLab` 是 v1.2.0、工作树有大量未提交修改，且没有官方 Humanoid AMP 目录。对它执行最小空场景命令时出现 `ModuleNotFoundError: No module named 'omni.isaac'`。
- **R1 路线已选定。** 使用独立的 Isaac Lab v2.1.1 checkout、Isaac Sim 4.5.0、Kit Python 3.10.15、skrl 1.4.3。官方 v2.1.1 tag 的完整 SHA 是 `90b79bb2d44feb8d833f260f2bf37da3487180ba`。
- **R2 尚未选定可直接运行的 G1 上游。** `legged_lab` 的 AMP 代码和 G1 配置存在，但动作片段与 USD 资产是 Git LFS 对象，本机没有 `git-lfs`，同时上游版本面向 Isaac Lab 2.3.x / Isaac Sim 5.1。动作数据的原始来源与授权也需要单独核实。R2 保持 `PARTIAL/BLOCKED`，不能把候选仓库的结果当作本项目结果。

## 1. 仓库与审计命令

仓库已克隆到本项目工作目录下的 `work/Learning-for-AMP`。审计开始时执行了：

```bash
git rev-parse HEAD
git remote -v
git status --short --branch
df -h / /home/ubuntu2004
free -h
uname -a
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader
nvidia-smi -q
nvcc --version
python3 --version
python3 -m pip list
```

当前仓库远端是 `https://github.com/dreamerToTOE/Learning-for-AMP.git`。本报告不提交训练 checkpoint、视频或外部动作资产。

## 2. 硬件与系统实测

| 项目 | 实测值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | `6.8.0-149-generic` |
| CPU | Intel Core i7-10700F，8 核 16 线程 |
| RAM | 48,106 MB，总体可用约 28–33 GiB，随进程变化 |
| Swap | 2 GiB |
| GPU | NVIDIA GeForce RTX 3070，compute capability 8.6 |
| GPU 显存 | 8,192 MiB。审计初始空闲约 4.1 GiB，smoke 结束后约 6.8 GiB |
| NVIDIA driver | `580.178.04` |
| nvidia-smi CUDA 字段 | `13.0` |
| CUDA Toolkit | `nvcc` 未安装，只有驱动 runtime 和 Python CUDA wheels |
| 根盘 | 724 GiB，总体已用约 470–471 GiB，可用约 216–218 GiB |
| Docker | `/usr/bin/docker` 可用，但没有 Isaac Lab 专用镜像 |
| Git LFS | 未安装 |

审计期间系统中曾有其他 Isaac Sim 进程占用 GPU。smoke 结束后没有残留训练进程。不能把当前空闲显存当作长期保证，R1 运行前要记录一次新的 `nvidia-smi` 快照。

## 3. Python、Isaac Sim 与 Isaac Lab 版本矩阵

### 3.1 已存在环境

| 环境 | Python | 关键包与结果 | 结论 |
|---|---|---|---|
| `/usr/bin/python3` | 3.10.12 | 未找到 torch、Isaac Lab、Isaac Sim、skrl | 不能运行仿真 |
| Conda `isaac_env` | 3.10.20 | Isaac Sim `4.5.0.0`，torch `2.11.0+cu130`，CUDA 可用；没有 `isaaclab`、`omni.isaac.lab`、skrl | Isaac Sim 包存在，Isaac Lab 未接入 |
| Conda `isaaclab` | 3.11.15 | Isaac Lab 包 `0.54.3`，Isaac Lab RL `0.5.1`，skrl `2.1.0`，rsl-rl-lib `5.0.1`，torch `2.7.0+cu128`；不能直接 import `omni` 或 Isaac Sim | 与官方 v2.1.1 AMP 组合未验证，不作为 R1 环境 |
| Conda `unitree_rl_mjlab` | 3.11.15 | MuJoCo `3.5.0`、mujoco-warp `3.5.0`、rsl-rl-lib `5.0.1`、torch `2.13.0+cu130`，没有 Isaac Sim/Lab | 只作为 MuJoCo 相关环境，和 R1 隔离 |
| Standalone Kit Python | 3.10.15 | `/home/ubuntu2004/isaacsim-4.5.0/python.sh`，Kit 内 torch `2.5.1+cu118`，通过外部路径加载 skrl `1.4.3` | 已完成官方 AMP smoke |

`nvidia-smi` 报告的 CUDA 13.0 是驱动接口字段，不等于本机存在 CUDA 13.0 Toolkit。官方 smoke 使用的 torch CUDA wheel 是 Kit 内的 `cu118`，在当前 580 驱动上已完成最小验证。

### 3.2 现有 Isaac Lab 工作树

`/home/ubuntu2004/GRScenes/a/IsaacLab` 的实测信息：

- Git tag/版本为 `v1.2.0`，commit `e00d62561a1b4ab19985bc15b40ac77c10bd5e33`。
- README 标注 Isaac Sim 4.2.0 与 Python 3.10，不是官方 AMP v2.1.1 的版本组合。
- 工作树存在大量未提交修改，不能作为本项目依赖源。
- `source/isaaclab_tasks/.../direct/humanoid_amp` 不存在。AMP 关键词主要来自本机复制的 `robot_lab-main` 扩展。
- 使用 `isaac_env` 执行 `/home/ubuntu2004/GRScenes/a/IsaacLab/isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py --headless`，得到 `ModuleNotFoundError: No module named 'omni.isaac'`。
- 该工作树的 `_isaac_sim` 链接指向不存在的旧路径，不能通过修链接来替换本项目的环境隔离方案。

## 4. 官方 Humanoid AMP 核验

### 4.1 源码与任务入口

官方 Isaac Lab v2.1.1：

- Git tag：`v2.1.1`
- 完整 SHA：`90b79bb2d44feb8d833f260f2bf37da3487180ba`
- 许可证：Isaac Lab 仓库 BSD-3-Clause
- AMP 环境目录：`source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/`
- 注册任务：`Isaac-Humanoid-AMP-Walk-Direct-v0`、`Isaac-Humanoid-AMP-Run-Direct-v0`、`Isaac-Humanoid-AMP-Dance-Direct-v0`
- 环境入口：`humanoid_amp_env.py`
- 配置入口：`humanoid_amp_env_cfg.py`
- skrl 配置：`agents/skrl_walk_amp_cfg.yaml`、`skrl_run_amp_cfg.yaml`、`skrl_dance_amp_cfg.yaml`
- 训练入口：`scripts/reinforcement_learning/skrl/train.py`
- 训练脚本可选 `--algorithm AMP` 或 `--algorithm PPO`，AMP 使用 skrl 的 `AMP` agent
- 训练脚本只检查 skrl 最低版本 `>=1.4.2`。配置字段与 skrl 2.x 的新 API 不同，已实测可运行版本是 `skrl==1.4.3`，R1 应锁定精确版本

Smoke 使用的 `/home/ubuntu2004/WorkBuddy/2026-10-22-16-52-24/IsaacLab-v2.1.1` 是无 Git metadata 的 v2.1.1 源码副本。上面的完整 tag SHA 来自独立的官方 v2.1.1 sparse checkout，后续 R1 需要改用带 Git metadata 的独立 checkout 并在日志中保存实际 SHA。

### 4.2 环境与 AMP 数据接口

v2.1.1 配置实测为：

- 28 个动作自由度，policy observation 81 维
- `num_amp_observations=2`，AMP 判别器输入为两个连续 AMP observation，展平后 162 维
- 仿真 `dt=1/60`，decimation=2，环境步长约 1/30 秒
- 默认并行环境数 4096，episode length 10 秒，跌倒高度阈值 0.5 米
- Actor 使用 policy observation，Critic 使用 value model，Discriminator 使用独立 deterministic model
- 官方 Walk YAML 的 policy/value/discriminator MLP 都是 `[1024, 512]`
- Walk YAML 默认 rollout 数 16、trainer timesteps 80000、AMP motion dataset memory 200000、replay buffer 1000000
- Walk YAML 默认 `task_reward_weight=0.0`、`style_reward_weight=1.0`、`discriminator_reward_scale=2.0`
- 环境动作由关节限位中心和范围缩放到 position target，部署时保留 Actor，训练期的判别器不等同于部署模块

官方动作文件随源码提供，当前 checkout 中可直接读取：

| 文件 | 大小 | 帧数 | DOF/Body | fps |
|---|---:|---:|---:|---:|
| `humanoid_walk.npz` | 159,670 B | 154 | 28 / 15 | 60 |
| `humanoid_run.npz` | 87,382 B | 82 | 28 / 15 | 60 |
| `humanoid_dance.npz` | 910,662 B | 902 | 28 / 15 | 60 |

README 定义了 `fps`、`dof_names`、`body_names`、DOF 位置/速度、body 位置、`wxyz` 四元数、线速度和角速度字段。官方动作文件适合 R1 的可复现基线。该 checkout 的 motions README 和相关提交没有记录原始动作数据来源与单独许可，R1 仅作为随仓库提供的官方基线输入，若对外发布需再核对上游资产条款，不能自动推导出 G1 动作数据授权。

本次静态 checkout 的动作文件 SHA256 如下：Walk `91a7af4e7fa59d8a3a2bb2c6e0dad45d73f6e5b5ee1a284d733435e32c418542`，Run `98dd7cf75b130a593d5f0254d5ece6a2123c2fefad0ee02dcb13ec6d57db99c1`，Dance `2f2b07fd6f2c12f89b60690f9937cf82a9d613e91eaec7e6ca29e126925f9d12`。

### 4.3 已执行 smoke

完整终端输出保存在项目外部的 `../r0_smoke_output.txt`，没有把 126 MB 左右的日志和 checkpoint 纳入 Git。实际命令为：

```bash
export TERM=xterm-256color
export PYTHONPATH=/home/ubuntu2004/WorkBuddy/2026-10-22-16-52-24/IsaacLab-v2.1.1/source/isaaclab:/home/ubuntu2004/WorkBuddy/2026-10-22-16-52-24/IsaacLab-v2.1.1/source/isaaclab_rl:/home/ubuntu2004/WorkBuddy/2026-10-22-16-52-24/IsaacLab-v2.1.1/source/isaaclab_tasks:/home/ubuntu2004/WorkBuddy/2026-10-22-16-52-24/IsaacLab-v2.1.1/source/isaaclab_assets:/home/ubuntu2004/anaconda3/envs/isaaclab/lib/python3.10/site-packages:$PYTHONPATH
/home/ubuntu2004/isaacsim-4.5.0/python.sh /home/ubuntu2004/WorkBuddy/2026-10-22-16-52-24/IsaacLab-v2.1.1/scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Humanoid-AMP-Walk-Direct-v0 \
  --algorithm AMP \
  --num_envs 4 \
  --max_iterations 1 \
  --headless \
  --device cuda:0
```

关键输出：

- `Using device: cuda:0`
- `Motion loaded ... humanoid_walk.npz ... duration: 2.55 sec, frames: 154`
- policy、value、discriminator 三个模型完成实例化
- 进度条完成 `16/16`
- `EXIT_CODE=0`
- 运行约 50 秒，日志目录为 `/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/logs/skrl/humanoid_amp_walk/2026-11-22_23-43-34_amp_torch/`
- 完整 smoke 输出文件 SHA256：`1a84a21634bc672256026a30b361497109463461e7ad6f27e01e483efb6e6cee`

运行中出现 cuBLAS 当前 CUDA context、Gymnasium wrapper deprecation、OmniHub 不可访问和 `omni.isaac.dynamic_control` deprecated 警告。本次均未阻止进程完成。这个 smoke 只验证启动、数据加载、rollout 和一次更新，不代表训练收敛、动作质量、独立评估或 AMP 优于 PPO。

## 5. G1 上游与资产审计

以下 SHA 使用 2026-11-22 执行的 `git ls-remote` 或上游 checkout 记录。仓库许可证只覆盖仓库代码，不能替代人体动作、机器人 USD 或 checkpoint 的单独授权。

`legged_lab` 和 `AMP_Running_baseline` 的 README license badge 与仓库实际 `LICENCE` 文件存在不一致。报告暂按实际许可证文件记录，后续复用前仍需逐项核对许可证文本和第三方资产条款。

| 项目 | 版本与许可证 | 已核对内容 | R0 判断 |
|---|---|---|---|
| [`legged_lab`](https://github.com/zitongbai/legged_lab) | main `69b7f73d912a9d5695a60b5bed41d0785c8b302e`，Apache-2.0 | 有 `LeggedLab-Isaac-AMP-G1-v0`、G1 29 DOF AMP 配置、任务奖励和 style reward。README 面向 Isaac Sim 5.1.0 / Isaac Lab 2.3.2 / Python 3.11。依赖 `zitongbai/rsl_rl` 的 `feature/amp` 分支 `6ddda9bace990a7e2cb9c317b94f783259f6436b`，BSD-3-Clause。30 个 motion `.pkl` 和 G1 USD 是 Git LFS 对象，本机没有 git-lfs。默认配置混合 walk/run/turn/side-step，不能直接称为单技能 Walk。 | 代码候选保留。资产、动作文件和版本环境待补齐 |
| [`AMP_Running_baseline`](https://github.com/Jiarui-Xie/AMP_Running_baseline) | main `4a4dff791ec9ba376a39894d60832754124247de`；tag `v0.1-baseline` `4abbc5b626f5122012d6db0deac324cb3e177522`；仓库声明 Apache-2.0 | 面向 Isaac Sim 5.1 / Isaac Lab 2.3.1 / Python 3.11，包含 G1 BASE 运行配置。motion clips、checkpoint、USD 和视频使用 Git LFS。README 提到 PMY9527 与 LaFan1 片段，但没有给出足以复核的 LaFan1 授权和数据 hash。 | 只作参考，不纳入默认依赖 |
| [`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab) | main `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`，Apache-2.0 | 支持 G1 29 DOF，但当前代码没有 AMP 任务。需要外部 `unitree_model` 或 `unitree_ros` 资产。 | 可用于核对 G1 常规模型接口，不是 AMP 实现 |
| [`unitree_model`](https://github.com/unitreerobotics/unitree_model) | main `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7`，BSD-3-Clause，README 标记 deprecated | 包含 G1 29 DOF USD，但对象使用 Git LFS。本机没有 git-lfs。未来维护迁移到 Unitree Hugging Face 入口。 | 资产来源候选，需单独下载和记录 hash |
| [`GMR`](https://github.com/YanjieZe/GMR) | master `25f56db79ac411258b377b6d4ccba0c8ff3372ac`，MIT | 支持 G1 29 DOF 的人体动作 retarget 工具。MIT 代码许可证不授予 AMASS、OMOMO 或 LaFAN1 原始数据再分发权。 | 仅作为 R2 数据适配工具候选 |

本机 `/home/ubuntu2004/GRScenes/robot_lab-main` 不是 Git 工作树。其 README 标注 Isaac Sim 4.5 / Isaac Lab 2.0，源码有 RSL-RL AMP，但 AMP 任务只覆盖 Unitree A1，专家数据是 A1 mocap 文本和 A1 URDF。该目录中的 G1 只有常规 PPO 配置和 G1 模型引用，没有 G1 AMP 任务或 G1 AMP 专家动作，不能当作 R2 已完成实现。

### G1 数据可用性结论

1. 官方 Humanoid `.npz` 只能证明 28 DOF Humanoid 的 R1 数据管线可用，不能直接喂给 G1。
2. `legged_lab` 与 `AMP_Running_baseline` 的 G1 动作和 USD 在当前机器上只有 LFS pointer 或未取得对象，不能开始可复现的 G1 实验。
3. 原始人体数据、retarget 输出、G1 USD、训练 checkpoint 必须分别记录来源、许可证、commit 和 hash。没有这些证据时，不上传、转换或发布外部资产。

## 6. 环境隔离方案

### A：R1 官方基线

- 固定 Isaac Lab v2.1.1 SHA `90b79bb2d44feb8d833f260f2bf37da3487180ba`，独立 checkout，不使用 `/home/ubuntu2004/GRScenes/a/IsaacLab`。
- 固定 Isaac Sim 4.5.0 standalone 与 Kit Python，固定 `skrl==1.4.3`。不要让 `pip install skrl` 无上限升级到 2.x，因为官方 v2.1.1 YAML 使用的是 skrl 1.x AMP 字段。
- 运行前保存 `nvidia-smi`、Python 包版本、IsaacLab commit、动作文件 SHA256、环境数量和 seed。初始 smoke 使用 4 环境，完成后再以 8、16、32 逐级测峰值显存。
- 所有日志、checkpoint、视频放到项目外部工件目录，只在仓库提交小型配置、指标表和 hash。

### B：R2 G1 候选环境

- 单独准备 Isaac Lab 2.3.x / Isaac Sim 5.1 兼容环境，或对候选代码做明确的 v2.1.1 适配。两种方案都不能升级现有 `isaac_env` 或修改其他仓库。
- 安装 git-lfs 前先确认磁盘预算和资产许可。拉取后记录每个 motion、USD 和 checkpoint 的 SHA256。
- 先把混合 motion 配置拆成单技能 Walk 子集，冻结命令分布和数据 hash，再做 R2 单技能 smoke。不要直接从混合 walk/run/turn 配置推导单技能结论。

## 7. 最小可复现 smoke 计划

下一阶段 R1 的第一条命令沿用已验证入口，先保持 4 个环境：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Humanoid-AMP-Walk-Direct-v0 \
  --algorithm AMP \
  --headless \
  --num_envs 4 \
  --max_iterations 1 \
  --seed 42 \
  --device cuda:0
```

执行前要确认 `skrl==1.4.3`、IsaacLab SHA、动作文件 hash 和 GPU 空闲显存。R1 正式实验前先完成官方 Walk AMP 与同环境、同预算的 PPO 对照，再决定是否扩大训练规模。smoke 的退出码不能替代独立评估。

## 8. 阻塞项与下一动作

| 阻塞项 | 影响 | 解除条件 |
|---|---|---|
| 实机只有 RTX 3070 8 GiB | 默认 4096 环境和大 batch 存在 OOM 风险 | 以 4 环境起步并实测峰值显存，必要时降低 batch 和并行数 |
| `/home/ubuntu2004/GRScenes/a/IsaacLab` v1.2.0 工作树脏且无官方 AMP | 不能复用为 R1 基线 | 使用独立 v2.1.1 checkout |
| Conda 中有 skrl 2.1.0，但官方 YAML 需要 skrl 1.x 字段 | 可能在 AMP agent 初始化时报字段不兼容 | 锁定并记录 `skrl==1.4.3` |
| G1 motion/USD 为 LFS 或未验证来源 | R2 无法形成可复现且可发布的数据输入 | 安装 git-lfs，确认许可证、对象 hash、retarget 记录 |
| G1 候选主要面向 Isaac Lab 2.3.x / Isaac Sim 5.1 | 与现有 4.5/2.1.1 组合不直接兼容 | 另建版本隔离环境并做最小导入/场景 smoke |
| 其他 Isaac Sim 进程会占显存 | 测试结果随桌面状态变化 | 运行前记录并尽量释放其他进程 |

**R0 状态：PARTIAL。** 官方 AMP 路线已通过最小真实 smoke，足以进入 R1。G1 资产、动作授权和独立兼容环境尚未闭合，因此 R2 不能开始，更不能写成已完成 G1 训练。

## 9. 证据索引

- 项目规划：`README.md`、`docs/ROADMAP.md`、`docs/EXPERIMENT_PROTOCOL.md`、`docs/HANDOFF.md`、`docs/REFERENCES.md`、`PROJECT_STATUS.md`
- 官方 v2.1.1 静态 checkout：`/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/work/upstream/IsaacLab-v2.1.1/`
- 官方 smoke 完整输出：`/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/work/r0_smoke_output.txt`
- 官方 smoke 日志与 checkpoint：`logs/skrl/humanoid_amp_walk/2026-11-22_23-43-34_amp_torch/`（被 `.gitignore` 忽略）
- 目标仓库审计基线：`803f84f55af0044a8649d0fad5c0c4205ca406ca`

