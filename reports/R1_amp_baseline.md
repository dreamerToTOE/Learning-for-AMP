# R1 官方 Humanoid AMP / PPO 最小 Smoke

运行日期：2026-11-23（Asia/Seoul）
阶段状态为 **PARTIAL**。AMP 与同环境 PPO 已完成短训和独立评估 smoke，风格奖励消融已完成资源 pilot。
运行主机：RTX 3070 8 GiB，driver `580.178.04`。
源码基线：官方 Isaac Lab `v2.1.1`，SHA `90b79bb2d44feb8d833f260f2bf37da3487180ba`。

## 1. 锁定的环境

R1 使用独立 Git checkout：

```text
/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/work/r1/IsaacLab-v2.1.1
```

该 checkout 的 Git 状态为干净的 detached tag `v2.1.1`。Isaac Sim 和运行时版本如下：

| 项目 | 锁定值 |
|---|---|
| Isaac Lab | v2.1.1，`90b79bb2d44feb8d833f260f2bf37da3487180ba` |
| Isaac Sim | `/home/ubuntu2004/isaacsim-4.5.0`，`4.5.0-rc.36+release.19112.f59b3005.gl` |
| Python | Kit Python `3.10.15` |
| PyTorch | `2.5.1+cu118` |
| skrl | `1.4.3` |
| Gymnasium | `0.29.1` |
| Hydra Core | `1.3.2` |
| NumPy | `1.26.0` |
| CUDA | `torch.cuda.is_available() == True`，`torch.version.cuda == 11.8` |
| GPU | NVIDIA GeForce RTX 3070，8192 MiB，compute capability 8.6 |
| Driver | `580.178.04` |

早期 R1 smoke 曾通过 `PYTHONPATH` 加载已存在的只读依赖目录 `/home/ubuntu2004/anaconda3/envs/isaaclab/lib/python3.10/site-packages`。本报告后面的最终结果改用真正独立的 venv，并通过 `scripts/r1_python.sh` 清除该路径。独立 venv 结果是本轮交付的复现基线。

Walk 专家动作文件为官方 v2.1.1 checkout 内的 `humanoid_walk.npz`，SHA256 为 `91a7af4e7fa59d8a3a2bb2c6e0dad45d73f6e5b5ee1a284d733435e32c418542`。文件为 60 fps、154 帧、28 DOF、15 bodies。

## 2. 公平 Smoke 设计

两组都使用 `Isaac-Humanoid-AMP-Walk-Direct-v0` 和同一个 `HumanoidAmpWalkEnvCfg`，因此保持相同的 28 维动作、81 维 policy observation、`dt=1/60`、decimation=2、10 秒 episode、跌倒终止条件、Walk 动作文件和随机 reset 管线。

两组固定以下参数：

```text
seed=42
num_envs=4
device=cuda:0
headless=true
rollouts=16
max_iterations=1
steps_per_env=16
total_vectorized_transitions=64
```

两组使用相同的 policy/value MLP `[1024, 512]`、ReLU、`initial_log_std=-2.9`、固定 log standard deviation、learning rate `5e-5`、6 learning epochs 和 2 mini-batches。

- AMP 使用官方 `skrl_walk_amp_cfg.yaml`，包含独立 discriminator、motion dataset 和 replay buffer，默认 `task_reward_weight=0`、`style_reward_weight=1`。
- PPO 使用仓库内 `configs/r1/skrl_ppo_amp_smoke.yaml`。该配置去掉 discriminator、motion dataset 和 replay buffer，保留相同 policy/value 网络与 16 rollouts。PPO 只使用该环境的任务奖励。
- 官方 AMP Walk 任务本身没有 `skrl_cfg_entry_point`。`scripts/r1_ppo_amp_smoke.py` 在 AppLauncher 启动后向同一 Gym task 临时注入 PPO YAML，然后执行未修改的上游 `train.py`。独立 checkout 没有被改写。

PPO 适配文件 SHA256：

```text
configs/r1/skrl_ppo_amp_smoke.yaml  5b695f5b0a9979a3295090ba14995145174eb2c88b2690903219dfa3eb88210c
scripts/r1_ppo_amp_smoke.py         5cb94f99bd4d39a9acd3c1bbc56c009efce61ee6b72ac0a50cd70619f95b1df4
```

## 3. 运行结果

GPU 监控每 0.5 秒采样一次 `nvidia-smi`。审计期间另有 Isaac Sim 进程占用 GPU，峰值显存包含该外部占用，不能直接解释为单个训练进程的 allocator 峰值。

| run_id | 算法 | 命令关键参数 | 退出码 | 采样 wall time | 监控峰值显存 | 结果 |
|---|---|---|---:|---:|---:|---|
| `amp_20261123T0006+0900_seed42_env4_steps16` | AMP | `task=Isaac-Humanoid-AMP-Walk-Direct-v0`, `max_iterations=1` | 0 | 约 35.0 s | 7203 MiB | Walk 动作加载，policy/value/discriminator 初始化，16/16 rollout |
| `ppo_20261123T0010+0900_seed42_env4_steps16` | PPO | 同一 task，临时 PPO entry，`max_iterations=1` | 0 | 约 29.8 s | 6475 MiB | Walk 动作加载，policy/value 初始化，16/16 rollout |

AMP 运行关键日志和 checkpoint：

```text
runs/r1_smoke/amp_20261123T0006+0900_seed42_env4_steps16/stdout.log
runs/r1_smoke/amp_20261123T0006+0900_seed42_env4_steps16/gpu_before.csv
runs/r1_smoke/amp_20261123T0006+0900_seed42_env4_steps16/gpu_memory.csv
runs/r1_smoke/amp_20261123T0006+0900_seed42_env4_steps16/gpu_after.csv
runs/r1_smoke/amp_20261123T0006+0900_seed42_env4_steps16/logs/skrl/humanoid_amp_walk/2026-11-23_00-04-15_amp_torch/
```

PPO 运行关键日志和 checkpoint：

```text
runs/r1_smoke/ppo_20261123T0010+0900_seed42_env4_steps16/stdout.log
runs/r1_smoke/ppo_20261123T0010+0900_seed42_env4_steps16/gpu_before.csv
runs/r1_smoke/ppo_20261123T0010+0900_seed42_env4_steps16/gpu_memory.csv
runs/r1_smoke/ppo_20261123T0010+0900_seed42_env4_steps16/gpu_after.csv
runs/r1_smoke/ppo_20261123T0010+0900_seed42_env4_steps16/logs/skrl/humanoid_amp_ppo_smoke/2026-11-23_00-05-35_ppo_torch/
```

最终 checkpoint hash：

```text
AMP agent_16.pt  e0c57aa7127a445226ae9409214c4c76bc3136fed35941cfe0b616fc2973ab9d
PPO agent_16.pt  56aef81bbca0d401568580ba4752ae76448036acd48fa8e6ec676066d95b5a68
```

首次 fresh checkout 尝试因稀疏 checkout 缺少 `source/isaaclab_mimic` 而被 Isaac Sim 扩展依赖解析阻止，退出码为 1。该失败日志保留在：

```text
runs/r1_smoke/amp_20261123T0003+0900_seed42_env4_steps16/
```

随后扩大独立 checkout 的 `source` 范围，使用相同版本和参数重跑成功。该修复没有修改官方源码内容。

## 4. 真正独立 venv 重跑

为避免复用 R0/R1 之前的 anaconda site-packages，创建了独立 venv：

```text
/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/work/r1/venv
```

创建入口为 `scripts/create_r1_venv.sh`，运行入口为 `scripts/r1_python.sh`。运行入口设置 `PYTHONNOUSERSITE=1`，清除继承的 `PYTHONPATH`，只加入该 venv 和独立 Isaac Lab source checkout。验证时 `sys.path` 不含 anaconda、用户 `.local` 或 `/opt/ros` 路径。

venv 锁文件为 `configs/r1/requirements-venv.lock`，SHA256 为 `858d385cb68b9103a6dfab650a170feb343bf9d60fafb76853a0c8fb8671f1bb`。venv 内固定 `skrl==1.4.3`、`hydra-core==1.3.2`、`omegaconf==2.3.0`、`gymnasium==0.29.1`、`numpy==1.26.0`、`tensorboard==2.19.0`、`flatdict==4.0.1`、`h5py==3.13.0` 及其余列出的传递依赖。Isaac Sim Kit Python 提供 `torch==2.5.1+cu118` 和原生 simulator extensions，未安装到 venv。

独立 venv 的版本探针输出为 Python 3.10.15、PyTorch 2.5.1+cu118、skrl 1.4.3、Gymnasium 0.29.1、Hydra 1.3.2、OmegaConf 2.3.0、TensorBoard 2.19.0、NumPy 1.26.0，CUDA 可用。

第一次真正隔离启动暴露了旧 site-packages 隐式提供的 `flatdict`，补齐后又暴露 `h5py`。两者已按精确版本安装，失败目录保留在：

```text
runs/r1_venv_smoke/amp_20261123T0030+0900_seed42_env4_steps16/
```

补齐依赖后的最终重跑结果如下：

| run_id | 算法 | 退出码 | 采样 wall time | 监控峰值显存 | 结果 |
|---|---|---:|---:|---:|---|
| `amp_20261123T0040+0900_seed42_env4_steps16` | AMP | 0 | 约 40.3 s | 7132 MiB | 16/16 rollout，policy/value/discriminator 初始化 |
| `ppo_20261123T0045+0900_seed42_env4_steps16` | PPO | 0 | 约 25.8 s | 6360 MiB | 16/16 rollout，policy/value 初始化 |

最终 venv checkpoint hash 与之前锁定环境的 smoke 一致：AMP `agent_16.pt` 为 `e0c57aa7127a445226ae9409214c4c76bc3136fed35941cfe0b616fc2973ab9d`，PPO `agent_16.pt` 为 `56aef81bbca0d401568580ba4752ae76448036acd48fa8e6ec676066d95b5a68`。

venv 运行证据目录如下：

```text
runs/r1_venv_smoke/amp_20261123T0040+0900_seed42_env4_steps16/
runs/r1_venv_smoke/ppo_20261123T0045+0900_seed42_env4_steps16/
```

## 5. 结论边界

- 这两次运行证明官方 Walk AMP 的 policy/value/discriminator 链路和同环境 PPO 的 policy/value 链路可以在当前独立 checkout 中启动、加载动作数据、完成一次更新并保存 checkpoint。
- 两次运行都只有 16 steps/env，不能作为训练收敛、步态稳定性、速度误差、风格相似度或 AMP 优于 PPO 的证据。
- 尚未执行独立评估，因此 `success_rate`、跌倒率、速度 MAE、足端滑移和 style metric 均为未测量。
- 最终独立 venv AMP 的峰值监控显存约 7132 MiB，当前 RTX 3070 8 GiB 上不适合直接复制官方 4096 环境配置。后续短训前要先释放外部 Isaac Sim 进程或把并行环境降到 2，并重新记录峰值。

## 6. 两环境短训 smoke（2026-11-23）

运行前项目 Git SHA 为 `9f38f2037274561391eba02db45f57057f7e99d5`，独立 Isaac Lab checkout 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`，两者工作树均干净。沿用第 4 节的 venv、Kit 运行时和 Walk 动作文件。启动探针复核了 Python 3.10.15、PyTorch 2.5.1+cu118、skrl 1.4.3、Gymnasium 0.29.1、Hydra 1.3.2、NumPy 1.26.0 和 TensorBoard 2.19.0，skrl 来源为 R1 venv。

两组顺序运行，固定同一 Walk 任务、seed 42、2 个 CUDA 并行环境、每次更新 16 步、5 次更新。每组共 80 步/环境，即 160 个环境交互步，约 2.67 秒模拟时间/环境。每次更新记录 TensorBoard 标量，第 80 步保存 checkpoint。两组导出的 `params/env.yaml` 字节一致。

### 复现命令

从项目仓库根目录执行。每组通过独立工作目录保存上游生成的日志。

```bash
repo="$PWD"
lab="$(realpath "$repo/../r1/IsaacLab-v2.1.1")"
run_root="$repo/runs/r1_short_train/$(TZ=Asia/Seoul date +%Y%m%dT%H%M%S%z)_seed42_env2_steps80"
mkdir -p "$run_root/amp" "$run_root/ppo"
common=(
  --task Isaac-Humanoid-AMP-Walk-Direct-v0
  --num_envs 2 --max_iterations 5 --seed 42 --headless --device cuda:0
  agent.agent.experiment.write_interval=16
  agent.agent.experiment.checkpoint_interval=80
)
(
  cd "$run_root/amp"
  timeout --kill-after=10s 180s "$repo/scripts/r1_python.sh" \
    "$lab/scripts/reinforcement_learning/skrl/train.py" --algorithm AMP "${common[@]}" > stdout.log 2>&1
)
(
  cd "$run_root/ppo"
  timeout --kill-after=10s 180s "$repo/scripts/r1_python.sh" \
    "$repo/scripts/r1_ppo_amp_smoke.py" --algorithm PPO "${common[@]}" > stdout.log 2>&1
)
```

### 训练和资源记录

| 算法 | 退出码 | 完成步数/环境 | 用时（含启动/退出） | 吞吐（交互步/s） | 运行前显存 | 采样峰值显存 | 运行后显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMP | 0 | 80/80 | 18.575 s | 8.61 | 991 MiB | 4393 MiB | 925 MiB |
| PPO | 0 | 80/80 | 12.110 s | 13.21 | 925 MiB | 3607 MiB | 925 MiB |

显存由 `nvidia-smi` 以 500 ms 间隔采样，AMP 有 37 个样本，PPO 有 24 个样本。数值是 RTX 3070 整卡显存用量。吞吐按 160 个交互步除以进程总用时计算，包含 Kit 启动和退出开销。

| 每环境步数 | AMP Policy loss | AMP Value loss | AMP Discriminator loss | PPO Policy loss | PPO Value loss |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.336944 | 3.266759 | 3.519190 | 0.300324 | 3.553840 |
| 32 | -0.027583 | 1.800663 | 2.927908 | -0.011569 | 2.961762 |
| 48 | -0.109444 | 1.171814 | 2.583831 | -0.058895 | 3.197326 |
| 64 | -0.080848 | 1.120824 | 2.312198 | 0.519977 | 2.788157 |
| 80 | -0.128962 | 2.981960 | 2.427591 | -0.082491 | 1.443087 |

两组记录的瞬时环境奖励均为 1，策略标准差均为 0.055023。Walk 环境每步返回 1，因此 TensorBoard 的回合总环境奖励等于已结束回合的长度。每环境步数 16、32、48、64、80 对应的回合长度均值，AMP 为 16、14、24、21、22，PPO 为 12、17.5、未记录、24.45、未记录。这些是 skrl 的训练日志统计。AMP 在内部使用风格奖励训练，权重为任务 0、风格 1。本次事件文件没有单独记录风格奖励。五次更新的损失、所有已记录标量，以及最终 checkpoint 中的 policy/value/discriminator 参数均通过有限值检查。

训练指标用于确认更新和日志链路。独立评估为 0 次，本轮尚无步态质量、收敛或算法优劣结论。

### 本地日志和 checkpoint

运行根目录为 `runs/r1_short_train/20261123T002939+0900_seed42_env2_steps80/`。其 `amp/` 和 `ppo/` 子目录各有实际命令 `command.txt`、`stdout.log`、`timing.txt`、`exit_code.txt`、`gpu_before.csv`、`gpu_memory.csv` 和 `gpu_after.csv`。各组的 TensorBoard 原始事件、参数 YAML 和最终 checkpoint 位于：

- AMP：`amp/logs/skrl/humanoid_amp_walk/2026-11-23_00-29-47_amp_torch/`，`checkpoints/agent_80.pt` 的 SHA256 为 `5b16ae810a913716526d4e2f99ae35acb46d4643818562bfcddeae19ad98fbaa`。
- PPO：`ppo/logs/skrl/humanoid_amp_ppo_smoke/2026-11-23_00-30-05_ppo_torch/`，`checkpoints/agent_80.pt` 的 SHA256 为 `608e8e8daf43b82314f65630d9df4c4425af2137d34e0a0cfecab76e8a6a1ae4`。

中断前的 AMP 尝试只有 GPU 采样和空 stdout，无退出码或训练证据。本节仅使用恢复后完整运行的结果。

## 7. 独立评估 smoke（2026-11-23）

使用第 6 节的两份 `agent_80.pt`。每组在同一个官方 Walk 环境中逐个评估 10 回合，重新设定 seed `1000–1009`。评估器加载各自的 Actor 与观测归一化状态，使用高斯策略均值动作。10 对初始 policy observation 的 SHA256 和躯干初始位置全部一致。

记录自动重置前的终止状态。跌倒对应躯干高度低于 0.5 m，超时使用上游标志，前进距离沿参考动作的世界坐标 +X。位移和水平速度均为实测值，当前任务没有速度指令。Wasserstein 距离使用每组所有实际执行步的躯干世界坐标 `v_x`，与官方 Walk 片段的参考速度分布比较。

| 策略 | 回合数 | 跌倒 / 超时 | 存活时间（s，均值 ± SD） | +X 净位移（m，均值 ± SD） | 水平速率（m/s，均值 ± SD） | 参考速度 W₁（m/s） |
|---|---:|---:|---:|---:|---:|---:|
| 官方 AMP | 10 | 9 / 1 | 1.793 ± 2.879 | 0.490 ± 0.267 | 1.201 ± 0.495 | 1.203 |
| 同环境 PPO | 10 | 10 / 0 | 0.870 ± 0.337 | 0.507 ± 0.306 | 1.252 ± 0.238 | 0.859 |

AMP 的唯一超时案例为 seed 1002，存活 9.97 s，净前进 0.456 m。两份 checkpoint 均来自 seed 42 的 5 次更新，且训练奖励不同。这些数据只用于核对推理、配对和指标链路。独立重跑的逐回合统计一致。

- 可追溯数据为 [AMP 回合 JSON](r1_eval/amp_episodes.json) 和 [PPO 回合 JSON](r1_eval/ppo_episodes.json)，包含重置 seed、初始观测哈希、终止原因和逐步 `v_x`。运行前项目 HEAD 为 `65f7a8d9baa1d9b55e66a8941b50ddfc5807d78b`，评估器为本次提交新增。
- 本机原始日志、命令和视频位于 `runs/r1_eval/20261123T004510+0900_seed1000_n10/`。RTX 相机尝试因着色器编译等待触发超时。最终两段 MP4 使用仿真连杆坐标绘制侧视骨架，固定选择 seed 1000，包含终止帧。AMP 与 PPO 分别验证可解码 27 和 41 帧，均为 30 fps。

复现时从仓库根目录运行，将变量指向第 6 节相应 checkpoint，输出目录使用尚未存在的路径：

```bash
scripts/r1_python.sh -u scripts/r1_eval_walk.py --headless --device cuda:0 \
  --checkpoint "$checkpoint" --output "$output_dir" --seed 1000 --episodes 10 --video
```

## 8. 等预算奖励消融资源 pilot

按 [正式对照协议](../docs/EXPERIMENT_PROTOCOL.md#6-r1-walk-正式对照预注册)，两组使用同一 skrl AMP 实现、相同环境和优化配置，任务奖励权重均固定为 1。seed 42，2 环境，每组 50 次更新，共 1,600 个交互步。唯一训练配置差异为风格权重。两组均退出码 0，50 次更新中的 Policy、Value 和判别器损失均为有限值。

| 组别 | 风格权重 | 用时（含启动/退出） | 整卡显存采样峰值 |
|---|---:|---:|---:|
| 任务奖励 | 0.0 | 23.088 s | 4274 MiB |
| 任务加风格 | 1.0 | 23.576 s | 4290 MiB |

本机日志和最终 checkpoint 位于 `runs/r1_reward_pilot/20261123T004402+0900_seed42_env2_steps800/`。其 `task/` 和 `task_style/` 子目录保留完整命令、TensorBoard 事件、参数和 500 ms GPU 采样。两组 `agent_800.pt` 的 SHA256 分别为 `481442a04c1dd7d037eb91154f40ad9c467792a8610e0048b5cf8c0c7abecefb` 和 `982b91840d6528ecc79df641666ec34b483a40d0105e1df1819d6bbf1a8f9097`。

## 9. 正式三 seed 对照结果

按预注册协议，两种奖励配置各用训练 seed `42, 43, 44`，2 环境、500 次更新、8,000 steps/env，即每组 16,000 个环境交互步。所有六组最终退出码为 0，损失和 checkpoint 参数通过有限值检查。seed 44 的任务奖励组第一次因另一个 Isaac Sim 进程占用 3.3 GiB 显存而 OOM，外部进程结束后用相同命令重跑成功。该失败目录保留，正式结果使用重跑目录。

六组训练的最终记录如下。峰值为整卡 `nvidia-smi` 显存用量，外部进程已经结束后重跑的组有更充足的可用显存。

| 训练组 | 用时 | 峰值显存 | 最终 Policy loss | 最终 Value loss | 最终 Discriminator loss |
|---|---:|---:|---:|---:|---:|
| task / seed 42 | 158.1 s | 7080 MiB | -0.0421 | 0.2887 | 1.0977 |
| task / seed 43 | 140.1 s | 7089 MiB | -0.1093 | 0.2787 | 1.1790 |
| task / seed 44 | 161.2 s | 6992 MiB | -0.0920 | 0.2304 | 0.9938 |
| task + style / seed 42 | 142.6 s | 6966 MiB | -0.0310 | 0.3937 | 1.0242 |
| task + style / seed 43 | 151.2 s | 7714 MiB | -0.0526 | 0.7222 | 1.1859 |
| task + style / seed 44 | 145.7 s | 7602 MiB | -0.0866 | 0.4666 | 0.9542 |

| 训练 seed | 风格权重 | 跌倒 / 超时（100 回合） | 存活时间均值 | +X 位移均值 | 水平速率均值 | 参考速度 W₁ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 100 / 0 | 0.475 s | 0.554 m | 1.530 m/s | 0.424 m/s |
| 43 | 0 | 100 / 0 | 0.679 s | 0.718 m | 1.281 m/s | 0.604 m/s |
| 44 | 0 | 100 / 0 | 0.529 s | 0.932 m | 1.918 m/s | 0.352 m/s |
| 42 | 1 | 100 / 0 | 0.508 s | 0.635 m | 1.552 m/s | 0.463 m/s |
| 43 | 1 | 100 / 0 | 0.576 s | 0.649 m | 1.391 m/s | 0.546 m/s |
| 44 | 1 | 100 / 0 | 0.562 s | 0.738 m | 1.487 m/s | 0.378 m/s |

每个回合的跌倒率都是 100%，没有回合达到超时。风格权重为 1 相对权重为 0 的三个成对差值如下。跌倒率差值为 0 个百分点，存活时间差值的均值为 -0.012 s，+X 位移差值均值为 -0.061 m，水平速率差值均值为 -0.100 m/s，参考速度 W₁ 差值均值为 +0.002 m/s。三 seed 间这些差值的样本标准差分别为 0、0.079 s、0.138 m、0.290 m/s、0.053 m/s。

逐回合 JSON、汇总 CSV 和成对差值位于 [formal evaluation artifacts](r1_formal_eval/)。六段固定 seed 42000 的首回合视频位于项目外部 `outputs/R1-formal-*.mp4`。骨架视频显示策略在很短时间内失去直立姿态，符合 100% 跌倒统计。视频只用于动作过程定性核对，位移和速度以 JSON 指标为准。

这组结果说明 16,000 个环境交互步在当前 2 环境设置下不足以得到稳定 Walk 策略。它不能支持“风格奖励无效”的结论，也不能把零超时解释成可控行走。当前 R1 仍为 PARTIAL，R2 不启动。

## 10. 六组全跌倒诊断（2026-11-23）

### 10.1 先复现症状

正式评估的六个 JSON 共 600 回合，均为 `terminated=true`、`truncated=false`，没有 NaN。训练和评估 checkpoint、初始 observation hash 以及 100 个评估 seed 已成对核对。这个结果可以稳定复现为“全跌倒”，因此后续检查针对训练和控制链路，不把评估配对错误作为首要假设。

### 10.2 预算

每组只完成 500 次更新、8,000 steps/env、16,000 个环境交互步。10 s episode horizon 对应 300 个 policy steps，而正式评估的存活步数只有约 14–20 步，策略没有接近一个完整 episode。训练 checkpoint 的单步探针还显示动作幅度随训练增加。任务奖励组的 agent 800、4000、8000 首个动作范围分别为 `[-0.531, 0.567]`、`[-1.048, 0.998]`、`[-1.372, 1.727]`，对应最大目标关节偏差分别为 2.73、4.97、7.83 rad。预算不足仍然是必要条件层面的解释，但单纯增加步数前必须先处理动作目标范围，否则会继续训练同一个失稳控制接口。

### 10.3 初始化状态

官方 reset 代码把参考躯干高度额外增加 0.15 m。100 个正式评估初态的躯干 z 范围是 1.153–1.244 m，均值 1.203 m。`humanoid_walk.npz` 参考躯干 z 范围是 0.998–1.094 m，均值 1.049 m。初态 x 范围与参考动作均为约 -2.080–1.807 m，且 reset 直接从参考姿态、速度和根部状态构造。当前证据支持“高度偏移是上游明确设计”，没有发现初态随机化把人形放到地面或错位的证据。后续修正动作接口后再用 `random-start` 和随机时间各做短对照，确认 0.15 m 提升和接触建立是否影响第一步。

### 10.4 AMP 风格奖励尺度

任务每步奖励固定为 1。AMP 配置的 `discriminator_reward_scale=2.0`、风格权重为 1 时，最终 task+style seed42 checkpoint 在同一初态上的判别器 logit 为 2.10，对应风格奖励 4.43，约为任务奖励的 4.4 倍。该组 agent800 的风格奖励已为 2.84。这个量级足以让风格项主导早期优势估计，说明奖励尺度需要单独校准。正式三 seed 中权重 0 与 1 两组均为 100% 跌倒，位移和速度差异方向不一致，因此目前只能判定为高风险混杂因素，不能宣称风格奖励单独造成全跌倒，也不能宣称它无效。

### 10.5 动作、接触和动力学

官方环境使用 `target = action_offset + action_scale * action`，其中 `action_scale` 直接取软关节上下限之差，而 agent 配置将 `clip_actions` 设为 `false`。任务 seed42 agent8000 的确定性动作已经超出 [-1, 1]，首步目标关节范围达到 -5.49–7.96 rad，最大目标与当前关节差 7.83 rad。CPU 物理探针同时报告 PhysX spherical joint 的目标角超出 [-π, π]。把动作临时裁剪到 [-1,1] 后首步最大偏差降到 4.97 rad，把 scale 再减半后降到 3.01 rad。相同 checkpoint 的 60 步单环境探针中，未裁剪、裁剪、裁剪加半范围分别在 17、14、22 步触发跌倒。裁剪和半范围能延长探针存活，但仍未形成稳定行走，说明动作范围是首要控制接口问题，同时还存在策略训练和动力学匹配问题。

环境没有接触奖励或足底接触传感器，地面材质为静态/动态摩擦 1.0、恢复系数 0.0。当前物理日志中的直接异常是关节目标角越界，而不是接触求解器发散。接触动力学需要在动作范围修正后再判断，不能把当前的 PhysX 目标角错误归因于地面接触。

探针 JSON 和完整 Kit/PhysX stdout 保存在 `runs/r1_diagnostics/`。其中 `task_seed42_probe_cpu.json`、`task_style_seed42_probe_cpu.json`、`task_seed42_raw_60.json`、`task_seed42_clip_60.json` 和 `task_seed42_clip_half_60.json` 对应上述数字，`*_probe_cpu.console.log` 保留目标角越界日志。`runs/` 按仓库规则忽略，报告中的结论和脚本可在 GitHub 复核，原始运行目录留在本机用于重放。

### 10.6 诊断结论和顺序

当前排序为：动作目标范围与未裁剪策略输出是首要原因，风格奖励尺度是高风险放大因素，训练预算不足是未满足稳定学习的条件，初始化状态暂未发现明确错误，接触模型暂未出现独立异常。六组全跌倒不能作为模式坍塌证据，也不能支持 R1 已经学会 Walk。

下一轮只做小规模可归因验证，顺序固定为：先固定动作接口并保留原始配置作为对照，再将风格奖励改为与任务奖励同量级，最后在同一 seed 和同一评估初态上延长预算。每个改动先跑 2 环境短训和 10 回合评估，只有跌倒率明显下降后才进入三 seed 正式重跑。

## 11. 动作范围修正短对照（2026-11-23）

在不修改独立 Isaac Lab checkout 的前提下，新增了项目侧可选 patch。它把动作裁剪到 `[-1, 1]`，并把 action scale 改为软关节上下限宽度的一半。修正组仍使用官方 AMP、2 个环境、seed 42、50 次更新、800 steps/env、任务奖励权重 1、风格奖励权重 0。原始组复用了同预算、同 seed 的既有 pilot checkpoint。两组使用相同的 10 个评估初态，初始 observation hash 全部一致。

| 组别 | action 接口 | 跌倒 / 超时 | 存活时间均值 | +X 位移均值 | 参考速度 W₁ | PhysX 目标角错误 |
|---|---|---:|---:|---:|---:|---:|
| 原始 pilot | 未裁剪、完整宽度 | 10 / 0 | 0.563 s | 0.554 m | 0.413 m/s | 0 |
| 修正 smoke | 裁剪、半宽度 | 10 / 0 | 0.747 s | 0.359 m | 0.977 m/s | 0 |

修正组训练和评估均退出码为 0，修正训练 stdout 没有 PhysX error。修正 checkpoint 的 CPU 单步探针首步最大目标偏差为 2.04 rad，目标关节范围为 -1.79–1.33 rad。存活时间提高但跌倒率没有下降，说明动作范围越界是实质性问题，却不是当前全跌倒的唯一原因。该结果不能支持增加训练预算已经有效，也不能支持 R1 通过。

训练日志、评估 JSON、GPU 记录和命令保存在 `runs/r1_actionrange_smoke/20261123T103000+0900_seed42_env2_steps800/`。可选 patch 位于 `scripts/r1_action_range_patch.py`，训练入口位于 `scripts/r1_amp_actionrange_smoke.py`，评估入口通过 `scripts/r1_eval_walk.py --action-range-correction` 启用。

## 12. 下一步

在修正动作接口下，运行风格权重为 1、`discriminator_reward_scale=0.5` 的同预算短对照，并使用同一 10 个评估初态。这个尺度把初始风格奖励拉近任务奖励量级。只有在跌倒率出现可重复下降后，才延长训练预算或恢复三 seed 正式实验。
