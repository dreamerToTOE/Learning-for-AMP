# R1 官方 Humanoid AMP / PPO 最小 Smoke

运行日期：2026-11-23（Asia/Seoul）
阶段状态：**PARTIAL**。AMP 与同环境 PPO 的最小 smoke 均完成，尚未进行长训、独立评估或性能结论。
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

运行时通过 `PYTHONPATH` 加载独立 checkout 的四个 source 目录，并加载已存在的只读 Python 依赖目录 `/home/ubuntu2004/anaconda3/envs/isaaclab/lib/python3.10/site-packages`。本轮没有修改该依赖目录，也没有升级现有 Conda 环境。

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

## 4. 结论边界

- 这两次运行证明官方 Walk AMP 的 policy/value/discriminator 链路和同环境 PPO 的 policy/value 链路可以在当前独立 checkout 中启动、加载动作数据、完成一次更新并保存 checkpoint。
- 两次运行都只有 16 steps/env，不能作为训练收敛、步态稳定性、速度误差、风格相似度或 AMP 优于 PPO 的证据。
- 尚未执行独立评估，因此 `success_rate`、跌倒率、速度 MAE、足端滑移和 style metric 均为未测量。
- AMP 的峰值监控显存约 7203 MiB，当前 RTX 3070 8 GiB 上不适合直接复制官方 4096 环境配置。后续短训前要先释放外部 Isaac Sim 进程或把并行环境降到 2，并重新记录峰值。

## 5. 下一步

1. 固定当前版本和 seed，选择低并行度，完成 AMP/PPO 各自的短训练 smoke。
2. 按实验协议建立独立评估入口，固定初态、seed、episode 数和动作/姿态指标。
3. 在有独立评估数据前，不报告 AMP/PPO 优劣或运动质量结论。
