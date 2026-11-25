# R5-0 G1 Motion Tracking 入门与复现审计

2026-11-26 · R4 归档后开始 R5。已核对本机参考跟踪入口和历史闭环数据，并在 CPU 上验证上游跟踪奖励。当前优先使用安装在独立 venv 中的 MimicKit DeepMimic 学习参考状态和评价。BeyondMimic 用于比较另一套全身跟踪实现。 [第一轮归档](FINAL_REPORT.md) · [R5 路线](../docs/ROADMAP.md#r5--g1-motion-tracking--beyondmimic当前主线)

## 已核验的环境和上游

| 入口 | 可复现证据和决策 |
| --- | --- |
| **MimicKit DeepMimic G1** | 本机 checkout 为 `a2c41528c2a34b638611f6c618e4a59f0e151179`，运行差异是归档中的两项 Newton 容量调整。venv 的系统包开关为 false，实读 Python 3.12.14、torch 2.8.0+cu128、Newton 1.6.1、Warp 1.17.0、MuJoCo/MuJoCo-Warp 3.12.0，与锁定文件一致。上游 G1 Walk 镜像权重已有本机闭环结果。MimicKit 代码是 Apache-2.0，G1 MJCF 资产目录为 BSD-3-Clause，镜像参考动作及权重的再分发权利另核。本轮只读取和核验既有证据。 [G1 配置](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/data/envs/deepmimic_g1_env.yaml) · [依赖锁](r2_g1_single_skill/requirements.lock.txt) · [资产来源和权重](R2_g1_single_skill.md#资产与动作) |
| **BeyondMimic 原跟踪工程** | 核对到 `9db915838b17785c6fec90c91fe80b95b147e1bd`，MIT 代码，已归档为只读。文档固定 Isaac Lab 2.1.0，训练/推理示例依赖 WandB 运动注册表或 run 路径。作者建议查看 mjlab 的移植。原作将 RL tracking 与后续蒸馏/扩散结合，本阶段先学 RL 跟踪。 [项目主页](https://beyondmimic.github.io/) · [作者代码及版本](https://github.com/HybridRobotics/whole_body_tracking/tree/9db915838b17785c6fec90c91fe80b95b147e1bd) |
| **mjlab 跟踪移植** | 核对到 `d872bfd2b048da657345c07669172fa284b38abf`，Apache-2.0。已注册 `Mjlab-Tracking-Flat-Unitree-G1`，转换器从重定向 CSV 生成按本引擎 body 顺序的 NPZ。其依赖要求 torch 2.14+、MuJoCo-Warp 4.0.x 和 RSL-RL 5.5.2，不能复用当前 MimicKit venv。若后续采用，先锁定版本和匹配的权重、动作、关节顺序及独立环境。此次未在本机运行。 [官方跟踪文档](https://github.com/mujocolab/mjlab/blob/d872bfd2b048da657345c07669172fa284b38abf/docs/source/training/motion_imitation.rst) · [依赖](https://github.com/mujocolab/mjlab/blob/d872bfd2b048da657345c07669172fa284b38abf/pyproject.toml) |
| **重定向与部署参考** | GMR `25f56db79ac411258b377b6d4ccba0c8ff3372ac` 的 MIT 工具支持 `unitree_g1` 29 DoF 和多种骨架格式。使用前逐份核对动作权利和骨架配置。Unitree RL Lab `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` 的 README 标注 Isaac Lab 2.3.0，可作为后续速度控制、仿真及部署接口参考。 [GMR](https://github.com/YanjieZe/GMR/tree/25f56db79ac411258b377b6d4ccba0c8ff3372ac) · [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3) |

GitHub 上游状态来自 2026-11-26 的仓库元数据与相应提交。`uv pip check` 核验现有 venv 的 151 个包，依赖检查通过。软件、参考动作、权重三个来源分别管理。

## 跟踪任务的输入和奖励

上游 `DeepMimicEnv` 在重置时从参考动作采样时间，并用参考根节点和关节的位姿、速度初始化。240 Hz 仿真、30 Hz 位置控制。Actor 同时读取当前状态，以及 MotionLib 的未来参考位姿，目标步长为 `[1, 2, 3]`，约对应 33、67、100 ms。配置 `global_obs=True`、`enable_tar_obs=True`、`enable_phase_obs=False`，会跟踪世界坐标里的根节点。R2 纯风格 AMP 的 Actor 用自身物理状态，训练时由 D 提供风格打分。这两种接口提供的任务信息不同，不能从两者的横漂数值判断单一组件贡献。 [DeepMimic 原论文](https://arxiv.org/abs/1804.02717) · [上游实现](https://github.com/xbpeng/MimicKit/blob/a2c41528c2a34b638611f6c618e4a59f0e151179/mimickit/envs/deepmimic_env.py)

这份 G1 跟踪配置用 PPO 优化参考误差奖励 $r=\sum_i w_i\exp(-k_i e_i)$。各项误差由关节旋转角或位置、速度差的平方或加权平方和计算。

| 项目 | 权重 | 指数尺度 |
| --- | ---: | ---: |
| 关节姿态 | 0.50 | 0.25 |
| 关节速度 | 0.10 | 0.01 |
| 根位姿 | 0.15 | 5.00 |
| 根速度 | 0.10 | 1.00 |
| 关键连杆相对根节点的位置 | 0.15 | 10.00 |

根位置和线速度项中还分别加入根旋转误差、角速度误差的 0.1 倍。检查观测接口时注意模型内部有 30 个非根旋转槽位，速度和控制只有 29 维，固定的 `head_link` 无可控 DoF。第一次离线构造把两者都按 29 维处理，断言未通过。改用 G1 MJCF 和上游权重映射得到正确维度。这个差别要体现在重定向、奖励诊断和部署接口检查里。

## 历史闭环与离线校验

- 镜像权重、`g1_walk.pkl`、G1 MJCF、原回放 JSON 与日志的本机 SHA-256 和大小均匹配归档清单。GitHub 的五回合评估 JSON 与本机原件字节相同。 [通用工件索引](../docs/AMP_G1_ARTIFACT_INDEX.md#r2) · [原始五回合 JSON](r2_g1_single_skill/deepmimic_reference_eval.json)
- 从原 JSON 重新汇总，seed 43000–43004 的五个参考初态散列各异，均到达 300 步、10 秒。平均前进 10.134 m，平均绝对横移 0.020 m，每回合 19 次独立足支撑交替，未检出 5 N 以上非足端接触。这是有未来参考状态的单片段历史结果。原配置 `log_tracking_error=False`，原 JSON 无逐步跟踪误差。下一轮跟踪评估应给出根与连杆误差。
- CPU 直接调用锁定上游的 `compute_reward` 和 G1 正运动学。完全匹配时奖励 1.00000000。参考根节点与连杆整体沿世界 X 平移 0.25 m 时，全局跟踪奖励 0.95974243，关闭世界根位置跟踪得 1.00000000。这个探针校验奖励对全局位移的敏感性，无新仿真回合或训练。

## 最小离线命令

从仓库根使用既有独立 venv。读取已校验的 G1 MJCF 和配置，只计算姿态与奖励。
```bash
upstream=$(realpath ../r2/upstream/MimicKit)
env -u PYTHONHOME CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 \
  PYTHONPATH="$upstream/mimickit" ../r2/mimickit-venv/bin/python - "$upstream" <<'PY'
from math import exp, isclose
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import yaml

from anim.mjcf_char_model import MJCFCharModel
from envs.deepmimic_env import DeepMimicEnv, compute_reward

upstream = Path(sys.argv[1])
cfg = yaml.safe_load((upstream / "data/envs/deepmimic_g1_env.yaml").read_text())
kin = MJCFCharModel("cpu")
kin.load(str(upstream / cfg["char_file"]))
weights = SimpleNamespace(_kin_char_model=kin, _device="cpu")
DeepMimicEnv._parse_joint_err_weights(weights, cfg["joint_err_w"])
assert (kin.get_num_joints() - 1, kin.get_dof_size()) == (30, 29)
assert kin.get_joint_dof_dim(kin.get_body_id("head_link")) == 0
root = torch.tensor([[0.0, 0.0, 0.8]])
rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
zero = torch.zeros_like(root)
dof = torch.zeros(1, kin.get_dof_size())
joint_rot = kin.dof_to_rot(dof)
body_pos, _ = kin.forward_kinematics(root, rot, joint_rot)
key_pos = body_pos[:, [kin.get_body_id(name) for name in cfg["key_bodies"]]]
terms = ("pose", "vel", "root_pose", "root_vel", "key_pos")
reward_weights = [cfg[f"reward_{term}_w"] for term in terms]
reward_scales = [cfg[f"reward_{term}_scale"] for term in terms]

def evaluate(shift, global_root):
    offset = torch.tensor([shift, 0.0, 0.0])
    return compute_reward(
        root, rot, zero, zero, joint_rot, dof, key_pos,
        root + offset, rot, zero, zero, joint_rot, dof, key_pos + offset,
        weights._joint_err_w, weights._dof_err_w, True, global_root,
        *reward_weights, *reward_scales,
    ).item()

nominal = evaluate(0.0, True)
global_shift = evaluate(0.25, True)
local_shift = evaluate(0.25, False)
root_w, root_scale = reward_weights[2], reward_scales[2]
expected = 1 - root_w + root_w * exp(-root_scale * 0.25**2)
assert isclose(nominal, 1.0, abs_tol=1e-6)
assert isclose(global_shift, expected, abs_tol=1e-6)
assert isclose(local_shift, 1.0, abs_tol=1e-6)
assert not torch.cuda.is_initialized()
print(f"pose={joint_rot.shape[-2]}, dof={dof.shape[-1]}, device={dof.device}")
print(f"match={nominal:.8f}, global_x_0.25={global_shift:.8f}, local_x_0.25={local_shift:.8f}")
PY
```

## 接下来的 R5

R5-0 已确定参考跟踪入口与可复用证据，R5-1 先找一条可修改和发布派生结果的短动作。CMU 官方动捕原始文件是候选，发布方明确允许复制、修改、再分发。需检查具体试次、格式及骨架，原 ASF/AMC 要确认兼容的转换路径。GMR 支持的 LAFAN1 BVH 与发布方的 CC BY-NC-ND 4.0 条款需要合并考虑，派生动作未作为可发布素材。HOLD 是已核验许可的 G1 重定向轨迹，可独立验证跟踪数据输入。 [CMU 数据条款](https://mocap.cs.cmu.edu/faqs.php) · [LAFAN1 数据条款](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) · [HOLD 已核定片段](R3_walk_jog_baseline.md#r3-0-专家数据)

用一条动作先核对根位姿和速度、关节轴、单位、限位、接触脚滑与可视化。跟踪复现时在独立输出目录记录根位置/朝向、关节或连杆误差、足端接触、回合完成、异常终止和实际视频。后续 AMP-G1 继续以冻结模型和 R7 验收协议为准。 [R7 恢复入口](../docs/AMP_G1_FUTURE_PLAN.md#二轮恢复入口)
