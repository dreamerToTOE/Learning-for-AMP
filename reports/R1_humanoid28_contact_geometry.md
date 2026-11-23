# R1 Humanoid28 接触几何与质量来源审计

日期：2026-11-23（Asia/Seoul）  
环境：Isaac Sim 4.5.0、Isaac Lab v2.1.1、CUDA `cuda:0`、seed `1000`  
回放条件：Walk motion、起始帧 `30`、stride `2`、摩擦 `0.5`、solver `16/4`、不做 reset lift、USD drive override

## 审计结论

左右脚碰撞盒来自 `Props/instanceable_meshes.usd`。原始 Cube extent 为 `±(0.0885, 0.045, 0.0275) m`，局部平移为 `(0.045, 0, -0.0225) m`，未考虑姿态时盒体底面位于 foot link 原点下方 `0.05 m`。frame 30 的参考姿态叠加脚链接旋转后，右脚和左脚碰撞盒底部约为 `-0.0295 m` 和 `-0.0641 m`，存在明显地面穿透风险。

运行时路径为：

```text
/World/envs/env_0/Robot/right_foot/collisions/right_foot
/World/envs/env_0/Robot/left_foot/collisions/left_foot
```

两个 prim 都是 instance proxy。直接修改 prototype 变换会触发 USD instancing 限制，运行时修改没有形成可复现的 PhysX 资产修复。

## 质量与惯量来源

Humanoid28 body prim 的 authored `physics:mass` 和 `physics:diagonalInertia` 为零值或无穷中心值，mesh payload 没有显式质量字段，body 密度字段为零。PhysX 加载碰撞几何后生成了有限 runtime mass/inertia。当前 runtime 总质量为 `35.731690 kg`，15 个 body 的质量和左右肢体惯量对称，证据在 `runs/r1_structure/humanoid28_physics_audit.json`。

改变碰撞几何会同时影响接触形状和 PhysX 推断的质量/惯量。局部复制 payload 即使保持可见几何和 runtime mass 数值，也可能改变资产组合和物理初始化路径，需要单独验证。

## 几何修复试验

| 资产路径 | 处理 | 结果 |
|---|---|---:|
| 官方远程资产 | 原始 payload、CUDA、全状态有限 | 63 步，step 62 终止 |
| 展平 USD，offset 0 | 展平所有 payload 与 prototypes | 24 步，状态有限 |
| 展平 USD，脚底上移 5 cm | 预物理修改展平 prototype | 24 步，状态有限 |
| 保留 root→mesh 分层，offset 0 | 本地复制 root 与 mesh，保留实例拓扑 | 63 步，状态有限 |
| 分层 mesh，脚底上移 2.5 mm | 仅修改本地 mesh collision transform | 56 步，状态有限 |
| 分层 mesh，脚底上移 5 mm | 仅修改本地 mesh collision transform | 57 步，状态有限 |
| 分层 mesh，脚底下移 5 mm | 仅修改本地 mesh collision transform | 52 步，状态有限 |
| 远程 payload，脚底上移 4 cm | 预物理修改解析层 | 23 步，状态有限 |
| 远程 payload，脚底下移 2 cm | 预物理修改解析层 | 25 步，状态有限 |

统一上移或下移都没有恢复完整 77 步。展平资产的物理组成变化更大，不能作为最终修复资产。

## 当前可复现基线

`runs/r1_structure/replay_cuda_baseline_finite.json` 是当前有效的有重力参考回放基线：

```text
steps_requested: 77
steps_completed: 63
terminated_step: 62
all_finite: true
seed: 1000
start_frame: 30
ground_friction: 0.5
solver: 16/4
```

回放脚本现在逐步检查 root、feet 和 joint state 的有限性。遇到 NaN 或 Inf 时记录 `nonfinite_step` 并停止，旧的“77 步但状态已经 NaN”结果不再作为成功证据。

## 阶段判定

接触几何 payload、实例原型关系和质量/惯量来源已经完成审计，预物理局部修改工具已经加入脚本。几何改动没有达到 77 步有重力稳定回放，执行器和接触动力学的联合失稳仍然存在。AMP 短训保持关闭，后续继续围绕物理初始化、接触参数和执行器映射做小规模诊断。

证据脚本：

- `scripts/r1_inspect_humanoid_payload.py`
- `scripts/r1_probe_collision_path.py`
- `scripts/r1_make_humanoid28_collision_override.py`
- `scripts/r1_patch_local_humanoid28_mesh.py`
- `scripts/r1_reference_replay.py`
