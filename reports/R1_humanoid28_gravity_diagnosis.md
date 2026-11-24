# R1 Humanoid28 有重力失稳诊断

日期：2026-11-23（Asia/Seoul）  
固定条件：CUDA、seed `1000`、摩擦 `0.5`、起始帧 `30`、frame stride `2`、solver `16/4`、不做 reset lift、77 步目标回放

## 红色反馈回路

判定同时检查终止步和 root、feet、joint state 的有限性。当前官方远程资产基线为 `63/77` 步，step 62 终止，所有状态有限。所有变体都沿用这套判定。

## 接触参数

runtime root PhysX view 暴露 18 个 shape。默认 contact offset 为 `0.00545 m`，rest offset 为 `0 m`。

| 变量 | 配置 | 完成步数 | 状态 |
|---|---:|---:|---|
| contact offset | 0.001 / 0.003 / 0.00545 / 0.008 / 0.01 / 0.02 m | 46 / 48 / 49 / 48 / 53 / 56 | 全有限 |
| rest offset | 0 / 0.0005 / 0.001 / 0.002 m | 63 / 47 / 47 / 47 | 全有限 |
| friction combine | average / min / multiply / max | 63 / 63 / 20 / 63 | 全有限 |
| restitution `0.12` | average / min / multiply / max | 68 / 63 / 63 / 57 | 全有限 |

在 `restitution=0.12`、average combine 下，起始帧 `28/30/32` 的结果为 `26/68/27` 步。围绕 `0.12` 的细扫 `0.1175/0.1185/0.1195/0.1205` 为 `68/68/68/67` 步，说明当前 68 步结果不是单点数值偶然性。

`restitution=0.12` 配合 average combine 是当前接触参数中最好的结果，但仍未完成 77 步。rest offset 的微小正值会明显缩短回放，说明当前脚底穿透和接触约束之间存在敏感耦合。

## 质量与惯量推断

runtime mass 与碰撞体积乘 `1000 kg/m³` 精确对应。示例包括：

- foot Cube 体积 `0.00087615 m³`，runtime mass `0.87615 kg`
- hand sphere 半径 `0.04 m`，runtime mass `0.26808 kg`
- head sphere 半径 `0.095 m`，runtime mass `3.59136 kg`

这证明当前 body 的质量来自 PhysX 几何推断，authored body prim 的质量、惯量和密度字段没有提供有效数值。

显式写入当前 runtime mass 和 diagonal inertia 后，回放仍为 `63/77`。把所有 body 的质量和惯量同时缩放为 `0.5` 与 `2.0` 后，结果分别为 `41/77` 与 `32/77`。质量和惯量会显著影响稳定性，当前推断值仍是测试范围内的最佳值，质量来源本身不是唯一失稳原因。

仅写入 mass 和 diagonal inertia 不能完全复现所有 runtime 惯性。torso 的 COM 和 principal axes 仍沿用 USD 的无穷或零值语义，因此显式覆盖结果更适合作为敏感性探针。

进一步提取 runtime 的 COM、完整 inertia tensor 和 principal axes，并全部写入 USD 后，回放降为 `38/77`。这说明显式质量属性的坐标语义或 PhysX 资产重建路径会改变动力学，完整覆盖目前不能作为修复方案。

保持 body-frame diagonal inertia、写入 COM 并使用 identity principal axes 的变体为 `49/77`，仍低于原始几何推断资产的 63 步。

## 执行器映射

原始 runtime actuator tensor 中四个关节的配置为零：

```text
right_elbow, left_elbow, right_knee, left_knee
stiffness = 0
damping = 0
effort_limit = 0
```

USD joint prim 仍包含 elbow `150/15`、knee `300/30` 的 drive 参数。首个参考目标帧中，原始配置的四个计算 torque 均为零。repair 或 override 后，四个 torque 变为约 `4978, -10529, -12273, -6484`，但三条完整回放的 root、feet、target delta 逐步完全一致，均为 `63/77`。

源码审计显示 `ImplicitActuator` 将位置/速度目标直接交给 PhysX，计算 torque 主要用于 tensor 记录。当前 Humanoid AMP 任务的实际物理驱动仍由 USD implicit drive 主导，因此 actuator tensor 增益修正没有改变真实轨迹。

切换到明确 effort 输出的 IdealPD 路径后，低增益 `(10, 1)` 和 `(25, 2)` 分别只完成 13 和 21 步，说明直接 torque 控制路径需要单独的稳定化验证。

## 结论

1. 接触参数对稳定性敏感，restitution average 的 68 步是当前最佳接触变体。
2. 质量和惯量来源已确定为几何体积乘默认密度，当前推断值与显式锁定值给出相同基线。
3. USD drive 到 actuator tensor 的零值是实际缺陷，但 Isaac Lab actuator gain 修正无法改变当前任务的 PhysX 轨迹。
4. 三条链路都无法得到 77 步有重力稳定回放，AMP 短训保持关闭。

下一步需要在 USD 场景创建前验证真实 joint drive 的显式覆盖，或构建明确输出 effort 的 actuator 路径，再复测同一回放条件。

证据脚本和结果：

- `scripts/r1_reference_replay.py`
- `scripts/r1_probe_actuator_mapping.py`
- `scripts/r1_make_humanoid28_mass_override.py`
- `scripts/r1_extract_humanoid28_mass_properties.py`
- `scripts/r1_make_humanoid28_full_mass_override.py`
- `scripts/r1_patch_local_humanoid28_usd_drives.py`
- `runs/r1_structure/replay_cuda_restitution_average.json`
- `runs/r1_structure/replay_explicit_mass_cuda.json`
- `runs/r1_actuator/probe_base.json`
- `runs/r1_actuator/probe_override.json`
