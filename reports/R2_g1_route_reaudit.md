# R2 G1 路线复审

日期：2026-11-24（Asia/Seoul）

## 结论

G1 路线具备可行候选，但当前仍需独立环境和资产闭环。现有 R1 Isaac Sim 4.5 / Isaac Lab 2.1.1 环境不应直接复用候选 G1 工程。

## 候选项目

| 项目 | AMP/G1 状态 | 版本与资产 | 当前判断 |
|---|---|---|---|
| `zitongbai/legged_lab` | 有 `LeggedLab-Isaac-AMP-G1-v0` 和 G1 29 DoF 配置，使用 forked `rsl_rl` AMP 分支 | README 要求 Isaac Lab 2.3.2，历史记录覆盖 2.3.1，要求 Git LFS 下载运动数据和 USD | 最优代码候选，需新建 Isaac Sim 5.1 / Isaac Lab 2.3.2 隔离环境并验证 LFS 对象 |
| `Jiarui-Xie/AMP_Running_baseline` | 有 G1 AMP 任务、`checkpoints/model_6200.pt` 和演示视频 | README 说明 Isaac Lab 2.3.1、RSL-RL 3.2.0，motion/USD/checkpoint 使用 Git LFS，后续仿真 checkpoint 未完成真机验证 | 最优 checkpoint 验证候选，需审计资产 hash 和许可证 |
| `unitreerobotics/unitree_rl_lab` | 支持 G1-29DoF，但当前仓库没有 AMP 任务 | Apache-2.0，USD 从 `unitree_model` HuggingFace 数据集下载，URDF 路径要求 Isaac Sim 5.0 或更高 | 适合核对 G1 资产和常规任务接口，不作为 AMP 主路线 |
| `YanjieZe/GMR` | 支持 Unitree G1 29 DoF retarget | 支持 AMASS、LAFAN1 等输入，原始人体数据仍受各自许可证约束 | 作为专家动作 retarget 工具，不是训练环境 |

## 证据与限制

- `legged_lab` README 明确要求 Isaac Lab `v2.3.2`、Git LFS 和 forked `rsl_rl` AMP 分支，并给出 `LeggedLab-Isaac-AMP-G1-v0` 的 `max_iterations 50000` 训练入口。
- `AMP_Running_baseline` README 给出 G1 BASE checkpoint 和 AMP play/train 命令，但区分了已展示的 BASE checkpoint 与尚未真机验证的后续仿真 checkpoint。
- `unitree_rl_lab` README 要求单独下载 `unitree_model` USD 或 `unitree_ros` URDF，并明确 URDF 方法适用于 Isaac Sim 5.0 及以上。
- GMR README 列出 Unitree G1 29 DoF 和多种原始动作格式，代码许可证不能替代 AMASS、OMOMO、LAFAN1 等原始数据授权。

## 建议的切换门槛

1. 独立安装 Isaac Sim 5.1、Isaac Lab 2.3.2、对应 RSL-RL AMP 分支和 Git LFS。
2. 记录候选仓库 commit、LFS object hash、USD 与 motion 文件 hash、许可证和下载来源。
3. 先运行 `AMP_Running_baseline` 或 `legged_lab` 的官方 G1 checkpoint play/import smoke。
4. 验证 G1 平地 AMP 视频或轨迹与 checkpoint 能在本机复现，再决定是否切换 R1 后续路线。

当前不把候选 README、演示视频或 checkpoint 名称当作本项目训练结果，也不把 G1 仿真验证写成真机验证。

参考链接：

- <https://github.com/zitongbai/legged_lab>
- <https://github.com/Jiarui-Xie/AMP_Running_baseline>
- <https://github.com/unitreerobotics/unitree_rl_lab>
- <https://github.com/YanjieZe/GMR>
