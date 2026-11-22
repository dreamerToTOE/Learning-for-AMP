# 论文与上游参考（候选，需 R0 校验）

> 本文件只汇总学习与潜在代码复用来源。**仓库列出 ≠ 兼容性已验证，也不代表用户已获得对应资产授权。**

| 项目 | 参考入口 | 拟使用场景 | 接手时需要核对 |
|---|---|---|---|
| 原始 AMP / Peng 等（2021） | https://arxiv.org/abs/2104.02180 | 算法基线：判别器风格奖励、任务奖励、混合专家运动 | 论文版本、原始实现、许可证 |
| Multi-AMP / 2022 | https://arxiv.org/abs/2203.14912 | 多技能运动风格与条件化/多先验思路 | 是否适合 G1，不能直接声称创新 |
| ASE / 2022 | https://arxiv.org/abs/2205.01906 | 多技能分布覆盖与技能潜变量（拓展阅读） | 不作为 R0–R2 必做 |
| Isaac Lab 官方 | https://github.com/isaac-sim/IsaacLab | 官方 Humanoid AMP 入口及训练示例 | 与本机 checkout tag/commit 的实际对应 |
| Isaac Lab v2.1.1 任务索引 | https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/environments | 版本参考 | 是否实际安装了示例及运动数据 |
| legged_lab | https://github.com/zitongbai/legged_lab | G1 AMP/DeepMimic 候选工程 | 对应版本、修改的 RL 框架、License、资产 |
| AMP Running baseline | https://github.com/Jiarui-Xie/AMP_Running_baseline | G1 跑步相关方案候选 | 数据、运动任务、资产与依赖 |

R0 必须补齐：
1. 具体选中上游的完整 Git SHA、tag、许可证、修改许可；
2. 实际可用的模型与专家运动数据下载/权限；
3. Isaac Sim / Isaac Lab / skrl 或 RSL-RL 的兼容版本矩阵；
4. 可重复验证的训练/测试入口；
5. 是否需要下载额外大文件、是否有潜在收费或登录限制。

参考材料只供设计与对照，未经验证不得把它们的结果写入本仓库的实验结果。
