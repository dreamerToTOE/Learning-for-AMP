# R1 40.96M checkpoint 冻结评估

日期：2026-11-25

评估对象是从 20.48M checkpoint 恢复训练后累计 40.96M 环境交互步得到的最终 checkpoint `agent_80.pt`。评估协议与 20.48M 阶段一致，保留 42000–42009 纵向基准组，并使用 45000–45099 共 100 个此前未参与评估的独立 seed。

## 结果

| 评估组 | 回合数 | 跑满约 10 秒 | 跌倒 | 完整运行率 | 95% Wilson 区间 | 平均存活(s) | 平均前进(m) | W1 速度距离 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 纵向基准 42000–42009 | 10 | 9 | 1 | 90.0% | 59.6%–98.2% | 9.24 | 10.43 | 0.382 |
| 新增独立 45000–45099 | 100 | 97 | 3 | 97.0% | 91.5%–99.0% | 9.71 | 11.26 | 0.354 |
| 两组合计 | 110 | 106 | 4 | 96.4% | 91.0%–98.6% | 9.67 | 11.19 | — |

新增 100 个独立 seed 的完整运行率为 97%，95% 置信区间下限约 91.5%。合并 110 个回合的完整运行率为 96.4%。完整回合平均前进约 11.3 m，平均前进速度约 1.13 m/s。

## 目标角超限和失败原因

新增组目标角超限比例均值为 0.777%，单回合最高为 2.029%。逐关节统计中 `left_ankle_y` 平均约 10.42%、`right_ankle_y` 平均约 8.69%，`right_shoulder_z` 约 1.58%。这些比例已纳入后续监控，当前没有证据表明它们导致多数回合失败。

新增组 3 个失败 seed 为 `45033、45062、45099`，均因躯干高度低于 0.5 m 触发跌倒终止。动作推理全部有限，没有 NaN、Inf 或其他独立的严重物理异常信号。

## 验收结论

40.96M checkpoint 满足当前冻结阶段的稳定行走验收标准。100 个独立 seed 中至少 90% 回合持续约 10 秒，当前点估计为 97%，置信区间下限也高于 90%。本阶段不继续推进 81.92M，先归档当前策略和评估证据。

## 原始数据和视频

- [汇总结果](r1_eval_freeze_40m/summary.json)
- [纵向基准逐回合数据](r1_eval_freeze_40m/baseline_42000/episodes.json)
- [新增 100 个 seed 逐回合数据](r1_eval_freeze_40m/new_independent_45000/episodes.json)
- [成功回合视频](/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/work/Learning-for-AMP/reports/r1_eval_freeze_40m/success_45000/episode_001.mp4)
- [失败回合视频](/home/ubuntu2004/Documents/Codex/2026-11-22/learning-for-amp-github-https-github/work/Learning-for-AMP/reports/r1_eval_freeze_40m/failure_45033/episode_001.mp4)
