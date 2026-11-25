# 成员贡献表

| 成员 | 主要职责 | 负责实验 | 对应代码/配置 | 交叉复现 | commit hash |
|---|---|---|---|---|---|
| A | 公共评测流程 + 严格格式提示 | `exp01_strict_prompt` | `configs/experiments/exp1_strict_prompt.yaml` | 复现 B 的 few-shot |  |
| B | 数据与少样本示例 | `exp02_fewshot3` | `configs/experiments/exp2_fewshot.yaml` | 复现 C 的 self-check |  |
| C | 自检流程 + 评分检查 | `exp03_selfcheck` | `configs/experiments/exp3_selfcheck.yaml` | 复现 D 的 vote3 |  |
| D | 采样投票 + 消融 | `exp04_vote3` / `abl_vote*` | `configs/experiments/exp4_vote3.yaml` | 复现 A 的 strict prompt |  |

## 填写说明

- 每人至少负责一项改进的假设、实现、基线对照、结果和错误分析。
- 每人至少复现另一人的一项关键实验；复现时使用 README，不依赖口头补充。
- 记录对应代码、实验编号、commit hash 和复现结果。
- 如果某人同时承担公共模块，写清楚具体模块，不要只写“写代码”。
