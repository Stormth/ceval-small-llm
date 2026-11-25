# 固定评测子集说明

## 来源

- 数据集：`ceval/ceval-exam`
- 官方项目：https://github.com/SJTU-LIT/ceval
- 数据页面：https://huggingface.co/datasets/ceval/ceval-exam
- 使用分割：`val`（答案公开）
- 下载方式：Hugging Face datasets-server rows API
- 数据集 revision：`617524a00b307ff6f9933702f724131fe12ca7ce`（2025-07-27）

`test` 分割的标签不公开，因此本课题不使用 `test` 作为评分数据。

## 固定规则

- 种子：`42`
- 每科题数：`25`
- 学科数：`8`
- 总题数：`200`
- 抽样方式：每个学科独立 RNG，`seed = "{42}:{subject}:val"`，无放回均匀抽样
- 抽样完成后按 `subject, source_id` 排序并冻结
- 后续基线、改进、消融全部使用这一份 `subset.jsonl`

## 学科与题量

| 学科 | 中文名 | 类别 | val 可用 | 选取 |
|---|---|---|---:|---:|
| college_economics | 大学经济学 | Social Science | 55 | 25 |
| accountant | 注册会计师 | Other | 49 | 25 |
| college_programming | 大学编程 | STEM | 37 | 25 |
| electrical_engineer | 注册电气工程师 | STEM | 37 | 25 |
| art_studies | 艺术学 | Humanities | 33 | 25 |
| teacher_qualification | 教师资格 | Social Science | 44 | 25 |
| environmental_impact_assessment_engineer | 环境影响评价工程师 | Other | 31 | 25 |
| civil_servant | 公务员 | Other | 47 | 25 |

## 文件

- `subset.jsonl`：200 道评测题
- `fewshot_examples.jsonl`：24 条 `dev` 示例（8 科 × 3 条），仅用于少样本实验
- `subset_manifest.json`：抽样配置、题量、SHA256、创建时间

当前哈希：

```text
subset.jsonl          sha256 = 92c3e860b1e5cb6fc84b120175e01fa85549c9828b0469182747ee6c568522a1
fewshot_examples.jsonl sha256 = f30496edc83aaaa2545ac70607a631e2906f7008175075f6815a1c62cfb98c43
```

校验命令：

```bash
python scripts/check_subset.py
```

## 重建子集

只有在确认需要重建时才运行：

```bash
python scripts/make_subset.py --config configs/subset.yaml --force
```

重建会改变 `subset.jsonl` 的哈希，之前的实验结果不能直接与新子集比较。正式实验开始后不要重建。

## 许可

C-Eval 数据集遵循 **CC BY-NC-SA 4.0**。使用时需注明来源，不得用于商业用途。模型和推理框架的使用条件以各自官方仓库为准。
