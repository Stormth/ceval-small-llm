# C-Eval 小语言模型评测工程

这是一个面向课程课题的**可复现 C-Eval 评测工程**：在本地或集群上用 PyTorch + Transformers 加载一个小型中文指令模型，完成固定子集上的基线评测、成员改进和消融分析。

> 结论口径：本项目报告的是 **“C-Eval 本组自选子集”** 上的结果，不是完整 C-Eval 官方成绩，也不是独立测试集成绩。

## 1. 已固定的实验要素

| 项目 | 当前设置 |
|---|---|
| Baseline | `Qwen/Qwen2.5-0.5B-Instruct` |
| 评测集 | C-Eval `val` 分割，固定 200 题 |
| 学科 | 8 个学科 × 每科 25 题 |
| 随机种子 | `42`，按学科独立抽样 |
| 评分 | 生成答案 + 最后一行固定规则解析 |
| 解码 | greedy，`max_new_tokens=128` |
| 运行 | 每张 GPU 一个进程，按题目分片 |

子集已经生成并纳入仓库：

- `data/subset.jsonl`：200 道固定题目
- `data/subset_manifest.json`：种子、学科、题量、数据集 revision、SHA256
- `data/fewshot_examples.jsonl`：24 条 dev 示例，供少样本实验使用
- 这些文件在跑模型前就已经冻结，后续不得按结果换题

如果要把 baseline 换成 1.5B，复制一份配置并只改 `experiment.id` 和 `model.name_or_path`，然后用同一份 `data/subset.jsonl` 重新跑基线。不同模型的结果不能直接混在同一张对照表里。

## 2. 目录结构

```text
ceval-small-llm/
├── ceval_eval/                 # 数据、提示、解析、推理、指标
├── configs/
│   ├── base.yaml               # 模型与默认生成配置
│   ├── subset.yaml             # 固定子集抽样配置
│   ├── experiments/            # 基线 + 4 项改进
│   └── ablations/              # 采样次数消融
├── data/                       # 固定子集与少样本示例
├── scripts/
│   ├── make_subset.py          # 重建子集（一般不需要重跑）
│   ├── check_subset.py         # 校验 SHA256 和题量
│   ├── run_eval.py             # 单分片评测入口
│   ├── run_parallel.sh         # 多卡数据并行
│   ├── merge_shards.py         # 合并分片并生成汇总
│   ├── collect_results.py      # 生成实验对比表
│   └── slurm_*.sbatch / *.pbs  # 调度器模板
├── tests/                      # 解析、提示、指标、子集测试
├── docs/                       # 实验协议与分析模板
└── outputs/                    # 运行结果（不提交 Git）
```

## 3. 集群环境安装

> 完整的从零到跑完实验的步骤、时间预估、调度器提交和报错排查，见 **[docs/cluster_runbook.md](docs/cluster_runbook.md)**。下面是精简版。

下面是通用 Conda 流程，具体 CUDA 版本按集群 `nvidia-smi` 调整。

```bash
git clone <your-github-repo-url>
cd ceval-small-llm

conda create -n ceval python=3.11 -y
conda activate ceval

# 先安装与集群 CUDA 匹配的 PyTorch，例如 CUDA 12.4：
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 再安装其余依赖
pip install -r requirements.txt
pip install -e .
```

如果集群已经预装 CUDA 版 PyTorch，跳过第一行 `pip install torch`，先确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

必须看到 `True` 和大于 0 的 GPU 数量。也可以把环境信息保存下来：

```bash
python scripts/env_report.py | tee data/env_report.json
```

模型下载到共享目录后，记录文件校验值：

```bash
python scripts/model_manifest.py \
  --model-dir /shared/models/Qwen2.5-0.5B-Instruct \
  --output data/model_manifest.json
```

## 4. 第一次运行：先做不加载模型的冒烟测试

```bash
python scripts/check_subset.py
python -m pytest -q

python scripts/run_eval.py \
  --config configs/experiments/baseline.yaml \
  --mock --limit 8 \
  --run-dir outputs/smoke_baseline --overwrite

python scripts/merge_shards.py --run-dir outputs/smoke_baseline
```

多进程冒烟测试可以加：

```bash
EXTRA_ARGS="--mock" bash scripts/run_parallel.sh \
  configs/experiments/baseline.yaml 2 outputs/smoke_parallel
```

`--mock` 只验证数据、提示、解析、日志、汇总链路，**不能当作实验结果**。

## 5. 真实基线

单卡：

```bash
python scripts/run_eval.py \
  --config configs/experiments/baseline.yaml \
  --run-dir outputs/exp00_baseline
```

多卡（例如 4 张卡）：

```bash
bash scripts/run_parallel.sh \
  configs/experiments/baseline.yaml 4 outputs/exp00_baseline
```

多卡脚本的行为：

- 每张 GPU 启动一个独立进程；
- 每个进程加载一份模型，按 `sample_id` 分片处理题目；
- 每个进程写自己的 `per_question.shard-XXX.jsonl`；
- 全部结束后自动合并为 `per_question.jsonl`、`summary.json`、`summary.md`。

0.5B 模型每卡一份副本压力很小；如果换成 1.5B/3B，先确认单卡显存和加载时间。

## 6. 实验矩阵

| 实验 | 配置 | 改动 | 负责人 |
|---|---|---|---|
| 基线 | `baseline.yaml` | 官方风格零样本提示 | 共享 |
| 改进 1 | `exp1_strict_prompt.yaml` | 严格输出格式，只允许选项字母 | A |
| 改进 2 | `exp2_fewshot.yaml` | 每科 3 条 dev 少样本示例 | B |
| 改进 3 | `exp3_selfcheck.yaml` | 两轮：作答 + 复核修正 | C |
| 改进 4 | `exp4_vote3.yaml` | 每题采样 3 次，多数投票 | D |
| 消融 1 | `ablations/vote1.yaml` | 采样 1 次 | D |
| 消融 2 | `ablations/vote3.yaml` | 采样 3 次 | D |
| 消融 3 | `ablations/vote5.yaml` | 采样 5 次 | D |

运行全部实验：

```bash
bash scripts/run_parallel.sh configs/experiments/exp1_strict_prompt.yaml 4 outputs/exp01_strict_prompt
bash scripts/run_parallel.sh configs/experiments/exp2_fewshot.yaml 4 outputs/exp02_fewshot3
bash scripts/run_parallel.sh configs/experiments/exp3_selfcheck.yaml 4 outputs/exp03_selfcheck
bash scripts/run_parallel.sh configs/experiments/exp4_vote3.yaml 4 outputs/exp04_vote3
bash scripts/run_parallel.sh configs/ablations/vote1.yaml 4 outputs/abl_vote1
bash scripts/run_parallel.sh configs/ablations/vote5.yaml 4 outputs/abl_vote5
```

汇总对比表：

```bash
python scripts/collect_results.py
# outputs/comparison.csv
# outputs/comparison.md
```

查看某项改进相对基线的“改对/改错”题目：

```bash
python scripts/compare_runs.py \
  --baseline outputs/exp00_baseline \
  --candidate outputs/exp01_strict_prompt
```

输出默认写到 `outputs/exp01_strict_prompt/comparison_vs_baseline.md`，可直接用于错误案例分析。

## 7. 离线或本地模型

如果计算节点不能联网，先在可联网节点下载模型，或把模型目录同步到共享存储：

```bash
python scripts/run_eval.py \
  --config configs/experiments/baseline.yaml \
  --model-path /shared/models/Qwen2.5-0.5B-Instruct \
  --offline \
  --run-dir outputs/exp00_baseline
```

也可以设置：

```bash
export HF_HOME=/shared/hf-cache
export HF_HUB_OFFLINE=1
```

`--offline` 会设置 `local_files_only`，避免计算节点访问网络。

## 8. 调度器

仓库提供了模板，不绑定具体集群：

- `scripts/slurm_single_gpu.sbatch`
- `scripts/slurm_multi_gpu.sbatch`
- `scripts/pbs_example.pbs`

提交前需要改三处：

1. `#SBATCH` / `#PBS` 的资源行；
2. `module load` 或 `conda activate`；
3. `CONFIG` 和 `RUN_DIR` 环境变量。

Slurm 单卡示例：

```bash
CONFIG=configs/experiments/baseline.yaml \
RUN_DIR=outputs/exp00_baseline \
sbatch scripts/slurm_single_gpu.sbatch
```

Slurm 多卡示例：

```bash
CONFIG=configs/experiments/baseline.yaml \
RUN_DIR=outputs/exp00_baseline \
NUM_SHARDS=4 \
sbatch scripts/slurm_multi_gpu.sbatch
```

如果集群使用 `sbatch --array`，也可以把每个分片作为一个独立任务提交，最后统一运行 `merge_shards.py`。

## 9. 结果文件

每次运行会生成：

```text
outputs/exp00_baseline/
├── per_question.shard-000.jsonl   # 单分片逐题记录
├── per_question.jsonl             # 合并后的逐题记录
├── summary.json                   # 机器可读汇总
├── summary.md                     # 人可读汇总
├── summary.shard-000.json         # 分片临时汇总
└── run_meta.json                  # 配置、模型、commit、环境
```

正式实验记录表模板在 `docs/experiment_log.csv`。逐题记录至少包含：

- 学科、样本编号、完整题目与选项；
- 完整输入 prompt；
- 原始输出；
- 提取选项、参考选项、是否正确；
- 格式是否有效、是否截断、耗时、调用次数；
- 异常信息（如果有）。

## 10. 评分规则

`ceval_eval/parsing.py` 的规则固定为：

1. 只看最后一行非空文本；
2. 匹配“答案：X”“最终答案是 X”“所以选 X”或单独一行 `X`；
3. 找不到唯一选项就记为格式无效，并计为错误；
4. 不在整段推理中搜索字母，避免把中间提到的选项误当最终答案。

这个规则在基线前已经冻结。所有实验共用同一份 `subset.jsonl` 和同一个解析器，不能按结果修改。

## 11. 复现纪律

- 每次正式运行都记录 `run_meta.json` 里的 git commit；
- 不根据结果换题、删题或改分母；
- 失败、截断、超时、解析失败都要保留在日志里；
- 少样本示例只能来自 `data/fewshot_examples.jsonl`，不能来自评测子集；
- 如果要调整方法，记录调整过程，不要只报告最有利的一次；
- 结论限定在 200 题目集上，不表述为完整 C-Eval 成绩；
- 使用 AI 编程辅助时，在报告中说明用途，成员仍需能解释代码和评分规则。

## 12. 数据与模型来源

- C-Eval 官方项目：https://github.com/SJTU-LIT/ceval
- C-Eval 数据集：https://huggingface.co/datasets/ceval/ceval-exam
- Qwen2.5-0.5B-Instruct：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

C-Eval 数据集遵循 CC BY-NC-SA 4.0；模型使用条件以官方模型卡为准。代码与数据来源需在最终报告中标注。
