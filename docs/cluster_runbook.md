# 集群运行手册

本手册给出一条从零到跑完所有实验的完整路径。所有命令都在仓库根目录执行。

> 前提：本机不安装模型、不跑推理。真实实验全部在集群完成。

---

## 0. 上机先做三项检查

```bash
# 1) GPU 是否可见、驱动是否正常
nvidia-smi

# 2) 是否有 conda
conda --version

# 3) 计算节点能否访问 Hugging Face
curl -I https://huggingface.co --max-time 10
```

判断方式：

- `nvidia-smi` 能看到卡、显存和 CUDA 版本 → 记录右上角的 `CUDA Version`，第 2 步要用。
- `curl` 超时 → 计算节点大概率离线，走第 5 步的「离线模型」方案。

---

## 1. 获取代码

```bash
git clone https://github.com/Stormth/ceval-small-llm.git
cd ceval-small-llm
```

---

## 2. 环境安装

### 方式 A：自己建 Conda 环境（通用）

```bash
conda create -n ceval python=3.11 -y
conda activate ceval
```

PyTorch 必须和集群驱动匹配。按第 0 步看到的 CUDA 版本选索引：

| 集群 CUDA | 安装命令 |
|---|---|
| 12.4 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

然后装其余依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

### 方式 B：集群已有 PyTorch 模块

```bash
module avail 2>&1 | grep -i -E "pytorch|cuda|python"
module load pytorch/2.x.x        # 按实际模块名填写
```

这种情况跳过 `pip install torch`，直接：

```bash
pip install -r requirements.txt
pip install -e .
```

### 验证环境

```bash
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); print(transformers.__version__)"
```

必须看到 `True` 和大于 0 的卡数。**如果只有 `False`，不要继续往下跑**，说明装成了 CPU 版 PyTorch。

---

## 3. 国内集群：配置 Hugging Face 镜像

如果直连 `huggingface.co` 超时，在**登录节点**设置镜像后再下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

建议写进 `~/.bashrc` 或作业脚本，否则每个新 shell 都要重设。

---

## 4. 预下载模型（推荐）

先下到共享目录，避免每个计算任务重复下载、也避免计算节点无网。

```bash
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="/shared/models/Qwen2.5-0.5B-Instruct",
)
print("done")
PY
```

记录校验值，写进报告：

```bash
python scripts/model_manifest.py \
  --model-dir /shared/models/Qwen2.5-0.5B-Instruct \
  --output data/model_manifest.json
```

同时固定模型版本（`revision` 建议填 commit hash，而不是 `main`）：

```yaml
# configs/base.yaml
model:
  revision: "<model-commit-hash>"
```

---

## 5. 自检（不加载模型）

```bash
python scripts/env_report.py | tee data/env_report.json
python scripts/check_subset.py
python -m pytest -q
```

期望结果：

- `env_report.json` 里有 GPU 型号和显存；
- `check_subset.py` 输出 `OK`，200 题、8 学科、SHA256 匹配；
- `pytest` 全绿（当前 14 项）。

---

## 6. 冒烟测试（不加载真实模型）

先验证数据 → 提示 → 解析 → 日志 → 汇总整条链路，再花时间跑真模型。

```bash
python scripts/run_eval.py \
  --config configs/experiments/baseline.yaml \
  --mock --limit 8 \
  --run-dir outputs/smoke_baseline --overwrite

python scripts/merge_shards.py --run-dir outputs/smoke_baseline
```

`--mock` 的输出**不能当实验结果**，只用于确认代码能跑通。

---

## 7. 正式基线

### 单卡 + 在线下载

```bash
python scripts/run_eval.py \
  --config configs/experiments/baseline.yaml \
  --run-dir outputs/exp00_baseline
```

### 单卡 + 本地模型（离线节点用这个）

```bash
python scripts/run_eval.py \
  --config configs/experiments/baseline.yaml \
  --model-path /shared/models/Qwen2.5-0.5B-Instruct \
  --offline \
  --run-dir outputs/exp00_baseline
```

跑完检查：

```bash
cat outputs/exp00_baseline/summary.md
```

重点看四个数：`accuracy`、`format_valid_rate`、`avg_latency_sec`、`avg_calls_per_question`。

**这一步的结果就是全组的基线。** 先不要急着跑其余实验，先确认：

1. 200 题全部有记录（`per_question.jsonl` 行数 = 200）；
2. 格式有效率不是 0（如果接近 0，说明提示模板或解析规则有问题）；
3. `run_meta.json` 里记录了 git commit。

---

## 8. 多卡运行

每张卡一个进程，按 `sample_id` 分片，跑完自动合并。

```bash
bash scripts/run_parallel.sh \
  configs/experiments/baseline.yaml 4 outputs/exp00_baseline
```

三个参数依次是：配置、分片数（等于卡数）、输出目录。

常用变体：

```bash
# 指定 Python 解释器（conda 环境没激活时）
PYTHON=/opt/conda/envs/ceval/bin/python \
bash scripts/run_parallel.sh configs/experiments/baseline.yaml 4 outputs/exp00_baseline

# 冒烟：多进程 + mock
EXTRA_ARGS="--mock" bash scripts/run_parallel.sh \
  configs/experiments/baseline.yaml 2 outputs/smoke_parallel
```

如果某个分片失败，先看日志：

```bash
tail -30 outputs/exp00_baseline/shard-*.log
```

---

## 9. 跑完所有实验

建议顺序：先基线，再四个人各自的改进，最后消融。

```bash
# 基线
bash scripts/run_parallel.sh configs/experiments/baseline.yaml 4 outputs/exp00_baseline

# 改进 1：严格输出格式
bash scripts/run_parallel.sh configs/experiments/exp1_strict_prompt.yaml 4 outputs/exp01_strict_prompt

# 改进 2：3-shot 少样本
bash scripts/run_parallel.sh configs/experiments/exp2_fewshot.yaml 4 outputs/exp02_fewshot3

# 改进 3：两轮自检
bash scripts/run_parallel.sh configs/experiments/exp3_selfcheck.yaml 4 outputs/exp03_selfcheck

# 改进 4：3 次采样投票
bash scripts/run_parallel.sh configs/experiments/exp4_vote3.yaml 4 outputs/exp04_vote3

# 消融：采样 1/5 次（3 次已在 exp04 得到，可直接复用）
bash scripts/run_parallel.sh configs/ablations/vote1.yaml 4 outputs/abl_vote1
bash scripts/run_parallel.sh configs/ablations/vote5.yaml 4 outputs/abl_vote5
```

汇总所有结果：

```bash
python scripts/collect_results.py
# outputs/comparison.csv
# outputs/comparison.md
```

逐题对比（看改对/改错了哪些题）：

```bash
python scripts/compare_runs.py \
  --baseline outputs/exp00_baseline \
  --candidate outputs/exp01_strict_prompt
```

### 时间预估（0.5B，200 题，单卡）

| 实验 | 每题调用次数 | 预估耗时 |
|---|---:|---|
| 基线 / 严格提示 / 少样本 | 1 | 5—15 分钟 |
| 自检 | 2 | 10—30 分钟 |
| 投票 3 次 | 3 | 15—45 分钟 |
| 投票 5 次 | 5 | 25—75 分钟 |

实际取决于卡型和排队情况。多卡分片会近似按卡数线性加速。

---

## 10. 用调度器提交

### Slurm 单卡

```bash
CONFIG=configs/experiments/baseline.yaml \
RUN_DIR=outputs/exp00_baseline \
sbatch scripts/slurm_single_gpu.sbatch
```

### Slurm 多卡

```bash
CONFIG=configs/experiments/baseline.yaml \
RUN_DIR=outputs/exp00_baseline \
NUM_SHARDS=4 \
sbatch scripts/slurm_multi_gpu.sbatch
```

提交前必须改脚本里的三处：

1. `#SBATCH` 资源行（`--gres`、`--mem`、`--time`、`--partition`）；
2. `module load` / `conda activate`（取消注释并改成实际名称）；
3. 需要离线时，给 `run_eval.py` 加 `--model-path` 和 `--offline`。

查看作业状态：

```bash
squeue -u $USER
```

### PBS 集群

```bash
qsub scripts/pbs_example.pbs
```

脚本里的 `#PBS -l` 资源行和 `conda activate` 同样要改。

---

## 11. 归档结果（重要）

`outputs/` 在 `.gitignore` 里，不会被提交。但课题要求提交逐题记录，所以要单独归档：

```bash
mkdir -p results/exp00_baseline
cp outputs/exp00_baseline/summary.json \
   outputs/exp00_baseline/summary.md \
   outputs/exp00_baseline/per_question.jsonl \
   outputs/exp00_baseline/run_meta.json \
   results/exp00_baseline/

git add results/ && git commit -m "Add baseline results (200 questions)"
git push
```

每个实验都按这个方式归档一份，最终 `results/` 就是全部实验证据。

---

## 12. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `bash\r: No such file or directory` | 脚本被转成 CRLF | 仓库已用 `.gitattributes` 固定 LF；若仍出现，执行 `sed -i 's/\r$//' scripts/*.sh` |
| `CUDA out of memory` | 显存不足或残留进程 | 0.5B 不该出现；检查是否别的进程占卡，`nvidia-smi` 确认后换卡 |
| `torch.cuda.is_available()` 为 `False` | 装了 CPU 版 PyTorch | 按第 2 步重装对应 CUDA 版本 |
| 下载模型卡住或超时 | 节点无法访问 HF | 设 `HF_ENDPOINT=https://hf-mirror.com`，或改用 `--model-path` + `--offline` |
| 格式有效率接近 0 | 提示模板或解析规则不匹配该模型 | 先看 `per_question.jsonl` 里的 `raw_output`，**不要直接改解析器提分**，改动要作为独立实验并记录 |
| `No summary.json files found` | `collect_results.py` 在没有结果时运行 | 先跑完至少一个实验再汇总 |
| 分片数大于题数 | 分片过细 | 分片数不超过 8，200 题除以卡数要 ≥20 才有意义 |

---

## 13. 跑之前最后确认

- [ ] `torch.cuda.is_available()` 为 `True`
- [ ] `check_subset.py` 通过，200 题未变
- [ ] 模型 `revision` 已固定并写进 `run_meta.json`
- [ ] 基线跑完再开始改进实验
- [ ] 所有实验共用同一份 `data/subset.jsonl` 和同一个解析器
- [ ] 每份结果都归档到 `results/` 并提交 Git
