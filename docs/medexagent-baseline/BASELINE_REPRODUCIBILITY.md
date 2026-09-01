# MedExAgent Baseline 复现与阻断报告

日期：2026-08-30
代码固定：`EndlessCG/medexagent` @ `e5b0699`（`git ls-remote` 确认 HEAD == `e5b0699eb109b8dd9f503b503a7b6d68c2c652ae`）
本地只读克隆：`/Users/xr-12345/Desktop/medexagent-baseline`（detached HEAD @ e5b0699）

本阶段目标：建立可比较实验底座；**不实现 Reliability Controller，不修改权重，不进行 SFT/RL**。

---

## 1. 结论速览（逐项状态）

| # | 任务 | 状态 | 说明 |
|---|---|---|---|
| 1 | gated 权重可访问性 | **阻塞** | 模型 gated；本网络无法直连 HF；无 token |
| 2 | 记录依赖版本 | 完成 | 见 §4 |
| 3 | clean 推理 smoke test | **部分** | 噪声模块 smoke test 通过；8B 推理阻塞 |
| 4 | 验证提问/检查/[DIAGNOSIS] 流程 | 代码级完成 | 运行时验证随权重阻塞 |
| 5 | 保存逐病例对话/诊断/工具/轮数/截断 | 未运行 | 评估 harness 已支持 per-sample 字段，缺数据与权重 |
| 6 | 测试时噪声协议 | 完成 | 见 `TESTTIME_NOISE_PROTOCOL.md` |
| 7 | 禁止读取 noise_type/干净回答/疾病 | 完成 | 协议中硬性约束，见协议 §6 |
| 8 | shadow 控制器 I/O | 完成 | 见 `SHADOW_CONTROLLER_INTERFACE.md` |

---

## 2. 硬件 / 环境

| 项 | 值 |
|---|---|
| 主机 | Apple M5 Pro（arm64），Darwin 25.4.0 |
| CPU / 内存 | 18 核 / 64 GB（`hw.memsize=68719476736`） |
| GPU | **无 NVIDIA GPU，无 CUDA**（`nvcc`、`nvidia-smi` 均不存在） |
| PyTorch 加速 | `cuda_available=False`，`mps_available=True`（仅 Apple MPS） |

## 3. Python / 依赖版本（实测）

| 解释器 | 版本 | torch | transformers | verl | accelerate |
|---|---|---|---|---|---|
| `/usr/bin/python3` | 3.9.6 | 2.8.0 (MPS) | 已损坏* | 未安装 | 未安装 |
| miniforge `base` | 3.13.13 | 未安装 | — | — | — |
| conda envs | 3dgs / ai_scientist / vsibench | — | — | — | — |

\* `/usr/bin/python3` 的 `transformers` 导入失败：`tokenizers==0.21.4` 与 `transformers` 要求 `0.22–0.23` 冲突。

**仓库要求（`requirements.txt`，未在本机就绪）**：Python 3.11（README 推荐）、`torch==2.9.1`、`transformers==4.57.1`、`accelerate==1.12.0`、`peft==0.18.1`、`datasets==4.5.0`、`sentence-transformers==5.2.2`、`verl==0.7.0`、`ray==2.53.0`、`sglang==0.5.8.post1`、`openai==2.6.1`、`aiohttp==3.13.3`。

**关键代码级约束**：`data/noise.py` 使用 PEP 604 语法 `int | None`（第 136 行），**要求 Python ≥ 3.10**；系统 Python 3.9.6 因此无法导入该模块（实测 `TypeError: unsupported operand type(s) for |`）。

## 4. 网络连通性（实测）

| 主机 | 结果 |
|---|---|
| github.com | 200 OK（可达，clone 成功） |
| pypi.org | 200 OK |
| huggingface.co | **000 / 15s 超时（阻断）** |
| hf-mirror.com | 200 OK（镜像可达） |
| api.github.com | 403（未认证限流） |

HF 凭证检查：无 `HF_TOKEN` 环境变量、无 `~/.huggingface/token`、无 `~/.cache/huggingface/token`、无 `~/.netrc`、无代理环境变量、无已缓存权重（`~/.cache/huggingface/hub` 不存在）。

## 5. 权重可访问性（任务 1）

- 模型：`medagent/MedExAgent-8B`
- README 明确标注：**"The model repository is gated for research-use acknowledgement."**
- 结论：**当前不可访问**。三重阻断：
  1. `huggingface.co` 本网络不可达；
  2. gated 模型即使走 `hf-mirror.com` 也需 read token；
  3. 本机无任何 HF token。
- 解除路径（外部条件，非本机可解决）：
  1. 在 HF 申请 `medagent/MedExAgent-8B` 访问授权；
  2. 生成 read token 并 `export HF_TOKEN=...`；
  3. 设 `export HF_ENDPOINT=https://hf-mirror.com`（或等效代理）后 `python chat.py --hf-model medagent/MedExAgent-8B` 拉取。

## 6. Smoke test 结果（任务 3）

**A. 8B 推理 smoke test：阻塞**（无权重、无 token、无 CUDA；参考栈 vLLM/SGLang 需 CUDA）。

**B. 噪声模块 smoke test：通过**（`data/noise.py` 纯 stdlib，用 miniforge Python 3.13 运行，无需权重）。已验证：

- `PATIENT_NOISE_TYPES` = 7 类；`EXAM_NOISE_TYPES` = 3 类；
- `plan_patient_noise(symptoms, 0.3, rng=Random(42))` → `noise_turns={'severity_change':4,'temporal_change':3,'symptom_confusion':2}`；
- `get_patient_turn_hint(spec, turn, ...)` 在正确 turn 返回 `(hint_text, noise_type)`，例如 turn=4 → `('When answering: report ... as moderate instead of severe.', 'severity_change')` —— 这即隐藏噪声标签的机制；
- `apply_exam_noise(forced_noise_type=...)` 三型均生效：`body_part_swap`（`lung→pleura`）、`omission`（删除一个分句）、`ambiguity`（套 `Findings are equivocal — ...`）；
- `plan_dataset_noise_assignments` 的 fraction 语义正确：100 患者 @ level 0.1/0.2/0.3 → 10/20/30 名患者被选中；100 患者 ×2 检查槽 @ level 0.1/0.2/0.3 → 20/40/60 槽。

结论：**测试时噪声原语已就绪并可用**；唯一缺口是模型推理与评测数据。

## 7. 流程验证（任务 4，代码级）

由既有只读审计与本次代码复核确认：

- 患者问答入口：`evaluation/generation.py:get_patient_response`（评测侧）与 `train/rl/interaction.py:PatientInteraction.generate_response`（训练侧）。
- 检查解析：`evaluation/generation.py:parse_tool_call_tag_from_output`（`<tool_call>...</tool_call>`）；执行：`get_tool_response` 对 GT findings 解析，命中返回真实 findings。
- 诊断/终止：`[DIAGNOSIS: ...]` 正则（`evaluation/data.py:extract_diagnosis`）；评测循环停止原因集合见 `evaluation/runners.py`：
  `diagnosis` / `tool_call` / `patient_reply` / `max_turns` / `no_tool_no_diagnosis` / `no_patient_client`。
- **注意**：评测循环**没有** RL 侧那种 `truncated_by_length`（token 截断）显式检测；当前唯一"截断"信号是 `stop_reason != "diagnosis"`（即未诊断即结束，含 max_turns 撞顶）。协议据此定义"截断率"。

运行时验证随权重/数据阻塞。

## 8. 冻结命令（解除阻塞后执行）

```bash
# 1) 环境（Python 3.11 venv）
cd /Users/xr-12345/Desktop/medexagent-baseline
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
export HF_TOKEN=<read-token> HF_ENDPOINT=https://hf-mirror.com   # gated 权重
export OPENAI_API_KEY=<judge+patient LLM key>

# 2) 本地 checkpoint 推理（clean）
python -m evaluation \
  --model medagent/MedExAgent-8B \
  --data dd=data/conversations/processed/ddxplus_test_conversations_clean.jsonl \
        pmc=data/conversations/processed/pmc_test_conversations_clean.jsonl \
  --eval-types diagnosis conversation \
  --judge-model gpt-4.1-mini --patient-model gpt-4.1-mini \
  --max-turns 15 --max-new-tokens 4096 \
  --no-wandb --output eval_results.json
```

（或 `--api-model <served>` + `--api-base-url http://...:8000/v1`，经远程 GPU 服务。）

## 9. 完整阻塞清单（验收「权重/数据/API/硬件缺失需明确报告」）

| 阻塞 | 影响 | 解除条件 |
|---|---|---|
| gated 权重 + HF 不可达 + 无 token | 无法加载 8B 模型 | 申请授权 + read token + hf-mirror/代理 |
| 测试数据缺失 | `data/conversations/processed/*_clean.jsonl` 未随仓库发布（gitignore） | 生成（`python -m data.pipeline`）或从作者处获取 |
| judge/patient LLM 无 key | gpt-4.1-mini 不可用 | `OPENAI_API_KEY` 或兼容端点 |
| 无 CUDA | vLLM/SGLang/verl 不可用；8B 推理仅 MPS/CPU（慢） | 远程 GPU 或接受慢速 MPS |
| 无匹配 venv | 系统 py 3.9 过旧，miniforge 缺 torch | 建 Python 3.11 venv + `pip install -r requirements.txt` |

## 10. 验收对照

| 验收标准 | 现状 |
|---|---|
| 原始 baseline 能独立运行 | ❌ 阻塞（权重+数据+API+硬件） |
| clean/noisy 使用相同病例 | 协议已固化（固定病例集复用） |
| 每条患者回答保留隐藏噪声标签、不给模型 | 协议已固化（side-channel label） |
| 可统计准确率/轮数/截断率/噪声类型 | 协议已固化（指标表）；未运行 |
| 命令/配置/版本/失败均记录 | ✅ 本文件 + 两协议文件 |
