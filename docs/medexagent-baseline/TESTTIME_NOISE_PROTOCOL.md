# 测试时噪声评估协议（MedExAgent Baseline）

日期：2026-08-30
目标：在同一病例集、同一轮数/token 预算下，测量测试时噪声对原始 MedExAgent-8B 的影响，
为后续 shadow-mode Reliability Controller 提供可比较的 clean / noisy 底座。

本协议**只定义评估方案**，不实现控制器、不修改权重、不做 SFT/RL。

---

## 1. 核心不变量（所有条件共享）

1. **同一病例集**：所有条件（clean 与 noisy）使用完全相同的一组病例（见 §3 病例采样）。
2. **同一轮数预算**：`max_turns = 15`（与官方评测默认一致，`evaluation/cli.py:DEFAULT_MAX_TURNS`）。
3. **同一 token 预算**：`max_new_tokens = 4096`（`DEFAULT_MAX_NEW_TOKENS`），patient/judge 同样固定。
4. **同一模型与采样**：`temperature = 0.1`，不做任何权重改动。
5. **同一 judge / patient LLM**：`judge-model = gpt-4.1-mini`，`patient-model = gpt-4.1-mini`（或统一替换为同一兼容端点）。
6. **隐藏标签**：噪声类型只写入 side-channel，绝不进入模型可见的 `messages`（见 §6）。

## 2. 条件矩阵（任务 6）

| 条件 | 噪声 | 用途 | 噪声来源 |
|---|---|---|---|
| **C0 clean** | 无 | baseline | — |
| **C1 p=0.1** | 患者噪声 level 0.1 | 率扫 | 训练分布 7 型 |
| **C2 p=0.2** | 患者噪声 level 0.2 | 率扫 | 训练分布 7 型 |
| **C3 p=0.3** | 患者噪声 level 0.3 | 率扫（对齐论文 p_conv=0.3） | 训练分布 7 型 |
| **C4 seen（per-type）** | 每型强制单型注入 | 分型消融 | `forced_noise_types={type}` |
| **C5 unseen** | 训练分布外新噪声型 | OOD 泛化 | 新定义（§4） |
| **C6 unseen 组合** | 训练分布外组合/错配 | 组合 OOD | §4 |

**"seen" 定义**：`data/noise.py` 的 10 型 —— 患者 7 型 `body_part_swap / symptom_confusion / severity_change / temporal_change / omission / self_diagnosis / vague_answer`，检查 3 型 `body_part_swap / omission / ambiguity`。这是训练时注入的完整集合（训练噪声 `(p_conv, p_exam)=(0.3, 0.1)`，见论文）。

## 3. 病例与噪声种子固定

- **病例集**：从测试集（`*_test_conversations_clean.jsonl`）按 `seed=2026` 无放回抽样固定 `N` 个病例（建议 `N=100`，与项目 DDXPlus 约定一致，见 `docs/EXPERIMENT_PROTOCOL.md`）。抽出的 `case_id` 列表写盘，**全条件复用**，保证 paired-case 可比。
- **噪声种子**：master `noise_seed = 2026`；条件级派生种子明确记录（如 `C1→20261001, C2→20261002, …`），保证复现。
- **多种子区间**：对每个条件额外跑 `seed ∈ {2026, 2027, 2028}`（与项目 `--seeds 2026 2027 2028` 一致），报均值区间与 paired-case 统计。
- **同 turn 预算下的截断语义**：任何条件命中 `max_turns` 或 `max_new_tokens` 而未输出 `[DIAGNOSIS: ...]`，一律记为"截断"，不得在 noisy 条件放宽预算。

## 4. unseen 噪声定义（C5/C6，建议未实现）

新噪声型需**不在** `PATIENT_NOISE_TYPES` / `EXAM_NOISE_TYPES` 内，且明确写为"拟议、未实现"：

- `contradiction`：患者后文推翻先前陈述（"其实我没有胸痛"）。
- `over_report`：患者补充一个档案中不存在但貌似合理的假症状（与 omission 相反）。
- `left_right_swap`：左右/部位混淆（训练型是相邻部位 swap，此为新维度）。
- `unit_mismatch`：单位/量纲错配（如时长 "3 周"→"3 小时"这类训练 `temporal_change` 未覆盖的量纲错配）。
- `hallucinated_allergy`：虚构过敏史。
- `numeric_inversion`：数值倒置。

组合（C6）示例：同 turn 叠加 `omission + body_part_swap`；或把 seen 型施加到训练期**不满足 eligibility** 的病例上（如单症状病例上做 omission，训练期 `check_patient_noise_eligibility` 会判不适用）。

## 5. 注入机制（对接既有代码，不新增训练路径）

- **患者噪声**：在评测循环 `evaluation/runners.py:_eval_conversation_sample`（及 `run_evaluation`）中，患者回复由 `get_patient_response(...)` 产生。协议做法：每病例先 `plan_patient_noise(symptoms, level, rng, forced_noise_types)`（固定种子），在每 turn 调用前用 `get_patient_turn_hint(spec, turn_idx, doctor_msg, rng)` 得到 `(hint_text, noise_type)`，把 hint 注入**患者视角**系统消息（对齐 `train/rl/interaction.py:_llm_response` 的做法）。**噪声标签只记 side-channel。**
- **检查噪声**：检查结果来自 `get_tool_response(...)`；协议做法：命中后对返回的 findings 调 `apply_exam_noise(exam_name, findings, level, rng, forced_noise_type)`，同样记录标签。
- 两者均复用 `data/noise.py` 的既有原语（本阶段已 smoke-test 通过，见 reproducibility 报告 §6）。

## 6. 隐藏标签与禁止读取（任务 7，硬性约束）

每个病例的评估产物为两个独立结构：

1. **模型可见轨迹** `messages`：只含 noisy 的观察文本（noisy 患者回复、noisy 检查 findings）。**不含** noise_type、不含干净回答、不含真实疾病。
2. **评估侧标签** `noise_log`（side-channel，仅评估器读）：
   `[{case_id, turn, noise_type, (clean_text, noisy_text)}]`，外加 `ground_truth_diagnosis`、`gt_tool_findings`（干净 findings 仅用于打分）。

**预测模块（模型 + 控制器）严禁读取**：`noise_type`、干净回答 `clean_text`、真实疾病 `ground_truth_diagnosis`。这些只出现在评分/事后分析里。协议以代码审查点强制：注入函数返回的 `noise_type` 不得回填进 `messages`。

## 7. 指标（可统计性）

每条件 × 每种子输出以下聚合：

| 指标 | 定义 | 来源字段 |
|---|---|---|
| 诊断准确率 strict / lenient | LLM judge Jaccard / 宽松匹配 | `diagnosis_accuracy_strict` / `_lenient` |
| 诊断产出率 | `diagnosis_found_rate` = 产出 `[DIAGNOSIS:]` 的病例占比 | `diagnosis_found` |
| 工具调用奖励 | ToolRL 三档工具匹配 | `tool_call_reward` |
| 工具幻觉率 | 调用不在可用集的比例 | `tool_hallucination_rate` |
| 平均轮数 | `num_turns` = assistant 消息数 | `num_turns` |
| 截断率 | `stop_reason != "diagnosis"` 的病例占比 | `stop_reason` |
| 噪声类型分布 | 每型注入次数 | `noise_log` |

必报：multi-seed 均值区间、paired-case 差值（同一 case 在 clean vs noisy 下的指标差）、每噪声型的效果分解（C4）。

## 8. 复现性记账要求

- 冻结病例清单、每条件噪声种子、每条件完整命令、模型 commit、依赖 `pip freeze`、judge/patient 模型名与端点、全部失败信息 —— 全部落盘到 `docs/medexagent-baseline/` 或 `artifacts/medexagent-baseline/`。
- 未解除阻塞前，**不产出任何数字**，只产出本协议与 reproducibility 报告。
