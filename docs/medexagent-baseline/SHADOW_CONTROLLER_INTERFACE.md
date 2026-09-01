# Shadow-mode Reliability Controller 输入输出接口

日期：2026-08-30
用途：定义后续可靠性控制器的 **I/O 契约**，供 shadow-mode（只观测、不改轨迹）评估其判别能力。
本文件**只定义接口，不实现控制器**。

---

## 1. 设计原则

1. **Shadow = 旁路**：控制器在每个患者观察点输出判断，但不影响模型实际收到的消息，不改权重、不打断 `[DIAGNOSIS: ...]`。
2. **信息隔离**：控制器与预测模块一样，**看不到** `noise_type` / 干净回答 / 真实疾病（见 §5）。
3. **可评分**：控制器输出的置信度可用 hidden `noise_log` 事后算 AUROC / AP，验证"能否察觉噪声"。
4. **可接入**：接口落在既有边界上，未来接进 `train/rl/interaction.py:generate_response` 与 `evaluation/runners.py:_eval_conversation_sample` 无需改动模型。

---

## 2. 输入（每个患者观察点调用一次）

```python
@dataclass
class ControllerInput:
    # —— 模型侧合法可见的历史（doctor 视角） ——
    history: list[dict]            # 完整 messages（system/user/assistant/tool）
    doctor_last: str               # 当前医生回合文本（含 <tool_call> 若有）
    patient_reply: str             # 当前患者观察文本（可能已含噪声）
    turn_index: int                # 当前患者回合序号（从 0 起）

    # —— 合法可用的环境元数据（不含噪声/GT） ——
    available_tools: list[str]     # 系统提示中的可用检查名（含 distractor）
    exam_results_so_far: list[str] # 已返回的检查 findings（可能已含噪声）
    patient_profile: dict          # demographics/medical_history/self_reported_symptoms
                                   #   （即患者"已知"信息，不含真实疾病）
```

## 3. 输出（控制器一次判断）

```python
@dataclass
class ControllerOutput:
    reliability: float             # 对 patient_reply 可信度的估计 [0,1]
    decision: Literal["trust", "distrust"]   # 二值判定
    distrust_reason: str | None    # 可选：判不可信的理由（供分析，非硬约束）
    suggested_action: Literal["none", "reask", "verify", "ignore"]
                                   # shadow 下仅记录，不执行
    confidence: float              # 控制器对自身判定/可靠性估计的置信度 [0,1]
```

- `verify` 指"建议追加确认性检查"；`reask` 指"建议追问患者"；`ignore` 指"建议不采纳该观察继续推理"。
- shadow 模式下这些 `suggested_action` **仅记录**，用于与真实噪声标签对照，验证控制器"该触发时是否触发"。

## 4. shadow-mode 评分侧（仅评估器可见，禁止回传控制器）

```python
@dataclass
class ShadowGroundTruth:           # 来自 hidden noise_log，评分专用
    turn_index: int
    noise_type: str | None         # None = clean 观察
    clean_text: str                # 干净回答（仅评分用）
    noisy_text: str                # 注入噪声后的回答
    ground_truth_diagnosis: str    # 真实疾病（仅评分用）
```

事后指标：以 `noise_type is not None` 为阳性标签，算控制器 `reliability` / `decision` 的 **AUROC / AP / 校准（ECE）**；并统计 `suggested_action` 在噪声 turn 上的触发率（召回）与在 clean turn 上的误触发率（假阳）。

## 5. 禁止读取清单（与协议 §6 一致）

控制器输入中**不得出现**：`noise_type`、`clean_text`、`ground_truth_diagnosis`、`gt_tool_findings`（干净 findings）、`noise_log` 本身。这些字段只在 `ShadowGroundTruth`（评估器）里存在。

## 6. 接入点（未来落地，非本阶段实现）

- 患者观察点：`evaluation/runners.py:_eval_conversation_sample`（评测）与 `train/rl/agent_loop.py:_handle_interacting_state`（训练）之间的 `generate_response` 边界。
- 检查观察点：`evaluation/generation.py:get_tool_response` / `train/rl/tool_agent.py:ToolAgent.execute` 返回值。
- 奖励整合（可选，后期）：`train/rl/reward.py:compute_score` 增加可靠性项，但 shadow 阶段不动训练目标。

## 7. 与本地 powerful_medrag 的映射（供对齐）

| MedExAgent 控制器概念 | 本地既有实现（`src/powerful_medrag`） | 备注 |
|---|---|---|
| `reliability` / `decision` | `gating.py`（heuristic/sparse/history gate） | 本地为结构化特征 gate |
| shadow 判别评分 AUROC/AP | `gate_analysis.py` | 可复用评估口径 |
| 学习式控制器 | `gate_learning.py`（Platt 校准） | 后期可选 |
| `suggested_action=verify/reask` | `clarification.py` / `decision.py` | 本地为结构化 verify 动作 |

**注意**：MedExAgent 是自然语言 LLM 智能体（Meditron3-8B），本地是结构化贝叶斯访谈（DDXPlus）。二者指标口径不同（LLM-judge 诊断 vs Top-k/EIG/Brier）。建立"可比较底座"时须显式冻结**指标映射**，不能直接把两边数字并列，详见 reproducibility 报告与 `docs/EXPERIMENT_PROTOCOL.md` 的 preregistered 约定。
