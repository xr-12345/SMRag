# VeriMedRAG 最小 RAG 接入 — Phase 1（模块 + 测试 + N=1 smoke）

日期：2026-08-31 ｜ HEAD：e661311 ｜ Python 3.13.13（miniforge）

## 0. 一句话结论

最小 RAG 已打通端到端：BM25 医学检索被接入了结构化 AskNew/VerifyOld/Stop 决策循环，**且检索影响确实改写了 VerifyOld 的排序**（同一快照下，删除某份陈述会让检索结果重排 → 该份陈述的核验分被抬高）。数据、运行时间、反事实影响量均可复现。但**在本次 N=1 简单病例（Anemia）上，对话在 2 问后即以 99.67% 置信度停止、从未触发核验**，所以检索对决策的端到端影响只体现在「VerifyOld 排序」层面，尚未在「实际核验动作」层面验证——这是进入 N=20 前要解决的首要阻塞项。

---

## 1. 代码接入点

| 文件 | 改动 | 作用 |
|---|---|---|
| `src/powerful_medrag/retrieval.py` | **新增** | 纯 Python Okapi BM25（无 Java/Pyserini）；`MedicalRetriever`（加载 corpus + 内存倒排索引 + query→hits 缓存）；`build_retrieval_query`（疾病名 + 阳性证据名 + 冲突名）；`apply_counterfactual`（delete/weaken/flip）；`jaccard`/`rbo`/`ndcg`/`counterfactual_retrieval` |
| `src/powerful_medrag/decision.py` | `RetrievalMode`（no_rag/static_rag/dynamic_rag） | 三模式开关 |
| | `ReliabilityAwarePolicyConfig` + `retrieval_mode` / `retrieval_impact_weight` / `retrieval_top_k` / `query_disease_top_k` | 配置项（默认 no_rag、weight=0.0，向后兼容） |
| | `ReliabilityAwareActionPolicy.retriever` | 持有检索器 |
| | `_retrieval_impacts()` | 逐报告「删除反事实 → 检索改变量（1−Jaccard）」 |
| | `_verification_scores()` 末尾 | 把 `retrieval_impact_weight × impact` **加进**每份报告的核验分（无论是否用 learned gate） |
| | `run_reliability_aware_dialogue()` | static 模式对话开始时检索一次、dynamic 模式每答一问重检索一次，写入 `retrieval_log` |
| | `ReliabilityAwareDialogueResult.retrieval_log` | 落盘每轮 query + hit_ids + hit_titles |
| | `_readable_feature_names()` | 证据代码 `E_146` → 官方 `question_en` 可读文本（无 ASCII 词时回退 feature name） |
| `tests/test_retrieval.py` | **新增 11 测试** | 回答变化改变 query/检索、三反事实、feature_names 映射、检索影响进 VerifyOld、三模式 retrieval_log |

关键设计决定：`FeatureKey.name` 是 DDXPlus 证据**代码**（如 `E_146`），不可改（条件图/身份映射依赖它）。可读临床词存在 `VariableSpec.question`（官方 `question_en`），故通过 `_readable_feature_names` 把代码映射成可读问句再喂给 BM25。

---

## 2. 数据规模

- 官方 MedRAG Textbooks 语料：**18 本教科书、125,847 snippets、209 MB**（`data/medrag-textbooks/*.jsonl`）。
- DDXPlus 模型：49 疾病、889 证据特征（`data/ddxplus/model-full.json`）。
- 输入切分：仅 `release_validate_patients.zip`（未碰 test split）。

## 3. 运行时间（M5 Pro）

| 步骤 | 耗时 |
|---|---|
| 语料加载 + BM25 内存索引构建（一次性） | **~3.8 s** |
| 单次检索（k=10） | **~18 ms** |
| N=1 对话 × 3 模式（含 static 首次检索） | no_rag 0.05s / static 2.8s（含索引）/ dynamic 0.05s |

结论：索引构建是唯一的一次性开销；运行时检索 ~18ms，可承受 N=20（甚至全 validate）规模。

---

## 4. 检索是否真正影响决策

**是，但仅在 VerifyOld 排序层面（N=1 未触发核验）。**

同一快照（4 份已答陈述、top-1 疾病 Anemia）下，VerifyOld 排序被检索改写：

| report | 证据（question_en） | retrieval_impact | no_rag 分 | +检索 分 |
|---|---|---|---|---|
| 1 | "Have you ever had a diagnosis of anemia?" | **0.5714** | 1e-6 | **0.2857** |
| 2 | "Is your BMI < 18.5 …?" | 0.4615 | 1e-6 | 0.2308 |
| 0 | "Are you taking new oral anticoagulants?" | 0.3333 | 1e-6 | 0.1667 |
| 3 | (absent) | 0.0 | 0.0 | 0.0 |

- base_order `[1,0,2,3]` → rag_order `[1,2,0,3]`：检索影响把「贫血史」和「低 BMI」两条陈述的核验优先级抬到「抗凝药」之前——语义上正确（贫血相关的证据对 Anemia 鉴别最有信息量）。
- 反事实影响量（对 top-impact 陈述 E_24）：delete/weaken/flip 均使 `topk_jaccard=0.43、rbo=0.55`（检索明显重排），疾病证据得分 0.143→0.095，top-1 疾病后验变化 ~1e-4~7e-4。

**局限**：本病例噪声 0.2、证据强，对话 2 问即达 99.67% 置信度停止、0 次核验，所以检索影响**没能端到端改变一次真实核验动作**（它只作用于「该核验哪份报告」的排序，不作用于「是否核验」的门控）。

---

## 5. 进入 N=20 validation 前的阻塞项

1. **（首要）检索只进排序、不进门控。** `retrieval_impact_weight` 只加到 `_verification_scores` 的分数上，而 `verification_gate_open` 仍由历史 cue + learned gate 决定。要让检索「真正触发核验」端到端，需决定：检索影响是否应打开门控（并做消融，避免重蹈 learned-gate「更激进但不更聪明」的覆辙）。
2. **`retrieval_impact_weight` 未调参。** 本次用 0.5（ad hoc）。N=20 前需小 sweep（如 0/0.1/0.5/1.0）。
3. **query 用整句 `question_en`，冗长。** DDXPlus 无官方「简洁症状名」字段；可读词只存在于问句里。需评估是否要加一层轻量名词抽取（或接受整句作为 BM25 词袋）。
4. **N=1 未触发核验。** 需在中高噪声（0.2/0.3）× 多 seed × 多病例上跑，才能观察检索对「实际核验次数/命中率」的影响。
5. **反事实 flip 对多值证据是 no-op。** 只有二元 present↔absent 能翻转；多值（M/C 型）翻转无定义，已按原值返回。
6. **疾病证据得分是粗略 overlap。** 未按先验/信息量加权；后验变化在置信病例上极小，需确认这些影响量在 N=20 尺度下是否够敏感。

---

## 6. 产物清单

`CONFIG.json` ｜ `smoke_metrics.json`（完整逐项指标）｜ `run_smoke.py` ｜ 本报告 ｜ `tests/test_retrieval.py`（11 测试，全量 71/71 通过）。

## 7. 未做（红线内）

不访问 test split；不做大规模 MedCorp；不训练 LLM/dense retriever；不 SFT/RL；策略不读 latent state（无 oracle）；未覆盖旧 artifacts；未 git commit。
