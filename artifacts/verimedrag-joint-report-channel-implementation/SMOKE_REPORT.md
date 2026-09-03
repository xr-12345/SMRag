# Phase 8B — 机制 smoke 报告 (Cases A–E)

范围：toy model（`generate_toy_cases(cases_per_disease=100)`，3 疾病 × 6 特征），
`rho=0.5`，冻结 `AnswerChannel`。**不碰 test split，不跑 N=5/N=20。**

## 结论

**5/5 机制全部通过**（`all_mechanisms_ok = true`）。

| Case | 机制 | 关键指标 | 通过 |
| --- | --- | --- | --- |
| A | 单答正确 → 后验集中 + 低 p_mode | top=influenza (0.681)，p_mode=0.0403 < 0.05 | ✅ |
| B | 矛盾再问 → 概率化解而非 UNKNOWN 压缩 | influenza 0.089→0.306，answers 保留 [absent, present] | ✅ |
| C | 一致回答 → 单因子 + 确认（正确）模式 | p_mode 0.0207→0.0104（重复确认正确） | ✅ |
| D | rho=0 → 独立再问 = 边际乘积 | joint 0.038576 == product 0.038576 | ✅ |
| E | 可靠度门控接入 + 真触发 | max p_mode=0.2913 > 0.05，reliability_ready=False | ✅ |

## 逐 Case 说明

### Case A — 单答正确

观察 `fever=present`（influenza 真实症状）。疾病后验 top1=influenza (0.681)，
`p_mode_misreported(fever)=0.0403`，低于可疑阈值 0.05。→ 单答路径在共享模型下
正确集中后验，且把可靠度读为「可信」。

### Case B — 矛盾再问被概率化解，而非压缩

首答 `fever=absent`，再问 `fever=present`（正确）。旧 `resolve` 会把矛盾压成
`UNKNOWN`（influenza 后验停在 0.089）。联合模型保留两个回答，并用联合似然
`P(absent, present | z, v)` 概率化解：influenza 从 0.089 → 0.306。→ 矛盾被
「解释为一次错误报告后被纠正」，而不是被丢弃。

### Case C — 一致回答确认模式，而非盲目当作错误报告

首答 `fever=absent`，再问 `fever=absent`。在 `rho=0.9` 下，两次一致回答最可能
由 `z=absent`（allergic_rhinitis）解释（两者皆正确），故 `p_mode` 从 0.0207
降到 0.0104。→ 一致性降低错误报告的怀疑，符合「持续模式」语义。

### Case D — 独立再问是联合模型特例

`rho=0` 时 `joint_probability(present, absent | present) = 0.038576 ==
single(present)·single(absent) = 0.038576`。→ 独立再问是联合模型的退化特例。

### Case E — 可靠度门控接线 + 真实触发

先观察 3 个强 influenza 证据（fever/dry_cough/myalgia=present → influenza 0.932），
再观察矛盾答案 `itchy_eyes=present`（influenza 下出现率仅 0.05）。联合模型把
`p_mode_misreported(itchy_eyes)` 推到 0.2913 > 阈值 0.05，决策层
`reliability_ready=False`、`base_stop_ready=False`。→ Stop 被可靠度门控阻止，
门控确由联合误报后验驱动（`max_mode_misreport == 0.2913`）。

## 产物

`smoke_metrics.json`（含逐 Case 指标与 `_summary.all_mechanisms_ok`），由
`run_smoke.py` 生成。
