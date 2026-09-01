# FAILURE LOG — Prompt #7 Learned Gate 独立评估

日期：2026-08-31 ｜ HEAD：e661311

本文件记录评估过程中遇到并已修复的问题，以及一条经判定为**良性**的运行时告警。所有问题均已修复、复跑出结果，**未影响 runner 产物（train/validation 事件 CSV 与 learned_gate.json）**。

---

## 1. evaluate_gate.py 分析脚本 bug（4 处，均已修复）

分析脚本 `config/evaluate_gate.py` 在首次运行时暴露 4 处拼装 bug，全部为脚本自身缺陷，与事件采集 / fit-gate 产物无关。

### 1.1 `decision_sensitive_wrong` 标签未定义导致 `getattr` 崩溃
- **现象**：`AttributeError: 'GateEvent' object has no attribute 'decision_sensitive_wrong'`。
- **原因**：exploratory 标签由 `wrong_report AND diagnostic_impact ≥ q75` 动态计算，但脚本在 `train_prevalence` 与 surprisal 校准器拟合里直接 `getattr(event, label)`，而 `q75` 尚未计算。
- **修复**：先计算 `q75`，定义 `label_value()` 闭包，`train_prevalence` 与 `_fit_univariate_surprisal` 统一改走 `label_value`（后者改为接收 `label_fn` 可调用）。

### 1.2 `_condition_mask` 参数名不一致
- **现象**：`NameError: name 'events' is not defined`（`"all"` 分支）。
- **原因**：把 records 从 `GateEvent` 对象改为 `dict` 后，`_condition_mask` 未同步，且 `"all"` 分支残留旧变量名 `events`。
- **修复**：`_condition_mask` 统一按 `r["noise_rate"]` 取值，`"all"` 分支改为 `len(records)`。

### 1.3 bootstrap 索引越界
- **现象**：`IndexError: index 144810 is out of bounds for axis 0 with size 132300`。
- **原因**：`scores = score_array(...)[idx]` 已按 `idx` 切片（长度 132300），但 `case_event_idx` 里存的是**绝对**索引 `j`，`boot_idx` 用绝对索引去取已切片数组，越界。
- **修复**：`case_event_idx` 改存**相对**位置（`enumerate(idx)` 的 `rel_j`）；同时引入 `y_sub = y[idx]`，bootstrap 内统一用 `y_sub`/`scores`/`probs`（三者长度一致）。

### 1.4 surprisal 校准器标准化残留（pre-review 自查）
- **现象**：标准化后 `x` 已≈N(0,1)，再取 `x.mean()/x.std()` 会污染返回值。
- **修复**：在标准化**前**保存 `mu/sigma`，返回 raw surprisal 空间的 `(intercept_raw, slope_raw)`；calibration slope 的梯度去掉非标准的 `mean(logit_p^2)` 归一化。

---

## 2. 良性告警（无需修复）

### 2.1 `RuntimeWarning: All-NaN slice encountered`（nanquantile）
- **来源**：noise_rate=0.0 时 `latent_misreport`/`harmful_misreport` 全为负类（患病率 0），AUROC/AUPRC 无定义 → NaN；case-clustered bootstrap 的 `nanpercentile` 对全-NaN 切片告警。
- **判定**：**预期且无害**。脚本用 `np.nanpercentile` 处理，正确输出 NaN 指标；不影响其他条件。
- **去向**：主标签在 noise 0.0 的 AUROC/AUPRC 记为 NaN（符合预注册语义——无误报事件则无可区分性）。

---

## 3. fit-gate 慢（非错误）

fit-gate 对 176,400 事件 × 2000 次迭代的纯 Python 梯度下降超出前台 120s 超时，移至后台运行后正常完成（hold-out 校准集 AUROC=0.850）。非缺陷。

---

## 4. 结论

上述问题均为**分析脚本层面**，与冻结数据、事件采集、fit-gate 产物无涉。修复后 `evaluate_gate.py` 复跑一次产出全部 6 个结果文件，指标稳定（learned gate AUROC 0.844 复现）。runner 产物未被修改或重跑。
