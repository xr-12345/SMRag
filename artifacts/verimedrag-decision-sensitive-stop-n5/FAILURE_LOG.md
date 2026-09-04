# Phase 22B — 失败日志（FAILURE_LOG）

## 已记录的问题（全部已修复）

### F1 — `analyze_n5_screen.py` 回归检查用 set 包 dict

- **现象**: 首次运行分析脚本时报 `TypeError: unhashable type: 'dict'`，定位在
  `_regression_check` 的 `mine = {r for r in rows if r["config"] == strat}`。
- **根因**: 误用集合推导（`{...}`）去收集 `csv.DictReader` 的 dict 行，dict 不可哈希。
  此处只需列表，`[r for r in ... if ...]`。
- **处理**: 改为列表推导。重跑后 `baseline_regression_check.txt` 正常产出
  （L-H/P-H 与 Phase 8E 冻结结果 **0 不匹配**，PASS）。
- **状态**: 已修复，分析全量跑通。

## 当前无未解决失败

- 完整测试套件 `python -m unittest discover -s tests -v` → **Ran 259 tests / OK**（exit 0）。
- 完整实验 5880 轨迹跑通，分析脚本全量产出无异常。
