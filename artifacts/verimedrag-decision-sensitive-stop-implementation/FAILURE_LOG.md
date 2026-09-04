# Phase 22A — 失败日志（FAILURE_LOG）

## 已记录的问题（全部已修复）

### F1 — Phase 8E 测试 test_07 / test_11 与 Phase 22A 日志冲突

- **现象**: Phase 8E 的 `tests/test_unknown_semantics.py::test_07` / `test_11` 用
  `mock.patch.object(tracker, "p_wrong", side_effect=AssertionError(...))` 断言「策略**从不**调用
  `p_wrong`」。Phase 22A §七 要求每轮记录 `max_p_wrong`，必然调用 `p_wrong` → 两测试报错。
- **根因**: Phase 22A 的日志需求与 Phase 8E 的「p_wrong 零调用」性质互斥。后者的**意图**是
  「p_wrong 的值不驱动决策」，而非字面的「不被调用」。
- **处理**: 按规格（Phase 22A 优先）更新两测试，保留原意图：
  - test_07 → patch `p_wrong` 返回 `None`、`p_mode_misreported` 返回 0.5，断言 `max_p_wrong is None`、
    `stop_reliability_ready is False`（决策仍由 p_mode 驱动）。
  - test_11 → 跑基线 + `p_wrong` 强制 `None` 两条轨迹，断言动作序列**逐字相同**。
- **状态**: 已修复，全绿。

## 当前无未解决失败

完整套件 `python -m unittest discover -s tests -v` → **Ran 259 tests / OK**（exit 0）。
