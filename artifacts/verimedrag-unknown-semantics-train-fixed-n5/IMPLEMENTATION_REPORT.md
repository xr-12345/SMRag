# Phase 8E — Implementation Report（UNKNOWN 语义修复 + protocol_fixed 先验集成）

## 1. 改动概览

| 文件 | 改动 |
|---|---|
| `src/powerful_medrag/joint_reliability_belief.py` | `p_wrong` 对 UNKNOWN 返回 `None`（签名 `float | None`）；新增 `is_nonresponse` |
| `src/powerful_medrag/channel.py` | 新增 `protocol_fixed_prior()` 与 `protocol_fixed_channel_parameters()` |
| `tests/test_joint_report_channel.py` | 更新 UNKNOWN 断言（`1.0` → `None` + `is_nonresponse`） |
| `tests/test_unknown_semantics.py` | 新增 11 个测试方法，覆盖 §五 的 12 类要求 |

## 2. UNKNOWN 语义修复（§二）

### 修改前（`joint_reliability_belief.py`）

```python
def p_wrong(self, key) -> float:
    ...
    if original_value == UNKNOWN:
        return 1.0  # UNKNOWN is never the true state, so it is always "wrong"
    return 1.0 - self.state_posterior(key).get(original_value, 0.0)
```

### 修改后

```python
def p_wrong(self, key) -> float | None:
    """p_i^wrong = P(Z_i != Y_i | H_t) for the first answer Y_i.
    Returns None when the first answer is UNKNOWN ..."""
    bundle = self.memory.get(key)
    if bundle is None:
        return 0.0
    if bundle.original.value == UNKNOWN:
        return None
    return 1.0 - self.state_posterior(key).get(bundle.original.value, 0.0)

def is_nonresponse(self, key) -> bool:
    """u_i = 1[Y_i = UNKNOWN] ..."""
    bundle = self.memory.get(key)
    return bundle is not None and bundle.original.value == UNKNOWN
```

- 明确回答 `Y_i ∈ {PRESENT, ABSENT, categorical}`：`p_wrong = P(Z≠Y | H_t)` 不变。
- `Y_i = UNKNOWN`：返回 `None`（不可定义），**既不是 1 也不是 0**（红线 §九）。
- `u_i = 1[Y_i=UNKNOWN]` 由 `is_nonresponse` 显式给出，确定性标记，不伪装成错误概率。
- `Optional[float]` 由调用方显式处理：无 `or 0.0` 静默转换；日志空值；汇总单独统计
  UNKNOWN rate（§二）。

## 3. 策略语义不被改变（§三）

- Stop 审计用 `p_mode = P(E=MISREPORTED | H_t)`（`p_mode_misreported`），**不是**
  `p_wrong`（`joint_channel_policy.py:208-211` 的 `max_mode_misreport`）——保持不变。
- `_verifyold_brier_values` 本已排除 UNKNOWN 候选（`joint_channel_policy.py:128`）——保持不变。
- `p_wrong` 在生产代码中**无任何调用方**（仅定义 + 测试），故语义修复只影响日志/校准/
  解释/接口，不改变动作轨迹。逐病例逐轮 byte-identical 回归见
  `baseline_regression_check.txt`（与 Phase 8C joint matched 0/0 比对）与
  `test_11`（mock `p_wrong` 抛异常仍正常跑完对话）。

## 4. protocol_fixed（train_fixed）先验集成（§四）

- 新增 `channel.protocol_fixed_prior()`：对协议噪声集合 `{0.2, 0.3}` 的
  `PatientProfile.from_noise_rate` 取平均，得到 `.75/.10/.075/.075`（非 cue 条件）。
- 新增 `channel.protocol_fixed_channel_parameters()`：保留默认 `rates`，把每个
  certainty cue 的先验替换为同一 protocol 先验。
- 改名：`train_fixed` → `protocol_fixed_prior`（因为它是人工计算的平均值，非学习）。
- 配置机制：`legacy_prior`（默认 `ChannelParameters()`，MISREPORTED=0.03，未动）与
  `train_fixed_prior`（显式构造 `AnswerChannel(protocol_fixed_channel_parameters())`）。
  legacy 默认**不删除、不覆盖**（§四）。
- `cue_conditioned` 本阶段只作为 Phase 8D 消融结论保留，不作为默认部署配置（§四）。

## 5. 测试（§五，12 类）

`tests/test_unknown_semantics.py`（11 方法）+ 既有套件：

| # | 要求 | 测试 |
|---|---|---|
| 1 | PRESENT/ABSENT p_wrong 为 [0,1] 概率 | `test_01` |
| 2 | categorical 正常计算 | `test_02` |
| 3 | UNKNOWN 返回 None | `test_03` |
| 4 | UNKNOWN 不进入 wrong-report Brier | `test_04` |
| 5 | UNKNOWN rate 单独统计 | `test_05` |
| 6 | UNKNOWN 非 VerifyOld 候选 | `test_06` |
| 7 | Stop 由 p_mode 审计 | `test_07` |
| 8 | legacy 旧路径回归 | `test_08` + `test_joint_report_channel.py::test_13` |
| 9 | protocol_fixed 来源固定 | `test_09` |
| 10 | 推断不读真值/噪声/干净回答 | `test_10` |
| 11 | 轨迹逐位一致 | `test_11` |
| 12 | 所有旧测试通过 | 全量套件 240 tests OK（`TEST_LOG.txt`） |

## 6. 实验设计（§六）

- 配置：`joint_legacy_prior` vs `joint_train_fixed_prior`（matched 0/0，
  `rho_env=0, rho_model=0`）。
- 冻结：validate split、N=20 manifest 前 5 例（245 例）、noise `{0.2,0.3}`、
  seeds `{2026,2027,2028}`、预算/阈值与 Phase 8C 完全一致。
- 每配置跑**自己的完整对话**（同 seed 同患者，仅推断先验不同），故既比诊断
  （Top-1/Brier/问题数）也比可靠度校准（p_wrong/p_mode ECE/Brier + UNKNOWN rate）。
- 未访问 test split、未调阈值、未跑 N=20。
