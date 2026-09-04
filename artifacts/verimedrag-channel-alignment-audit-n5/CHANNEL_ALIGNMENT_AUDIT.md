# Channel Alignment Audit (Phase 8D §二)

静态参数审计：环境 vs 推断的 5 个通道组件。

## 组件 1 — 报告模式先验 P(E)

| 来源 | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |
|---|---|---|---|---|
| env_noise_0.2 | 0.8000 | 0.0800 | 0.0600 | 0.0600 |
| env_noise_0.3 | 0.7000 | 0.1200 | 0.0900 | 0.0900 |
| inference_default | 0.8200 | 0.1000 | 0.0500 | 0.0300 |

## 组件 2 — cue 机制 P(E|cue)


### cue = none

| cue | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |
|---|---|---|---|---|
| env_noise_0.2 | 0.0828 | 0.2899 | 0.5839 | 0.0435 |
| env_noise_0.3 | 0.0500 | 0.3002 | 0.6047 | 0.0450 |
| inference_default | 0.8200 | 0.1000 | 0.0500 | 0.0300 |

### cue = uncertain

| cue | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |
|---|---|---|---|---|
| env_noise_0.2 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| env_noise_0.3 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| inference_default | 0.1800 | 0.7000 | 0.0900 | 0.0300 |

### cue = certain

| cue | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |
|---|---|---|---|---|
| env_noise_0.2 | 0.9302 | 0.0000 | 0.0042 | 0.0655 |
| env_noise_0.3 | 0.8861 | 0.0000 | 0.0069 | 0.1070 |
| inference_default | 0.9300 | 0.0300 | 0.0100 | 0.0300 |

## 组件 3 — 首答通道 P(Y|Z,E)（env 与推断相同，未失配）

| mode | correct | unknown | wrong |
|---|---|---|---|
| certain | 0.98 | 0.01 | 0.01 |
| uncertain | 0.55 | 0.35 | 0.1 |
| unknown | 0.04 | 0.94 | 0.02 |
| misreported | 0.08 | 0.07 | 0.85 |

## 组件 4 — 复问转移先验 P(e')（(1-rho) 项）

| 来源 | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |
|---|---|---|---|---|
| env_noise_0.2 | 0.8000 | 0.0800 | 0.0600 | 0.0600 |
| env_noise_0.3 | 0.7000 | 0.1200 | 0.0900 | 0.0900 |
| inference_reask_mode_prior | 0.8200 | 0.1000 | 0.0500 | 0.0300 |

## 失配摘要（KL env→infer / max|Δ|）

| noise | P(E) KL | P(E\|NONE) KL | P(E\|CERTAIN) KL | P(E\|UNCERTAIN) KL |
|---|---|---|---|---|
| noise_0.2 | 0.0149 | 1.5696 | 0.0478 | 0.3567 |
| noise_0.3 | 0.0629 | 1.7158 | 0.0907 | 0.3567 |
