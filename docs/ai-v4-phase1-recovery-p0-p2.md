# V4 Phase 1 P0–P2 recovery 실행 결과

이 문서는 [Phase 1 model-gate 진단](ai-v4-phase1-model-gate-diagnosis.md)의 P0–P2를
구현하고, clean 20-game smoke corpus에서 bounded diagnostic 실행한 결과를 기록한다.
P0–P2 결과는 모두 diagnostic/non-promotable이며 Phase 2 strength gate를 대체하지 않는다.

## 1. 실행 범위와 상태

| 항목 | 결과 |
|---|---|
| P0 metric contract | 구현·테스트 완료 |
| P0 canonical gate | 6-game smoke는 `inconclusive`, `promotable=false` |
| P1 shadow-negative | λ `0, 0.01, 0.05, 0.1` 1-epoch CUDA smoke 완료 |
| P2 recovery sweep | 4 seeds × 2 learning rates, 1-epoch bounded CUDA smoke 완료 |
| parity/export | P1/P2 diagnostic run에서는 production parity/export 없음 |
| source replay | 읽기 전용, source hash 불변 확인 |
| promotion | 금지 |

현재 worktree가 dirty인 상태에서 실행했으므로 두 artifact 모두 smoke로 표시했고,
`allow_dirty_smoke`를 명시했다. 이는 production/promotion 증거가 아니다.

## 2. P0 — metric/gate contract

P0은 metric을 다음처럼 분리했다.

- Primary: opened-support path normalization
- Secondary: all-legal runtime calibration, `gate_effect=false`
- Modal tie: all-modal hit와 `1/k` fractional score
- Tail: 모든 tied modal을 제외한 macro/micro recall
- Evidence: 기본 `minimum_game_blocks=30`, hard lower bound `8`
- 6-game smoke: `inconclusive`, `promotable=false`

구현 파일:

- [`phase1_ab.py`](../train/v4/phase1_ab.py)
- [`test_phase1_ab.py`](../train/v4/test_phase1_ab.py)

P0 focused test는 15개, V4 전체는 P2 추가분 포함 86개 통과했다.

P0 계약으로 실제 1-epoch A/B smoke도 실행했다.

- artifact: [`ab-summary.json`](../../augmentia-v4-artifacts/phase1-p0-contract-smoke-20/ab-summary.json)
- summary seal: `5992b23e486daa12bf6bc15a21aafce5296b1917b3d03fb500185b07b9d767da`
- plan seal: `fb2f0836f5291b701b51b9daf6784b5d2a8664d9d3e12042f34d69d5c44ea910`
- opened-support conditional-KL improvement: `+0.020681`, bootstrap CI `[+0.001175, +0.039775]`
- opened-support all-modal top-1 improvement: `+0.031250`
- opened-support soft-NLL improvement: `+0.020681`
- all-modal ECE regression: `+0.012635`, tolerance `0.01` 초과
- tail macro/micro improvement: `-0.008929 / -0.010101`
- outcome: `inconclusive`
- reasons: `minimum-game-blocks-not-met`, `primary-bootstrap-evidence-incomplete`

즉 P0는 metric 정의와 low-power guard를 정상화했지만, 6-game smoke를 conclusive
pass로 승격하지 않는다.

## 3. P1 — mixed-row shadow-negative ablation

P1 loss는 다음 형태다.

```text
total = opened-only policy CE
      + lambda * all-legal CE on mixed opened/closed rows
      + value_weight * value CE
```

All-closed/padding row는 shadow gradient를 주지 않는다. Production export semantics는
계속 `augmentia.policy-target.opened-mask.v1`이다.

구현 파일:

- [`loss.py`](../train/v4/loss.py)
- [`train.py`](../train/v4/train.py)
- [`test_replay_trainer.py`](../train/v4/test_replay_trainer.py)

bounded diagnostic artifact:

- [`shadow-summary.json`](../../augmentia-v4-artifacts/phase1-p1-shadow-ablation-smoke-20/shadow-summary.json)
- summary seal: `c4a234386c89c1e691c5dd59333e177f2440bd5b6e59ade10bb78836ef3f1551`
- plan seal: `18cd1767ce4df7289f22b80eff377f2b6957c8f6a55ffa201aded301fa986a8a`
- initial state: `eaead42774f303ce50686353d964584e37b21428b6b314b374a74b35230bd4b9`
- validation blocks: 6
- `evidence_sufficient=false`

### P1 결과

각 cell은 동일 corpus/split/initialization, 1 epoch, CUDA, opened-only arm이다.

| λ | Conditional KL ↓ | Opened-support soft NLL ↓ | All-modal top-1 ↑ | Fractional score ↑ | All-modal ECE ↓ | Tail macro |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.703282 | 2.647586 | 0.218750 | 0.109375 | 0.069189 | 0.284226 |
| 0.01 | 0.702572 | 2.646876 | 0.218750 | 0.109375 | 0.069271 | 0.284226 |
| 0.05 | 0.700025 | 2.644329 | 0.208333 | 0.104167 | 0.059833 | 0.284226 |
| 0.10 | 0.697882 | 2.642185 | 0.208333 | 0.104167 | 0.059891 | 0.284226 |

### P1 해석

- λ가 커질수록 conditional KL과 opened-support soft NLL은 이 bounded run에서 개선
- all-modal ECE도 일부 개선
- 반대로 all-modal top-1과 fractional score는 λ `0.05`, `0.1`에서 악화
- tail recall은 모든 λ에서 동일
- 따라서 mixed-row negative hypothesis는 **부분적으로 지지**되지만, blindly increasing
  λ가 gate를 통과한다고 결론 내릴 수 없음
- 6-game validation과 1 epoch이므로 evidence는 exploratory뿐

P1 runner:

- [`phase1_shadow_ablation.py`](../train/v4/phase1_shadow_ablation.py)
- [`test_phase1_shadow_ablation.py`](../train/v4/test_phase1_shadow_ablation.py)

## 4. P2 — optimization sweep

P2는 다음 bounded matrix를 실행했다.

- Seeds: `2026091501`, `2026091502`, `2026091503`, `2026091504`
- Learning rates: `3e-4`, `1e-3`
- Shadow λ: `0.05`
- Fixed updates: `14` (`1` epoch, 현재 224 train examples / 14 batches)
- Cosine LR schedule
- Early-stop wiring: enabled, patience `2`
- Validation blocks: `6`
- `promotable=false`

Artifact:

- [`sweep-summary.json`](../../augmentia-v4-artifacts/phase1-p2-recovery-smoke-20/sweep-summary.json)
- summary seal: `236d6d4cd489df8b728b20d8bdd595fcf98b9e06462878b03999b8caaca4d45a`
- plan seal: `a3d37eaf4ba0c978c08a1216fc90979af1a3b9256db4d6a7ad4692e3308befa1`
- source immutability: `true`
- `evidence_sufficient=false`

### P2 learning-rate group 결과

| Learning rate | Conditional KL mean ± SD | Opened-support NLL mean ± SD | All-modal top-1 mean ± SD |
|---:|---:|---:|---:|
| `3e-4` | `0.690227 ± 0.008172` | `2.634531 ± 0.008172` | `0.234375 ± 0.032387` |
| `1e-3` | `0.689086 ± 0.009837` | `2.633390 ± 0.009837` | `0.239583 ± 0.032940` |

- `1e-3`가 평균 KL/NLL/top-1에서 약간 유리
- 하지만 seed SD가 LR 차이보다 큼
- paired seed 비교에서 `1e-3`가 KL/NLL에서 이긴 seed는 `2/4`, top-1은 `1/4`
- best single cell: `seed-2026091502-lr-1.00000000e-03`
  - conditional KL `0.679830`
  - opened-support soft NLL `2.624135`
  - all-modal top-1 `0.25`
- 1 epoch budget이라 early stopping이 실제로 발생하지는 않음
- best cell은 diagnostic ranking일 뿐 promotion 후보가 아님

P2 runner:

- [`phase1_recovery_sweep.py`](../train/v4/phase1_recovery_sweep.py)
- [`test_phase1_recovery_sweep.py`](../train/v4/test_phase1_recovery_sweep.py)

## 5. P0–P2 전체 결론

P0–P2는 다음을 확인했다.

1. Metric gate의 support/tie/evidence 문제를 코드와 테스트로 고칠 수 있다.
2. Mixed-row shadow loss는 λ `0.05–0.1`에서 conditional KL/NLL을 일부 개선할 수 있다.
3. 다만 top-1/fractional score와 calibration 사이에는 trade-off가 있다.
4. Learning-rate `1e-3`의 평균 우위는 seed variance보다 크지 않다.
5. 20-game/6-validation-block corpus는 여전히 gate-grade evidence가 아니다.
6. 따라서 P0–P2는 **진단 가설의 방향을 확인한 bounded recovery**이며, model gate
   recovery 완료나 promotion 승인으로 해석할 수 없다.

## 6. 검증

- Python 전체 unittest: **167 passed**
- Python V4 unittest: **86 passed**
- P0 focused: **15 passed**
- P1/P2 focused: **15 passed**
- Rust workspace: `augment-ai` **112 passed**
- Rust Clippy `-D warnings`: passed
- P0/P1/P2 plan/result/evaluation seals: independently verified
- P2 source immutability: verified
- 모든 diagnostic artifact: `promotable=false`

## 7. 다음 조건

다음 실행 전 최소 조건:

- diagnostic: 30 independent games 이상
- gate-grade: 80–100 held-out games 이상
- practical starting point: 200 train + 100 held-out games
- longer/full-game horizon과 terminal/value corpus 확보
- 4개 이상 seed와 고정 update budget 유지
- shadow λ, learning rate, early-stop 조건을 사전 등록
- 별도 Phase 2 paired strength/CI gate 수행

현재 P0–P2 artifact의 `promotable`은 계속 `false`다.
