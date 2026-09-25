# V4 Phase 1 baseline model-gate 진단

이 문서는 Phase 1 current-code baseline paired A/B의 `model gate: fail` 원인을
Agent Team이 독립적으로 감사한 결과를 기록한다. 이 진단은 canonical artifact를
읽기 전용으로 분석했으며, source code·checkpoint·외부 artifact를 변경하거나 재학습하지
않았다.

## 1. 대상과 결론

대상 artifact:

- run: `phase1-current-paired-smoke-20`
- corpus: `phase1-clean-v3-smoke-20`
- replay/target: `augmentia.replay.game.v5` / `augmentia.factorized-mcts-visits.v3`
- A/B arms: `opened-only` / `legacy-all-open-experiment`
- validation: 14 train games / 6 held-out games, 224 / 96 decisions
- A/B plan seal: `876efcdfd67aa662d5ee30a503ab2f52679c567cd438d889fdd871c6a046742e`
- A/B summary seal: `5615005b097388ab2295d15d4ce24f5bec4d41f6a97fe333056807ebea6326b9`
- model gate: `fail`, `promotable=false`

### 최종 판정

현재 실패는 **실행 실패가 아니다**. A/B, held-out evaluation, bootstrap, export,
eager/ONNX/Rust parity까지 정상적으로 실행되었다.

그러나 사전등록된 gate는 opened-only의 이득을 증명하지 못했고, non-degradation
기준도 충족하지 못했다. 독립 감사로 다음 결론을 분리했다.

1. canonical metric의 산술·부호·checkpoint·seal 오류는 확인되지 않았다.
2. all-legal complete-action metric이 opened-only training support와 부분적으로
   맞지 않아 secondary gate에 정의상 bias가 있다.
3. validation 96 decisions는 독립 game 6개뿐이어서 population-level 회귀를 확정할
   수 없다.
4. mixed row에서 closed-negative supervision을 받는 all-open control이 shared scorer에
   더 강한 regularization을 주는 것이 가장 유력한 학습 인과 가설이다.
5. 42 optimizer step, 고정 learning rate, 작은 correlated corpus가 training 결과를
   불안정하게 만들었을 가능성이 높다.

따라서 현재 gate 실패를 “opened-mask 개념이 실패했다”거나 “population 성능이
열세다”라고 해석해서는 안 된다. 정확한 표현은 다음과 같다.

> 사전등록된 Phase 1 gate를 통과하지 못했고, 현재 smoke에서는 opened-only 이득이
> 보이지 않았으며, 일부 complete-action metric은 현재 6-game 평가에서 일관된
> regression을 보였다. population-level model failure는 미확정이다.

## 2. 실행·무결성 감사

Agent Team은 canonical checkpoint를 CUDA eager로 다시 추론하고 저장된 evaluation과
비교했다.

- conditional metric 최대 절대 오차: `1.21e-7`
- complete-action soft NLL 최대 절대 오차: `3.70e-7`
- modal NLL, top-1, confidence, tail metric: exact match
- CE = target entropy + KL: 재현
- decision 내부 prefix visit-mass 합산 후 decision-level 평균: 재현
- improvement delta 방향: 재현
- 6-game / 1000-resample / seed `2026133768` bootstrap: 재현
- plan, run, evaluation, checkpoint, parity seal: 재검증 통과

따라서 이번 gate fail의 원인은 stale prediction, 부호 반전, aggregate 실수,
parity 실패가 아니다.

관련 실행 코드는 [`phase1_ab.py`](../train/v4/phase1_ab.py)와
[`loss.py`](../train/v4/loss.py)에 들어 있다.

## 3. Metric 정의 감사

### 3.1 Opened training support와 all-legal runtime 경로

Opened-only loss는 다음 mask를 사용한다.

```text
training_mask = legal_mask & opened_mask
```

즉 closed child는 denominator에서 제외되고 직접 gradient를 받지 않는다.

반면 complete-action metric과 Rust runtime은 각 prefix에서 all legal child로
softmax를 계산한다. 따라서 opened-only arm의 closed logits는 직접 학습되지 않은
상태로 all-legal complete-action metric에 참여한다.

관련 runtime 계약은 [`ai-v4-factorized-policy.md`](ai-v4-factorized-policy.md)에
정의되어 있다.

동일 checkpoint를 **opened-support path distribution**으로 다시 계산하면 다음
결과를 얻었다. 이는 canonical gate를 대체하는 승인된 metric이 아니라, metric
정의 sensitivity를 확인하기 위한 진단 재계산이다.

| Opened-support metric | Opened-only | All-open control | Canonical 대비 |
|---|---:|---:|---|
| Soft NLL ↓ | 2.641025 | 2.633401 | Opened 열세 |
| Modal top-1 ↑ | 0.052083 | 0.020833 | Opened 열세 |
| MCTS-modal ECE ↓ | 0.112551 | 0.136478 | Opened 열세 |

즉 canonical all-legal 평가에서 실패한 non-degradation 3개 항목은 opened-support
정의에서는 방향이 뒤집힌다. 그럼에도 conditional KL의 bootstrap improvement가
음수이므로 최종 gate 결과 자체는 바뀌지 않는다.

해석은 다음과 같다.

- all-open control은 production semantics가 아니다.
- all-open arm은 mixed row의 closed-negative를 직접 학습하는 diagnostic control이다.
- 현재 비교에는 objective support와 runtime evaluation support 사이의 불일치가 있다.
- 이 불일치를 해결하지 않은 채 complete-action metric를 promotion gate로 사용해서는
  안 된다.

### 3.2 MCTS modal tie

Held-out 96개 decision 모두에서 최다 visit action이 정확히 두 개씩 동률이었다.
현재 구현은 `torch.argmax`로 첫 번째 legal action을 modal로 선택한다.

동률 처리 정책에 따라 top-1, ECE, tail 결과가 크게 달라진다.

- First-index top-1: opened `0.020833`, control `0.041667`
- Any-modal hit: opened `0.177083`, control `0.208333`
- Fractional tie score: opened `0.088542`, control `0.104167`

두 modal을 모두 tail에서 제외한 recall은 opened `0.214286`, control `0.220982`로,
opened-only 열세가 더 명확해진다. Modal 정의는 action order에 독립적이어야 한다.

### 3.3 Tail recall 집계

현재 tail recall은 decision-macro다. Validation에서는 다음 분포다.

- 84 decisions: tail action 15개
- 12 decisions: tail action 2개
- sparse 12 decisions에서는 두 arm 모두 recall `1.0`

그래서 sparse decision이 macro 평균에 과도한 비중을 갖는다. Action-micro 값도
방향은 같지만, 다음 gate에서는 두 정의를 모두 기록해야 한다.

### 3.4 Gate의 일반적 취약점

`evidence_sufficient`가 mean와 CI의 존재만 확인하고 최소 game/block 수를 요구하지
않는다. 일반적인 1-game validation도 bootstrap CI가 만들어지면 통과할 수 있다.
현재 6-game smoke artifact에는 직접 영향을 주지 않지만, 다음 gate에서는 최소
독립 game 수와 최대 block 수를 명시해야 한다.

## 4. Training dynamics 진단

### 4.1 Optimizer와 learning curve

- Parameter 수: `3,266,821`
- Batch: 16
- Examples: 224
- Epoch: 3
- Total Adam step: 42
- Learning rate: `1e-3`
- Schedule, decay, weight decay, early stopping: 없음
- `value_weight=0`

Opened-only history:

```text
epoch       1        2        3
train       2.66460  2.63873  2.62370
validation  2.63496  2.65581  2.64102
```

Train loss는 1.54% 감소했지만 validation loss는 0.23% 악화되었고, best validation은
epoch 1이었다. 이는 단순 underfit보다는 small-data noisy optimization과 mild
overfit이 함께 나타난 패턴이다.

Epoch 3 parameter movement:

- Opened-only L2 delta: `10.0182`, 초기 norm의 `3.244%`
- Control L2 delta: `10.2261`, 초기 norm의 `3.312%`
- Arm-final checkpoint 간 L2 delta: `9.4416`
- Arm-final checkpoint cosine similarity: `0.999533`

Model capacity 부족보다는 update 수와 data diversity를 먼저 검증해야 한다.

### 4.2 Mixed-row negative supervision

Training corpus:

- Mixed rows: `1,136`
- Closed negative edges: `1,799`
- Closed child mass: `578.24`
- All-closed rows: `4,888`

Validation corpus:

- Mixed rows: `490`
- Closed negative edges: `917`
- Closed child mass: `340.35`
- All-closed rows: `2,434`

Opened-only arm은 closed child를 softmax에서 mask하므로 mixed row에서도 해당
gradient가 없다. Control은 같은 row에서 closed child를 zero-target negative로
압축한다.

Optimizer proxy:

- Control `exp_avg_sq` L2: `0.0003319`
- Opened-only `exp_avg_sq` L2: `0.0001441`
- 비율: **2.30배**
- Control `exp_avg` L2: opened-only의 **1.40배**

Heldout output에서도 raw opened-minus-closed logit gap은 다음과 같다.

- Opened-only: `+0.35760`
- Control: `-0.02861`

이는 all-open arm이 opened action의 semantic target 자체를 더 잘 학습했다는
증거라기보다, mixed row의 추가 negative supervision이 shared scorer에 전이되어
conditional calibration을 개선했을 가능성을 지지한다.

All-closed row는 all-open control에서도 visit target가 0이어서 직접 gradient가 없다.
따라서 설명의 핵심은 “모든 closed row”가 아니라 **mixed row의 denominator
calibration**이다.

## 5. Corpus·split 검출력

현재 corpus는 다음 특징을 가진다.

- 20 games / 320 decisions
- 모든 game이 16-step `max_decisions` limit
- Terminal games: 0
- Supervised value labels: 0
- Train/validation: 14/6 games
- Validation: 96 decisions
- Root opened action 비율: `59.79%`
- All-open 변환으로 root opened slot이 `3,220 → 5,273`, 즉 `+63.8%`
- Mixed decision 비율: `87.5%`
- Validation의 closed mass/game가 train보다 약 `37.34%` 높음
- Game별 closed support와 mixed row의 heterogeneity가 큼

### Independent block 해석

Bootstrap resample는 1,000개지만 독립 block은 6개다. 96 decisions는 6개
trajectory 안의 반복 decision이므로 독립 sample 96개로 취급할 수 없다.

- Conditional KL: 5/6 games에서 opened-only 열세, exact sign-flip `p=0.125`
- Tail recall: 3개 개선 / 3개 악화, `p=0.625`
- Soft/modal NLL: 6/6 games에서 opened-only 열세, exact `p=0.03125`지만 최소
  6-block 결과
- Top-1: 두 game만 개선, 나머지 tie/열세, `p=0.5`

따라서 현재 데이터로 가능한 결론은 다음뿐이다.

> 사전등록된 gate는 통과하지 못했고, 현재 smoke 평가에서 opened-only 이득은
> 관찰되지 않았다. 하지만 population regression은 확정할 수 없다.

## 6. 인과 우선순위

| 우선순위 | 가설 | 현재 근거 | 신뢰도 |
|---:|---|---|---|
| 1 | Mixed-row closed-negative supervision 부재 | 1,799 closed edges, optimizer proxy 2.30배, control calibration 우위 | 높음 |
| 2 | 42-step small/noisy optimization | train↓/validation↑, best epoch 1, LR schedule 없음 | 높음 |
| 3 | 6-game/96-decision 낮은 검출력 | block CI, sign-flip p, game heterogeneity | 높음 |
| 4 | Opened-only/all-legal support 불일치 | alternate metric에서 non-degradation 방향 역전 | 높음 |
| 5 | Absolute underfit | train KL은 높지만 validation은 이미 불안정 | 중간 |
| 6 | Random batching/padding | 게임 혼합은 양호, padding은 runtime 문제일 가능성 | 낮음 |
| 제외 | Value objective | supervised value 0건 | 배제 |

## 7. 다음 실험 순서

### P0 — Gate와 artifact semantics 고정

- `opened-only` primary metric을 opened-support와 명시적으로 alignment
- all-legal runtime calibration은 별도 secondary metric으로 유지
- modal tie는 first-index가 아니라 all-modal/fractional policy를 사전 등록
- tail macro와 action-micro를 모두 seal
- `evidence_sufficient`에 최소 independent game 수를 추가
- `legacy-all-open-experiment`는 계속 experiment-only로 유지

### P1 — Causal shadow-negative ablation

동일 corpus/init에서 primary opened-only CE를 유지하고, mixed row에만 작은
all-open shadow loss를 더한다.

```text
lambda = 0, 0.01, 0.05, 0.10
```

조건:

- export semantics는 `opened-only`로 고정
- common evaluation도 opened-only view로 고정
- 동일 split/init 유지
- closed-negative가 control 우위의 원인인지 직접 비교

### P2 — Optimization sweep

최소 4개 seed로 다음을 비교한다.

- Learning rate: `3e-4`, `1e-3`
- Early stopping
- Learning-rate decay
- Epoch별 checkpoint
- 총 update 수를 고정하여 corpus coverage와 step 수를 분리

### P3 — Corpus power

- Diagnostic minimum: 30 independent games
- Gate-grade: 80–100 held-out games
- Practical starting point: 200 train + 100 held-out = 300 games
- 16-step-only 대신 longer/full-game horizon
- opening/color/phase/length stratification
- strength 판단을 위한 terminal/value corpus 별도 확보

### P4 — Capacity/Hybrid

Mixed-negative ablation, multi-seed optimization, corpus power가 해결된 뒤에만
model enlargement와 Hybrid를 비교한다. 현재 capacity는 첫 번째 suspect가 아니다.

## 8. Promotion 정책

현재 canonical artifact는 계속 다음과 같이 분류한다.

```text
promotable=false
strength_claim=none
web_default=unchanged
self_play_producer_promotion=forbidden
```

All-open control의 상대 우위가 production semantics의 승인이 아니다. 다음 gate를
재실행하기 전에는 checkpoint 승격, web default 변경, self-play producer 변경을
하지 않는다.

## 9. 관련 문서

- [Phase 1 결과](ai-v4-phase1-status.md)
- [V4 구현 상태](ai-v4-implementation-status.md)
- [V4 factorized policy 계약](ai-v4-factorized-policy.md)
- [Phase 1 A/B runner](../train/v4/phase1_ab.py)
- [Opened training loss](../train/v4/loss.py)
- [Rust runtime](../crates/augment-ai/src/runtime_v4.rs)
- [Legacy quarantine 정책](ai-v4-legacy-artifact-quarantine.md)
