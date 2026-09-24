# Augmentia V4 개선 우선순위 조사

## 1. 조사 성격과 결론

이번 조사는 Agent Team의 아키텍처/MCTS, 데이터/학습, 런타임/제품, 독립
Skeptic 관점을 합쳐 현재 `augmentia` 작업 트리의 V4를 읽기 전용으로 검토한
결과이다. 조사 기준 HEAD는 `f2c7afa52af140b35299edafb173dc4e011bc30c`이며,
이번 조사에서는 소스 파일을 수정하지 않았다.

### 한 줄 결론

**현재 V4의 가장 우선 개선 과제는 모델 용량이나 Hybrid 인코더를 늘리는 것이
아니라, MCTS의 실제 `opened` 후보 집합을 replay target과 학습 loss 끝까지
보존하도록 target/replay 계약을 수정하는 것이다.**

이 작업은 단순한 성능 최적화가 아니라 **policy target correctness의 P0
수정**이다. 현재 계약과 실제 학습 경로가 어긋나므로, 이 문제를 해결하지 않은
상태에서 데이터·모델·탐색을 함께 변경하면 인과를 분리할 수 없고 잘못된
target을 더 크게 학습할 위험이 있다.

## 2. Agent Team 의견 요약

| 관점 | 우선 권고 | 이유 |
|---|---|---|
| 아키텍처/MCTS | `opened_mask` end-to-end 보존 | MCTS support와 replay target support가 다르면 policy loss가 선택 편향으로 오염된다. |
| 데이터/학습 | 데이터 규모·capacity보다 target-contract 수정 우선 | 기존 target의 semantics가 명시적이지 않은 상태에서 corpus를 늘리면 오염을 복제한다. |
| 런타임/제품 | target 수정 후 fail-closed 승격 gate | 현재 arena는 측정 보고서일 뿐 promotion 판정을 강제하지 않고 p95/same-wall-clock도 없다. |
| Skeptic/portfolio | 증거 체인 복구 후 baseline-first | Hybrid·모델 확대·MCTS rewrite는 clean data와 고정 평가가 있을 때만 정보량이 생긴다. |

네 관점의 표현은 달랐지만 실행 순서에는 큰 충돌이 없다.

```text
opened-mask correctness
  -> clean 0915 native corpus
  -> baseline Model-1
  -> ONNX/Rust parity
  -> fixed-simulation + same-wall-clock paired gate
  -> 그다음 MCTS ablation 또는 Hybrid A/B
```

## 3. P0: `opened_mask` target 계약 불일치

### 3.1 문서가 요구하는 semantics

V4 정책 계약은 factorized child를 만들 때 **legal이면서 실제로 opened된
child만 loss의 후보 집합에 넣도록** 정한다. legal이지만 닫힌 child는
`visit=0`이라고 해서 부정확한 negative target으로 사용하면 안 된다.

- 계약: [`docs/ai-v4-factorized-policy.md`](ai-v4-factorized-policy.md#L38-L43)
- loss의 의도: [`train/v4/loss.py:32-50`](../../train/v4/loss.py#L32-L50)

### 3.2 Rust 검색이 보관하는 정보와 replay가 보관하는 정보가 다르다

1. 기본 MCTS는 `gumbel_candidates=16`이다.
   - [`crates/augment-ai/src/mcts_v4.rs:23-53`](../../crates/augment-ai/src/mcts_v4.rs#L23-L53)
   - root candidate 수는 legal action 수, 16, simulation 수 중 작은 값으로 제한된다.
   - [`crates/augment-ai/src/mcts_v4.rs:873-905`](../../crates/augment-ai/src/mcts_v4.rs#L873-L905)

2. 선택된 root action의 trie 경로만 `opened=true`가 된다.
   - [`crates/augment-ai/src/mcts_v4.rs:506-524`](../../crates/augment-ai/src/mcts_v4.rs#L506-L524)

3. 그러나 `SearchDiagnosticsV4`와 `SearchResultV4`에는 최종 root
   action별 `opened` mask가 없다.
   - [`crates/augment-ai/src/mcts_v4.rs:199-214`](../../crates/augment-ai/src/mcts_v4.rs#L199-L214)
   - [`crates/augment-ai/src/mcts_v4.rs:1027-1085`](../../crates/augment-ai/src/mcts_v4.rs#L1027-L1085)

4. factorized target record도 prefix, children, visit count만 가진다.
   - [`crates/augment-ai/src/factorized_v4.rs:58-63`](../../crates/augment-ai/src/factorized_v4.rs#L58-L63)
   - [`crates/augment-ai/src/data_v4.rs:20-31`](../../crates/augment-ai/src/data_v4.rs#L20-L31)

5. self-play는 aggregate된 complete visits만 replay에 기록한다.
   - [`crates/augment-ai/src/self_play_v4.rs:115-149`](../../crates/augment-ai/src/self_play_v4.rs#L115-L149)

즉, 검색 내부의 `opened` 상태가 replay 경계에서 사라진다.

### 3.3 Python trainer가 누락된 mask를 all-open으로 복원한다

- replay example에는 opened mask 필드가 없다.
  - [`train/v4/replay.py:24-32`](../../train/v4/replay.py#L24-L32)
  - [`train/v4/replay.py:79-103`](../../train/v4/replay.py#L79-L103)
- trainer는 complete visits만 collator에 전달한다.
  - [`train/v4/train.py:251-260`](../../train/v4/train.py#L251-L260)
  - [`train/v4/train.py:395-406`](../../train/v4/train.py#L395-L406)
- collator의 기본값은 모든 legal action을 opened로 둔다.
  - [`train/v4/batch.py:143-175`](../../train/v4/batch.py#L143-L175)
- replay tensorization도 opened mask를 생략한다.
  - [`train/v4/batch.py:249-265`](../../train/v4/batch.py#L249-L265)
- loss는 `legal_mask & opened_mask`를 softmax 후보로 사용한다.
  - [`train/v4/loss.py:32-43`](../../train/v4/loss.py#L32-L43)

Python unit test에는 opened mask의 의미가 반영되어 있지만, 현재 테스트는
명시적으로 mask를 collator에 넣는 수준이고 실제 replay→trainer 경로의
end-to-end 회귀 테스트는 아니다.

- [`train/v4/test_batch.py:48-63`](../../train/v4/test_batch.py#L48-L63)

### 3.4 실제 production corpus에서 확인되는 신호

현재 256-game native corpus의 한 decision은 다음 상태를 보인다.

- legal complete action: **28개**
- positive visit: **16개**
- replay record에 opened metadata: **없음**

- 원본 위치: [`game-00000000.json`](../../augmentia-v4-artifacts/m5-v41-model0-prod-s00-20260919/games/m5-v41-model0-prod-s00-20260919-game-00000000.json#L151898-L152159)

positive visit 16개가 기본 Gumbel cap 16과 일치한다는 점은 opened
정보가 누락됐을 때 방문하지 않은 legal action을 0으로 보게 된다는
신호다. 다만 원본 record만으로 각 action의 정확한 opened 여부를
소급해 증명할 수는 없다.

영향 범위는 다음과 같이 표현해야 한다.

- 28개 각각이 독립적인 negative label이라는 뜻은 아니다.
- 모든 descendant가 닫힌 row는 `row_visits=0`, `prefix_mass=0`이므로 현재
  loss 기여가 0이다.
- 실제 문제는 **positive sibling이 있는 mixed prefix row**에서 closed
  sibling logits가 all-legal softmax normalizer에 포함되는 것이다.
- 그 결과 opened sibling의 학습 확률이 낮아지는
  false-negative/censoring이 발생한다.

즉, target vector의 opened sibling 비율 자체가 항상 바뀌는 것은 아니어도,
모델이 전체 legal support에 대해 정규화되도록 강제되어 closed sibling을
억제하는 selection bias가 생긴다. 이것은 MCTS training semantics에 직접
영향을 주며 schema 위반 여부는 코드 자체로 확정된다. 전체 corpus에서 mixed
row가 얼마나 많은지는 별도 계측해야 한다.

## 4. 현재 모델의 강도·배포 상태

### 4.1 epoch2는 유망하지만 승격 증거는 아니다

웹에 제공 중인 Model-1 epoch2와 Model-0의 combined arena는 다음과 같다.

- 48 paired seeds / 96 games
- candidate 55승, opponent 41승
- pair score mean: **57.29%**
- 95% CI: **48.54%–66.04%**

- [`combined48-aggregate.json`](../../augmentia-v4-artifacts/m7-v41-model1-epoch2-vs-model0-extra32pairs-20260921/combined48-aggregate.json#L12-L25)

CI가 50%를 포함하므로 통계적으로 명확한 superiority가 아니다. V3과의
동일 ruleset 비교도 아니며, same-wall-clock 조건과 trained-model p95가
없으므로 promotion 근거로 사용할 수 없다.

### 4.2 최종 epoch5는 명백히 불안정하다

- epoch2 validation loss: 약 `4.0994`, value loss: 약 `0.99997`
- epoch5 validation loss: 약 `4.9235`, value loss: 약 `1.82185`
- epoch5 16-pair arena: 14승 18패, score `43.75%`

- [`model-1-v41-prod/history.json:33-48`](../../augmentia-v4-artifacts/model-1-v41-prod/history.json#L33-L48)
- [`model-1-v41-prod/history.json:98-120`](../../augmentia-v4-artifacts/model-1-v41-prod/history.json#L98-L120)
- [`m7-v41-model1-final-vs-model0-prod-16pairs-20260921/aggregate.json`](../../augmentia-v4-artifacts/m7-v41-model1-final-vs-model0-prod-16pairs-20260921/aggregate.json#L11-L21)

현재 웹 기본 경로는 epoch2다. 이는 runtime 통합과 preview 운영이지 공식
champion 승격이 아니다.

- [`crates/augment-web/src/app.rs:16-33`](../../crates/augment-web/src/app.rs#L16-L33)
- [`crates/augment-web/src/app.rs:1092-1147`](../../crates/augment-web/src/app.rs#L1092-L1147)

### 4.3 현재 benchmark는 strength benchmark가 아니다

0915 benchmark의 674–702 simulations/s, 285–297 ms/search 수치는
synthetic regression fixture의 재현성·throughput 기록이다. trained V4의
실전 strength와 p95 latency 증거로 해석하면 안 된다.

- [`docs/ai-v4-baseline.md:100-139`](ai-v4-baseline.md#L100-L139)
- [`docs/ai-v4-baseline.md:172-178`](ai-v4-baseline.md#L172-L178)

## 5. 2순위 이상에서 다룰 문제

### P1-A: 승격 gate를 fail-closed로 만들기

현재 `arena_v4`는 report만 만들며, 결과가 나빠도 성공 종료할 수 있다.
truncated/nonterminal을 기록할 뿐 terminal rate, complete pair ratio,
recall, p95, CI 하한에 대한 promotion 판정이 없다.

- [`crates/augment-cli/src/bin/arena_v4.rs:115-120`](../../crates/augment-cli/src/bin/arena_v4.rs#L115-L120)
- [`crates/augment-cli/src/bin/arena_v4.rs:354-419`](../../crates/augment-cli/src/bin/arena_v4.rs#L354-L419)

### P1-B: trained model의 동일 wall-clock·p95 측정

현재 benchmark schema/report는 mean·median 위주이고 warm repeat·reference
search를 latency sample에서 제외한다. p95와 동일 wall-clock calibration이
없다.

- [`crates/augment-cli/src/bin/benchmark_v4_search.rs:25-57`](../../crates/augment-cli/src/bin/benchmark_v4_search.rs#L25-L57)
- [`crates/augment-cli/src/bin/benchmark_v4_search.rs:188-216`](../../crates/augment-cli/src/bin/benchmark_v4_search.rs#L188-L216)

### P1-C: 학습 checkpoint와 value objective 안정화

현재 V4 trainer에는 production-strength를 위한 robust best-checkpoint
selection/early stopping과 scheduler가 충분히 성숙하지 않다. 최종 epoch가
train loss를 낮추면서 validation을 크게 악화시켰다. value label은 terminal
game outcome을 사용하며, `root_value`는 pre-search NN raw value이므로
그대로 value target으로 바꾸는 것은 self-label 위험이 있다.

- [`train/v4/replay.py:89-103`](../../train/v4/replay.py#L89-L103)
- [`crates/augment-ai/src/mcts_v4.rs:479-480`](../../crates/augment-ai/src/mcts_v4.rs#L479-L480)

### P1-D: 웹 preview의 fail-closed/health 상태

웹은 모델 로드 실패를 `Result`로 저장하고 서버를 계속 띄우며, view
evaluation 실패를 0.0으로 삼킨다. `/api/suggestions`는 동기 MCTS를 실행하므로
long request, stale request, provider/p95 상태가 구분되지 않는다.

- [`crates/augment-web/src/app.rs:57-63`](../../crates/augment-web/src/app.rs#L57-L63)
- [`crates/augment-web/src/app.rs:1190-1198`](../../crates/augment-web/src/app.rs#L1190-L1198)
- [`crates/augment-web/src/main.rs:42-50`](../../crates/augment-web/src/main.rs#L42-L50)

## 6. 권장 실행 순서

### Phase 0 — 1~3일: contract correctness

1. target schema를 bump한다. 예: `augmentia.factorized-mcts-visits.v3`.
2. replay schema도 새 target lineage를 명시적으로 요구하도록 bump한다.
3. MCTS가 root action별 `opened_action_mask`를 반환한다.
4. factorized prefix record의 child와 같은 길이의 `opened_masks`를 저장한다.
5. hash, validation, Rust/Python round-trip에 mask를 포함한다.
6. Python replay example와 trainer가 mask를 누락하지 않게 한다.
7. production 경로에서 mask 누락을 fail-closed로 처리한다.
8. 28+ legal action, 16 candidates, mixed prefix row를 사용하는 property test를 추가한다.

### Phase 1 — 1~2주: clean corpus와 최소 A/B

1. 기존 V2 corpus와 그 target으로 학습한 checkpoint를 artifact 보존 +
   `legacy/non-promotable/contaminated-target-semantics`로 격리한다.
2. producer bundle, seed, config가 있으면 deterministic 재생성을 우선한다.
3. 단순 재사용이 아니라, 20–50개 native smoke로 무결성부터 통과한다.
4. 동일 producer/seed/trajectory/init/split에서 legacy all-open arm과
   opened-only arm을 비교한다.
5. baseline encoder만 사용해 Model-1을 학습한다.
6. eager/Python ONNX/Rust parity를 수행한다.

### Phase 2 — 2~4주: strength 증거 봉인

1. 최소 100 paired seeds와 색 교환을 실행한다.
2. fixed-simulation과 fixed-wall-clock을 각각 기록한다.
3. candidate recall, tactical quality, terminal-value calibration, coverage,
   truncated rate, p50/p95, paired score CI를 하나의 report로 봉인한다.
4. 사전에 정한 superiority threshold를 만족할 때만 registry/web default
   승격을 허용한다.
5. epoch2는 그 전까지 experimental preview로 유지한다.

### Phase 3 — 조건부 후속 작업

다음 항목은 clean corpus와 고정 평가가 있는 경우에만 순서대로 진행한다.

1. Gumbel candidate 16/32/64 및 simulation budget ablation
2. per-worker runtime/session과 row batching
3. Hybrid encoder A/B
4. 모델 capacity 확대
5. adaptive compute/search-risk predictor

## 7. 성공 기준

### Contract gate

- positive visit는 반드시 opened이다.
- opened zero-visit child는 허용된다.
- closed child는 mixed prefix의 loss denominator에서 제외된다.
- prefix/action conservation, simulation 합, legality, path 1:1 오류는 0이다.
- 같은 seed/config에서 canonical replay가 재현된다.
- legacy target과 새 target은 trainer에서 명시적으로 섞이지 않는다.

### Model/evaluation gate

- opened-only arm이 legacy arm보다 held-out conditional KL 또는 tail recall을
  일관되게 개선한다.
- 완전 action NLL/top-1/ECE는 비열화한다.
- paired score의 사전 정의된 CI 기준을 통과한다.
- terminal/truncated, p95, worker scaling, CPU/CUDA provider가 report에 있다.
- gate 통과 checkpoint hash만 registry와 web default에 들어간다.

## 8. 가장 강한 반대와 기각 조건

### 반대: “모든 legal action을 0 target로 두는 것이 더 강한 지도일 수 있다”

이 주장이 성립하려면 해당 action이 실제로 검색에서 열리고 비교되었어야
한다. 현재 Gumbel cap과 all-open 기본값은 그 조건을 만족하지 않으며,
문서 계약도 closed action을 negative로 쓰지 않도록 정한다.

### 반대: “MCTS candidate cap 16을 먼저 늘려야 한다”

wide stress에서는 484 legal action 중 16 candidate만 열리고 root coverage가
약 3.31%로 낮다. 따라서 검색 병목 가능성은 진짜다.

- [`docs/ai-v4-baseline.md:102-109`](ai-v4-baseline.md#L102-L109)
- [`docs/ai-v4-baseline.md:135-137`](ai-v4-baseline.md#L135-L137)

다만 다음 조건을 모두 만족하기 전에는 search-first로 전환하지 않는다.

- clean 0915 holdout을 사용한다.
- 동일 model·seed·search semantics로 16/32/64를 비교한다.
- held-out regret/optimal mass가 개선된다.
- paired arena도 개선된다.
- p95/VRAM budget을 넘지 않는다.

synthetic coverage나 speed만 좋아지고 real holdout strength가 개선되지 않으면
그 search 최적화 주장을 기각한다.

## 9. 조사 한계

- 이번 조사는 read-only code/document/artifact 감사이며 새로운 테스트
  실행이나 모델 학습을 수행하지 않았다.
- 실제 corpus 전체의 mixed-row 수와 false-zero가 loss에 미치는 총 mass는
  다음 단계에서 계측해야 한다.
- 기존 V2 record에는 opened metadata가 없으므로 공식 opened-only 학습에
  그대로 사용하면 안 된다. deterministic 재생성이나 명시적 migration이
  가능하더라도 새 schema와 새 provenance로 검증해야 한다.
- 0915 ruleset의 source coverage와 live-site parity는 별도 gate로 남아 있다.

## 최종 결정

**다음 작업 이름은 “V4 model enlargement”가 아니라
“V4 opened-mask target contract recovery and promotion-gate restoration”로
등록한다.**

이 gate를 통과하기 전에는:

- 기존 Model-1을 promotion/self-play producer로 승격하지 않는다.
- epoch2 web 제공은 preview로만 취급한다.
- Hybrid·capacity·synthetic search coverage를 모델 개선의 주 근거로
  사용하지 않는다.

이 gate를 통과한 뒤에야 clean native data와 baseline-first 실험을 통해
모델 구조, MCTS budget, runtime 최적화의 실제 효과를 비교한다.
