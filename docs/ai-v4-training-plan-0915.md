# Augmentia V4 실제 학습 계획 — 0915 native data

작성일: 2026-09-19  
기준 seal: `ai/v4-training-base-20260919`  
기준 commit: `f4e64c35f5adcb7f361ed845fb55eb4d123b370b`  
대상 ruleset: `augment_chess_20260915`

이 문서는 V4 factorized policy를 **실제 학습 가능한 운영 파이프라인**으로 연결하고,
0915 규칙에서 생성한 native self-play만으로 첫 실전 V4 모델을 만드는 계획을 고정한다.

[V4 factorized policy contract](ai-v4-factorized-policy.md),
[V4 구현 현황](ai-v4-implementation-status.md),
[V4 training base seal](ai-v4-training-base-20260919.md)을 전제로 한다.

## 1. 핵심 결정

### 1.1 0818 policy/value target을 0915 학습에 사용하지 않는다

`augment_chess_20260818` self-play의 MCTS visit과 value/terminal target은
`augment_chess_20260915` V4 학습 데이터로 변환하거나 혼합하지 않는다.

카드 메커니즘이나 transition semantics가 변경된 경우, 현재 decision의 합법 행동 집합이
우연히 같더라도 MCTS target은 안전하다고 볼 수 없다. MCTS visit과 root value는
미래 rollout의 상태 전이와 terminal value에 의존하므로, 탐색 중 한 번이라도 이전
ruleset semantics를 사용하면 현재 root의 target까지 영향을 받을 수 있다.

따라서 다음은 금지한다.

- 0818 `mcts_visits`를 V4 prefix target으로 변환해 0915 policy loss에 사용
- 0818 terminal result/root value를 0915 value loss에 사용
- ruleset provenance가 다른 replay를 하나의 V4 training dataset으로 묵시적으로 혼합
- 합법 행동 집합이 같다는 이유만으로 policy target을 호환된 것으로 간주

### 1.2 허용 가능한 과거 자산

과거 모델이나 데이터의 재사용은 **규칙 의미와 독립적임을 별도로 증명한 경우**에만 허용한다.

예:

- 일반 체스 보드 representation pretraining
- 규칙과 무관한 저수준 encoder 초기화
- synthetic fixture를 이용한 tensor/ONNX/runtime 회귀 검증

이 경우에도 0915 self-play target과 동일한 provenance로 취급하지 않는다.
초기화 source와 hash를 checkpoint metadata에 별도로 남긴다.

첫 0915 V4 baseline은 가능한 한 이 예외에도 의존하지 않고
0915 native data만으로 policy/value target을 구성한다.

## 2. 목표

첫 번째 목표는 즉시 강한 모델을 만드는 것이 아니라 다음 end-to-end 경로를
재현 가능하게 만드는 것이다.

```text
0915 Engine
  -> V4 self-play
  -> strict replay validation
  -> V4 training dataset
  -> CUDA training
  -> checkpoint
  -> split ONNX export
  -> Python/ONNX/Rust parity
  -> paired arena
  -> 다음 세대 self-play
```

모든 단계는 ruleset, source seal, model, seed, search 설정을 추적할 수 있어야 한다.

## 3. 공통 provenance 계약

모든 기록형 self-play와 training run은 최소 다음 identity를 보존한다.

- source seal tag와 exact Git commit
- ruleset ID와 manifest SHA-256
- model/checkpoint SHA-256
- exported V4 bundle metadata SHA-256
- state encoder/scorer graph SHA-256
- seed 또는 seed domain
- MCTS simulations
- chance outcome sample 수
- Gumbel/root-selection 설정
- temperature 및 적용 decision 범위
- replay schema와 factorized-policy schema ID
- trainer configuration
- CUDA/PyTorch/ONNX Runtime 환경

dirty worktree에서 **기록형 self-play 또는 공식 training run을 시작하지 않는다**.
개발 smoke test는 예외지만 artifact 이름과 metadata에서 명확히 `smoke`로 표시하고
promotion/arena 기준 데이터로 사용하지 않는다.

## 4. 단계별 계획

### M0 — 0915 V4 native self-play CLI

현재 `crates/augment-ai/src/self_play_v4.rs`의 library-level 생성기를
운영 가능한 CLI에 연결한다.

CLI는 다음을 지원해야 한다.

- ruleset 선택. 첫 production 경로는 `20260915`만 허용
- V4 ONNX bundle 또는 명시적인 bootstrap evaluator 선택
- 시작 game ID / game count
- deterministic per-game seed
- simulations
- chance outcome samples
- temperature / temperature decision cutoff
- max decisions
- output directory
- worker 수
- overwrite 방지
- provenance 출력

기록은 `augmentia.replay.game.v4`와
`augmentia.factorized-mcts-visits.v2` 계약을 그대로 사용한다.

#### M0 gate

- 같은 seed/config로 게임 record가 bitwise 또는 canonical JSON 의미상 재현 가능
- 모든 decision의 complete visits 합이 completed simulations와 동일
- played action은 양의 visit을 가짐
- legal complete action과 factorized trie path가 1:1
- illegal action 0
- chance/search seed가 live RNG/private future state에 의존하지 않음
- dirty worktree production 실행 차단

### M1 — 소규모 0915 native replay 생성

첫 smoke dataset은 대규모가 아니라 계약 검증용으로 만든다.

초기 권장 범위:

- 20~50 games
- 100~200 simulations/decision
- chance outcome samples 8
- 고정된 seed range
- baseline evaluator 한 종류만 사용

이 단계에서는 게임 strength보다 데이터 무결성을 우선한다.

#### 검증

각 game과 전체 corpus에 대해 다음을 검사한다.

- replay schema validation
- action/observation schema validation
- decision index 연속성
- visit conservation
- factorized prefix target 재계산 일치
- trie SHA-256 일치
- played action / visit 일치
- terminal result / actor perspective 일치
- ruleset ID/hash 일치
- game seed 중복 없음
- model provenance 일치

M1 dataset은 이후 trainer smoke에 사용할 수 있지만,
규모가 작다면 공식 strength 평가 데이터로 사용하지 않는다.

### M2 — production V4 trainer CLI

현재 `train/v4/train.py`의 library trainer를 실제 운영 CLI로 확장한다.

필수 기능:

- replay/dataset 입력
- train/validation 분리
- CUDA device 선택
- batch size, LR, epochs
- deterministic seed
- baseline/hybrid encoder 선택
- checkpoint 저장 주기
- resume
- validation loss
- history/metrics JSON
- Git/ruleset/dataset/checkpoint provenance

#### 0915 vocabulary 고정

현재 example에서만 vocabulary를 만들어서는 안 된다.
학습 vocabulary는
`train/v3/fixtures/0915/runtime_vocabulary_v3.json`의 full runtime vocabulary를
기준으로 고정한다.

학습 example에 등장하지 않은 token도 embedding index를 유지해야 하며,
ONNX exporter가 요구하는 runtime vocabulary와 동일한 index contract를 가져야 한다.

#### resume 계약

checkpoint sidecar에 저장된 vocabulary/model config를 실제 model 재구성에 사용한다.
새 input example에서 vocabulary를 다시 생성해서 strict-load하는 방식은 허용하지 않는다.

resume 시 최소 다음이 일치해야 한다.

- ruleset identity
- vocabulary
- model config
- encoder kind
- factorized policy schema
- source checkpoint hash

### M3 — CUDA overfit / end-to-end smoke

RTX 3060 12GB에서 작은 native dataset을 의도적으로 overfit한다.

초기 baseline config:

- encoder: `baseline`
- channels: 128
- model_dim: 128
- residual_blocks: 6
- runtime_message_passes: 2
- optimizer: Adam
- initial LR: 1e-3
- batch size: VRAM 측정 후 8~32 범위 결정

목표는 strength가 아니다.

#### M3 gate

- CUDA training이 안정적으로 실행
- policy loss 감소
- value-supervised sample이 있는 경우 value loss 감소
- checkpoint 저장/재로드 후 동일 validation 결과
- resume 후 vocabulary/index 변화 없음
- NaN/Inf 없음
- memory growth/leak 없음

작은 데이터에서 overfit조차 하지 못하면 장기학습으로 넘어가지 않는다.

### M4 — checkpoint -> split ONNX -> Rust parity

학습 checkpoint를 다음 bundle로 export한다.

- state encoder ONNX
- factorized scorer ONNX
- V4 bundle metadata

#### M4 gate

동일 sample에서:

- PyTorch eager logits/value
- Python ONNX logits/value
- Rust `V4OnnxRuntime` logits/value

가 허용 오차 내에서 일치해야 한다.

추가로 확인한다.

- full 0915 runtime vocabulary
- graph hash
- checkpoint hash
- ruleset manifest hash
- legal-only conditional normalization
- complete action probability sum
- destination 63 / promotion regression
- lazy prefix scorer 경로

M4까지 통과한 모델만 self-play producer로 사용할 수 있다.

### M5 — 첫 real 0915 bootstrap corpus

M4를 통과한 Model-0으로 수백 game 규모의 native corpus를 생성한다.

초기 목적은 첫 학습 가능한 분포를 확보하는 것이다.
Model-0 자체가 약하더라도 ruleset semantics가 정확한 것이 우선이다.

생성 시 dataset manifest에 producer model을 명시한다.

```text
producer_model = Model-0
ruleset = augment_chess_20260915
source_seal = ai/v4-training-base-20260919
```

truncated game은 value supervision에서 terminal game과 구분한다.
현재 trainer의 value 계약을 임의로 바꿔 root bootstrap value를 terminal label처럼
사용하지 않는다.

### M6 — 첫 실제 Model-1 학습

M5 corpus로 첫 실전 baseline V4를 학습한다.

첫 세대에서는 구조 변경을 최소화한다.

- baseline encoder 유지
- factorized scorer 유지
- architecture sweep 최소화
- 데이터/학습/runtime pipeline 문제와 모델 구조 문제를 분리

checkpoint마다 다음을 저장한다.

- train/validation loss
- policy/value sample 수
- examples/games 수
- optimizer/LR
- elapsed training steps
- dataset manifest hash
- source seal
- model SHA-256

### M7 — paired arena 및 다음 세대

Model-0과 Model-1을 동일한 조건에서 paired arena한다.

비교 시 최소 다음을 고정한다.

- ruleset
- start seed/state pairing
- search budget
- chance outcome samples
- temperature
- color assignment
- wall-clock 또는 simulation budget 기준

arena 결과는 모델 promotion을 자동 의미하지 않는다.
게임 수와 통계적 불확실성을 함께 기록한다.

Model-1이 self-play producer로 사용 가능한 수준이라고 판단된 경우:

```text
Model-1
 -> 0915 native self-play generation 2
 -> Model-2 training
 -> paired arena
 -> repeat
```

로 세대를 진행한다.

## 5. replay buffer 운영 원칙

self-play 세대가 늘어나도 모든 데이터를 동일 비율로 무한 누적하지 않는다.

manifest 수준에서 최소 다음을 추적한다.

- producer model generation
- model/checkpoint hash
- game ID range
- seed range
- ruleset identity
- search configuration
- terminal/truncated 비율
- policy/value eligible decisions
- split
- artifact SHA-256

향후 mixture 전략은 별도 실험으로 결정한다.

후보:

- 최근 세대 중심 sliding window
- 최근 세대 높은 가중치 + 이전 세대 일부 유지
- tactical/rare-state replay 별도 quota
- draft/card mechanic coverage quota

어떤 mixture도 dataset provenance 없이 자동 병합하지 않는다.

## 6. Hybrid encoder 정책

Hybrid encoder는 첫 production baseline을 만드는 선행 조건이 아니다.

진행 순서:

1. baseline encoder로 전체 pipeline 안정화
2. 동일 native dataset으로 baseline 모델 학습
3. 동일 split/seed/budget으로 Hybrid 학습
4. latency, memory, validation, arena를 A/B

V3 encoder checkpoint를 Hybrid에 부분 로드했다는 이유만으로
학습된 Hybrid라고 간주하지 않는다.

Hybrid 채택은 parameter count가 아니라
**strength-per-compute와 runtime cost**로 판단한다.

## 7. promotion 전 필수 gate

V4를 기본 모델로 승격하기 전에 최소 다음을 통과해야 한다.

- 0915 native replay만으로 재현 가능한 training lineage
- replay/trie/visit conservation 오류 0
- full runtime vocabulary 호환
- checkpoint resume 검증
- eager/ONNX/Rust parity
- self-play runtime 안정성
- paired arena 결과와 불확실성 기록
- fixed-budget latency/p95 측정
- 실제 학습 모델로 V4 baseline 재봉인
- Web/CLI 배포 경로에서 model/ruleset mismatch 차단

## 8. 명시적으로 하지 않는 것

첫 V4 production cycle에서는 다음을 동시에 진행하지 않는다.

- 0818 replay target bootstrap
- Hybrid를 기본 encoder로 즉시 승격
- NN-only beam/exact fast path 도입
- adaptive compute controller 도입
- search-risk predictor 도입
- tactical verifier 구조 변경
- 여러 architecture를 한꺼번에 sweep

먼저 native data -> training -> ONNX -> Rust -> arena의 단일 경로를 안정화한다.

## 9. 실행 순서

현재 구현 순서는 다음으로 고정한다.

| 순서 | milestone | 산출물 |
|---:|---|---|
| 1 | M0 | `self_play_v4` production CLI |
| 2 | M1 | 0915 native smoke replay |
| 3 | M2 | production trainer CLI |
| 4 | M3 | CUDA overfit checkpoint |
| 5 | M4 | trained split ONNX bundle + parity report |
| 6 | M5 | 첫 real native bootstrap corpus |
| 7 | M6 | Model-1 checkpoint/bundle |
| 8 | M7 | paired arena + generation-2 replay |
| 9 | 이후 | baseline/Hybrid A/B 및 반복 학습 |

현재 다음 구현 작업은 **M0 — 0915 V4 native self-play CLI**다.
