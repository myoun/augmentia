# Augmentia V4 구현 현황

이 문서는 V4 factorized search의 구현 상태를 기록한다.
설계 계약은 [V4 factorized policy contract](ai-v4-factorized-policy.md)를 참고한다.
2026-09-19 현재 Phase 0/1 이후 lazy prefix scoring, Hybrid Encoder 구현과 0915 고정 예산 baseline 재봉인까지 반영했다.
실제 학습 파이프라인의 실행 순서와 데이터 오염 방지 원칙은 [0915 native V4 training plan](ai-v4-training-plan-0915.md)에 고정한다.

## 요약

- **Phase 0 — chance 정확성: 완료.**
- **opened-mask target contract recovery: 완료, contract gate 통과.**
- **구현 Phase 1 — Rust split ONNX 런타임 및 MCTS 연결: 완료, 검증 통과.**
- **개선 Phase 1 — legacy quarantine와 clean 0915 smoke: 완료. baseline paired A/B: 실행 완료, model gate 실패, strength gate 미진행.**
- **Lazy prefix scoring: 완료, 실제 ONNX 경로 검증 통과.**
- **Hybrid Encoder: opt-in 구현 및 eager/ONNX 계약 검증 완료.**
- **0915 고정 예산 baseline: 재봉인 완료.** [baseline 기록](ai-v4-baseline.md)
- V3 대비 대국 품질 우위와 학습된 Hybrid의 우위는 아직 검증하지 않았다.
- 기본 모델 승격, CLI/Web 전체 배포 통합을 완료했다는 의미는 아니다.

## 유지하는 설계 원칙

1. `Engine::legal_actions()`가 합법 행동의 유일한 기준이다.
2. `LegalActionSetV3`와 `FactorizedTrieV4`는 이미 합법인 완전한 행동을 분해한다.
   trie나 scorer가 새로운 Engine 행동을 생성하지 않는다.
3. 가상 prefix에서는 Engine을 적용하거나 actor·backup 부호를 바꾸지 않는다.
4. 탐색의 chance seed와 모델 입력에는 live RNG나 숨겨진 미래 정보를 사용하지 않는다.
5. V3 flat policy와 V4 factorized policy의 모델·학습 target 계약을 구분한다.

## Phase 0: 확률적 전이

- `ObservationV3`와 정확한 합법 행동 집합으로 공개 정보 key를 만든다.
- chance seed에서 `GameState::fingerprint()`와 live RNG 의존성을 제거했다.
- 확률적 행동마다 여러 search-only outcome을 만들고 lane 방문을 고르게 배분한다.
- `chance_outcome_samples`는 `1..=64`, 기본값은 8이다.
- Gumbel seed와 chance/search seed의 도메인을 분리했다.
- 실제 플레이는 `Engine::apply`, 탐색은 `Engine::apply_for_search`를 사용한다.
- replay target은 `augmentia.factorized-mcts-visits.v3`이며,
  chance sample 수와 search semantics provenance 및 root `opened_action_mask`를 기록한다.

## Opened-mask target contract recovery

- MCTS `SearchResultV4`는 legal action 순서의 `opened_action_mask`를 반환한다.
- replay record는 `augmentia.replay.game.v5`로 bump하고 V3 target schema를 명시한다.
- target은 root mask와 child-aligned prefix `opened_masks`를 모두 보존하며,
  checksum은 두 정보를 함께 봉인한다.
- positive visit는 opened, opened zero-visit는 허용, closed positive-visit는 거부한다.
- loss denominator는 `legal_mask & opened_mask`만 사용한다.
- Python replay example과 trainer는 root mask를 필수로 전달한다.
- legacy V2 corpus/checkpoint는 새 trainer에 fail-closed되며 명시적 migration 없이
  승격이나 학습에 사용할 수 없다.

## Clean corpus와 baseline Phase 1

- legacy artifact 8 runs, 304 games, 74 checkpoints를 read-only registry로 격리했다.
- 동일 producer/seed/config로 v5/v3 20-game clean smoke를 재생성했다.
- seed와 trajectory는 legacy reference와 20/20 일치하고, visit target도 320/320 일치한다.
- 7,694 legal action 중 4,600 opened, 3,094 closed였고 mixed prefix row는 1,626개였다.
- 동일 split/init/epoch의 baseline A/B에서 opened-only가 held-out conditional KL,
  complete-action NLL/ECE와 top-1을 개선했다.
- 두 실제 arm의 Python eager/ONNX/Rust parity가 tolerance `2e-4` 안에서 통과했다.
- 20 games가 모두 truncated smoke이고 strength sample이 아니므로 모든 결과는
  `promotable=false`다. 상세 seal은 [Phase 1 결과](ai-v4-phase1-status.md)에 있다.

## 구현 Phase 1: split ONNX 런타임

### 평가 경로

`V4OnnxRuntime`은 `FactorizedEvaluatorV4`를 구현한다.
`MctsV4Tree`는 이 evaluator를 통해 실제 ONNX 추론을 실행한다.

1. 공개 관측값을 V3 tensor 계약으로 변환한다.
2. state encoder가 state embedding, board tokens, piece/card entities, value를 만든다.
3. factorized scorer가 합법 trie prefix의 조건부 child logits를 계산한다.
4. 각 prefix의 합법 child에만 log-softmax를 적용한다.
5. 완전한 행동의 경로를 따라 조건부 log probability를 합산한다.

따라서 `action_logits`는 완전한 행동의 log probability이다.
지수화한 행동 확률의 합은 1이어야 한다.

완전한 정책 평가 API는 parity와 검증을 위해 유지하지만, MCTS 내부 ONNX 경로는
`prepare_prefix_evaluation_v4`와 `PrefixScorerV4`를 통해 **실제로 방문한 합법 prefix row만 lazy 평가**한다.
single-child prefix는 scorer 호출 없이 처리하고, 평가한 row는 search node에서 재사용한다.

### 캐시와 검증

- encoder 캐시 key는 직렬화한 `ObservationV3`이다.
- 캐시에 전체 `GameState`나 live RNG를 저장하지 않는다.
- 최대 64개 상태를 보관하며, 한도에 도달하면 캐시를 비운다. LRU는 아니다.
- 같은 공개 관측값에서 live RNG만 달라지면 encoder를 재사용한다.
- schema ID, ruleset/manifest identity, 전체 런타임 어휘, graph SHA-256을 검증한다.
- 선언된 ONNX 입출력 계약과 실제 dtype/rank를 검사한다.
- 실행 출력의 shape, 유한성, illegal child mask와 확률 합을 검사한다.
- 런타임 옵션은 `V3RuntimeOptions`를 재사용하지만 V3 모델 loader로 읽지 않는다.

### 수정한 주요 오류

**불완전한 fixture 어휘**

초기 fixture는 일부 사례에서 수집한 어휘만 포함했다.
Rust의 필수 런타임 토큰 검증과 불일치하여 로딩이 실패했다.
필수 토큰을 모델 생성 전 어휘에 합치고 모델·ONNX graph·기대 출력을 함께 재생성했다.
exporter는 필수 토큰이 누락된 어휘를 거부한다.
metadata의 토큰 목록만 바꾸면 embedding 인덱스가 맞지 않으므로 그렇게 수정하면 안 된다.

**비승격 인덱스의 embedding Gather 오류**

승격이 아닌 `destination=63`도 piece-kind embedding에 먼저 들어가던 오류가 있었다.
embedding 크기는 59이므로 기존 ONNX graph에서 범위 오류가 발생했다.
`train/v4/model.py::encode_factor_edges`에서 다음과 같이 수정했다.

```python
promotion_indexes = torch.where(is_promotion, raw_index, torch.zeros_like(raw_index))
```

사용하지 않는 embedding 인덱스는 **조회 전에** 마스킹해야 한다.
조회 후 결과에 mask를 곱하거나 `clamp(min=0)`만 적용해도 잘못된 양수 인덱스는 막지 못한다.
실제 승격 인덱스의 범위 검증은 유지했다.
기존 번들에서 실패를 재현한 뒤 새 번들로 같은 Rust 테스트가 통과하는 것을 확인했다.

## 검증 기록

| 검증 | 결과 |
|---|---|
| `cargo test --workspace --exclude augment-snapshot` | 통과, `augment-ai` 112개 포함 |
| `cargo clippy --workspace --all-targets --exclude augment-snapshot -- -D warnings` | 통과 |
| Python 전체 unittest | 167개 통과 |
| Python V4 unittest | 86개 통과 |
| P0–P2 focused tests | phase1_ab·shadow trainer·recovery sweep 포함 V4 회귀에 포함 |
| current-code full A/B | 20 games, 224 train / 96 held-out decisions, 3 epochs, model gate `fail` |
| P0 contract A/B smoke | 1 epoch, 6 validation blocks, gate `inconclusive`, `promotable=false` |
| P1 shadow-negative smoke | λ `0/0.01/0.05/0.1`, 1 epoch, 4 sealed cells, evidence 부족 |
| P2 recovery sweep | 4 seeds × 2 learning rates, 1 epoch/cell, 8 sealed cells, evidence 부족 |
| 실제 opened-only checkpoint eager/Python ONNX/Rust parity | tolerance `2e-4` 통과; all-open experiment export 차단 확인 |
| Rust/Python shared v3 target fixture | 동일 hash `e2c30b...9605`, 양쪽 round-trip 통과 |
| clean v5 producer → Rust validator → Python audit | 20 games / 320 decisions, seed·trajectory 20/20 일치 |
| 0915 fixed-budget baseline | standard/draft/wide 재현성 delta 0으로 봉인 |
| `git diff --check` | 통과, 줄바꿈 경고만 발생 |

전체 `cargo test --workspace`의 유일한 실패는 현재 작업 트리에 없는
`rulesets/augment_chess_20260915/sources/raw/main-6qeAcxbX.js`를 요구하는
`augment-snapshot`의 기존 로컬 입력 테스트다. Phase 0 변경 crate와 CLI/web/core
회귀는 위 exclude 실행에서 모두 통과했다.

V4 ONNX 테스트는 **실제 ONNX Runtime으로 실행**했다. skip 결과가 아니다.
다음 항목을 확인했다.

- Python eager와 Rust ONNX의 value 및 완전한 행동 log probability 수치 일치
- Engine이 생성한 `destination=63` 합법 행동의 정상 평가
- export 이후 바뀐 큰 비승격 인덱스의 정상 ONNX 평가
- 공개 관측값 캐시 hit/miss 및 합법 행동 순서 변경 시 올바른 출력 순서
- 별도 cold runtime 간 live RNG 독립적인 MCTS 결과
- 실제 ONNX MCTS에서 chance outcome sampling과 방문 수 보존

기존 PyTorch legacy ONNX exporter의 deprecation/tracing 경고는 남아 있다.
테스트 통과는 모든 입력 분포나 경기 성능 검증을 대신하지 않는다.

## 재현 명령

저장소 루트에서 Rust 테스트를 실행한다.
이 환경에서 확인한 ONNX Runtime DLL 설정 예시는 PowerShell 기준이다.
다른 환경에서는 해당 플랫폼의 라이브러리 경로를 사용한다.

```powershell
$env:ORT_DYLIB_PATH = "D:/augmentia/augmentia/train/.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
cargo test -p augment-ai runtime_v4 -- --nocapture
cargo test --workspace --exclude augment-snapshot
cargo clippy --workspace --all-targets --exclude augment-snapshot -- -D warnings
```

V4 Rust ONNX 테스트는 `ORT_DYLIB_PATH`가 없을 때만 skip한다.
변수가 있는데 라이브러리나 번들이 잘못되면 실패해야 한다.

Python 전체 검증과 fixture 재생성은 `train/`에서 실행한다.

```text
uv run python -m unittest discover -s . -p "test_*.py" -v
uv run python -m unittest v4.test_quarantine_legacy v4.test_audit_phase1_corpus v4.test_phase1_ab v4.test_replay_trainer v4.test_phase1_shadow_ablation v4.test_phase1_recovery_sweep -v
uv run python -m v4.export_fixture_bundle
```

fixture 재생성은 파일을 변경한다. 재생성 후 Rust parity를 다시 실행한다.
fixture는 작은 테스트 모델이며 배포용 학습 모델이나 품질 baseline이 아니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `crates/augment-ai/src/mcts_v4.rs` | Factorized MCTS, seed 도메인, chance sampling |
| `crates/augment-ai/src/runtime_v4.rs` | split ONNX loader, tensor 구성, 조건부 확률 합성 |
| `crates/augment-ai/src/runtime_v4_integration_tests.rs` | 실제 ONNX 캐시·MCTS 통합 검증 |
| `crates/augment-ai/src/self_play_v4.rs` | self-play 및 탐색 provenance |
| `crates/augment-ai/src/data_v4.rs` | V4 replay target 계약 |
| `train/v4/model.py`, `train/v4/batch.py` | 모델과 factor edge tensor 계약 |
| `train/v4/hybrid.py` | opt-in typed-token attention Hybrid Encoder |
| `train/v4/onnx_export.py` | split graph export 및 metadata |
| `train/v4/export_fixture_bundle.py` | ruleset별 Python/Rust 공유 fixture bundle 재생성 |
| `train/v4/quarantine_legacy.py` | legacy v4/v2 artifact read-only quarantine registry |
| `train/v4/audit_phase1_corpus.py` | producer/trajectory/target regeneration audit |
| `train/v4/phase1_ab.py` | 동일 corpus baseline paired A/B, P0 metric/gate, export, parity CLI |
| `train/v4/phase1_shadow_ablation.py` | mixed-row shadow-negative bounded ablation runner |
| `train/v4/phase1_recovery_sweep.py` | 4-seed·2-LR fixed-update recovery sweep runner |
| `train/v3/fixtures/v4_bundle/` | 0818 회귀 테스트 graph, metadata, parity 기대값 |
| `train/v3/fixtures/0915/`, `train/v3/fixtures/v4_bundle_0915/` | 0915 baseline fixture와 split ONNX bundle |

## 남은 작업과 진행 순서

다음 순서는 [V4 개선 우선순위 조사](ai-v4-improvement-priority-audit.md)의
Phase 1~3과 [0915 native V4 training plan](ai-v4-training-plan-0915.md)을 따른다.

1. **Opened-mask target contract recovery: 완료.** v5/v3 lineage, producer→replay→
   trainer 전파, fail-closed legacy 차이와 contract regression을 봉인했다.
2. **Clean 0915 native corpus: 완료.** legacy v4/v2 corpus/checkpoint를 격리하고
   v5/v3 producer로 20-game deterministic smoke를 재생성·검증했다.
3. **Baseline Model-1 A/B: 실행 완료, model gate 실패, 비승격.** 동일 producer,
   trajectory, split, baseline encoder와 initial hash로 opened-only와 all-open experiment을
   비교하고 opened-only eager/Python ONNX/Rust parity를 통과했다. 그러나 conditional KL,
   visited-tail recall, complete-action NLL/top-1/ECE gate를 통과하지 못했다.
4. **P0–P2 bounded recovery: 완료, gate recovery 미확정.** metric contract, mixed-row
   shadow ablation, multi-seed learning-rate sweep을
   [`ai-v4-phase1-recovery-p0-p2.md`](ai-v4-phase1-recovery-p0-p2.md)에 봉인했다.
   λ=0.05–0.1은 현재 smoke에서 KL/NLL을 일부 개선했지만 top-1 trade-off가 남았고,
   6-game validation은 여전히 evidence 부족이다. gate를 다시 통과하기 전에는
   Phase 2 strength 실행도 승격 근거로 사용할 수 없다.
5. **Strength evidence와 조건부 최적화.** model gate recovery 뒤에만 최소 100 paired
   seeds/색 교환/fixed-simulation/fixed-wall-clock/p95/CI를 봉인하고, 그 다음 Gumbel
   16/32/64, worker scaling, Hybrid A/B, capacity, adaptive compute를 비교한다.

이 gate가 통과되기 전에는 epoch2를 experimental preview로 유지하고 기존 V4 corpus
또는 checkpoint를 승격용 self-play producer로 사용하지 않는다.
