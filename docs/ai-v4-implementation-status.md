# Augmentia V4 구현 현황

이 문서는 V4 factorized search의 구현 상태를 기록한다.
설계 계약은 [V4 factorized policy contract](ai-v4-factorized-policy.md)를 참고한다.
2026-09-19 현재 Phase 0/1 이후 lazy prefix scoring, Hybrid Encoder 구현과 0915 고정 예산 baseline 재봉인까지 반영했다.

## 요약

- **Phase 0 — chance 정확성: 완료.**
- **Phase 1 — Rust split ONNX 런타임 및 MCTS 연결: 완료, 검증 통과.**
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
- replay target은 `augmentia.factorized-mcts-visits.v2`이며,
  chance sample 수와 search semantics provenance를 기록한다.

## Phase 1: split ONNX 런타임

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
| `cargo test --workspace` | 통과, `augment-ai` 107개 포함 |
| Python 전체 unittest | 101개 통과 |
| Rust V4 runtime / MCTS / integration | 통과, 실제 ONNX Runtime 포함 |
| 0915 fixed-budget baseline | standard/draft/wide 재현성 delta 0으로 봉인 |
| `git diff --check` | 통과, 줄바꿈 경고만 발생 |

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
$env:ORT_DYLIB_PATH = "D:/augmentia/train/.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
cargo test -p augment-ai runtime_v4 -- --nocapture
cargo test --workspace
```

V4 Rust ONNX 테스트는 `ORT_DYLIB_PATH`가 없을 때만 skip한다.
변수가 있는데 라이브러리나 번들이 잘못되면 실패해야 한다.

Python 전체 검증과 fixture 재생성은 `train/`에서 실행한다.

```text
uv run python -m unittest discover -s . -p "test_*.py" -v
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
| `train/v3/fixtures/v4_bundle/` | 0818 회귀 테스트 graph, metadata, parity 기대값 |
| `train/v3/fixtures/0915/`, `train/v3/fixtures/v4_bundle_0915/` | 0915 baseline fixture와 split ONNX bundle |

## 남은 작업과 진행 순서

1. **0915 고정 예산 baseline: 재봉인 완료** — [baseline 기록](ai-v4-baseline.md).
   lazy scoring 적용 상태에서 standard/draft와 wide stress fixture를 봉인했고 warm repeat visit delta는 모두 0이다.
   V3 대비 paired arena 우위 비교는 아직 없다.
2. **Hybrid Encoder: 구현/계약 검증 완료, 학습 A/B 남음.**
   baseline encoder와 동일 입력 계약을 유지한 채 실제 학습 모델로 latency/strength를 비교해야 한다.
3. **NN-only exact/beam fast path**: Engine 합법 행동 공간 안에서 동작시킨다.
4. **Search-risk predictor**: 일반 confidence가 아니라 regret/필요 예산을 예측한다.
   먼저 shadow 모드에서 검증한다.
5. **Adaptive compute controller**: NN-only 선택, 탐색 진입, 탐색 계속 여부를 제어한다.
   공개 정보만 사용하고 선택한 예산과 결정 경로를 기록한다.
6. **Tactical verifier**: Engine 전이와 합법성 계약을 유지한다.

이 단계들이 검증되기 전에는 V4를 기본 모델로 승격하지 않는다.
