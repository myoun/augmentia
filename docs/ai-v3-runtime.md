# V3 ONNX 추론 런타임

10단계는 Python `OnnxRuntimeV3`와 Rust `augment_ai::V3OnnxRuntime`에 구현되어 있다. 두
경로 모두 생성 시 ONNX session을 한 번만 만들고 이후 batch마다 재사용한다. 모델 hash,
ObservationV3/ActionV3 schema, ruleset ID와 manifest hash, vocabulary, tensor input/output
contract가 하나라도 다르면 로드 또는 추론을 실패시킨다. V2 heuristic으로의 암묵적
fallback은 없다.

Rust runtime은 `V3RuntimeOptions`로 `cpu`/`cuda` execution provider와 intra-op thread 수를
명시한다. 기본값은 재현 가능한 CPU 1-thread이며, CUDA 요청은 provider 등록에 실패하면 즉시
오류를 반환해 CPU fallback을 숨기지 않는다. 동적 ONNX Runtime DLL은 실행 환경의
`ORT_DYLIB_PATH`로 고정한다.

Rust `evaluate_batch`는 여러 `GameState`를 한 번에 ObservationV3와 가변 LegalActionSetV3로
바꾸고 하나의 ONNX call로 평가한다. 반환 policy는 각 상태의 실제 합법 행동 길이에만
softmax하며 ActionV3 index와 원래 `Action`을 함께 보존한다. terminal state는 모델 입력이
아니므로 호출자가 엔진 종료 결과로 처리해야 한다. V3 batch leaf queue와 탐색 의미는
11단계 `MctsV3Tree`에 연결했다. 자세한 계약과 측정은 `ai-v3-search.md`를 참고한다.

재현 가능한 교차 언어/성능 명령:

```powershell
cd train
uv run python -m v3.benchmark_runtime --iterations 100 --soak 1000 --rust
```

이 명령은 임시 랜덤 V3 ONNX를 export하고 같은 Rust-authored `standard_playing` fixture에
대한 Python/Rust policy와 value가 허용 오차 안에서 같은지 검사한다. 모델은 임시
directory에만 존재하고 Git에 남지 않는다. batch 1/8/32의 p50/p95와 처리량, session
instance/run 수, Rust soak 전후 working set을 출력한다.

2026-08-21 이전 0807 observation schema의 로컬 debug build, CPU,
2,841,732-parameter network, 20회 측정 결과:

| 경로 | batch | p50 | p95 | positions/s |
|---|---:|---:|---:|---:|
| Python ONNX-only | 1 | 2.87 ms | 3.10 ms | 377.4 |
| Python ONNX-only | 8 | 12.44 ms | 12.83 ms | 656.6 |
| Python ONNX-only | 32 | 48.72 ms | 50.53 ms | 656.7 |
| Rust encode+ONNX | 1 | 8.96 ms | 9.17 ms | 111.1 |
| Rust encode+ONNX | 8 | 67.56 ms | 68.99 ms | 118.3 |
| Rust encode+ONNX | 32 | 288.59 ms | 308.37 ms | 109.4 |

Python end-to-end batch-one은 p50 29.45 ms/p95 35.15 ms였다. Rust batch-one은 session을
매번 다시 만들던 V2 `/api/state` 기준 551~563 ms보다 크게 낮다. 다만 V2 수치는 HTTP
endpoint/release 구성이고 새 수치는 직접 runtime/debug 구성이라 절대적인 동급 비교가
아니다. batch throughput이 Python ONNX-only처럼 증가하지 않는 주된 부분은 현재 Rust의
ObservationV3 재인코딩이다. 세션 재사용과 batching 계약은 완료했지만, tree 안에서 동일
state를 실제로 방문할 때만 전이·인코딩하는 지연 생성과 leaf batching은 11단계에
구현했다. subtree는 실제 다음 상태가 직전 루트의 자식과 정확히 일치할 때 재사용한다.

별도 1,000회 Rust batch-one soak에서 session instance는 계속 1개였고 총 run 수는
1,034회였다. working set은 측정 전 174,899,200 bytes, 측정 후 173,588,480 bytes로
1,310,720 bytes 감소해 반복 횟수에 비례하는 메모리 증가는 관찰되지 않았다.

## 웹 V2/V3 동시 제안

`augment-web`은 매 fingerprint에서 V2 기준 모델과 V3 모델을 같은 simulation 수로 병렬
탐색하고 각 모델의 최종 제안 하나만 반환한다. `/api/suggestions`는 모델별 action index,
표시 label, 방문 수, root value, 계산 시간을 함께 반환하며 어느 모델이 실패해도 다른 모델
결과를 보존한다. V2 ONNX 추론 실패가 감지되면 휴리스틱 결과를 제안으로 위장하지 않는다.
브라우저는 두 제안을 항상 나란히 표시하고, 수동 선택 또는 모델별 자동 진행에서 현재
fingerprint와 일치하는 action만 `/api/action`으로 적용한다. 자동 진행 모델은 백·흑 양쪽을
두지만 다른 모델의 제안도 매 상태마다 계속 계산·표시한다.

2026-08-22 실제 V2 baseline과 V3 step-500 smoke artifact를 첫 드래프트 상태에 연결했다.
20 simulations에서 두 모델 모두 드래프트 제안을 만들었고, 150 simulations에서는 V2
1,591 ms, V3 2,489 ms, 전체 HTTP wall time 2,497 ms로 합계가 아니라 느린 탐색 시간에
수렴해 병렬 실행을 확인했다. 각 모델을 백·흑 모두에 사용해 4개 action씩 자동 적용했으며
양쪽 모두 드래프트에서 실제 playing phase까지 진행했다.

Lichess-only checkpoint vocabulary에는 현재 ruleset의 일부 runtime text token이 없으므로
export는 기존 token index를 유지한 채 Rust encoder가 생성한
`train/v3/fixtures/runtime_vocabulary_v3.json` 계약의 token을 뒤에 추가하고 새 embedding row를
0으로 초기화한다. 골든 position에 우연히 등장한 token만 수집하지 않으므로 희귀 카드 상태의
`square` 같은 중첩 record field도 빠지지 않는다. 추가 token과 방식은 ONNX metadata의
`vocabulary_extension`에 기록한다.

이는 미학습 카드 의미를 학습된 것으로 만들지 않으며, smoke 모델이 증강 상태를 구조적으로
입력받아 통합 경로를 검증하기 위한 호환 확장이다.

선택지의 사용자 표시 문구는 보드 상황에 따라 동적으로 만들어지므로 모델 vocabulary에
넣지 않는다. ObservationV3와 ActionV3에는 `choice:{handler_id}:{option_index}` 형식의 안정된
token만 기록하고, 실제 문구는 `GameState`와 웹 UI 경로에만 남긴다. 트롤리의 희생 기물 구성은
별도 runtime 구조와 option index로 전달되므로 모델 정보는 잃지 않는다. Python verifier는
원시 표시 문구가 V3 학습 record에 들어오면 fail-closed로 거부한다.
