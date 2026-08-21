# V3 ONNX 추론 런타임

10단계는 Python `OnnxRuntimeV3`와 Rust `augment_ai::V3OnnxRuntime`에 구현되어 있다. 두
경로 모두 생성 시 ONNX session을 한 번만 만들고 이후 batch마다 재사용한다. 모델 hash,
ObservationV3/ActionV3 schema, ruleset ID와 manifest hash, vocabulary, tensor input/output
contract가 하나라도 다르면 로드 또는 추론을 실패시킨다. V2 heuristic으로의 암묵적
fallback은 없다.

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

2026-08-21 로컬 debug build, CPU, default 2,841,732-parameter network, 20회 측정 결과:

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
