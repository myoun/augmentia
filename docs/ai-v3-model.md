# 첫 V3 신경망

상태: 8단계 모델 완료, 9단계 auxiliary/export 확장 (`2026-08-21`)

구현: `train/v3/model.py`, `train/v3/tensorize.py`

## 역할과 경계

`AugmentNetV3`는 `ObservationV3`와 현재 `LegalActionSetV3`를 받아 합법 행동마다 logit
하나와 actor 관점 value 하나를 출력한다. 모델 가중치는 랜덤 초기화한다. 대규모 replay
loader, 장기 학습, checkpoint 재개와 ONNX export는 9단계 모듈로 분리되어 있으며 V2 tensor,
고정 `ActionIndex`, heuristic 또는 ONNX fallback은 없다. 9단계에서 solver의
`forced_win_distance_decisions`를 위한 scalar auxiliary head가 추가되어 기본 parameter
수는 2,864,132개가 되었다.

일반 체스 `board_pretraining` record는 shared state encoder와 move action scorer의 policy만
학습한다. value label이 없으므로 value loss에서 제외하고, 일반 체스 결과를 Augmentia
value로 사용하지 않는다.

## 입력 표현

- spatial board: `[batch, 116, 8, 8]` residual CNN;
- global dense state: `[batch, 32]`, exact mode/applied-rule tokens와 local entity counts;
- piece/card records: exact kind/definition ID, owner, local slot과 수치 feature를 embedding한
  뒤 masked pooling;
- phase: phase kind, local piece/card reference와 순서가 있는 exact text token;
- runtime: 169 field의 재귀 값을 field/type/text/value/parent/depth/sibling/local-reference
  node로 평탄화하고 두 번의 parent message passing 뒤 attention pooling;
- actions: move, ability, card, draft, choice의 typed field와 from/to board context, local
  piece/card reference, exact text ID, 순서가 있는 target square sequence.

가변 길이 목록은 batch 안에서만 padding한다. padding은 별도 mask를 가지며 policy logit은
현재 합법 행동에만 존재한다. `ActionTargetV3::Squares`는 64-square membership과 순서가 있는
sequence를 함께 보존하므로 순서가 다른 descriptor를 같은 입력으로 만들지 않는다.

문자열은 해시하거나 unknown bucket으로 보내지 않는다. `V3Vocabulary`는 index 0을
`<none>`으로 두고 ruleset card catalog와 해당 run에서 검증한 exact token을 정렬해 만든다.
없는 token은 오류다. 9단계에서는 이 vocabulary를 학습 run과 checkpoint provenance에
고정해야 한다.

## 기본 구조와 출력

기본 설정은 128 CNN channels, residual block 6개, model dimension 128, runtime parent
message pass 2회다. 모든 state branch를 128차원으로 결합하고 각 action embedding과
state/action interaction을 MLP로 점수화한다. value head는 `tanh`로 `[-1, 1]`을 출력한다.

`compute_v3_loss`는 policy와 value supervision mask를 독립적으로 받는다. policy target이
불법/padding action에 확률을 주거나 합이 1이 아니면 실패한다. value가 없는 standard-chess
sample은 value target을 읽지 않는다.

## 검증 결과

golden fixture 7개 상태로 다음을 고정했다.

- 모든 5개 ActionV3 종류와 0~144개 가변 합법 행동 처리;
- 한 상태의 서로 다른 fixture action이 model tensor에서 충돌하지 않음;
- terminal의 빈 행동 집합과 batch padding 차단;
- piece/card record 순서 변경 불변성과 동일 입력 결정성;
- 순서가 다른 multi-square target 구별;
- vocabulary 밖 문자열 거부;
- policy/value supervision 독립성;
- 4개 tiny fixture의 policy/value 의도적 overfit.

기본 모델은 2,825,091 parameters, FP32 parameter 11,300,364 bytes다. Windows 11,
PyTorch 2.11.0+cu128, runtime-rich fixture(4 pieces, 2 cards, 298 runtime nodes, 144 legal
actions), batch 1에서 측정한 기준은 다음과 같다.

| device | 조건 | p50 | p95 |
| --- | --- | ---: | ---: |
| CPU | 1 thread, 100 iterations | 8.16 ms | 8.74 ms |
| RTX 3060 | 100 iterations, synchronized | 9.97 ms | 13.81 ms |

이 입력 batch tensor는 112,852 bytes였고 CUDA 측정 구간의 peak allocated memory는
22,754,304 bytes였다. CPU는 allocator 전체 peak 대신 FP32 parameter와 입력 tensor byte를
재현 가능한 최소 메모리 기준으로 기록한다.

이는 Python eager의 batch-one 구조 기준이며 최종 Rust/ONNX 성능 주장이 아니다. GPU가
batch one에서 느린 결과는 작은 embedding/message-passing 연산의 kernel launch 비용을
보여준다. 9~10단계에서 ONNX 동등성과 batch별 처리량을 별도로 측정한다.

재현:

```powershell
cd train
uv sync --frozen
uv run python -m unittest v3.test_model -v
uv run python -m v3.benchmark_model --case runtime_rich --device cpu --threads 1
```
