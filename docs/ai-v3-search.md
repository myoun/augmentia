# V3 MCTS 탐색

11단계는 V3의 가변 합법 행동 정책과 batch ONNX 런타임을 실제 탐색에 연결한다.
`augment_ai::MctsV3Tree`는 각 노드의 정책을 전역 고정 인덱스가 아니라 해당 상태의
`LegalActionSetV3` 순서에 맞춘 local action index로 저장한다. 합법성과 종료 여부는
항상 `Engine::legal_actions`, `Engine::apply`, `GameStatus`로 판정한다.

## 의미 변경

기존 V2 MCTS의 다음 우회 로직은 제거했다.

- 루트에서 즉시 킹 포획을 발견하면 탐색 없이 1,000회 방문으로 꾸미는 처리;
- leaf의 즉시 킹 포획을 네트워크 평가보다 우선하는 처리;
- 최종 후보에서 다음 수에 킹을 잃는다고 추정한 수를 제거하는 처리.

이 로직들은 카드 효과와 다단계 행동을 포함한 실제 엔진 전이를 대신할 수 없었다.
예를 들어 보호된 킹을 공격하는 행동은 엔진상 진행 중인 상태인데도 가짜 terminal로
처리될 수 있었다. V3 탐색은 모델의 prior/value를 사용하되 승패는 오직 엔진 결과에서
얻는다.

value는 평가한 상태의 `state.turn` 관점이다. 백업할 때 부모와 자식의 차례가 실제로
바뀐 경우에만 부호를 반전한다. 따라서 한 플레이어가 카드와 후속 대상을 연속으로
선택하는 phase에서도 관점이 뒤집히지 않는다.

## 탐색과 배치

한 번의 탐색 반복에서 서로 다른 미평가 leaf를 최대 `leaf_batch_size`개 예약한 뒤
한 번의 `evaluate_batch`로 평가한다. 예약 중에는 virtual visit/loss를 적용해 같은
leaf에 batch가 몰리지 않게 한다. terminal leaf는 모델에 보내지 않는다.

확장 시 모든 합법 행동에 대해 `Engine::apply`를 미리 실행하지 않는다. action과 prior만
노드로 만들고 실제로 선택된 자식의 상태를 지연 생성한다. 이 변경은 batch 평가 호출
감소가 엔진 전이 비용에 묻히지 않도록 한다.

prior가 작은 행동도 탐색에서 완전히 굶지 않도록 `root_min_visits`를 둔다. 시뮬레이션
예산이 루트 합법 행동 수 이상이면 각 행동에 최소 방문 수를 우선 배정한다. 예산이 더
작으면 모든 행동 방문을 보장할 수 없으므로 높은 prior 순으로 제한된 기회를 배정한다.

root Dirichlet noise는 탐색 전 prior에만 적용하며, temperature는 탐색이 끝난 방문 횟수에서
행동을 표본 추출할 때만 사용한다. 둘은 각각 명시적인 설정과 seed를 가지며 서로의 역할을
대신하지 않는다.

## 재사용과 보류한 최적화

직전 루트의 자식과 다음 실제 `GameState`가 정확히 같으면 해당 subtree를 새 루트로 압축해
재사용한다. 도달할 수 없는 이전 노드는 함께 버려 메모리가 계속 증가하지 않게 한다.

transposition table은 활성화하지 않았다. 표준 시작 상태에서 100 simulations를 실행한
측정에서 leaf batch 1/8/16 모두 정확한 전체 상태 fingerprint 중복 후보가 0건이었다.
카드 runtime, phase, RNG 상태를 빠뜨린 근사 키는 잘못된 상태를 합칠 위험이 있으므로,
실제 중복이 관찰되기 전에는 복잡성과 위험을 추가하지 않는다.

V2 quiescence도 V3에 연결하지 않았다. 기존 구현은 material/capture 중심 휴리스틱으로
증강체스 카드 전술의 정확한 종료 의미를 보장하지 않는다. 학습된 V3 모델로 전술 세트와
동일 벽시계 arena 개선이 확인될 때 별도 실험으로 도입한다.

## 검증과 측정

회귀 테스트는 작은 prior의 강제승리, 실제 `SACRIFICE` 카드 희생승리, 보호된 킹 공격,
동일 플레이어 연속 phase의 value 부호, 전체 루트 방문, noise/temperature 분리와 subtree
재사용을 포함한다.

재현 명령:

```powershell
cd train
uv run python -m v3.benchmark_runtime --iterations 20 --soak 1000 --rust `
  --search-simulations 100 --search-iterations 5
```

2026-08-21 로컬 debug build, CPU, 임시 랜덤 V3 ONNX, 100 simulations와 2회 측정 결과:

| leaf batch | evaluator batches/search | mean search | simulations/s | root coverage |
|---:|---:|---:|---:|---:|
| 1 | 101 | 1,099.0 ms | 91.0 | 100% |
| 8 | 14 | 1,081.3 ms | 92.5 | 100% |
| 16 | 8 | 1,084.1 ms | 92.2 | 100% |

지연 상태 전이 전 같은 debug 환경의 처리량은 약 50~62 simulations/s였다. 위 수치는
검색 경로의 정확성과 처리량 기준이며 랜덤 모델의 경기력 지표가 아니다.
