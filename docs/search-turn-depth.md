# 탐색 깊이를 턴 기준으로 변경

## 구현

모델 없는 augment-search 기준선의 기본 깊이를 `turns`로 변경했다. 한 턴은 백+흑 한 쌍이 아니라 **한 플레이어가 턴을 완료하는 것**이다. 루트가 턴 중간이면 그 턴의 남은 부분이 첫 번째 깊이다.

- 같은 턴의 이동·카드·추가 행동·후속 선택에는 깊이를 차감하지 않는다.
- 엔진의 `Event::TurnEnded`를 세어 완료된 턴 수만큼 깊이를 차감한다. 드래프트나 임시 선택 때문에 `state.turn`이 바뀌는 것만으로는 깊이를 차감하지 않는다. 자동 건너뛰기로 한 전이에서 여러 턴이 끝나면 모두 반영한다.
- max/min은 여전히 각 상태의 실제 행동자를 기준으로 한다. 같은 턴의 액션을 강제 단일 경로로 바꾸거나 상대의 선택을 제거하지 않는다.
- 확률 분기는 독립 시드 표본의 평균과 full child window를 그대로 유지한다.
- 중간 액션과 상태 전이는 실제로 모두 실행한다. 압축되는 것은 **깊이 단위**이지 분기 수나 전이 호출 수가 아니다. 완전한 턴 액션 묶음의 선생성·상태 중복 병합은 이번 변경에 포함되지 않는다.
- 실제 선택 결과는 첫 번째 액션이다. live 게임에서 후속 선택은 해당 상태를 다시 탐색한다. 미리 계산한 모든 액션을 확률 결과와 무관하게 일괄 실행하지 않는다.

## 설정과 보고

```powershell
cargo run --release -p augment-search -- bench --depth-unit turns --depth 4 --time-ms 250 --seed 1
# 기존 액션 깊이 의미를 재현
cargo run --release -p augment-search -- bench --depth-unit actions --depth 4 --time-ms 250 --seed 1
```

`SearchConfig.depth_unit` 기본값은 `DepthUnit::Turns`다. JSON config와 각 report에 `depth_unit`을 기록한다. `completed_depth`는 완전히 완료한 반복의 턴/액션 깊이이며, `stats.max_action_path`는 중단된 반복까지 포함한 최대 실제 방문 액션 경로 길이다. 둘은 다른 통계다. 과거 최적화 보고서의 depth는 actions이므로 턴 깊이와 숫자만 직접 비교하면 안 된다.

시간·노드 예산은 유지한다. 추가로 경로당 64개 액션 제한을 두어 긴 같은 턴 연속 행동에서 스택 고갈을 방지한다. 제한에 걸리면 `action_path_limit`으로 반복을 중단하며, 잘린 상태를 평가해 턴 탐색 완료라고 보고하지 않는다. 깊이 1도 완료하지 못하면 기존 계약대로 합법 fallback과 `completed_depth=0`을 반환한다.

## 동일 250ms 예산 관측

같은 release 실행 파일로 actions/turns 모드를 직렬 교대 실행했다. 최대 깊이 4, seeds 1/2/3 각각 3회, 노드 100000, chance_samples=4, search_seed=19다. 별도 폐기 warmup 없는 탐색 의미 비교이며 통계적 속도 우위를 주장하지 않는다. 아래는 9회 중앙값이다.

| 국면 | 기존 완료 액션 깊이 | 새 완료 턴 깊이 | 새 최대 방문 액션 경로 |
|---|---:|---:|---:|
| 즉시 승리 fixture | 4 | 4 | 4 |
| 같은 플레이어 추가 행동 | 4 | 4 | 5 |
| 표준 보드/카드 없음 | 4 | 4 | 4 |
| 초기 드래프트 | 4 | 2 | 7 |
| 카드 진행 decision 4 | 4 | 3 | 5 |
| decision 8 | 4 | 4 | 4 |
| decision 16 | 4 | 4 | 4 |
| decision 24 | 3 | 2 | 6 |

추가 행동 fixture에서는 깊이 1턴이 두 액션을 포함한다. seed 1 단일 확인에서 깊이 1턴의 초기 드래프트는 최대 4액션, decision 24는 최대 3액션을 탐색했다. 깊이 숫자가 작아져도 액션 단위 지평선이 얕아졌다는 뜻은 아니다. 액션 경로에는 중단된 더 깊은 반복도 포함되므로 특정 완료 깊이의 전 분기 보장으로 해석하지 않는다.

## 검증

- 깊이 1액션에서는 놓치는 **같은 턴 두 번째 액션의 승리**를 깊이 1턴에서 찾는다. 입력 상태 불변도 검사한다.
- 완료 턴 카운터를 사용한 독립 unpruned 참조와 점수를 대조한다. 턴당 한 액션인 fixture는 기존 액션 깊이와 수·점수·노드가 같다.
- 완료 턴 이벤트만 깊이를 차감하며 다중 자동 턴 종료의 깊이 차감도 검사한다.
- 확률 평균의 finite window 정확성을 actions/turns 양쪽에서 검사한다.
- 시간·노드 중단과 실제 긴 액션 연쇄 제한이 거짓 완료 깊이를 보고하지 않는지 검사한다.
- 기존 최종 구조 최적화 release의 seed 1/depth 3 고정 탐색 8개와 새 actions 모드의 행동·점수·깊이·루트 범위·기존 비시간 카운터가 일치한다.

전체 workspace **350개 테스트**, strict workspace/all-target Clippy, fmt/diff 검사, 최종 release 빌드가 통과했다. 반복 측정 당시 경로 제한은 256이었으며 최종 안전 제한은 64로 강화했다. 비교 측정은 이 제한에 도달하지 않았고, 최종 빌드의 기본 turns 모드와 legacy actions 모드는 별도 smoke로 재확인했다. 최종 actions 모드도 기존 고정 탐색 8개의 비시간 결과가 일치했다.

당시 비교 대상이던 ONNX MCTS와 웹 모델 제안은 이후 제거했다. 현재 구조는 [README](../README.md)를 따른다.

[모드별 관측](<../artifacts/turn-depth-audit/comparison.json>) · [구현](<../crates/augment-search/src/lib.rs>)
