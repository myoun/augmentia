# 엔진 구조 최적화 검증 보고서

## 범위와 기준

이번 작업은 이전의 작은 최적화가 **이미 적용된 실행파일**을 기준으로 추가 구조 최적화를 수행했다. 과거 측정과 이번 수치를 곱해 누적 배속을 주장하지 않는다. 탐색 평가·행동 순서·가지치기·확률 샘플 수는 바꾸지 않았다.

- 기준 release: [before.exe](<../artifacts/engine-structural-optimization/before.exe>), SHA256 `8C35F26D8DFF8057BEA9C254D1817BB1B1717B04F0E49AD1C33E1CEF0CA89AD3`.
- 최종 release: [after-verified.exe](<../artifacts/engine-structural-optimization/after-verified.exe>), SHA256 `E8B8E980415BD50215861291390FA9353BC0D7F5B87869D186B94E6D8C8D2F6A`. 정확한 SHA256와 환경은 [측정 결과](<../artifacts/engine-structural-optimization/measurement-verified/comparison.json>)와 [환경 기록](<../artifacts/engine-structural-optimization/measurement-verified/environment.json>)에 있다.
- 사용자 코드·이전 변경·학습 산출물을 보존했으며, API의 외부 입력 검증을 제거하지 않았다.

## 구현

### 1. 정확한 상태에 묶인 합법 행동 배치

[엔진](<../crates/augment-core/src/engine.rs>)에 `LegalActionBatch`를 추가했다. 엔진·원본 상태를 불변 참조하고, 생성한 행동 목록은 비공개로 소유한다. 배치가 허가한 행동은 목록 재생성 없이 기존 전이 파이프라인으로 실행된다. 다른 상태나 다른 엔진을 인자로 끼워 넣을 수 없고, 원본을 변경한 뒤 배치를 재사용하는 코드도 borrow checker가 거부한다.

일반 `Engine::apply`는 여전히 규칙·종료 상태·행동 합법성을 검사한다. 내부 검증 완료 실행 경로는 비공개다. 배치는 **열거된 행동만** 허가한다. 열거 밖의 동적 CLEANUP/HYPOCRISY 목표 집합은 기존 checked API를 사용해야 한다. 불완전한 행동 열거를 완전한 것으로 간주하지 않는다.

[알파베타 탐색](<../crates/augment-search/src/lib.rs>)은 노드당 배치를 생성하고 각 간선에서 재사용한다. MCTS V3(과거 구현·삭제됨)와 MCTS V4(과거 구현·삭제됨)도 같은 부모 행동의 확률 샘플들에 한 배치를 재사용한다. 직렬화된 행동 목록 자체는 신뢰하지 않으며, 실제 부모 상태에서 새 배치를 생성한다.

### 2. 내장 규칙의 확률 전이를 한 번 실행

효과·선택 handler registry가 **모두 비어 있는 경우에만** 독립 탐색 seed로 전이를 한 번 실행한다. RNG가 소비되지 않으면 결과의 원래 live RNG를 복원하고 탐색 chance context를 그대로 둔다. RNG가 소비되면 context/event index를 정확히 한 번 진행한다. 내장 규칙의 위치 반복 키와 replay 정보는 RNG를 포함하지 않는다. 전체 상태 fingerprint는 RNG를 포함하므로 결정론적 전이에서는 복원이 중요하다.

사용자 handler는 RNG를 직접 관찰할 수 있으므로 registry가 비어 있지 않으면 기존 실제 RNG probe/독립 seed 재실행 경로를 유지한다. handler를 임의로 순수하다고 가정하지 않았다.

### 3. 이동 존재 여부만 필요할 때 조기 종료

행동 불능 판정의 이동 확인을 전체 행동 목록 생성·정렬·중복 제거 대신 `has_move_action`으로 바꿨다. Checker의 별도 강제 포획 pre-pass는 유지한다. 일반 이동과 quantum shadow를 순서대로 검사해 첫 행동에서 종료한다. Scarecrow 강제 포획은 이미 생성된 비어 있지 않은 부분집합을 선택할 뿐이므로 존재 여부를 바꾸지 않는다.

### 4. 이벤트 handler snapshot 비용 축소

이벤트별로 실제 handler가 등록된 카드만 snapshot/정렬한다. handler가 없으면 카드·이벤트 clone을 생략한다. `(owner.index(), instance_id)` 실행 순서, 이벤트별 snapshot 시점, 이후 handler가 보는 변경된 상태, 큐 진행·상한은 유지한다. 처리되지 않는 카드나 charge가 0인 카드를 이유 없이 의미론적으로 제거하는 별도 필터는 넣지 않았다.

## 최종 release 전후 측정

[비교 스크립트](<../scripts/compare-search.ps1>)로 warm-up 이후 **직렬 AB/BA 교대 실행**했다. 고정 작업/250ms는 seed 1/2/3 각각 3회, 1000ms는 seed 1을 3회 반복했다. 빌드·테스트·다른 benchmark와 겹치지 않았다. 다음 시간은 한 bench 프로세스가 여러 국면을 실행한 **프로세스 wall-clock 중앙값**이며 시작·규칙 로딩 비용도 포함한다.

| 시나리오 | 이전 | 이후 | 이전/이후 |
|---|---:|---:|---:|
| 고정 작업, depth 3, 30초 비구속 ceiling | 344.707 ms | 163.658 ms | **2.106배** |
| 국면별 250ms 예산, 최대 depth 4 | 1174.110 ms | 818.247 ms | 1.435배 |
| 국면별 1000ms 예산, 최대 depth 4 | 2355.001 ms | 1109.238 ms | 2.123배 |

고정 작업의 프로세스 시간은 **52.52% 감소**했다. 72개 비종료 탐색 기록의 보고된 탐색 시간 합계도 2970.2804 → 1259.7347ms로 감소했다. 고정 작업의 행동·점수·완료 깊이·root 수·노드·합법 호출·전이 호출·chance 샘플·평가·cutoff 수가 모두 일치했다. 모두 depth 3까지 완료했고 stop reason은 `depth_limit`이었다.

국면별 고정 작업의 `report.elapsed_ms` 중앙값(프로세스 시간과 별개)은 다음과 같다.

| 국면 | 이전 ms | 이후 ms | 비율 |
|---|---:|---:|---:|
| immediate-win | 2.0988 | 1.8313 | 1.146배 |
| same-player-continuation | 5.2065 | 4.0903 | 1.273배 |
| seeded-opening-draft | 10.8278 | 6.3991 | 1.692배 |
| standard-board/no-cards | 8.8683 | 4.4617 | 1.988배 |
| trajectory-decision-16 | 116.3705 | 44.7582 | 2.600배 |
| trajectory-decision-24 | 55.0259 | 22.8643 | 2.407배 |
| trajectory-decision-4 | 46.4362 | 21.7996 | 2.130배 |
| trajectory-decision-8 | 45.8229 | 23.0653 | 1.987배 |

250ms 시나리오의 평균 방문 노드는 6898.40 → 10176.01, 평균 완료 깊이는 3.542 → 3.875였다. 이는 시간 제한 때문에 같은 작업 비교가 아니며 기력 증거가 아니다. 1000ms 시나리오는 이번 샘플에서 양쪽 모두 depth 4까지 완료했으므로 실제로 예산이 구속하는 사례라고 주장하지 않는다. 평균 노드 수도 양쪽 모두 11834였다.

첫 구조 후보의 별도 [중간 측정](<../artifacts/engine-structural-optimization/measurement/comparison.json>)은 고정 작업 1.384배였다. 이후 이동 존재 조기 종료·이벤트 clone 생략 등이 추가된 후보는 [첫 최종 측정](<../artifacts/engine-structural-optimization/measurement-final/comparison.json>)에서 2.123배였다. 문서·회귀 테스트를 마친 정확한 현재 release를 다시 보존·측정한 본 보고서의 최종 값은 2.106배다. 이는 개별 최적화의 독립 ablation 실험은 아니다. MCTS의 별도 처리량 개선율은 측정하지 않았다.

## 의미론·회귀 검증

- 전체 workspace 테스트 **344개 통과**(코어 189, 검색 15, AI 112 및 기타 테스트/compile-fail 포함), 전체 workspace 모든 target의 `clippy -D warnings`, `cargo fmt --check`, `git diff --check`가 통과했다. 상세 결과는 [검증 로그](<../artifacts/engine-structural-optimization/validation-final.log>)에 기록한다.
- 배치와 checked API의 모든 열거 행동을 다수 progression 상태에서 전체 상태·이벤트 단위로 비교했다. 독립적으로 구성한 기존 probe/reapply 참조 구현과도 탐색 결과·chance 판정·event index·context를 비교했다.
- 이동 존재 쿼리는 다양한 기물·양 플레이어·idle/zugzwang/socialism gate·quantum shadow 조합에서 전체 생성과 비교했다.
- 사용자 RNG 관찰 handler와 기존 clone-all 이벤트 snapshot 참조 구현으로 호환 경로·순서를 비교했다.
- 비회원 행동 거부, 동적 CLEANUP 행동의 checked/batch 경계, 실패 시 chance context 불변성, 원본 불변성 및 상태 borrow compile-fail을 확인했다.
- seed 1/2/3, 최대 96 decisions의 기준/최종 [trace 비교](<../artifacts/engine-structural-optimization/trace-verified-comparison.json>)는 전체 Debug 상태·이벤트·정렬된 합법 목록을 포함해 SHA256가 동일했다. 총 644 전이 기록이다.

**coverage 주의:** `bundled_20260807()`는 기존 호환 alias로 실제로는 20260818을 로딩한다. 세 생성자를 호출했지만 서로 다른 실제 규칙은 **20260818/20260915의 두 개**다. 644개 기록 중 alias 중복을 제외한 것은 418개다. 독립 0807 규칙을 검증했다고 주장하지 않는다. 이는 이번 최적화로 생성자를 변경한 것이 아니다.

## 남은 한계

- CLEANUP 1~3칸 및 HYPOCRISY 동적 목표 집합의 열거는 아직 부분집합이다. 이를 확대하면 탐색 작업·결과도 바뀌므로 이번 의미론 동일 처리량 비교에 섞지 않았다.
- 공격 판정은 아직 전체 상태 clone, king 이동 표시, Alibaba sentinel, 기물별 기존 생성기를 사용한다. 일반 체스 attack 식으로 대체하면 특수 기물·카드 규칙을 깨뜨릴 수 있어 검증 없이 바꾸지 않았다.
- 전체 `GameState` 및 runtime의 깊은 복사, 반복 키 계산, 여러 특수 기물의 생성기는 향후 profiling/별도 동등성 설계가 필요한 병목 후보다. 이번 결과를 이론적 최대 속도라고 주장하지 않는다.
- cooperative budget은 단일 엔진 호출 중간을 강제로 끊지 못한다. 완료된 iterative-deepening 결과만 반환한다.
- 유한한 테스트/trace 동등성은 모든 카드 조합의 증명이나 원본 사이트 parity가 아니다. 처리량 개선은 확인했지만 승률·Elo·학습된 AI의 기력 개선은 확인하지 않았다.

## 재현

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check

# 저장된 동일 release 바이너리를 이용한 비교. 다른 부하와 겹치지 않게 실행.
./scripts/compare-search.ps1 `
  -BeforeExecutable artifacts/engine-structural-optimization/before.exe `
  -AfterExecutable artifacts/engine-structural-optimization/after-verified.exe `
  -OutputDirectory artifacts/engine-structural-optimization/reproduction `
  -Repeats 3
```
