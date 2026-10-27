# 엔진 최적화 및 전후 재측정

> 이전 소규모 최적화의 역사적 측정 기록이다. 이후 구조 최적화는 [추가 보고서](<engine-structural-optimization.md>)를 참조한다. 범위 정정: 기존 `bundled_20260807()`는 0818 호환 alias이므로 세 생성자 호출이 서로 다른 세 규칙셋 검증을 뜻하지 않는다.

## 결론

동일 깊이 3 탐색 72개 결과의 **순수 탐색 시간 합계**는 4842.22ms → 3008.02ms로 **37.88% 감소 / 1.61배 처리 속도**였다. 프로세스 시작·룰셋 로딩을 포함한 9회 bench의 중앙값도 545.08ms → 338.27ms (1.61배)였다. 서로 다른 비용 정의이므로 구분한다.

이는 이번 머신·작업군에서 세 변경을 묶어 적용한 결과이며, 각 변경의 독립 효과나 AI 승률 향상을 뜻하지 않는다. 과거 감사의 시간값과 직접 나누지 않고 변경 전 엔진도 같은 세션에서 재측정했다.

## 실제 변경

- [캐슬링 생성](<../crates/augment-core/src/engine.rs>): 유효한 룩과 빈 경로가 있는 경우에만 공격 검사. 양쪽이 가능하면 왕 출발 칸의 검사는 한 번만 수행한다. 기존 공격 판정 로직은 그대로 사용한다.
- [이동 적용](<../crates/augment-core/src/engine.rs>): 0807/0915에서는 이동 전 카드 런타임을 불변 원본에서 빌린다. 0818의 replay용 복사와 중첩 replay 제거는 유지한다. 메뚜기 등 이동 후 효과의 원본 참조도 보존한다.
- [행동 불가 패배 검사](<../crates/augment-core/src/engine.rs>): 이동이 존재하면 카드 액션 목록을 생성하지 않는다. 이동이 없으면 기존 카드 생성/패배 판정을 그대로 수행한다. 기물 능력 관련 기존 판정 의미도 변경하지 않았다.
- [진단 CLI](<../crates/augment-search/src/main.rs>)에 transition-trace를 추가해 전체 Debug 상태·이벤트 및 정렬된 합법 목록을 기존/변경 엔진에서 비교한다. 검색 알고리즘·평가 함수·확률 표본 수는 변경하지 않았다.

합법 목록 재검증, 확률 액션의 probe/reapply, 일반 GameState 복사, 이벤트 처리 구조는 **아직 최적화하지 않았다**. CLEANUP·HYPOCRISY의 액션 조합 열거 누락도 그대로다. 이번에는 작은 변경의 효과와 의미론 보존을 먼저 확인했다.

## 방법과 환경

- 2026-10-01T07:28:28.6590301Z–2026-10-01T07:29:22.3075353Z; CPU AMD Ryzen 7 5800X 8-Core Processor, 16 logical processors; rustc 1.90.0 (1159e78c4 2025-09-14), Node v22.20.0.
- 동일 CLI를 먼저 빌드하고 엔진 변경 전 [before 실행파일](<../artifacts/engine-optimization-audit/before.exe>)을 보존했다. 변경 후 테스트를 통과한 [after 실행파일](<../artifacts/engine-optimization-audit/after.exe>)과 비교했다. 두 실행파일 모두 release.
- SHA256 before: `EF6AAA965D364F2052449B4A8FF985EF4DEB7780853E8D6DA84D9F06EA3E30CA`; after: `8C35F26D8DFF8057BEA9C254D1817BB1B1717B04F0E49AD1C33E1CEF0CA89AD3`.
- 규칙 0915, seeds 1/2/3, seed별 3회 반복. 확률 표본 4, search_seed 19, node ceiling 100000, 임의 합법 진행 최대 24행동에서 얻은 8개 작업 상태.
- 각 실행파일 1회 워밍업 후 측정. 다른 빌드/테스트와 겹치지 않게 직렬 실행. 홀수 반복 AB, 짝수 BA로 실행 순서를 바꿨다.
- 고정 작업: max depth 3 / time ceiling 30000ms. **전 결과 깊이 3 완료, depth_limit 종료**, 시간·노드 제한 미도달. 결과·점수·노드·호출 수 등 시간 제외 필드까지 72개 쌍이 일치했다. 동일 seed의 반복 간 결과도 일치했다.
- 시간 제한 작업: max depth 4, 250ms는 seeds 1/2/3 × 3회; 1000ms는 seed 1 × 3회. 중단된 다음 반복 노드도 nodes에 포함되며 completed_depth는 완료한 반복만 센다.
- 아래 시간은 process wall clock이 아닌 report.elapsed_ms의 중앙값이다. fixed/timed250은 시나리오당 9개, timed1000은 3개 표본이다.

## 고정 깊이 3: 같은 일을 처리하는 시간

| 시나리오 | 변경 전 ms | 변경 후 ms | 처리 속도 |
|---|---:|---:|---:|
| immediate-win | 2.39 | 2.16 | 1.10× |
| same-player-continuation | 5.37 | 5.03 | 1.07× |
| seeded-opening-draft | 17.55 | 10.94 | 1.60× |
| standard-board/no-cards | 16.62 | 9.11 | 1.82× |
| trajectory-decision-16 | 194.57 | 111.86 | 1.74× |
| trajectory-decision-24 | 71.93 | 53.03 | 1.36× |
| trajectory-decision-4 | 77.62 | 44.99 | 1.73× |
| trajectory-decision-8 | 81.54 | 48.00 | 1.70× |

즉시 승리·동일 플레이어 연속 행동의 작은 보드에서는 이득이 작다. 표준 보드와 대부분의 카드 진행 작업에서는 1.36–1.82배였다. 작업마다 효과가 다르므로 보편적인 1.61배를 보장하지 않는다.

## 250ms 제한: 같은 시간에 처리한 노드

| 시나리오 | 전 노드 중앙값 | 후 노드 중앙값 | 노드 비율 | 전 완료 깊이 범위 | 후 완료 깊이 범위 |
|---|---:|---:|---:|---:|---:|
| immediate-win | 851 | 851 | 1.00× | 4–4 | 4–4 |
| same-player-continuation | 5515 | 5515 | 1.00× | 4–4 | 4–4 |
| seeded-opening-draft | 3988 | 3988 | 1.00× | 3–4 | 3–4 |
| standard-board/no-cards | 2838 | 2838 | 1.00× | 4–4 | 4–4 |
| trajectory-decision-16 | 6000 | 9977 | 1.66× | 2–3 | 3–3 |
| trajectory-decision-24 | 5873 | 8378 | 1.43× | 3–3 | 3–4 |
| trajectory-decision-4 | 7218 | 12250 | 1.70× | 3–3 | 3–3 |
| trajectory-decision-8 | 5739 | 9319 | 1.62× | 3–3 | 3–4 |

trajectory 네 작업의 노드 중앙값은 약 1.43–1.70배였다. 빨리 끝나는 작업은 깊이 4 상한 때문에 노드 수가 같아도 실행 시간은 줄었다. 카드 작업의 완료 깊이는 일부만 늘었다. 250ms 그룹의 변경 후 최대 관측 초과는 0.06ms로, hard deadline 보장은 아니다.

## 1000ms 제한: seed 1, 3회

| 시나리오 | 변경 전 ms | 변경 후 ms | 전 완료 깊이 범위 | 후 완료 깊이 범위 |
|---|---:|---:|---:|---:|
| immediate-win | 6.55 | 5.88 | 4–4 | 4–4 |
| same-player-continuation | 39.77 | 35.12 | 4–4 | 4–4 |
| seeded-opening-draft | 146.98 | 82.86 | 4–4 | 4–4 |
| standard-board/no-cards | 84.25 | 43.09 | 4–4 | 4–4 |
| trajectory-decision-16 | 719.78 | 457.25 | 4–4 | 4–4 |
| trajectory-decision-24 | 1000.03 | 866.24 | 3–3 | 4–4 |
| trajectory-decision-4 | 1000.03 | 678.38 | 3–3 | 4–4 |
| trajectory-decision-8 | 420.56 | 247.34 | 4–4 | 4–4 |

seed 1의 decision 4/24는 세 반복 모두 깊이 3 → 4로 늘었다. 깊이 4가 상한이므로 이후 시간 절약은 더 깊은 탐색으로 이어지지 않는다. 임의 합법 진행은 AI 최선 수의 궤적이 아니다.

## 검증

- cargo test --workspace 통과: 코어 186개, 검색 15개, 기존 AI/CLI/웹/스냅숏 등의 테스트 포함.
- cargo clippy -p augment-core -p augment-search --all-targets -- -D warnings 통과.
- cargo fmt --all -- --check 및 git diff --check 통과. 기존 [Cargo.lock](<../Cargo.lock>)의 LF/CRLF 경고만 있다.
- 새 회귀 테스트: 3개 bundled 생성자(실제 규칙 0818/0915)·양쪽 색의 캐슬링 원점/경로 공격, 한쪽 차단, 이동한/누락된 룩; 행동 불가 검사와 이전 전체 생성 방식의 상태·이벤트 일치; 0915 메뚜기 소모·입력 불변성.
- 0807/0818/0915 생성자 × seeds 1/2/3의 최대 96행동 진행, 종료 시 조기 종료. **644개 전이 기록의 전체 상태·이벤트·합법 목록이 파일 SHA256까지 일치**했다. 0807 alias의 0818 중복을 제외하면 418개이며 독립 0807 규칙을 검증한 것은 아니다. [비교 결과](<../artifacts/engine-optimization-audit/trace-comparison.json>).
- 기존 원본 포 이동 3개 helper 대조 통과. 전체 사이트 일치, 모든 카드 상황 동등성, 완전한 액션 열거, 승률 개선을 증명하는 것은 아니다.

## 재현과 산출물

[측정 스크립트](<../scripts/compare-search.ps1>)는 빌드하지 않으며 입력 실행파일 두 개를 직렬 비교한다. CLI 옵션·해시·환경은 [environment.json](<../artifacts/engine-optimization-audit/environment.json>), 전체 통계는 [comparison.json](<../artifacts/engine-optimization-audit/comparison.json>)에 있다. 측정된 before/after 실행파일과 각 회차 JSON은 같은 artifact 디렉터리에 보존했다.

저장소 루트에서:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compare-search.ps1 `
  -BeforeExecutable artifacts\engine-optimization-audit\before.exe `
  -AfterExecutable artifacts\engine-optimization-audit\after.exe -Repeats 3

# 엔진 변경 전에 baseline 실행파일을 먼저 보존해야 한다.
# 두 실행파일 모두 아래 동일 CLI를 포함한 release 빌드여야 한다.
cargo build --release -p augment-search
.\target\release\augment-search.exe transition-trace --seed 1 --decisions 96
```

실행파일을 다시 만들면 비교 해시와 환경 기록도 다시 생성해야 한다. 모델 학습이나 다른 측정을 동시에 실행하지 않는다. 다음 구조적 후보는 검증된 액션 적용 경로와 확률 probe 중복 제거이며, 이번 보고서는 그 속도 개선을 포함하지 않는다.
