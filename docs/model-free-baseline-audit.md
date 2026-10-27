# 학습 없는 탐색 기준선과 엔진 감사

> 최초 도입 당시의 측정 기록이다. 현재 실행 구조와 검증 명령은 [README](../README.md)를 따른다.

## 결론

현재 엔진을 버리고 범용 규칙 DSL·규칙 임베딩·새 학습 파이프라인을 만드는 것은 권하지 않는다.
이미 행동 예산, 드래프트, 강제 이동, 후속 선택, 카드 상태와 공개 확률 전이를 다루는 코어가 있다.
다만 **로컬 구현·로컬 테스트 통과·원본 사이트 일치는 서로 다른 증거**다.

이번 작업은 신경망 없는 작은 실행 경로를 만들고 원본 대조를 시작한 것이다.
강한 AI를 완성하거나 기존 V4보다 높은 승률을 입증한 작업은 아니다.
기존 V2/V3/V4 모델, 학습 산출물, 웹 모델 제안 경로는 바꾸지 않았다.

| 항목 | 확인한 사실 | 아직 주장할 수 없는 것 |
|---|---|---|
| 0915 카드 구현 | manifest의 구현/behavioral-slice 수 240, site-parity 수 0 | 240장 전체 원본 일치 |
| 로컬 엔진 | 코어 라이브러리 183개 테스트 통과 | 전체 카드 상호작용/실사이트 정합성 |
| 새 탐색 | 회귀 테스트 15개, release 빌드, strict clippy 통과 | 실전 경쟁력 또는 V4 대비 우위 |
| 원본 대조 | 조건을 제한한 포 이동 3개 fixture 모두 일치 | 전체 포 카드, 포털, 과거 0915 규칙 전체 일치 |
| 짧은 실행 | 시드 1/2/3 각각 32 decisions, 일반 apply 재검증 성공 | 완주율·승률·강함 |

근거: [0915 manifest](../rulesets/augment_chess_20260915/manifest.json),
[원본 provenance](../rulesets/augment_chess_20260915/sources/source-manifest.json),
[탐색 구현과 테스트](../crates/augment-search/src/lib.rs).
manifest의 parity 수와 기존 카드 스냅숏은 수정하지 않았다.

## 1. 원본 로직을 그대로 쓸 수 있는가

[보관된 JS 번들](../rulesets/augment_chess_20260915/sources/raw/main-6qeAcxbX.js)은 독립 게임 엔진 모듈이 아니다.
상단에는 보관되지 않은 browser/auth sibling chunk import가 있고, 말 이동과 reset 경로에는 UI·DOM·worker·네트워크 호출이 있다.
전체 getLegalMoves도 전역 상태에 의존하고 일부 상태를 바꾼다. 번들 전체를 Node에서 import하는 방식은 바로 쓸 수 없다.
[스냅숏 선언 평가기](../crates/augment-app/src/snapshot/declarative.rs)는 JS 데이터 추출기이지 게임 실행기가 아니다.
당시 V3 검증(현재 제거)은 로컬 Rust/Python/ONNX 계약 검사이지 원본 JS differential test가 아니다.

그러나 순수하거나 의존성을 좁힐 수 있는 helper는 **원본 알고리즘을 실행하는 differential oracle**로 재사용 가능하다.
[포 대조 스크립트](../scripts/check-cannon-oracle.mjs)는 해당 파일의 포 이동 helper와 의존 선언만 런타임 추출한다.
전체 번들·브라우저·네트워크·별도 subprocess를 실행하지 않는다.

- 번들 SHA256: 60d5816e78896d4b5371c0d5cc37ef0a302aa0cc814c3b027be3936124878dfc
- 추출 범위: 1409, 2462–2464, 2553–2580, 2601–2684행.
- 추출 선언 SHA256: 904e540f0f7e937c23d0535340468365bba369b97338d26be68495cf44f25185
- no-rules/no-passives, 기본 보드 크기, 백 포 d4, 포털/투명 기물 없음.
- 빈 이동선: 양쪽 결과 없음. d5 폰 스크린+d7 적 룩: 양쪽 d6/d7. d5 포 스크린+d7 적 룩: 양쪽 결과 없음.
- 점유와 포획 허용 callback은 fixture adapter가 제공하므로 사이트의 전체 포획 규칙까지 검증한 것은 아니다.
- hash가 바뀌거나 목적지가 다르면 fail-closed한다. 목적지를 일부러 틀린 음성 fixture에서 exit 1도 확인했다.
  VM timeout은 보안 sandbox가 아니다.

대조 결과는 [원본 대조 JSON](../artifacts/engine-baseline-audit/cannon-oracle.json)에 있다.
사이트에서 수집한 live 번들에는 미공표 변경이 섞여 있고, 과거 0915 번들 자체는 보관되지 않았다.
따라서 이 실험을 historical 0915 parity로 승격하지 않는다.
원본 재사용·재배포 라이선스도 확인되지 않았으므로 별도 확인이 필요하다.

## 2. 새 기준선의 계약

새 [augment-search crate](../crates/augment-search/Cargo.toml)는 core/serde/serde_json만 의존한다.
ONNX, GPU, Python, 모델 파일이 필요 없다. 기존 V4도 이미 실제 행동자와 확률 샘플을 다뤘다.
이 의미론을 새로 발명한 것이 아니라 모델 성능과 전이 비용을 분리해 볼 독립 기준선이다.

- 반복 심화 minimax/alpha-beta. 노드의 실제 state.turn이 루트와 같으면 max, 다르면 min.
- 연속 행동·강제 이동·드래프트·후속 선택을 일반 체스처럼 무조건 부호 반전하지 않는다.
- 깊이는 수/턴 수가 아니라 **선택한 액션 수**다.
- 루트에서 이전 반복의 최선 액션을 먼저 탐색하고 alpha를 공유한다. 학습 policy로 후보를 제거하지 않는다.
- 기존 apply_for_search를 사용해 실제 게임 RNG는 보존하고, 별도의 공개 탐색 시드로 확률을 샘플링한다.
- 확률 전이는 기본 4개 표본의 평균이다. 표본 자식에는 full alpha-beta window를 사용한다.
  bound를 평균하지 않으며, 표본 처리 중 중단되면 부분 평균을 값으로 쓰지 않는다.
- 확률을 정확히 전수 열거한 expectimax가 아니다. 결정 위치별 고정된 표본 트리의 근사 결과다.
- 시간·노드 예산이 있고 마지막으로 **전체 열거 루트를 처리한 반복**만 채택한다.
  하나도 완료하지 못하면 합법 목록의 첫 액션을 fallback으로 반환하고 completed_depth=0으로 표시한다.
- deadline은 cooperative다. 단일 엔진 호출 내부를 중단하지 못하며 초기 루트 열거도 예산을 넘길 수 있다.
  실제 elapsed_ms/budget_overrun_ms를 기록한다. hard real-time을 보장하지 않는다.
- 재료 평가: 폰 1, 비숍/나이트 3, 룩 5, 퀸 9, 킹 100, 다른 특수 기물 기본 3, 승패 ±100000, 무승부 0.
  카드 가치, 왕권 교체, 별 판정, 장기 효과, 이동성 등은 제대로 평가하지 않는다.

### 반드시 알아야 할 액션 누락

[코어 액션 생성/적용](../crates/augment-core/src/engine.rs)은 서로 정확히 같은 집합을 표현하지 않는다.
apply는 목록에 없는 일부 동적 대상도 별도로 검증해 받아들인다.

- CLEANUP: 생성기는 단일 기물 대상만 제공하지만 apply는 유효한 1–3개 대상 조합을 허용한다.
- HYPOCRISY: 생성기는 첫 빈칸 4개 배치 하나만 제공하지만 apply는 다른 유효한 4칸 배치를 허용한다.
- HYPOCRISY에서 수십만 조합을 먼저 모두 생성하면 deadline 밖에서 비용이 폭발할 수 있다.
  이 작업에서는 코어 의미론을 바꾸거나 전수 열거를 급히 덧붙이지 않았다.

모든 결과에 action_space=engine-enumerated-subset을 기록한다.
root_actions/scored_root_actions는 **생성된 부분집합**의 크기이지 모든 유효 액션의 수가 아니다.
known_omitted_target_cards는 탐색 중 발견한 두 카드의 보수적인 알림이다.
빈 목록도 전체 액션 완전성의 인증이 아니다. 유효한 CLEANUP 다중 대상이 실제로 누락된다는 회귀 테스트를 추가했다.

### 보고서와 재현 한계

라이브러리 결과에는 실행 가능한 selected_action이 있지만 JSON 직렬화에서는 생략한다.
CLI의 Debug action 문자열과 결과 JSON은 진단 로그이지 replay 포맷이 아니다.
completed_depth=0일 때 score는 선택한 fallback의 값이 아니라 현재 루트의 재료 평가다.

nodes는 탐색 자식 표본 수, apply_calls는 wrapper가 호출한 search adapter 수다.
legal_calls/legal_ms는 명시적 목록 생성만 측정한다.
apply_ms에는 코어 내부의 반복 목록 생성, state clone, 이벤트 생성, 확률 probe/reapply 등이 포함된다.
내부 각 비용을 따로 계측하지 않았으므로 지금 숫자로 clone 또는 legal만이 원인이라고 단정할 수 없다.
시드·상태·노드 예산으로 선택은 재현할 수 있지만 wall-clock 중단 깊이는 시스템 부하에 따라 달라진다.

## 3. 검증과 직렬 측정

[단일 검증 스크립트](../scripts/verify-search.ps1)를 실행했다.
183개 코어 테스트, 15개 탐색 테스트, strict clippy, release 빌드, 원본 helper 대조가 통과했다.
탐색 테스트에는 백/흑 루트와 상대 최소화, 같은 플레이어 연속 행동, exhaustive minimax 대조,
유한 window의 서로 다른 확률 자식 평균, 확률 처리 중 노드 중단, 독립 RNG 결과,
마지막 완료 반복 보존, 정확한 노드 한도, 강제 이동, 프로모션 후속 선택,
terminal/무승부/빈 ongoing/엔진 오류 구별이 포함된다.

최종 수치는 작업 완료 버전의 release binary로 workload를 **직렬 실행**한 결과만 사용했다.
초기 병렬 실행 측정은 배제했다. 각 위치/시드 1회 관측이며 통계적 성능 비교는 아니다.
AMD Ryzen 7 5800X, 논리 프로세서 16, rustc 1.90.0, Node v22.20.0.
[환경 기록](../artifacts/engine-baseline-audit/environment.json)의 기준 commit은
5770ee7fb2c0319cd48fb8073bb90f7462180e37이며 이 작업의 미커밋 변경을 포함한다.
기본 depth 상한 4, 100000 nodes, chance_samples=4, search_seed=19.

### 250ms, 시드 1 대표 결과

| 위치 | 열거 루트 | 완료 깊이 | 탐색 노드 | 실제 ms | apply 시간 비중 |
|---|---:|---:|---:|---:|---:|
| 즉시 왕 포획 | 16 | 4 | 851 | 9.00 | 87.4% |
| 같은 플레이어 추가 행동 | 16 | 4 | 5515 | 58.95 | 89.9% |
| 표준 보드/카드 없음 | 20 | 4 | 2838 | 136.39 | 90.2% |
| 실제 초기 드래프트 | 3 | 4 | 3988 | 206.33 | 93.3% |
| 임의 합법 진행 decision 4 | 32 | 3 | 4590 | 250.06 | 93.8% |
| decision 8 | 47 | 3 | 4177 | 250.01 | 94.7% |
| decision 16 | 55 | 2 | 3451 | 250.08 | 95.8% |
| decision 24 | 3 | 3 | 4146 | 250.06 | 95.0% |

임의 진행은 AI 최선 수의 궤적이 아니고 저렴하게 다양한 상태를 얻는 고정 목록 인덱스 진행이다.
노드 수에는 중단된 다음 반복의 작업도 포함되며, 반환 깊이는 그 이전 완료 반복이다.
시드 1/2/3의 카드 포함 trajectory에서는 250ms에 대체로 깊이 2–3,
apply 비중 91.7–96.0%였다. 표준 no-card fixture는 세 실행 모두 깊이 4였다.
시드 1에서 예산을 1000ms로 늘리면 trajectory 4/8/16/24의 완료 깊이가 3/4/3/3이었다.
예산 4배가 깊이 4배를 의미하지 않고, 한 경우는 상한 4에 도달했다.

[250ms 시드 1](../artifacts/engine-baseline-audit/bench-250ms-seed1.json),
[시드 2](../artifacts/engine-baseline-audit/bench-250ms-seed2.json),
[시드 3](../artifacts/engine-baseline-audit/bench-250ms-seed3.json),
[1000ms 시드 1](../artifacts/engine-baseline-audit/bench-1000ms-seed1.json).
이 artifacts는 로컬 생성물이므로 checkout에 없으면 검증 스크립트로 재생성한다.

### 짧은 self-play smoke

시드 1/2/3에서 각각 32 decisions를 실행했다. 매 선택은 일반 engine.apply로 재검증했고,
검색 전후 원본 상태 fingerprint 불변을 검사했다. 카드 액션은 각각 3/2/4회 있었다.
세 실행 모두 Ongoing, truncated=true로 끝났으며 무승부로 기록하지 않았다.
완전한 대국 0개이므로 승률·완주율·카드 전략 평가 자료가 아니다.
후속 choice는 이 세 live trajectory에 나타나지 않았고, 별도 회귀 fixture로 검사했다.
관측한 250ms short-play 최대 초과는 약 0.617ms였다. 모든 상태에서의 최대 지연 보장은 아니다.

[시드 1 진행](../artifacts/engine-baseline-audit/play-250ms-seed1.json),
[시드 2 진행](../artifacts/engine-baseline-audit/play-250ms-seed2.json),
[시드 3 진행](../artifacts/engine-baseline-audit/play-250ms-seed3.json).

## 4. 다음 작업 순서

1. **원본 정합성과 액션 완전성.** 이번 원본 helper 대조를 난수 소비/후속 phase/카드 잠금/별 종료 등
   좁은 상태 전이 fixture로 확장한다. 원본 접근과 adapter가 지원하는 범위를 case마다 기록한다.
   CLEANUP·HYPOCRISY는 지연 생성 또는 명시적인 후보 제한 계약을 설계한다.
   제한된 후보면 계속 제한이라고 표시하고, 전체 열거에는 테스트와 시간 예산을 포함한다.
2. **전이 내부를 세분화해서 프로파일링.** apply_ms가 가장 큰 측정 구간이다.
   공개 RNG probe/reapply, legal 재생성, clone, 이벤트 비용을 나누고 동일 fixture/노드 수로 비교한다.
   이미 검증된 액션용 내부 적용 경로·캐시 등을 고려할 수 있으나 불법 액션을 허용하는 공개 우회 API는 만들지 않는다.
   재검증을 제거하기 전 parity/RNG 테스트를 통과시킨다. universal DSL로 이 비용이 자동 해결되지는 않는다.
3. **저렴한 평가와 전술 기준선.** 실제 royal piece, 즉시 위협·보호·이동성, 별 종료를 소수 특징으로 보완하고
   고정 전술/카드 선택 suite를 만든다. shallow search로 카드 가치가 드러나지 않는 드래프트에서 특히 현재 재료 평가는 약하다.
   현재 기준선과 동일 예산으로 측정하고, 연장전 포함 완주 smoke를 추가한다.
4. **교사 검증 뒤 학습 계약 설계.** 현재는 이전 세대 모델 경로를 제거했다. 새 탐색 교사의
   동일 시간 기력과 정보·coverage를 검증한 뒤 실제 필요한 정책·Value 학습 계약을 만든다.

## 재실행

저장소 루트에서:

```powershell
cargo run --release -p augment-search -- bench --time-ms 250 --seed 1
cargo run --release -p augment-search -- play --time-ms 250 --seed 1 --decisions 32
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify-search.ps1
# 빠른 테스트/빌드/원본 helper smoke만:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify-search.ps1 -SkipBenchmarks
```

bench는 CLI 내부 fixture를 사용한다. 기존 사용자 스냅숏을 불러오는 입력 API나 웹 UI 통합은 이번 범위에 없다.
범위에 ONNX/Python/전체 workspace 계약 변경이 없으므로 전체 V3 fixture 재생성/학습은 실행하지 않았다.
