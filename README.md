# Augmentia

증강체스 엔진과 탐색 교사를 만드는 작업 공간입니다. 먼저 학습 없이 강한 탐색을 만들고,
기력 검증 후 그 판단을 신경망에 학습시키는 Expert Iteration으로 진행합니다.

## 실행

```powershell
cargo run --release -p augment-app --bin web -- --no-open
cargo run --release -p augment-search -- bench --time-ms 250 --seed 1
cargo run --release -p augment-search -- tactics --depth 2 --nodes 100000
cargo run -p augment-app --bin play
```

웹은 [로컬 분석기](http://127.0.0.1:7878)에서 열립니다. `--no-open`을 생략하면 브라우저가
자동으로 열립니다. `--seed 1234`, `--port 9000`도 지원합니다. 모든 실행은 모델·Python·GPU 없이 동작합니다.

## 남긴 코드

| 경로 | 역할 |
|---|---|
| `crates/augment-core` | 상태·규칙·합법 행동·실제 전이·확률·종료 판정 |
| `crates/augment-search` | 턴 기준 alpha-beta·독립 확률 표본·시간/노드 예산·AND-OR 전술 진단 |
| `crates/augment-app` | `web` 분석기·`play` 수동 대국·`snapshot` 규칙 수집/비교 |
| `rulesets/` | 불변 규칙·원본 provenance·카드 카탈로그 |
| `scripts/` | 엔진/탐색·원본 helper 대조·분석기 검증 |

앱의 세 진입점은 `augment-app/src/bin/`에 있고, 각 구현은 `src/web/`, `src/play.rs`,
`src/snapshot/`에 둡니다. `cargo run -p augment-app`의 기본 실행은 `web`입니다.

기존 V2/V3/V4 신경망 실행·학습·승격·데이터 포맷·ONNX 의존성과 별도 MCTS V5 실험 구현은
삭제했습니다. 기존 AND-OR 알고리즘은 `augment-search::tactics`로 옮기고 엔진의
`Action`을 직접 사용하도록 줄였습니다. 새 학습 파이프라인은 교사의 기력이 검증된 후 설계합니다.

## 탐색과 전술 진단의 범위

현재 엔진은 `augment_chess_20260915` 카탈로그 240개 항목의 로컬 구현을 포함합니다.
원본 사이트 전체 일치나 모든 카드 상호작용의 정확성을 인증한 상태는 아닙니다.
이전 스냅숏은 엔진의 규칙 회귀와 원본 대조에 필요해 유지합니다.

- 일반 탐색은 실제 행동자 기준 max/min과 `TurnEnded` 기준 턴 깊이를 사용합니다.
  같은 턴의 카드·추가 행동·후속 선택은 턴 깊이를 줄이지 않습니다.
- 확률 분기는 live RNG와 독립된 표본 평균입니다. 예산이 끝나면 마지막 완료 반복을 반환합니다.
  깊이 1도 완료하지 못하면 미탐색 fallback과 `completed_depth=0`을 표시합니다.
- 점수는 재료·승패 기준 휴리스틱입니다. 승률이나 centipawn이 아닙니다.
- CLEANUP 다중 대상·HYPOCRISY 배치 등 일부 적용 가능한 행동이 열거에서 빠집니다.
  현재 탐색 결과는 `engine-enumerated-subset` 범위입니다.
- `tactics`는 결정론적 전수 응수 트리의 제한된 전술 진단입니다. 깊이는 액션 수입니다.
  확률 전이·알려진 대상 누락·노드 한도에서는 `unknown` 또는 오류를 반환합니다.
  공개/비공개 정보가 섞인 실제 상태를 받으므로 플레이어 정책이나 일반적인 안전 증명으로 쓰지 않습니다.
- 단일 엔진 호출은 중간에 취소하지 못합니다. 실제 시간 초과를 기록합니다.

[현재 구조](docs/architecture.md), [전환 후 작업 목록](docs/search-teacher-cleanup-backlog.md),
[턴 깊이 계약](docs/search-turn-depth.md), [관측 원칙](docs/observation-safety.md)을 참고하세요.
엔진 처리량의 과거 측정은 [누적 비교](docs/engine-cumulative-optimization.md)에 있습니다.

## 웹 분석기

- 현재 국면의 추천 행동·평가·깊이·시간·노드·확률 표본을 표시합니다. 추천은 선택해야 적용됩니다.
- 과거 수에서 분기해도 원래 경로를 보존합니다. 보드 편집과 수동 카드/룰 설치를 지원합니다.
- 카드 획득·기물·중복·상호 배제·runtime 조건이 맞지 않는 설정은 사유와 함께 거부합니다.
- ACG v1/v2를 검증해 재생하고 원본을 다시 저장합니다. 현재 엔진 분기는 공개 스냅숏
  `Partial`로 내보냅니다. 비공개 상태·RNG·일부 효과가 없어 완전한 엔진 replay는 아닙니다.
- 가져온 공개 ACG는 원래 엔진 상태로 분석하지 않습니다. 명시적인 보드 복사로 편집 분기를 만들 수 있습니다.
- 서버는 하나의 로컬 세션을 공유하고 fingerprint로 오래된 편집 요청을 거부합니다.

## 검증

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify-search.ps1 -SkipBenchmarks
```

전체 Rust 테스트·Clippy·fmt, 오프라인 분석기 검사, release 빌드, 전술 fixture와
원본 포 이동 helper 대조를 실행합니다. `-SkipBenchmarks`를 빼면 짧은 탐색/진행 측정도 실행합니다.
원본 helper 3개 사례의 일치는 전체 사이트 parity나 기력 증거가 아닙니다.

규칙 수집은 `cargo run -p augment-app --bin snapshot -- extract ...` 또는 `diff ...`를 사용합니다.
규칙 스냅숏의 불변성·revision 정책은 [버전 정책](docs/versioning.md)을 따릅니다.
