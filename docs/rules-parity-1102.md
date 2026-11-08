# 1102 규칙 대조표

기준은 [augmentchess.org/rules/](https://augmentchess.org/rules/)이며, 레거시 모드는 제외한다.
원본 권위는 `rulesets/augment_chess_20261102/sources/raw/main-COPLmLfz.js` 번들이다.

규칙 항목별로 **우리 구현 / 원본 번들 / 사이트 문서**를 나란히 적고, 불일치는 근거와 함께
남긴다.

## 원본 번들에서 확인한 상수

| 상수 | 값 | 의미 |
|---|---:|---|
| `STAR_LIMIT_TURNS` | 45 | 이 턴 수에 도달하면 연장전 시작 |
| `DEATHMATCH_INTERVAL_TURNS` | 10 | 연장전 판정 주기 |
| `DEATHMATCH_REVELATION_INTERVAL_TURNS` | 5 | `revelation` RULE 적용 시 주기 |
| `DEATHMATCH_WARNING_HALF_TURNS` | 2 | 종료 경고 시점 |
| `localRuleOptions` 기본값 | `{ deathmatchEnabled: true }` | **연장전 기본 활성** |

`deathmatchEnabled`가 `false`면 연장전 대신 별 판정(`resolveStarTiebreak`)으로 즉시 종료한다.

## 승리 조건

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 킹 포획 | 승리 (`king_capture_wins`) | `WinReason::KingCaptured` | 일치 |
| 합법 행동 없음 | 패배 (`immobility_loses`) | `WinReason::NoLegalActions` | 일치 |
| 카드 승리 조건 | 카드/RULE 우선 | `WinReason::CardEffect` | 일치 |
| 3회 반복 | 별 비교 → 적은 쪽 승리 | `repetition_tiebreak: "draw"` → 무승부 | **불일치** |
| 45수 도달 | 연장전 시작 | **없음** | **불일치** |
| 연장전 무진전 종료 | 별 적은 쪽 승리 | **없음** | **불일치** |

### 별 판정 방향

원본 `starWinner`: `totals.white < totals.black ? "white" : "black"` — **별이 적은 쪽이 승리**하고
동률이면 `null`(무승부)이다.

우리 `repetition_tiebreak: "draw"`는 별을 아예 비교하지 않는다.

## 무승부 조건

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 3회 반복 | 별 판정으로 종료 | `DrawReason::ThreefoldRepetition` | **불일치** |
| 50수 규칙 | `quietTurns / 100`, 상한 100 | `fifty_move_completed_turns: 100` | 일치 |
| 동률 별 | 무승부 | 미구현 | **불일치** |

50수 규칙의 단위를 확인했다. 원본 표시 코드가 `state.fiftyMoveQuietTurns / 100`으로 진행률을
계산하므로 상한은 **100 완료 턴**이고, 우리 `fifty_move_completed_turns: 100`과 같다.

## 연장전 (deathmatch)

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 시작 시점 | sharedTurn ≥ 45 | `deathmatch_enabled: false` | **불일치** |
| 판정 주기 | 10 하프턴 (`intervalHalfTurns`) | `deathmatch_quiet_move_count: 0` | **불일치** |
| 게이지 초기화 | 폰 이동, 기물 포획, 액티브 카드 사용 | `mark_deathmatch_progress` 구현됨 | 구현 있음 |
| 경고 | 종료 2 하프턴 전 | 미구현 | **불일치** |
| 종료 판정 | 별 적은 쪽 승리 | 미구현 | **불일치** |
| `revelation` 카드 | 주기 10 → 5 | 미구현 | **불일치** |

### 중요한 발견: 구현체는 있는데 설정으로 꺼져 있다

`DeathmatchState { started_at_shared_turn, quiet_full_turns, interval_full_turns,
progress_this_round }`가 `state.rs`에 있고, `mark_deathmatch_progress()`가 폰 이동·기물 포획·
카드 사용에서 게이지를 초기화한다. 전이 경로 곳곳에 `mark_deathmatch_progress()` 호출이 있다.

즉 **연장전 로직의 상당 부분이 이미 구현돼 있고**, `deathmatch_enabled: false` 설정과
`deathmatch_starts_at_move_count: 0`, `deathmatch_quiet_move_count: 0` 때문에 실행되지 않는다.

게다가 `engine.rs`의 `ensure_ruleset`에 이런 가드가 있다:

> `if state.deathmatch.is_some() { return Err(... "최신 룰셋에는 데스매치가 없습니다." ...) }`

이 가드는 **"1102에는 연장전이 없다"는 잘못된 전제**를 담고 있다. 원본은 연장전을 기본
활성(`localRuleOptions` 기본값 `{ deathmatchEnabled: true }`)으로 두고 45수부터 적용한다.

## 종료 사유 이름

원본은 종료 사유를 별도로 표기한다: `king_capture`, `no_legal_move`, `threefold_repetition`,
`deathmatch_inactivity`(연장전 무변화 종료), `star_limit`(45수 별 판정), `revelation`(계시 기물
점수 판정), `resign`, `timeout`, `abandonment`, `draw_agreement`.

우리는 `WinReason`/`DrawReason` enum으로 표현하며 `deathmatch_inactivity`, `star_limit`,
`revelation`에 대응하는 variant가 없다.

## 결론

**규칙 동질성은 확보되지 않았다.** 승리·무승부의 기본 축(킹 포획·합법 행동·카드)은 맞지만,
**연장전 전체가 빠져 있고 3회 반복이 별 판정이 아니라 무승부로 처리된다.**

이 두 축은 게임 종료 방식을 바꾸므로 전이·탐색·아레나 결과에 직접 영향을 준다.
