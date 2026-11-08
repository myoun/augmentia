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
| 3회 반복 | 별 비교 → 적은 쪽 승리 | `LowerStarsOnRepetition` | **일치 (수정됨)** |
| 45수 도달 | 연장전 시작 | `deathmatch_starts_at_move_count: 45` | **일치 (수정됨)** |
| 연장전 무진전 종료 | 별 적은 쪽 승리 | `LowerStarsAfterDeathmatch` | **일치 (수정됨)** |

### 별 판정 방향

원본 `starWinner`: `totals.white < totals.black ? "white" : "black"` — **별이 적은 쪽이 승리**하고
동률이면 `null`(무승부)이다. 우리 `resolve_lower_stars_*`가 같은 방향이며 `EqualStars*`가 동률을
무승부로 처리한다.

## 무승부 조건

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 3회 반복 | 별 판정으로 종료 | 별 판정으로 종료 | 일치 |
| 50수 규칙 | `quietTurns / 100`, 상한 100 | `fifty_move_completed_turns: 100` | 일치 |
| 동률 별 | 무승부 | `EqualStarsOnRepetition` / `EqualStarsAfterDeathmatch` | 일치 |

50수 규칙의 단위를 확인했다. 원본 표시 코드가 `state.fiftyMoveQuietTurns / 100`으로 진행률을
계산하므로 상한은 **100 완료 턴**이고, 우리 `fifty_move_completed_turns: 100`과 같다.

## 연장전 (deathmatch)

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 활성 여부 | 기본 활성 | `deathmatch_enabled: true` | **일치 (수정됨)** |
| 시작 시점 | sharedTurn ≥ 45 | `deathmatch_starts_at_move_count: 45` | **일치 (수정됨)** |
| 판정 주기 | 10 하프턴 | `deathmatch_quiet_move_count: 10` | **일치 (수정됨)** |
| 게이지 초기화 | 폰 이동, 기물 포획, 액티브 카드 사용 | `irreversible`(폰·포획) + 카드 경로 | **일치 (수정됨)** |
| 종료 판정 | 별 적은 쪽 승리 | `resolve_lower_stars_after_deathmatch` | **일치 (수정됨)** |
| 경고 (종료 2 하프턴 전) | UI 알림 | 미구현 | 표시 전용, 규칙 아님 |
| `revelation` 카드 | 주기 10 → 5 | `card_runtime.revelation` → 5 | 일치 |

### 수정 이력

연장전은 **구현돼 있었지만 설정으로 꺼져 있었다.** `DeathmatchState`,
`tick_deathmatch_after_turn`, `start_deathmatch_if_due`, `resolve_lower_stars_after_deathmatch`가
모두 존재했고 전이 경로에 `mark_deathmatch_progress()` 호출도 있었다. 막은 것은 세 가지였다:

1. `deathmatch_enabled: false`, 시작·주기 0 → 실제 값으로 수정
2. `ensure_ruleset`이 "최신 룰셋에는 데스매치가 없습니다"로 상태를 거부 → 비활성 룰셋에서만
   거부하도록 수정
3. 일반 이동 경로가 **모든 수**에서 게이지를 초기화 → 폰 이동·포획만 초기화하도록 수정.
   이것이 실제 버그였고, 고치기 전에는 연장전이 시작돼도 절대 끝나지 않았다.

## 종료 사유 이름

원본은 종료 사유를 별도로 표기한다: `king_capture`, `no_legal_move`, `threefold_repetition`,
`deathmatch_inactivity`(연장전 무변화 종료), `star_limit`(45수 별 판정), `revelation`(계시 기물
점수 판정), `resign`, `timeout`, `abandonment`, `draw_agreement`.

우리는 `WinReason`/`DrawReason` enum으로 표현하며 `deathmatch_inactivity`, `star_limit`,
`revelation`에 대응하는 variant가 없다.

## 결론

**승리·무승부·연장전 축은 일치한다.** 킹 포획·합법 행동·카드 승리, 3회 반복과 45수 별 판정,
50수 무승부, 연장전 시작·주기·종료 판정이 모두 원본과 같다.

남은 항목:

| 항목 | 상태 | 근거 |
|---|---|---|
| 연장전 경고 표시 | 미구현 | UI 알림이며 규칙 판정에 영향 없음 |
| 종료 사유 이름 (`deathmatch_inactivity`, `star_limit`, `revelation`) | 이름 미대응 | 우리는 `WinReason`/`DrawReason`으로 표현하며 판정 결과는 같음 |
| 레거시 모드 (별 합산 즉시 종료) | 미구현 | 범위에서 제외 |
| 드래프트 체크포인트 | `game_start`/`10`/`20`, 3장 중 1장 | 사이트 문서와 일치 |
| 턴 구조 | 행동 후 턴 종료, `END_TURN` 카드 | P0-2에서 대조 완료 |

**규칙 판정은 원본과 동일하다.** 종료 사유의 이름 체계만 다르며, 그것은 결과에 영향을 주지 않는
표현 차이다.
