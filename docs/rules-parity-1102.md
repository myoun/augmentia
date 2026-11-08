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

## 드래프트

원본 번들에서 확인한 트리거 (`@17822676`, `@17822873`):

```
sharedTurns >= 20 && middleDraftDone && !endDraftDone  → END 드래프트
sharedTurns >= 10 && !middleDraftDone                  → MIDDLE 드래프트
```

`sharedTurns = min(white.turnsTaken, black.turnsTaken)` — **양쪽이 모두** 해당 수를 마쳐야 열린다.

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 오프닝 드래프트 | 게임 시작 | `game_start` | 일치 |
| MIDDLE 시점 | 양쪽 10수 완료 | `shared_turn >= 10` | 일치 |
| END 시점 | 양쪽 20수 완료 | `shared_turn >= 20` | 일치 |
| 매 드래프트 제안 수 | 3장 | `bundle_count: 3` | 일치 |
| 매 드래프트 선택 수 | 1장 | `picks_per_round: 1` | 일치 |
| 총 획득 카드 | 3장 | 3 라운드 × 1 | 일치 |
| END 카드 첫 드래프트 제외 | 제외 | `DraftPhase` 분리로 처리 | 일치 |
| 그랜드 모드 풀 | 28장 (OPENING 4, MIDDLE 10, PIECE 7, END 7) | `pool_size: 28`, 동일 분포 | 일치 |

### 구조적 결함: 선언된 체크포인트를 엔진이 읽지 않는다

`ModeConfig.draft`의 `checkpoints: ["game_start", "move_count_10", "move_count_20"]`는
**선언만 되고 어떤 Rust 코드도 읽지 않는다**(grep 결과 `ruleset.rs`의 타입 정의 1회뿐).

엔진의 `start_due_milestone_draft`가 `10`과 `20`을 **하드코딩**한다:

```rust
let phase = if !completed[Middle] && shared_turn >= 10 { Some(Middle) }
    else if completed[Middle] && !completed[End] && shared_turn >= 20 { Some(End) }
    else { None };
```

**현재 값은 원본과 일치하므로 동작 차이는 없다.** 그러나 manifest가 드래프트 시점의 권위가
아니라는 뜻이며, 다른 스냅숏이 다른 체크포인트를 선언하면 조용히 무시된다.

## 턴 구조

P0-2에서 원본 번들을 실행해 대조한 결과를 옮긴다([전이 대조 기록](search-teacher-cleanup-backlog.md)).

| 항목 | 원본 | 우리 | 판정 |
|---|---|---|---|
| 턴 종료 판정 | `TurnEnded` 이벤트 | `Event::TurnEnded` | 일치 |
| 같은 행동자 연속 행동 | 턴 안에서 여러 액션 | `actions_remaining` | 일치 |
| `END_TURN` 카드 | 카드 사용이 턴을 종료 | `ends_turn` | 일치 |
| 다음 턴 카드 잠금 | `nextTurnPending` (dev 카드 예외) | `available_after_turn` | 일치 |
| 턴당 카드 예산 | v2 인벤토리 `usedAtPly` 기준 | `cards_used_this_turn` | 일치 |
| `retainV6Turn` | 액터 복원·앙파상 초기화·흑 풀무브 롤백 | 동일 | 일치 |

턴 종료 카드 4장(`brainwash`, `miracle`, `trojan-horse`, `zugzwang`)은 Rust 테스트가 실제
`TurnEnded` 발생을 확인한다.

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
