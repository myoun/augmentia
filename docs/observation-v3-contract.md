# ObservationV3 의미 계약

상태: 의미 계약 및 Rust 인코더 구현 완료 (`2026-08-21`)

스키마 ID: `augmentia.observation.v3`

이 문서는 ObservationV3가 표현해야 하는 게임 의미와 공개 정보 경계를 정의한다.
실제 tensor shape, padding 한도와 네트워크 입력 배치는 3단계에서 정한다. 따라서
물리 배치가 바뀌지 않고도 모델 구조는 실험할 수 있지만, 이 문서의 의미를 바꾸면
반드시 새 observation schema ID를 발급해야 한다.

Rust가 의미의 기준이다. 기계 판독 가능한 전체 필드 목록은
`crates/augment-ai/src/observation_v3.rs`의 `GAME_STATE_SOURCE_FIELDS` 18개와
`CARD_RUNTIME_SOURCE_FIELDS` 147개다.

## 1. 관측자와 공개 정보 경계

ObservationV3는 항상 현재 합법 행동 집합을 선택하는 actor 관점이다. 진행 중 상태의
actor는 `state.turn`이며, `Draft`, `AwaitingChoice`, `ForcedMove` payload의 owner와
piece owner는 이 관계를 검증하는 데도 사용한다. terminal 상태는 기록할 수 있지만
정책·가치 입력 샘플로 다시 평가하지 않는다.

현재 일반 모드는 완전정보 게임으로 취급한다.

- 양쪽 보드 기물, 카드 인스턴스, 별, charge와 공개 runtime 효과를 모두 관측한다.
- 드래프트 offer와 choice option은 그 결정을 내리는 actor의 관측에만 존재한다.
- 상대 차례용 별도 관측을 만들면서 actor 전용 payload를 복사하지 않는다.
- deterministic RNG의 내부 상태와 seed는 모델에 주지 않는다. 이를 노출하면 미래의
  주사위·카드 무작위 결과를 미리 알 수 있다.
- 이벤트 로그와 과거 전체 상태는 입력이 아니다. 현재 규칙 판정에 필요한 반복
  횟수만 `current_repetition_count`로 파생한다.

나중에 fog-of-war 규칙이 추가되면 같은 V3 ID를 재사용하지 않고 visibility mask를
포함하는 새 스키마를 만든다.

## 2. 관점 정규화

- `self`는 actor, `opponent`는 actor의 상대다. 원래 `White`/`Black` 값도 replay
  검증 메타데이터에 보존한다.
- actor가 흑이면 rank를 뒤집어 actor의 진행 방향을 항상 같은 방향으로 맞춘다.
  file은 뒤집지 않는다.
- square 변환은 보드, target, terrain, shadow, portal과 pending record에 동일하게
  적용한다.
- player pair `[white, black]`는 `[self, opponent]`로 재정렬한다.
- raw `piece_id`, `instance_id`, `choice_id`의 숫자 크기는 입력 feature가 아니다.
  한 관측 안에서 결정론적으로 만든 local slot으로 관계만 보존한다.
- local piece table은 현재 보드뿐 아니라 runtime과 pending record가 참조하는 off-board
  ID의 합집합도 포함한다. 원본 ID는 동일 entity를 join하는 데만 사용하고 값의 크기나
  생성 순서를 모델에 노출하지 않는다.
- 정렬은 의미가 있는 원래 순서를 우선하고, 집합·맵은 canonical square, stable kind
  code, owner, local slot 순으로 고정한다.

## 3. `GameState` 필드 계약

| 필드 | 의미 표현 | 처리 |
| --- | --- | --- |
| `ruleset_id` | vocabulary namespace와 provenance | 모델 수치가 아닌 필수 메타데이터 |
| `mode` | stable string ID | categorical token |
| `board` | square, owner, exact kind, capture count, local piece slot | 보드·기물 record |
| `turn` | actor와 원래 색 | 관점 메타데이터 |
| `ply` | 시작 이후 반수 | 음이 아닌 raw count와 bounded transform |
| `quiet_ply` | 무진행 반수 | 음이 아닌 raw count와 bounded transform |
| `actions_remaining` | 현재 행동 예산 | raw count; 2에서 clamp하지 않음 |
| `cards_used_this_turn` | 양쪽 이번 차례 카드 사용량 | self/opponent count |
| `phase` | playing/forced/draft/choice와 전체 payload | tagged record |
| `status` | ongoing/win/draw와 이유 | 샘플 메타데이터; neural input에서 제외 |
| `cards` | instance, exact definition ID, owner, 별, charge, unlock | card record sequence |
| `half_star_totals` | 양쪽 누적 반별 | self/opponent raw count |
| `turns_taken` | 양쪽 완료 차례 수 | self/opponent raw count |
| `completed_drafts` | opening/middle/end 완료 여부 | 고정 3개 boolean |
| `deathmatch` | 시작·quiet·interval·round progress | presence가 있는 record |
| `en_passant` | target, captured, pawn owner | presence가 있는 spatial record |
| `applied_rules` | exact rule definition ID 목록 | ruleset namespaced token set |
| `card_runtime` | 아래 147개 필드 | typed record collections |

추가 파생값으로 `shared_turn_count`와 `current_repetition_count`를 포함한다. 절대
deadline은 해당 player의 `turns_taken`, `ply` 또는 `shared_turn_count`를 기준으로
남은 시간으로 바꾸되, 원래 값이 없다는 뜻과 남은 시간이 0이라는 뜻을 presence로
구분한다.

## 4. `GamePhase` payload 계약

- `Playing`: payload 없음.
- `ForcedMove`: 강제 `piece_id` 관계, `Platform` 또는 exact effect ID인 source,
  `ResumeTurn`/`FinishTurn` continuation.
- `Draft`: `Opening`/`Middle`/`End`, 순서를 보존한 exact card ID offer,
  `resume_turn`.
- `AwaitingChoice`: local choice slot, owner, source card local slot, exact handler ID,
  순서를 보존한 option string, continuation.

handler/effect/option 문자열은 해시하지 않는다. 현재 ruleset vocabulary에 없는 값은
데이터 검증 오류이며 하나의 `unknown` 버킷으로 합치지 않는다.

## 5. 기물과 카드의 안정 ID

기물 종류는 `PieceKind::stable_code()`의 1~51을 그대로 사용한다. 51종 각각이 고유한
token이며 V2의 16개 catch-all 채널은 폐기한다. 새 기물은 기존 번호를 재사용하지 않고
새 번호를 추가하며 schema 검토를 요구한다.

카드 정의는 `(ruleset_id, definition_id)` 문자열 쌍이 안정 ID다. 256 bucket scalar
hash는 폐기한다. 카드 instance ID는 같은 정의 카드 여러 장과 runtime 참조를 연결하는
관계 키일 뿐 숫자 feature가 아니다. 카드 record에는 최소한 다음 의미가 존재한다.

- owner, exact definition ID, phase/activation과 ruleset 정의 feature;
- `half_stars`, `charges`;
- `available_after_turn`의 presence와 owner 기준 남은 차례;
- `opening_auto_card_ids` membership;
- 현재 합법 행동에 연결되는지 여부는 ActionDescriptorV3 단계에서 추가한다.

## 6. `CardRuntimeState` 147개 필드의 표현 규칙

모든 필드는 Rust의 원래 타입을 보존하는 아래 record family 중 하나로 변환한다.
필드 이름의 전체 목록과 순서는 코드의 `CARD_RUNTIME_SOURCE_FIELDS`가 고정한다.

| Rust 형태 | ObservationV3 의미 |
| --- | --- |
| `[bool; 2]`, `[u8/u32; 2]`, `[Option<T>; 2]` | self/opponent player feature 또는 player record |
| `BTreeSet<u32>` | local piece/card slot에 붙는 정확한 marker |
| `BTreeMap<u32, T>` | local piece slot과 typed payload record |
| `BTreeSet<Square>`, `Option<Square>`, square pair | canonical square/terrain record |
| `Vec<T>` | 원소 필드를 보존한 가변 record sequence |
| 전역 bool/count/option | presence와 payload가 분리된 global feature |
| `BTreeSet<String>` | exact ruleset-namespaced ID token set |

대표적인 의미 그룹은 다음과 같다.

- player passive/clock: `hill_king`, `initiative`, `dice_locks`, `skip_turns`,
  `royal_command`, `socialism`, `armistice_remaining_ply`, `gale_due_turns`,
  `hallucination_until_ply`, `prophecy`, `vanish` 등;
- piece marker/resource/clock: `moved_piece_ids`, `frozen`, `windmill_modes`,
  `wizard_mana`, `reaper_souls`, `large_pieces`, `merchant_gold`, `ice_sheet`,
  `imperial_moves`, `bear_retaliations`, `bunkers` 등;
- spatial rule/terrain: `platform`, `quantum_positions`, `barricades`, `palaces`,
  `portals`, `black_holes`, `crown_pos`, `football_pos`, `high_ground_squares`,
  `highways`, `rule_portals`, `rule_bombs` 등;
- multi-entity/pending relation: `delayed_hazards`, `pending_otherworld`,
  `feudal_contracts`, `pending_panic`, `pending_premoves`, `pending_scarecrows`,
  `twin_bonds`, `active_trolley`, `bribe_reverts`, `necromancy_reverts`,
  `vip_pending`, `chains`, `icbm_pending`, `crown_holder`, `spies` 등;
- global rule switches: `revelation`, `collapsed_files_ranks`, `acceleration`,
  `camouflage_color`, `conveyor`, `macho_chess`, `mistake_rule`,
  `monochrome_chess`, `recycling`, `saturation_rule`, `transcendence` 등.

한 필드가 여러 entity를 참조하면 scalar로 합치지 않고 각 참조와 payload를 보존한다.
예를 들어 `FeudalContract`는 owner, pawn slot, guardian slot 세 값을, `DelayedHazard`는
kind, owner, trigger player와 모든 square를, `PendingPremove`는 owner, trigger clock과
모든 `MoveAction`을 보존한다.

## 7. 수치와 optional 값

의미 계약의 원본 값은 정수/boolean/categorical이다. float 변환은 3단계 물리 인코더의
책임이며 다음 규칙을 지킨다.

- 임의 상수로 clamp해서 서로 다른 큰 값을 같게 만들지 않는다.
- bounded transform이 필요하면 raw bucket/overflow 또는 log-scale을 함께 둔다.
- absolute turn/ply deadline은 관점에 맞는 남은 시간으로 변환한다.
- `Option<T>`는 presence bit와 payload를 분리한다.
- count 0, missing, expired를 서로 구분한다.
- HP는 `hp`, `max_hp`를 모두 보존하고 비율만 남기지 않는다.

## 8. 명시적으로 제외하는 내부 상태

| 내부 값 | 제외 이유 | 공개 대체값 |
| --- | --- | --- |
| `rng`/seed | 미래 deterministic 무작위 결과 누출 | 없음 |
| `next_piece_id` | allocator 구현 세부 | local piece slot |
| `next_card_instance_id` | allocator 구현 세부 | local card slot |
| `next_choice_id` | allocator 구현 세부 | local choice slot |
| `turn_end_requested` | reducer 내부의 일시 상태 | resolve된 `phase`와 합법 행동 |
| 전체 `repetitions` map | 내부 position hash/history 표현 | `current_repetition_count` |

`fingerprint`, 이벤트 로그, UI 문자열, 탐색 통계와 모델 평가는 게임 상태 입력이 아니다.
이 값들은 provenance나 진단 데이터로만 저장한다.

## 9. 샘플 provenance 계약

모든 학습·평가 record는 최소한 다음 문자열 필드를 반드시 가진다.

- `ruleset_id`;
- `observation_schema_id`;
- `action_schema_id`;
- `model_id` 또는 데이터 생성 actor ID;
- record 자체의 `schema_version`.

V2 Gen2 audit generator에도 이 필드를 추가해 새 데이터가 출처 없이 생기지 않도록
했다. V3 encoder가 구현되기 전에는 어떤 샘플에도
`augmentia.observation.v3`를 기록하지 않는다.

## 10. 변경과 검증 규칙

- `GameState`나 `CardRuntimeState` 필드가 추가되면 exhaustive review test가 실패해야
  하며, 이 계약에서 encode/derive/exclude 중 하나를 선택해야 한다.
- 기존 stable piece code와 card definition ID의 의미는 변경하지 않는다.
- 공개 범위, 관점 변환, deadline 기준 또는 ID 의미가 바뀌면 새 schema ID를 만든다.
- 3단계 구현은 18개 `GameState` 필드, 147개 runtime 필드와 2개 파생값에 대한
  coverage test를 제공해야 한다.

## 11. Rust 물리 레이아웃

구현은 `crates/augment-ai/src/encoder_v3.rs`의 `encode_observation_v3()`다. 현재 V2
ONNX와 MCTS 입력은 아직 교체하지 않았으며, V3 학습·모델 경로가 준비될 때까지 두
인코더를 병행한다.

고정 dense 입력:

- `spatial`: `f32[102, 8, 8]`, 총 6,528개 값, 범위 `[0, 1]`;
- plane 0~50: self의 `PieceKind::stable_code() - 1`;
- plane 51~101: opponent의 `PieceKind::stable_code() - 1`;
- `global`: `f32[32]`, 범위 `[0, 1]`;
- count의 dense 사본은 `x / (x + 1)`을 사용하고 exact integer는 typed state에 함께
  보존한다.

`global` index:

| index | 의미 |
| --- | --- |
| 0~8 | 행동 수, ply, quiet ply, 양쪽 카드 사용·별·완료 turn count |
| 9~11 | opening/middle/end draft 완료 boolean |
| 12~16 | deathmatch와 en-passant presence/progress |
| 17~21 | 양쪽 card 수, rule 수, shared turn, 현재 repetition count |
| 22~25 | playing/forced/draft/choice one-hot |
| 26~29 | standard/chaos/grand/campaign one-hot |
| 30~31 | acceleration/revelation global switch |

정확한 board piece는 `BoardPieceV3`, 카드는 `CardRecordV3`, phase는 `PhaseV3`, 전역
정수와 terminal metadata는 `GlobalStateV3`에 보존한다. `CardRuntimeState`는 항상
147개의 `RuntimeFeatureV3`를 source 순서대로 내보내며 field ID는 1~147이다. 값은
`None`, boolean, signed/unsigned integer, relative player, local piece/card reference,
exact piece kind, canonical square, exact text, list, named record 중 하나다. 가변 목록에는
고정 길이 truncation이나 padding을 적용하지 않는다.

검증은 다음을 고정한다.

- dense shape, finite 여부와 `[0, 1]` 범위;
- 147/147 runtime field 이름과 ID 순서;
- 51종 기물 plane의 비충돌;
- 192개 card definition ID의 무손실 보존;
- raw piece/card ID와 RNG state 비노출;
- 색 교환·rank mirror 시 board, card와 대표 runtime relation의 동일 관측;
- 서로 다른 runtime 의미가 같은 관측으로 축약되지 않음.
