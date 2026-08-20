# V3 Rust/Python 골든 계약

상태: 구현 완료 (`2026-08-21`)

ObservationV3와 ActionDescriptorV3의 의미 기준은 Rust다. Python은 Rust 엔진의 카드
규칙, 합법 행동, 관점 변환이나 local ID 배정을 다시 구현하지 않고, Rust가 직렬화한
versioned record를 검증하고 학습 입력으로 소비한다.

## 구성

- Rust exporter: `crates/augment-cli/src/bin/export_v3_fixtures.rs`
- checked-in fixture: `train/v3/fixtures/observation_action_v3.json`
- Python validator: `train/v3/schema.py`
- Python parity tests: `train/v3/test_schema.py`

fixture ID는 `augmentia.golden.observation-action.v3`이며 내부에 다음 provenance를
필수로 둔다.

- ruleset: `augment_chess_20260807`;
- observation: `augmentia.observation.v3`;
- action: `augmentia.action.v3`.

## 대표 상태

fixture 순서는 고정한다.

1. `initial_draft`: exact card offer와 `DraftPick`;
2. `standard_playing`: 일반 보드 이동;
3. `forced_move`: local piece reference와 강제 이동 phase;
4. `awaiting_choice`: source card, handler, option과 `ResolveChoice`;
5. `runtime_rich`: 카드, 기물 능력, player/piece/spatial runtime record;
6. `promotion`: 네 가지 promotion descriptor;
7. `terminal_draw`: terminal metadata와 빈 합법 행동 목록.

## 검증 방향

Rust workspace test는 현재 encoder/exporter가 만든 compact JSON과 checked-in fixture를
byte-for-byte 비교한다. Rust 의미나 serialization이 달라지면 fixture를 명시적으로
재생성하지 않는 한 테스트가 실패한다.

Python validator는 다음을 검사한다.

- suite/ruleset/observation/action schema ID;
- `f32[102,8,8]` spatial과 `f32[32]` global shape, finite 및 `[0,1]` 범위;
- board/card local slot, square 0~63, piece kind 1~51;
- runtime field 147개와 연속 field ID 1~147;
- 모든 recursive runtime value type과 local reference 범위;
- Move, PieceAbility, Card, DraftPick, ResolveChoice descriptor payload;
- descriptor 중복, target 종류와 가변 square 목록;
- promotion 4종과 terminal empty-action case.

Python은 알 수 없는 schema, field type, descriptor kind, 범위를 벗어난 local reference,
shape 변경과 비정상 수치를 즉시 `SchemaError`로 거부한다. 오래된 데이터를 추측해서
읽거나 V2 형식으로 fallback하지 않는다.

## 재생성과 실행

저장소 루트에서 Rust fixture를 재생성한다.

```powershell
cargo run -p augment-cli --bin export_v3_fixtures
cargo test -p augment-cli --bin export_v3_fixtures
```

Python 테스트는 대형 학습 dependency 동기화가 필요 없으므로 `train`에서 실행한다.

```powershell
uv run --no-sync python -m unittest discover -s v3 -p "test_*.py"
```

fixture 변경은 observation/action schema 변경, 의도적인 직렬화 변경 또는 대표 상태
확장과 함께 검토한다. 단순히 drift test를 통과시키기 위해 자동 갱신하지 않는다.
