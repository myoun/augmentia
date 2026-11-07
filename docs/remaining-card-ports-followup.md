# 카드 이식 후속 조사 항목

2026-11-07 기준, 중반 4장과 후반 30장 이식 완료 직후 정리한 **미착수 조사 항목**이다.
아래 항목은 모두 "확인해보면 좋겠다" 수준이며, 이식된 카드가 동작하지 않는다는 뜻이 아니다.
현재 지원 카드 271장은 테스트로 동작이 고정돼 있다.

## 1. 공유 상태를 함께 갱신하는 카드 쌍 전수 조사

이번 이식에서 **같은 상태 필드를 두 카드가 함께 갱신하는 경우**에 실제 버그가 하나 나왔다.
`collapse` 카드와 `periodic-collapse` RULE이 `rule_runtime.collapse_depth`를 공유하는데,
같은 턴 경계에서 둘 다 정산돼 링이 두 번 제거됐다. 원본은 같은 경계에서
`pending-collapse`가 이미 처리됐으면 주기 정산을 건너뛰지만
(`main-RjrZ5fFP.js:213178`), Rust에는 그 가드가 없었다. 지금은 `d9d3f01`에서 고쳤다.

같은 종류의 누락이 남아 있을 수 있는 곳을 아직 조사하지 않았다. 조사 방법:

1. `engine/cards/rules.rs`, `rule_support.rs`의 `RuleRuntime` 필드와
   `state.rs`의 `CardRuntimeState` 필드 중 **두 개 이상의 카드 파일에서 대입되는 것**을 찾는다.
2. 각 필드마다 "두 카드가 같은 턴 경계에 동시에 정산될 수 있는가"를 확인한다.
3. 동시에 가능하면 원본이 순서·중복 제거를 어떻게 하는지 대조한다.

이번까지 확인된 공유 필드와 판단:

- `collapse_depth`: `collapse` 카드와 `periodic-collapse` RULE이 함께 갱신하고 같은 턴
  경계에서 둘 다 정산된다. **가드가 없어 실제 버그였고 `d9d3f01`에서 고쳤다.**
- `monster_last_halfmove`: `monster` RULE과 `black-magic` 아군 괴물이 공유한다. 한 번에
  하나의 정산만 실행되므로 문제 없음을 확인했다.
- `barricades`/`blockade_walls`: 삽입하는 카드가 많지만(`barricade`, `blockade`,
  `checker`, `conscription`, `conveyor`, `lobster`, `random-roulette`, `collapse`)
  제거는 공통 `remove_barricade` 경로라 서로 덮어쓰는 문제는 없다. 다만 **동시에 같은
  칸을 점유하려는 경우의 우선순위**는 대조하지 않았다.
- `large_pieces`: `judgment`, `random-roulette`, `shared.rs`가 생성하고 파괴는 공통
  경로다. 생성 측 충돌은 아직 조사하지 않았다.

## 2. `capture_count` 상한 차이

원본은 `capturesMade`를 `Math.min(3, ...)`으로 3에서 자른다
(`main-RjrZ5fFP.js:226387-226390`). Rust는 `saturating_add`로 상한이 없다.

`judgment`는 `>= 2`와 최댓값 비교만 쓰므로 상한 유무로 결과가 갈리지 않는다.
`saturation`은 `capture_count`가 아니라 별도 필드
`card_runtime.saturation_capture_counts`를 3으로 비교하므로
(`engine.rs:9384-9391`) 이 상한 차이의 영향을 받지 않는다.

따라서 지금 확인된 범위에서는 상한 차이가 동작을 바꾸는 경로가 없다. 다만
`capture_count`를 새로 참조하는 카드를 추가할 때 상한을 가정하지 않도록 주의한다.

## 3. 대형 기물 HP 피해 경로의 제한

`settle_collapse`와 HP 피해를 다루면서 확인한 사실이다. 최신 룰셋에서 `BigRook`과
`BigBishop`은 Rust의 HP 피해 경로에 도달하지 않는다. `engine.rs:3586-3588`이 두 종류를
명시적으로 제외하기 때문이다. HP 피해를 받는 대형 기물은 `Colossus` 등 나머지다.

**원본은 다르다.** `captureProtectionKind`가 `colossus`, `big-rook`, `big-bishop`을 모두
`health-piece`로 판정하고(`main-RjrZ5fFP.js:204992`), HP 피해도 그 보호 종류에 대해
적용된다(`main-RjrZ5fFP.js:209294-209295`). 즉 원본에서는 big rook과 big bishop도
HP를 가진 피해 가능 기물이다.

따라서 Rust의 제외는 **원본과 다를 수 있는 지점**이다. 다만 이 제외를 넣은 커밋과 이유를
아직 추적하지 않았고, big rook이 실제로 피해를 받아야 하는 상황(어떤 공격이 health-piece
피해를 주는지)도 함께 확인해야 한다. 이번에 추가한 캡처 집계 수정은 HP 경로에 실제로
도달하는 `Colossus`로 테스트했으므로, big rook이 피해를 받게 되면 그 경로도 집계 대상이다.

## 4. 아군 괴물의 남은 상호작용

`black-magic`의 아군 괴물은 이제 이동·포획한다(`5c8d9ed`). 확인하지 않은 것:

- 원본은 아군 괴물 포획 시 `poisoned-pawn` 대상이면 독 상태를 옮긴다
  (`main-RjrZ5fFP.js:210997`). 이 상호작용은 이식하지 않았다.
- 원본은 괴물 이동을 `v12NoteUltimatumMovement`에 기록한다
  (`main-RjrZ5fFP.js:211013`). `ultimatum`의 "움직이지 않은 기물" 판정에
  괴물 이동이 반영되는지는 확인하지 않았다.
- `dark-wizard`가 아군 괴물을 잡을 수 있는 예외(`main-RjrZ5fFP.js:205794`)는
  미이식 카드와의 상호작용이라 범위 밖이다.

## 5. 문서 수량 표기 (완료)

해소됐다. `remaining-card-ports.md`를 미지원 목록에서 **이식 현황**으로 다시 썼고,
`card-implementation.md`, `opening-card-ports.md`, `piece-card-ports.md`,
`rule-card-ports.md`, `rulesets/augment_chess_20261017/README.md`의 235/40 표기도
271/4로 맞췄다.

수량은 `test-latest-ruleset.mjs`가 `profile.pending.length`와
`implemented.cards.length`를 카탈로그와 대조하므로 불일치가 테스트에서 잡힌다.
분류별 표의 수치는 카탈로그 `phase` 기준으로 계산한 값이다
(OPENING 44, PIECE 43, MIDDLE 97, END 64, RULE 27).
