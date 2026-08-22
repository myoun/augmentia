# ActionDescriptorV3 계약

상태: Rust 구현 완료 (`2026-08-21`)

스키마 ID: `augmentia.action.v3`

구현: `crates/augment-ai/src/action_v3.rs`

ActionDescriptorV3는 V2의 고정 32,768칸 policy head와 action clamp를 대체한다.
정책 모델은 전역 action index를 예측하지 않고, 현재 상태에서 엔진이 반환한 합법
행동 descriptor 각각에 logit 하나를 출력한다.

## 1. 기본 원칙

- `Engine::legal_actions()`가 유일한 합법 행동 기준이다.
- descriptor 순서는 엔진의 현재 합법 행동 순서와 같다.
- 한 상태에서 서로 다른 합법 `Action`은 반드시 서로 다른 descriptor를 가진다.
- descriptor는 원래 `Action`으로 무손실 복원할 수 있어야 한다.
- raw `piece_id`, card `instance_id`, `choice_id` 숫자는 모델 입력에 노출하지 않는다.
- square는 ObservationV3와 동일하게 actor가 흑일 때 rank를 반전한다.
- 카드 definition, choice handler/option과 draft offer는 해시하지 않고 원문 ID를 쓴다.
- multi-square target은 길이와 순서를 모두 보존하며 잘라내거나 마지막 index로
  clamp하지 않는다.

## 2. Descriptor 종류

### Move

- ObservationV3와 같은 local piece slot;
- exact `PieceKind::stable_code()`;
- canonical `from`, `to` square;
- optional promotion stable code.

출발 square의 현재 기물 slot과 종류가 일치하지 않으면 stale descriptor로 거부한다.

### PieceAbility

- local piece slot, exact kind와 canonical current square;
- `WizardLightning`, `WizardShield`, `WizardMeteor`, `WizardTimeStop`,
  `ColossusSector`, `MerchantBuy` 중 정확한 ability;
- 구조화된 target.

### Card

- ObservationV3와 같은 local card slot;
- `(ruleset_id, definition_id)`의 exact definition ID;
- 구조화된 target.

같은 definition 카드가 여러 장이어도 local slot이 다르므로 충돌하지 않는다. 현재
card slot의 definition이 달라졌다면 stale descriptor로 거부한다.

### DraftPick

- `offer_index`;
- 해당 index의 exact `definition_id`.

현재 draft offer와 ID가 모두 일치해야 복원된다.

### ResolveChoice

- 현재 pending choice를 나타내는 local choice slot 0;
- exact `handler_id`;
- `option_index`와 exact option 문자열.

현재 choice payload와 하나라도 다르면 stale descriptor로 거부한다.

## 3. Target 종류

| 종류 | 보존 값 |
| --- | --- |
| `None` | payload 없음 |
| `Square` | canonical square 하나 |
| `PieceAt` | target 의미와 canonical square |
| `Squares` | 모든 canonical square의 순서 있는 가변 목록 |

`Square`와 `PieceAt`은 같은 칸이어도 다른 descriptor다. `Squares([])`, 한 칸, 두 칸,
세 칸 이상도 서로 다른 값이며 별도의 최대 길이를 두지 않는다.

## 4. Local reference

piece/card slot 생성은 ObservationV3의 `LocalIds`를 공유한다.

- board piece는 actor 관점 canonical square 순서로 slot을 배정한다.
- card는 self/opponent, exact definition, 별, charge, unlock 정보 순으로 배정한다.
- raw ID는 동일 entity를 join하는 데만 사용한다.
- 색을 교환하고 rank를 반전한 같은 의미의 상태·행동은 같은 descriptor가 된다.

descriptor는 상태 독립적인 영구 ID가 아니다. 반드시 자신을 생성한 상태와 함께
사용하며, state가 바뀌면 새 합법 행동 집합을 생성한다.

## 5. 가변 policy 인터페이스

`LegalActionSetV3::from_state(state, engine)`은 다음을 수행한다.

1. 엔진의 합법 행동 목록을 얻는다.
2. 각 행동을 descriptor로 변환한다.
3. 같은 descriptor가 두 번 나타나는지 검사한다.
4. 각 descriptor를 다시 `Action`으로 복원해 원본과 같은지 검사한다.

`policy_from_logits()`는 descriptor 수와 정확히 같은 길이의 finite logit만 받고 이
목록 위에서 softmax를 계산한다. 따라서 고정 action mask, 32,768 출력 head, overflow
index와 clamp가 없다. terminal 상태의 빈 합법 행동 목록에는 빈 policy를 반환한다.

## 6. 검증 게이트

- 모든 `Action` variant의 encode/decode 왕복;
- 네 가지 합법 promotion의 고유 descriptor;
- 빈 목록부터 세 칸 이상까지 multi-square target 비충돌;
- 208개 현재 카드가 생성하는 모든 합법 카드 행동의 고유 descriptor와 왕복;
- decode한 행동과 원본 행동의 `Engine::apply()` 전이 완전 일치;
- deterministic game prefix의 모든 상태에서 합법 행동 descriptor 고유성;
- 색 교환·rank mirror의 descriptor 대칭성;
- 오래된 card/draft/choice/piece descriptor 거부;
- policy logit 길이·finite 값·합 1 검증.

현재 V2 ONNX와 MCTS는 아직 `ActionIndex`와 고정 mask를 사용한다. V3 데이터·모델·탐색
경로가 준비될 때까지 V2 코드를 제거하지 않는다.
