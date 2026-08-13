# 증강체스 엔진 및 self-play 아키텍처

## 1. 목표와 현재 범위

이 프로젝트의 첫 번째 기준 버전은 `augment_chess_20260807`이다. 날짜는 규칙의
출처가 된 사이트 업데이트 날짜를 뜻하며, 엔진 릴리스 날짜나 모델 학습 날짜와는
독립적이다.

현재 구현은 다음 세 가지를 검증하는 스켈레톤이다.

1. 규칙 스냅숏을 대국 상태와 분리하고 매 대국에 고정할 수 있다.
2. 같은 상태와 행동열은 플랫폼과 실행 시점에 관계없이 같은 결과를 낸다.
3. 이동, 카드 사용, 드래프트, 후속 선택을 하나의 명령 경계로 확장할 수 있다.

192개 카드, 모든 특수 기물, 캠페인 규칙을 구현 완료한 버전은 아니다. 지원 범위는
각 스냅숏의 `manifest.json`과 테스트가 사실의 기준이다.

## 2. 버전 모델

서로 다른 세 버전을 한 문자열에 섞지 않는다.

| 값 | 예 | 의미 |
| --- | --- | --- |
| `ruleset_id` | `augment_chess_20260807` | 카드, 모드, 밸런스와 판정 규칙 |
| `schema_version` | `1` | ruleset 파일 형식 |
| `engine_api_version` | `1` | 상태·행동·관측 프로토콜 호환성 |

대국 기보, replay buffer, 모델 checkpoint, 평가 결과는 최소한 `ruleset_id`, 엔진
commit, seed를 기록한다. 스냅숏은 공개 후 수정하지 않고 새 ID로만 갱신한다. 향후
manifest에 정규화된 전체 카드 데이터가 들어오면 SHA-256 digest도 기록한다.

검증 완료 시 annotated Git tag `ruleset/augment_chess_YYYYMMDD-rN`으로 commit을
고정한다. 같은 날짜 자료의 해석 정정은 revision을 올리고, 사이트 동작 변경은 새 날짜
스냅숏을 만든다. 세부 release gate는 `docs/versioning.md`를 따른다.

권장 업데이트 흐름은 다음과 같다.

```text
사이트 원본 보존 -> 정규화 -> 이전 스냅숏과 diff -> 사람의 규칙 검토
               -> golden scenario 추가 -> 불변 스냅숏 공개 -> self-play 재평가
```

사이트 규칙 페이지는 실제 카드 효과가 계속 조정될 수 있고 게임 내 설명이 최신
기준이라고 밝힌다. 따라서 업데이트 로그만 수집해서는 부족하며, 카드 설명·RULE·
모드 설정의 원본도 함께 보존해야 한다.

## 3. 엔진 경계

`augment-core`는 UI, 네트워크, 벽시계, 학습 프레임워크를 모른다. 입력은 완전한
`GameState`와 `Action`, 출력은 다음 `GameState`와 발생한 `Event`뿐이다.

```text
Ruleset + seed
      |
      v
 GameState -- legal_actions() --> ordered Action list
      |                               |
      +--------- apply(action) <------+
                      |
                      v
             Transition { state, events }
```

상태에는 보드뿐 아니라 다음 값도 포함해야 한다.

- 적용 중인 `ruleset_id`와 모드
- 수순, 차례, 승패 상태
- 양측 카드 인스턴스와 별 합계
- 드래프트 또는 카드의 미완료 선택
- 엔진 소유의 결정론적 난수 상태
- 반복 판정용 position history

현재 스켈레톤은 표준 8x8 기물 이동, 프로모션, 킹 포획 승리, 움직일 수 없을 때 패배,
3회 동형 시 별 합계 판정을 구현한다. 캐슬링, 앙파상, 체크 안전성, 특수 기물, 실제
카드 효과, 드래프트 진행은 후속 구현 범위다.

### 기보 카탈로그

기물 기보 코드 역시 ruleset에 속한다. `notation.json`은 코드와 표시 이름을 보존하고
원본 HTML은 별도 source artifact로 둔다. 현재 범례에는 `M`과 `MI`, `S`와 `SK`처럼
prefix가 겹치는 가변 길이 코드가 있으므로 한 글자 enum이나 고정 폭 파서를 사용하지
않는다. 파서는 대문자 prefix 중 가장 긴 일치를 고르고, 저장된 기보는 대국 당시의
`ruleset_id`로 해당 카탈로그를 선택해야 한다.

새 스냅숏에서 코드가 바뀌더라도 과거 기보를 재작성하지 않는다. 이름 변경, 코드
재할당, 추가·삭제는 notation diff로 보고하고 과거 카탈로그를 계속 유지한다. 현재
`20260807` 데이터는 2026-08-13에 사용자 제공 HTML에서 정규화했으며 사이트 번들/API
대조가 끝나기 전까지 `unverified_against_site_bundle` 상태다.

## 4. 카드와 RULE 구현 방식

모든 카드를 범용 DSL 하나로 표현하면 복잡한 상호작용이 DSL 인터프리터 안으로
숨고 디버깅이 어려워진다. 반대로 카드마다 엔진을 직접 수정하면 업데이트 diff와
테스트가 불안정해진다. 따라서 혼합 방식을 사용한다.

- 이름, 단계, 별, 대상 조건, 수치 파라미터: ruleset 데이터
- 단순 spawn/replace/status 효과: 검증된 공통 effect opcode
- 고유한 카드: Rust의 결정론적 event handler
- 카드 ID를 handler ID에 연결하는 표: 해당 ruleset 스냅숏

핸들러는 직접 임의의 시스템 상태를 만지지 않고 `EffectCommand`를 반환한다. 엔진이
명령을 순서대로 reduce하므로 effect 순서와 replay 결과를 테스트할 수 있다. 실제
카드를 추가할 때는 최소한 다음 golden scenario를 둔다.

- 카드 단독 발동
- 대상이 없거나 사라졌을 때
- 양측 카드가 같은 이벤트에 반응할 때의 순서
- 킹 포획, 승패 변경과 동시에 발동할 때
- 저장한 기보를 replay했을 때 최종 fingerprint 일치

## 5. 모드 모델링

일반, 카오스, 그랜드는 별도 엔진이 아니라 `ModeConfig`로 시작한다. 번들 manifest는
현재 공개 규칙에 맞춰 일반 모드의 세 번 단일 카드 선택, 카오스의 두 장 묶음 선택,
그랜드의 28장 공용 풀과 각 6장 선택 구조를 보존한다. 모드가 행동 순서를 바꾸면
`GamePhase`와 `PendingChoice`가 이를 표현한다.

캠페인은 보드 크기나 승리 조건까지 바꿀 수 있으므로, 8x8 가정이 실제 장애가 되는
첫 캠페인을 구현할 때 별도 scenario/config 계층을 추가한다. 지금 미리 동적 보드로
일반 경로의 성능을 희생하지 않는다.

## 6. Self-play 인터페이스

초기 학습기는 Python으로 두되 simulation hot path는 Rust에 둔다. 첫 연동은 PyO3
batch API가 적합하다.

```text
reset_batch(ruleset_id, mode, seeds) -> observation batch
legal_actions(batch)                 -> 가변 길이 action 목록과 mask
step_batch(action_indices)           -> observation, reward, terminal
```

카드와 대상 선택 때문에 고정된 체스 move policy head만으로는 부족하다. 우선은
결정론적으로 정렬된 합법 행동을 토큰화하고, 모델이 상태별 후보 행동을 점수화하는
방식을 권장한다. 데이터가 쌓인 뒤 자주 쓰는 이동·카드·대상 공간을 분리한 head로
최적화한다.

확률 효과는 외부에서 샘플링하지 않는다. 모든 chance 결과는 `GameState`의 RNG에서
나오고 이벤트 로그에 결과가 남아야 한다. 그래야 MCTS, replay, 버그 재현이 같은
게임을 바라본다.

## 7. 구현 순서

1. 기본 이동과 판정의 perft/golden test 완성
2. 카드 백과 원본 수집기와 snapshot diff 도구
3. 공통 effect opcode 및 대표 카드군 구현
4. 일반 모드 드래프트와 카드 행동/후속 선택
5. 카오스·그랜드 및 RULE modifier
6. 사이트 기보와 엔진 replay의 교차 검증
7. Python batch binding, baseline bot, MCTS self-play
8. 전체 카드 coverage gate와 새 날짜 스냅숏 자동 생성
