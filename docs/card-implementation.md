# 카드 구현 위치와 추가 방법

최신 실행 프로필의 지원 카드 235장은 `crates/augment-core/src/engine/cards/`에 둔다.
파일명은 공식 ID의 `-`를 `_`로 바꾼 것이다. 예를 들어 `pawn-chain`은 `pawn_chain.rs`다.
카탈로그·지원 목록·원본 계약은 계속 `rulesets/augment_chess_20261017/cards/`에서 관리한다.
현재 미지원 40장의 최신 효과·변경 항목·이식 확인사항은 [남은 카드 목록](remaining-card-ports.md)을 참고한다.

## 파일의 책임

| 위치 | 책임 |
|---|---|
| `card.rs` | 공통 정의 타입과 룰셋별 정의 조회 |
| `card_ids.rs` | 버전과 실행 코드에서 공유하는 Rust 카드 ID 상수 |
| `latest.rs` | 최신 카탈로그와 실행 프로필에서 카드 정의 구성 |
| `compat/card_definitions.rs` | 과거 룰셋의 240장 메타데이터 |
| `engine/cards/<card_id>.rs` | 카드별 획득 조건, 획득 효과, 사용 효과, 대상 열거, 전용 상태 타입과 테스트 |
| `engine/cards/mod.rs` | 최신 지원 카드의 단일 등록부와 실행 연결 |
| `engine/cards/lifecycle.rs` | 여러 카드에 걸친 정산 순서: 턴 시작·종료, 체크 목표 등 |
| `engine/cards/shared.rs` | 공통 배치 가능 칸, 착지 처리, 실제 왕실 공격 검사와 이동 제약의 조합 |
| `engine/cards/opening_support.rs` | 최신 오프닝 기물 변환의 상태·포획 지연 처리 |
| `engine/cards/transform_support.rs` | 최신 기물 카드의 공통 대상 검사와 ID·일반 상태를 보존하는 변환 |
| `engine/cards/eligibility_support.rs` | 획득 조건의 공통 판정 도구 |
| `engine/cards/runtime.rs` | 카드 상태의 집계 저장소 |
| `engine/cards/rule_support.rs`, `chaos_pieces.rs` | 최신 RULE 예약·정산 순서와 카오스 기물 의존성 |
| `engine/cards/tests/` | 여러 카드의 상호작용, 원본 fixture 대조와 공용 테스트 보드 |
| `engine/compat/` | 과거 룰셋의 카드 분기와 설치 처리 |
| `card_eligibility/compat.rs` | 과거 룰셋의 획득 조건 |
| `engine/piece_effects.rs`, `turn_effects.rs`, `board_rules.rs` | 여러 카드가 공유하는 기물·턴·보드 처리 |
| `engine.rs` | 공통 행동 검증, 이동·잡기, 이벤트·추가 행동·턴 진행과 전이의 원자성 |

카드 전용 보조 처리도 해당 카드 파일에 둔다. 예를 들어 `seal.rs`에는 속박 예약·해제,
`ascension.rs`에는 복귀 예약과 상태 복원, `penta_check.rs`에는 체크 횟수 갱신과 승리 판정이 있다.
공통 이동·잡기 처리와 여러 효과를 결합하는 순서는 엔진과 공용 모듈이 소유한다.
`lifecycle.rs`의 호출 순서는 승리 우선순위·예약 소진·이벤트 순서에 영향을 주므로 유지해야 한다.

최신 정의의 이름·등급·활성화·턴 종료 정책은 해당 룰셋의 `catalog.json`에서 읽는다.
지원 여부와 `turn_exclusive`는 같은 룰셋의 `implemented.json`에서 읽는다.
`turn_exclusive`가 빠진 프로필은 로드에 실패한다. 과거 정의의 값으로 보충하지 않는다.
기존 crate 루트의 공개 타입·상수·조회 함수는 유지하며, `implemented_card()`와
`IMPLEMENTED_CARDS`는 과거 정의 조회다. 최신을 포함한 버전별 조회에는 `ruleset_card()`를 사용한다.

현재 상태 저장 필드 `card_runtime.october`와 `OctoberRuntimeState`의 이름·필드 순서는 유지했다.
기존 `position_key`가 이 구조의 Debug 표현을 해시하기 때문이다. 구조 정리 때문에 반복 판정과
탐색 상태 식별자가 달라지지 않도록 저장 계약은 그대로 두고 정의 위치만 옮겼다.

## 카드를 추가하거나 수정할 때

1. 공식 원본과 지원 범위를 확인하고 카탈로그·`implemented.json`·`ported.json`을 갱신한다.
2. `card_ids.rs`에 ID를 한 번 정의하고 카드 파일의 `ID`에서 해당 상수를 참조한다.
   필요한 함수를 작성한다. 획득 가능 여부와 실제 사용 가능 여부는 별도다.
3. `BEHAVIOR`를 다음 중 하나로 정의한다. 모든 생성자는 획득 조건 함수를 필수로 받는다.

   ```rust
   CardBehavior::active(ID, play, actions, eligibility)
   CardBehavior::passive(ID, Some(acquire), eligibility)
   CardBehavior::passive(ID, None, eligibility) // 카드 보유 자체를 읽는 효과
   CardBehavior::rule(ID, install, eligibility)
   ```

4. `mod.rs`의 모듈 선언과 ID 정렬된 `CARDS` 목록에 등록한다. 최신 카드가 미등록 상태로
   과거 handler에 빠지는 경로는 없다. 지원 목록과 등록부의 일치, 중복, 순서, 활성화 종류를 테스트한다.
5. 카드 전용 테스트는 카드 파일의 `#[cfg(test)]` 모듈에 둔다. 색상·대상·RNG 호출 수·턴 경계와
   실패 시 원본 상태 보존을 확인한다. 다른 카드와의 정산 순서는 공용 상호작용 테스트로 검증한다.
6. 공통 생명주기 처리가 필요하면 해당 카드에 처리 함수를 두고 `lifecycle.rs`나 기존 정산 지점에
   순서를 명시한다. 날짜별 이식 묶음을 새 실행 모듈로 만들지 않는다.

## 구조 변경의 회귀 기준

`tests/card_contracts.rs`는 리팩터링 당시 128장 각각을 공개 Engine API로 두 보드·두 색상에서 실행한다.
획득 조건, 획득 결과, 전체 합법 행동 목록, 이벤트, 상태 fingerprint와 대표 이동·카드 사용 전이를
정리 전 커밋 `c63c68a`에서 수집한 `tests/fixtures/card-contracts.json`과 비교한다.
2026-10-29 기물 카드 이식에서는 무덤이 생성되기 전의 포획도 기억하도록 전역 이력을 추가했다.
원본 `main-RjrZ5fFP.js:221498–221528`의 기억 계약에 따른 의도적인 상태 식별 변경이다.
기존 512개 사례를 유지하며 fingerprint 138개만 갱신했다. 행동·이벤트·나머지 값은 모두 이전 기준과 같다.
이력과 포획된 풍차 모드는 반복 키에 포함되며 별도 기물 카드 테스트로 검증한다.
일반 테스트 실행은 이 기준 파일을 수정하지 않는다. 의도적인 규칙 변경 때만 원본 계약에 근거해 갱신한다.

전체 검증은 루트에서 다음 명령으로 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify-search.ps1 -SkipBenchmarks
```
