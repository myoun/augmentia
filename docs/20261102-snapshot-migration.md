# 2026-11-02 스냅샷 이관 기록

2026-10-17(`augment_chess_20261017`)에서 2026-11-02(`augment_chess_20261102`)로
엔진 대상을 옮긴 작업의 기록이다. 카드별 이식 판단과 근거를 남긴다.

## 결과

| 항목 | 값 |
|---|---|
| 카탈로그 전체 | 281장 |
| 원본 활성 | 272장 |
| 실행 지원 | **272장** (활성 기준 100%) |
| 원본 비활성 | 9장 (실행 거부) |
| 소스 이식 기록 | 189장 (`cards/ported.json`) |
| 불변 계약 상속 | 83장 |

`+ 9 = 282`가 아니라 `margin`이 없는 이유: 비활성 9장은 지원 목록에 넣지 않고
`implemented.json`의 `pending`에만 남긴다. `지원 + pending = 281`이 성립한다.

즉 `272 + 9 = 281`이고, 이 9장은 카탈로그에는 있지만 원본에서 비활성이므로 실행
대상이 아니다.

## 정책 변경 7장 (ACTIVE → PASSIVE)

2026-10-27 밸런스 패스로 대상 선택형에서 **획득 시 무작위 변환**으로 바뀌었다.
`OCTOBER27_RANDOM_TRANSFORMS`(main 번들 1768)와 획득 리졸버(41613-41630)가 근거다.

| 카드 | 원본 기물 | 비고 |
|---|---|---|
| `barrier` | rook | |
| `berserker` | rook | |
| `killer` | bishop | |
| `siren` | rook | 원본이 queen에서 rook으로 변경됨 |
| `undead` | queen | |
| `hedgehog` | queen | 변환 시 반격 예산 2로 초기화 |
| `chameleon-mutation` | — | 최대 3개, 중복 없이 추첨. MIDDLE → OPENING |

공통 헬퍼는 `engine/cards/transform_support.rs`의 `acquire_random_transform`이다.
후보가 1개면 난수를 소비하지 않고, 0개면 Draft 단계에서는 조용히 넘어간다.

## 계약 변경 카드

| 카드 | 변경 내용 | 근거 |
|---|---|---|
| `overwhelm` | 킹만 차단 → **모든 마이너 기물** 차단, MIDDLE → OPENING | 6040-6045 |
| `ascension` | 3수 → **7수**, 착지 칸이 비어 있지 않아도 됨, MIDDLE → END | 49123-49125, 49208 |
| `death-squad` | 파일 전체 폰 → **선택한 폰 최대 3개** | 43459-43467 |
| `trojan-horse` | 나이트 1개 → **나이트 + 폰**, 폰을 소모, 턴 종료 | 47063-47082 |
| `summon-colossus` | 재활성. 이름 거신병 소환 → **메가체스트론**. 효과는 동일 | 5052-5101 |

## 신규 카드 6장

| 카드 | 동작 | 근거 |
|---|---|---|
| `citadel` | 홈랭크 킹사이드 비숍 → 룩 | 24364-24372 |
| `offering` | 획득 시 무작위 룩 → 제물 | 1768, 41613-41630 |
| `wanderer` | 획득 시 무작위 룩 → 방랑자 | 1768, 41613-41630 |
| `antichess` | 3턴 연속 손실 시 승리 | 29572-29593 |
| `bombardment` | 파일 지정 → 2수 뒤 붕괴 | 49154-49161, 58154 |
| `omega-chesstron` | 전 기물 희생 → 7수 뒤 3×3 소환 | 24374-24380, 452-475 |

`omega-chesstron`을 위해 `PieceKind`에 `Offering`, `Wanderer`, `OmegaChesstron`을
추가했다. 오메가는 3×3 대형 기물이라 랭크/파일 대시 이동을 별도로 생성한다.

## 원본 비활성 9장

`beacon`, `blue-jeans`, `chess-n-pow-30`, `democracy`, `martyrdom`,
`monochrome-chess`, `qxe1`, `royal-shield`, `underground-bunker`.

구현은 과거 룰셋을 위해 남겨두고 `#[allow(dead_code)]`로 표시한다. 현재 프로필은
이들을 거부한다.

## `draw_requirement`만 바뀐 카드

`clockwork`, `judgment`, `octopus`, `retreat`, `siege-ram`, `socialism`,
`transmission-tower` 7장은 이 필드만 변경됐다. 이 엔진은 `draw_requirement`를
**읽지 않는다**. `LatestCard` 구조체에도 없다. 드래프트 풀 구성용 메타데이터이므로
재이식 대상에서 제외하고 그대로 승계했다.

## 테스트 기준선 재설정

`crates/augment-core/tests/fixtures/card-contracts.json`은 fingerprint를 저장하는데,
fingerprint에 룰셋 ID가 포함된다. 엔진 대상을 바꾸면 전부 달라진다. 재설정 전에
동작 변화를 분리 검증했다.

- 결과 이벤트 목록 2,023개가 바이트 단위로 동일
- 유일한 동작 변화는 의도한 `ascension`의 7수 비행

`crates/augment-core/tests/fixtures/october-objectives.json`은 `model.rs`에서
`PieceKind`를 파싱해 마스크를 만든다. 기물 3종 추가로 76 → 79개가 됐고, 같은
1017 번들로 재생성했다.

## 재현 스크립트

`artifacts/updates-1102/`에 있다.

| 스크립트 | 역할 |
|---|---|
| `build-profile.mjs` | 1017 프로필을 1102로 승계. 기존 1102 기록은 보존 |
| `backfill-ports.mjs` | 이 브랜치에서 직접 이식한 카드의 출처 기록 추가 |
| `sync-registry-1102.mjs` | 프로필에서 `CARDS` 레지스트리를 유도 |
| `build-delta.mjs` | 두 카탈로그에서 `sources/0915-delta.json` 생성 |

`build-profile.mjs`는 `ported.json`을 1017 기록만으로 재작성하던 버그가 있었다.
1102 소스에 직접 대고 이식한 기록이 조용히 사라지므로, 기존 1102 기록을 먼저
읽어 보존하도록 고쳤다. 같은 이유로 `summon-colossus` 같은 재활성 카드가
`wasSupported` 검사에 다시 걸리던 문제도 함께 고쳤다.

## 남은 작업

없다. 원본 활성 카드는 전부 실행된다.

다만 **사이트 전체 동작 일치**는 주장하지 않는다. `manifest.coverage.site_parity_verified_card_count`
가 0이고 `status`는 `partial`이다. 카드별 상호작용·원본 동작 일치 검증은 별도
작업이며, 상세 조사 항목은 [후속 조사 항목](remaining-card-ports-followup.md)에 있다.
