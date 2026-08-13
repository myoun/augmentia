# augment_chess_20260807

2026-08-07 사이트 업데이트를 기준으로 만드는 최초의 규칙 스냅숏입니다.

현재 `manifest.json`은 공개된 기본 규칙과 세 가지 대전 모드의 드래프트 구조를
고정합니다. 192장 중 언덕의 왕, 제네바 협약, 패링, 빈 찬합, 발판의 behavioral
slice를 구현했으며 전체 특수 기물 및 카드 간 상호작용의 사이트 parity 검증은 아직
완료되지 않았습니다. 이 디렉터리를 완성된 규칙 구현으로 간주하면 안 됩니다.

엔진 실행 범위는 일반 모드뿐입니다. 백·흑 순서의 OPENING 드래프트와 양쪽 10수의
MIDDLE, 20수의 END 드래프트, MIDDLE/END 카드의 첫 턴 잠금, 앙파상, 45수부터 시작하는
10수 연장전 판정을 구현했습니다. 카오스와 그랜드 설정은 향후 호환성을 위해
manifest에 보존하지만 현재 `new_game`에서는 명시적으로 거부합니다.

`cards/implemented.json`은 다섯 카드의 번들 effect, 관련 hook, 재현해야 하는 quirk와
현재 검증 범위를 기록합니다.

`notation.json`에는 사용자 제공 HTML에서 정규화한 기물 기보 코드 52개가 들어
있습니다. 원본은 `sources/raw/notation-legend.html`에 그대로 보존했으며, 사이트
번들/API와 대조하기 전까지 provenance의 verification은
`unverified_against_site_bundle`입니다.

원본 기준:

- https://augmentchess.org/rules/
- https://augmentchess.org/updates/
- https://augmentchess.org/ 내 게임 카드 설명
