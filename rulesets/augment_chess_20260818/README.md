# augment_chess_20260818

2026-08-18 10:10 사이트 업데이트를 기준으로 고정한 규칙 스냅숏입니다.

`cards/catalog.json`은 원본 번들 `main-BeJ0OYoS.js`에서 직접 추출한 공개 카드 208개
(원시 정의 209개, 기본 숨김 `shotgun-king` 제외)를 보존합니다. 분포는 OPENING 31,
MIDDLE 75, PIECE 29, END 47, RULE 26입니다. 번들 길이와 SHA-256은
`sources/source-manifest.json`에 기록했습니다.

`cards/implemented.json`은 208개 카드 모두에 연결된 엔진 behavioral slice를 기록합니다.
0818 신규 카드 15종, `혼돈의 체스` RULE, 기존 카드의 행동 변경과 별점 조정을 포함합니다.
초기 이식에서 근사 처리했던 의심스러운 물약 28종, 트릭스터 43종, 리플레이 상태 복원,
`혼돈의 체스` 가치 DP·배치 알고리즘은 원본 번들 기준으로 보완했습니다. 현재
`compatibility_gaps`는 비어 있지만 전체 카드 조합의 전수 parity 검증은 아직 완료되지 않아
coverage 상태는 `partial`로 유지합니다.

일반 모드는 백·흑 순서의 OPENING 드래프트, 양쪽 10수의 MIDDLE 드래프트, 20수의 END
드래프트, MIDDLE/END 카드의 첫 턴 잠금, 앙파상, 반복 및 연장전 판정을 실행합니다.
카오스와 그랜드 모드의 드래프트 구조도 manifest에 고정되어 있습니다.

`notation.json`에는 사용자 제공 HTML에서 정규화한 기물 기보 코드 52개가 들어 있습니다.
원본은 `sources/raw/notation-legend.html`에 보존하며, 신규 0818 기물의 별도 기보 표기가
사이트 범례에 추가되기 전까지 기존 코드 집합을 유지합니다.

원본 기준:

- https://augmentchess.org/
- https://augmentchess.org/rules/
- https://augmentchess.org/updates/
