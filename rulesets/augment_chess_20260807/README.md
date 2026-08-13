# augment_chess_20260807

2026-08-07 사이트 업데이트를 기준으로 만드는 최초의 규칙 스냅숏입니다.

현재 `manifest.json`은 공개된 기본 규칙과 세 가지 대전 모드의 드래프트 구조만
고정합니다. 카드 192개의 설명과 효과는 아직 정규화하지 않았으므로 coverage가
`scaffold`로 표시됩니다. 이 디렉터리를 완성된 규칙 구현으로 간주하면 안 됩니다.

`notation.json`에는 사용자 제공 HTML에서 정규화한 기물 기보 코드 52개가 들어
있습니다. 원본은 `sources/raw/notation-legend.html`에 그대로 보존했으며, 사이트
번들/API와 대조하기 전까지 provenance의 verification은
`unverified_against_site_bundle`입니다.

원본 기준:

- https://augmentchess.org/rules/
- https://augmentchess.org/updates/
- https://augmentchess.org/ 내 게임 카드 설명
