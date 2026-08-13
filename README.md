# Augmentia

증강체스용 결정론적 게임 엔진과 self-play AI를 위한 초기 저장소입니다.

현재 범위는 `augment_chess_20260807`의 일반 모드 게임 코어입니다. 백·흑 순서의 게임
시작 드래프트, 양쪽 10수/20수 완료 시 드래프트, 다음 턴 카드 잠금, 기본 이동과
앙파상, 45수 연장전 및 별 판정을 실행합니다. 첫 behavioral slice로 언덕의 왕,
제네바 협약, 패링, 빈 찬합, 발판을 구현했습니다. 카오스·그랜드 실행과 나머지 187개
카드 및 전체 상호작용은 아직 구현되지 않았습니다.

## 빠른 실행

```powershell
cargo test --workspace
cargo run -p augment-cli
```

CLI 예제는 양쪽의 첫 드래프트에서 첫 후보를 선택한 다음 `e2-e4`를 적용합니다. 현재
정규화된 카드가 5장뿐이라 실제 사이트의 192장 후보 분포는 아직 재현하지 않습니다.

## 저장소 구조

```text
crates/augment-core/                 결정론적 게임 엔진 코어
crates/augment-cli/                  엔진 연동 예제
docs/architecture.md                 엔진, 규칙 버전, self-play 설계
rulesets/augment_chess_20260807/      불변 규칙 스냅숏
```

기보 코드도 규칙 데이터입니다. 현재 스냅숏은 사용자 제공 원본 HTML과 52개 코드의
정규화 카탈로그를 함께 보존하며, 한 글자와 두 글자 코드가 겹칠 때 longest-prefix로
해석합니다.

Git은 누적 개발 이력을, ruleset 디렉터리와 annotated tag는 불변 게임 버전을
관리합니다. 날짜별 장기 브랜치는 만들지 않습니다. 자세한 태그·revision·웹게임 버그
호환 정책은 [docs/versioning.md](docs/versioning.md)를 참고하세요.

규칙 스냅숏은 한번 공개되면 수정하지 않습니다. 같은 날짜에 정정이 필요하면 날짜를
바꾸거나 명시적인 revision을 추가하고, 기존 대국과 학습 산출물은 원래 ID를 계속
참조해야 합니다.
