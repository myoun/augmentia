# Augmentia

증강체스용 결정론적 게임 엔진과 self-play AI를 위한 초기 저장소입니다.

현재 범위는 `augment_chess_20260807`의 일반 모드 게임 코어입니다. 백·흑 순서의 게임
시작 드래프트, 양쪽 10수/20수 완료 시 드래프트, 다음 턴 카드 잠금, 기본 이동과
캐슬링·앙파상·프로모션, 행동 예산 기반 연속 행동, 강제 이동, 카드 후속 선택, 45수 연장전 및 별 판정을
실행합니다. 게임 시작 전 40% 오프닝 RULE 판정도 결정론적으로 실행합니다. 현재
스냅숏 카탈로그에 존재하는 192장 전체 카드를 100% 구현 완료했습니다. 여기에는 25개 RULE,
지속 상태 카드, 특수 기물 22종, 오프닝 카드 30장, 중반 카드 71장, 후반 카드 44장 및 메타 조작 카드가
포함되며, 결정론적 시드 기반 난수와 카드 액션이 모두 지원됩니다. 시작 드래프트에서 얻은 OPENING
카드는 자신의 첫 이동 직후 자동 발동합니다.

## 빠른 실행

```powershell
cargo test --workspace
cargo run -p augment-web
```

기본 실행은 `http://127.0.0.1:7878`의 로컬 웹 UI를 열어 백과 흑을 번갈아 조종하는
핫시트 게임을 시작합니다. 보드 클릭 이동, 프로모션, 드래프트, 카드 대상 지정, 기물
능력과 카드 후속 선택을 한 화면에서 처리합니다. `--seed 1234`를 전달하면 같은
드래프트와 확률 결과를 다시 재현할 수 있고, `--port 9000`으로 포트를 바꾸거나
`--no-open`으로 브라우저 자동 열기를 끌 수 있습니다.

터미널 UI가 필요하면 `cargo run -p augment-cli`를 사용할 수 있습니다. 드래프트·카드·후속
선택은 화면에 표시된 번호로 고르고, 보드 이동은 `e2e4` 또는 `e2-e4`로 입력합니다.
전체 공개 카드 192장의 메타데이터는 정규화되어
있지만 실제 효과를 실행할 수 있는 카드는 현재 124장이므로, 드래프트도 이 124장 중
RULE을 제외한 카드를 사용합니다. 오프닝 RULE 선택 풀에는 현재 실행 가능한 `발판`과
`계시`가 들어갑니다.

## 카드 스냅숏 갱신

사이트에서 발견한 JS 번들을 내려받은 뒤 전체 카드 카탈로그를 다시 생성할 수 있습니다.

```powershell
cargo run -p augment-snapshot -- extract `
  --bundle C:\path\to\main.js `
  --ruleset-id augment_chess_20260807 `
  --output rulesets\augment_chess_20260807\cards\catalog.json

cargo run -p augment-snapshot -- diff `
  --old rulesets\augment_chess_20260807\cards\catalog.json `
  --new C:\path\to\new-catalog.json
```

추출기는 번들 SHA-256, 원시·공개 카드 수, 숨김 카드, ID·이름·카테고리·별·effect·target·
설명을 보존합니다.

## 저장소 구조

```text
crates/augment-core/                 결정론적 게임 엔진 코어
crates/augment-web/                  로컬 핫시트 웹 UI와 엔진 API 어댑터
crates/augment-cli/                  터미널 핫시트 UI
crates/augment-snapshot/             카드 카탈로그 추출·diff 도구
docs/architecture.md                 엔진, 규칙 버전, self-play 설계
docs/ai-v2-baseline.md               V3 전환을 위한 V2 AI 기준선과 보존 정책
docs/ai-v3-roadmap.md                AI V3 전환 1~14단계와 단계별 완료 조건
docs/observation-v3-contract.md       ObservationV3의 의미·공개 범위·안정 ID 계약
docs/action-v3-contract.md            ActionDescriptorV3와 가변 policy 계약
docs/v3-rust-python-contract.md       V3 Rust/Python 골든 fixture와 검증 계약
docs/data-v3-format.md                V3 대국·독립 decision·shard·manifest 포맷
docs/bootstrap-v3-data.md             PGN board-pretraining과 proven 전술 생성 계약
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
