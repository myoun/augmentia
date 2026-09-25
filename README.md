# Augmentia

증강체스용 결정론적 게임 엔진과 self-play AI를 위한 초기 저장소입니다.

현재 범위는 `augment_chess_20260818`의 일반 모드 게임 코어입니다. 백·흑 순서의 게임
시작 드래프트, 양쪽 10수/20수 완료 시 드래프트, 다음 턴 카드 잠금, 기본 이동과
캐슬링·앙파상·프로모션, 행동 예산 기반 연속 행동, 강제 이동, 카드 후속 선택, 45수 연장전 및 별 판정을
실행합니다. 게임 시작 전 40% 오프닝 RULE 판정도 결정론적으로 실행합니다. 현재
스냅숏 카탈로그에 존재하는 192장 전체 카드를 100% 구현 완료했습니다. 여기에는 25개 RULE,
지속 상태 카드, 특수 기물 22종, 오프닝 카드 30장, 중반 카드 71장, 후반 카드 44장 및 메타 조작 카드가
포함되며, 결정론적 시드 기반 난수와 카드 액션이 모두 지원됩니다. 시작 드래프트에서 얻은 OPENING
카드는 자신의 첫 이동 직후 자동 발동합니다.

## 빠른 실행

```powershell
cargo run -q -p augment-cli --bin export_v3_fixtures
cargo test --workspace
cargo run -p augment-web
```

### V3 모델 계약 게이트

엔진(말 종류, 카드, phase, effect source)이나 V3 인코더 계약을 건드리는 변경은
커밋 전에 반드시 다음 게이트를 통과해야 합니다. 이 스크립트는 golden fixture와
runtime vocabulary를 현재 엔진 기준으로 재생성해 드리프트를 즉시 잡고, 전체 Rust
테스트와 전체 Python V3 테스트를 실행합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify-v3.ps1
```

새 카드·새 phase·새 특수 상태가 엔진에 추가되면 같은 PR에서 golden fixture case도
확장해야 합니다. `scripts/verify-v3.ps1`이 재생성 결과가 바뀌면 커밋을 막으므로,
fixture 갱신 없이는 룰셋 수정이 머지될 수 없습니다.

체크포인트 구조 변경(예: 말 종류 임베딩 확장)은 0 초기화 행 추가 방식의 마이그레이션으로만
처리하고, ONNX metadata(`piece_kind_vocabulary_size`, vocabulary 크기, schema ID)는
런타임이 fail-closed로 검증합니다. 축소·재배치 같은 비가역 변경은 금지이며 새
체크포인트로 시작합니다.

기록형 self-play는 clean worktree에서만 실행되며, 생성 중 오류 대국이 하나라도
나오면 즉시 실패합니다. 배포용 모델은 커밋 후 짧은 smoke self-play(수 시드, 소량
simulation)로 오류 0건을 확인한 뒤 사용합니다.

기본 실행은 `http://127.0.0.1:7878`의 로컬 웹 UI를 열어 백과 흑을 번갈아 조종하는
핫시트 게임을 시작합니다. 보드 클릭 이동, 프로모션, 드래프트, 카드 대상 지정, 기물
능력과 카드 후속 선택을 한 화면에서 처리합니다. `--seed 1234`를 전달하면 같은
드래프트와 확률 결과를 다시 재현할 수 있고, `--port 9000`으로 포트를 바꾸거나
`--no-open`으로 브라우저 자동 열기를 끌 수 있습니다.

웹은 매 상태에서 1번 V2 기준 모델과 2번 V3 모델을 같은 fingerprint·simulation 수로 병렬
탐색해 제안 하나씩을 나란히 보여줍니다. 어느 제안이든 사용자가 선택할 수 있고,
`1번 모델 자동 진행` 또는 `2번 모델 자동 진행`을 누르면 선택한 모델이 백·흑 양쪽을 계속
둡니다. 자동 진행 중에도 두 제안은 매 상태마다 함께 갱신됩니다. V2 ONNX 추론이 실패하면
휴리스틱으로 조용히 대체하지 않고 해당 제안을 오류로 표시합니다.

현재 smoke checkpoint를 기본 2번 모델 artifact로 내보내려면 다음을 실행합니다. export는
pre-tensor provenance를 유지하고, Rust가 생성한 현재 ruleset runtime vocabulary 계약에서
학습 vocabulary에 없던 token을 기존 index 뒤에 zero-initialized row로 추가합니다. 따라서 일반 체스에서 학습한 embedding은
그대로 유지하면서 드래프트·카드 상태도 fail-closed V3 런타임에 입력할 수 있습니다.
상황에 따라 달라지는 선택지 표시 문구는 AI vocabulary에서 제외하고 안정된 handler·option
식별자로 바꾸므로, 새로운 문구가 등장해도 `unknown token` 오류를 만들지 않습니다.

```powershell
cd train
uv run python -m v3.onnx_export `
  ..\runs\v3-smoke\lichess-board-pretrain-b1024-v1\checkpoints\step-000000000500.pt `
  ..\artifacts\models\v3-smoke-step500.onnx
cd ..
cargo run -p augment-web
```

기본 경로가 아닌 artifact를 비교하려면 `--model-1`, `--model-2`,
`--model-2-metadata`, `--ruleset-manifest`를 지정합니다. 현재 V3 smoke는 일반 체스
policy-only 모델이므로 이 화면은 통합·행동 비교용이며 카드 실력이나 value 품질을 증명하지
않습니다.

터미널 UI가 필요하면 `cargo run -p augment-cli`를 사용할 수 있습니다. 드래프트·카드·후속
선택은 화면에 표시된 번호로 고르고, 보드 이동은 `e2e4` 또는 `e2-e4`로 입력합니다.
전체 공개 카드 208장의 메타데이터를 정규화했고 모두 엔진 behavioral slice에
연결했습니다. 처음 남아 있던 0818의 네 근사 구현은 원본 알고리즘으로 교체했으며,
전체 상호작용의 parity 검증 상태는 최신 스냅숏의 `cards/implemented.json`에 기록합니다.

## 카드 스냅숏 갱신

사이트에서 발견한 JS 번들을 내려받은 뒤 전체 카드 카탈로그를 다시 생성할 수 있습니다.

```powershell
cargo run -p augment-snapshot -- extract `
  --bundle C:\path\to\main.js `
  --ruleset-id augment_chess_20260818 `
  --output rulesets\augment_chess_20260818\cards\catalog.json

cargo run -p augment-snapshot -- diff `
  --old rulesets\augment_chess_20260807\cards\catalog.json `
  --new rulesets\augment_chess_20260818\cards\catalog.json
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
docs/ai-v3-model.md                   첫 V3 신경망 구조·동적 policy·성능 기준
docs/ai-v3-hierarchical-sampled-search.md  계층형 표본 PUCT와 확률적 Draft 후속 설계
docs/ai-v4-factorized-policy.md       V4 factorized policy·조건부 visit target 계약
docs/ai-v4-implementation-status.md   V4 Phase 0/1 구현 현황·검증 결과·후속 계획
docs/ai-v4-phase1-status.md            opened-mask clean corpus·paired A/B·model gate 결과
docs/ai-v4-phase1-model-gate-diagnosis.md  model gate 실패 원인·metric bias·recovery 실험 순서
docs/ai-v4-phase1-recovery-p0-p2.md      P0–P2 shadow ablation·optimization sweep 결과
docs/ai-v4-p3-a-stage-diagnostic.md      30-game longer-horizon P3 A단계 결과
docs/ai-v4-p3-b300-gate-contract.md      300-game B단계 사전 동결 gate 계약
docs/ai-v4-p3-b300-gate-result.md        93-block B300 gate 결과 (evidence 충분, fail)
docs/ai-v4-legacy-artifact-quarantine.md  V2 target artifact 비파괴 격리 정책과 검증
docs/ai-v4-training-plan-0915.md       0915 native data 기반 실제 V4 학습 계획과 단계별 gate
rulesets/augment_chess_20260818/      최신 불변 규칙 스냅숏
rulesets/augment_chess_20260807/      이전 불변 규칙 스냅숏
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
