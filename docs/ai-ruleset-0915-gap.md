# 2026-09-15 규칙 격차 보고서 (읽기 전용 분석 결과)

출처: https://augmentchess.org/updates/ (2026.09.01/09.08/09.15),
라이브 번들 https://augmentchess.org/assets/main-6qeAcxbX.js.
현재 동결: 208 visible cards, 0 site-parity verified.

## 1. 신규 카드/RULE (전부 cards.rs와 frozen catalog에 없음 확인)

- 09.08 (15): majesty(위엄), resolve(결의), nullification(상쇄), infiltration(잠입),
  outpost(전초기지), hedgehog(고슴도치), princess(프린세스), big-bishop(BISHOP),
  vanguard(선봉), killer-king(킬러 킹), campfire(캠프파이어), reversal(반전),
  recurrence(회귀), miracle(기적), overtake(추월)
- 09.08 RULE: capture-the-flag(깃발 뽑기) — 홈 랭크에 랜덤 플래그 칸,
  적이 진입해 1턴 데드라인까지 유지하면 패배
- 09.15 (16): long-en-passant(롱파상), disassembly(분해), proficiency(숙련),
  mutation(변이), thief(도적), locust-swarm(메뚜기떼), false-start(부정출발),
  paladin(팔라딘), symmetry(대칭), metal(메탈), octopus(문어), parrot(앵무새),
  clockwork(태엽인형), extinction(멸종), brutus(브루투스), highlander(하이랜더)

합계: 카드 31 + RULE 1. 추가 게임플레이 규칙: 드래프트 카드 밴 필터,
카오스 번들 OPENING 1개 제한(카오스 모드 자체가 미구현 — Blocked).

## 2. 기존 핸들러 드리프트 (crates/augment-core/src 기준)

- 신규 변형 기물 캡처 금지: mark_fresh_no_capture(:3960)가 shared_turn_count>=2에서만
  lock 설치. 초기 턴에도 lock 필요. (캡처하는 것이 금지, 잡히는 것은 아님)
- queens-gambit: 두 번째 보호 파일 c/e 제외(engine.rs:604는 d만 제외).
  비승격 변신 시 보호 해제 필요(mark_transformed_piece:630, clear_piece_kind_runtime:4516).
  승격 예외는 유지. middle.rs/end.rs 정리 경로 포함.
- iron-monarch(친정): :3574 모든 직접 캡처에 왕 extra action → 캡처한 폰/광신도만.
  cards.rs 표시명 친정으로. 테스트 :11630 확장.
- lobster: 즉시 소환(engine/end.rs:275) → 1수 지연. pending 상태 필요.
- undead: engine.rs:714 due_ply=ply+9 → 3턴. 턴 규약 유의.
- fianchetto: engine.rs:7057 이동 기물 종류 필터 없음 → 폰만 차단으로.
- slime: 특수 이동 금지. Dragon 스왑 engine.rs:6158, middle.rs:1529 등 감사.
- cleanup: remove_piece_for_card가 캡처 반응/마나/사신 호출(engine.rs:677-696).
  독립 삭제 판정으로 분리. 09.15 최대 3개 제한(engine.rs:1877 validator).
- substitution: MIDDLE → END phase 변경.
- horse-riding: engine.rs:1483 단순 토글 → 사용 시 아군 나이트 전부 삭제.
- grasshopper: engine.rs:5717 own Rook → own minor.
- stealth: engine.rs:7834 왕실만 캡처 가능 + 통과/깔아뭉개기 불가 →
  상대 턴에 보드 부재 취급. occupancy view 필요(state.rs+middle.rs).
- black-magic 리메이크: middle.rs:541 대량 희생+왕 변신 → 킹 옆 아군 괴물 소환,
  END, 4성. blackTowerLegacy는 별도 유지.
- zugzwang: engine/end.rs:313 플래그만 토글, cards.rs ends_turn=false → 턴 소모.
- portal-gun: pending 포탈 이동 차단 없음. state.rs에 pending 라이프사이클 자체가 없음.
- missionary: 비숍 1 → 선교사 2.
- checker: engine.rs:338,:1104 자신 폰 → 적 폰.
- king-of-the-hill: engine.rs:6095 퀸 → 아마존 이동.
- macho-chess RULE: rule.rs:408 플래그만 있고 승리 훅 없음 → 반대 끝 랭크 도달 승리.

## 3. 별 드리프트 (half-star, cards.rs 기준)

horse-riding 8→5; zugzwang 7→6; portal-gun 7→8; grasshopper 6→7; stealth 6→8;
democracy 7→8; scarecrow 5→7; royal-shield 4→6; missionary 5→6; checker 3→4;
black-magic 9→8. 09.08 신규는 09.15 밸런스 값으로 추가.
라이브 번에는 미발표 override 존재(Highlander 5.0 등). 발표분과 라이브 감사 분리.

## 4. 구현 표면

- 전 카드: cards.rs 상수+IMPLEMENTED_CARDS, lib.rs export, ruleset.rs 로더/커버리지.
- 신규 기물: model.rs 안정 코드(번호 추가만, 재번호 금지), engine 변신/이동/가치,
  state.rs 런타임/직렬화, rule.rs 랜덤 카탈로그.
- Opening: princess/big-bishop/vanguard/thief/locust-swarm/false-start 디스패처.
- 새 state 필드는 fingerprint 직렬화(state.rs)와 롤백 정리(engine/end.rs) 필수.
- 핀 테스트: cards.rs implemented_ids_and_effects_are_unique는 implemented.json을
  쓰고 구 ruleset hash 하드코딩(2105-2112) — 순수화 필요.
  compiled_cards_match_the_snapshot_catalog, ruleset.rs manifest 테스트,
  every-card smoke 테스트 갱신 필요.

## 5. 스냅샷 추출기

augment-snapshot extract_catalog는 최상위 `{` 리터럴만 파싱.
라이브 CARD_DEFS는 spread + .map(internalBalanceCard) 구조라 32개 전부 누락.
선언적 해석(spread/factory/override/동적 category) + fail-closed 필요.

## 6. ruleset/AI 파장

- 0818은 불변 유지, 새 스냅샷 augment_chess_20260915 + 원본 바이트 해시.
- bundled_20260807은 0818 alias일 뿐임 — 버전 지원 방식을 명시적으로 결정.
- manifest 바이트 변경이 곧 identity 변경. (ruleset_id, manifest_sha256)로 검증하는
  runtime_v3/v4, selfplay/arena CLI 정렬 필요.
- golden: cargo run -p augment-cli --bin export_v3_fixtures 후 byte 비교 테스트.
- 어휘: encoder_v3가 IMPLEMENTED_CARDS 전체를 요구 → 신규 32 ID 전부 어휘 재생성.
  정렬 삽입으로 인덱스 이동 → metadata-only 패치 불가, 재export 필수.
- V4: export_fixture_bundle.py/_Identity와 integration 테스트가 0818+zero hash 하드코딩.
  재생성 순서: Rust fixture/어휘 → uv run python -m v4.export_fixture_bundle → parity 재검증.


## 7. 0915 구현 감사 부록 (진행 중)

- 구현 기준은 **발표된 2026.09.15**이다. 현재 라이브 전체를 소급 적용하지 않는다.
- `artifacts/ruleset-0915-card-definitions.json`은 신규 32개 정의의
  `source_definition`, `announced`, `effective_live`를 분리한다.
- 원본 JS 12,097,393 bytes의 SHA256:
  `60d5816e78896d4b5371c0d5cc37ef0a302aa0cc814c3b027be3936124878dfc`.
  원본과 발표 HTML은 새 스냅샷 `sources/raw/`에 보존한다.
- source base 정의의 princess=PIECE, vanguard=MIDDLE은 발표의 OPENING 변경을
  적용하기 전 값이다. 발표에서는 둘 다 OPENING이다. vanguard는 유일한 선두 폰만
  발동하고 앞으로 캡처할 수 없다. recurrence는 무작위 비폰 대상, miracle는 선택한
  비숍 희생, overtake는 active로 적용한다.
- Highlander의 발표에 없는 live 5성 override는 적용하지 않고 base 4.5성을 유지한다.
  역사적 0915 배포 JS를 확보하지 못했으므로 신규 카드 초기값은 현재 base 정의와
  명시적 발표 delta로 재구성했다. 완전한 역사적 site parity를 주장하지 않는다.
- 별도 라이브 감사: `rulesets/augment_chess_20260915/sources/live-unannounced-audit.json`.
  이 자료는 선언적 metadata 차이만 다루며 전체 UI mutation/gameplay 검증이 아니다.
- 현재 0915 스냅샷은 전체 240개 catalog 항목을 behavioral slice로 등록했다.
  `manifest.json`과 `cards/implemented.json`의 coverage는 240으로 맞춘다.
  이는 전체 site parity 완료를 뜻하지 않으며 `site_parity_verified_card_count`는 0이다.
  cross-card 상호작용과 live 동작의 완전 검증은 계속 별도 과제로 남는다.
- 기존 `augment_chess_20260818`의 7개 파일은 baseline SHA256으로 불변 검증한다.
  구 생성자는 유지하지만 버전별 옛 engine behavior/replay 호환을 새로 주장하지 않는다.
- E 단계 관측/어휘/번들 재생성은 별도 승인 전까지 보류한다.
