# augment_chess_20260818 카드 구현 로드맵

초기 실행 가능 카드 17장과 0818 신규 16개 정의를 포함한 공개 208장 전체에 behavioral
slice를 연결했다. 특수 기물, 오프닝, 중반, 후반, RULE, 메타 카드가 결정론적 엔진에
포함된다. 사이트 parity 검증 상태와 알려진 근사 구현은 스냅숏 compatibility gap을
따른다. 현재 0818 스냅숏의 알려진 gap은 비어 있으며, 전수 조합 parity 검증은 별도다.
primitive와 의존성을 기준으로 나눈다. 특수 기물과 일반 효과를 먼저 만들고, 다른 카드
또는 RULE을 호출하는 메타 카드는 마지막에 구현한다. 각 파트 안의 세부 묶음은 작업
순서일 뿐 별도 완료·커밋 단위가 아니다.

## 공통 완료 조건

각 파트는 다음 조건을 모두 만족해야 완료로 본다.

1. 사이트 번들 symbol과 실제 적용 순서를 `cards/implemented.json`에 기록한다.
2. 대상 선택, 지속 턴, 무작위 소비 순서와 알려진 버그를 엔진에 반영한다.
3. 카드 단독·경계 조건·seed 재현·승패 동시 발생 회귀 테스트를 추가한다.
4. 터미널에서 새 기물과 지속 상태를 식별할 수 있게 표시한다.
5. `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
   `git diff --check`를 통과한다.
6. 파트 하나당 독립 Git 커밋 하나를 만든다.

| 파트 | 기존 세부 묶음 | 범위 | 카드 수 |
| --- | --- | --- | ---: |
| Part 01 | 01–03 | 특수 기물 | 20 |
| Part 02 | 04–06 | 오프닝 | 25 |
| Part 03 | 07–11 | 중반 | 62 |
| Part 04 | 12–14 | 후반 | 39 |
| Part 05 | 15–16 | RULE | 23 |
| Part 06 | 17 | 메타 카드 | 6 |
| 합계 | | | 175 |

## Part 01 — 특수 기물 (20장, 구현 완료)

### 01-A — 기본 합성 기물 I (7장)

- `amazon` — 아마존
- `constitutional-monarchy` — 입헌군주제
- `dragon` — 드래곤
- `eastern-policy` — 동방견문록
- `herald` — 전령
- `knightmaster` — 기사단장
- `pegasus` — 유니콘

기본 leaper/slider 조합과 기물 변환 레지스트리를 먼저 완성한다.

### 01-B — 고유 행동 기물 (6장)

- `hook` — 구행
- `idol` — 아이돌
- `jester` — 광대
- `local-conscription` — 현지군 징병
- `reaper` — 사신
- `wizard` — 마법사

고유 대상 선택, 소환, 마법처럼 별도 행동 공간이 필요한 기물을 구현한다.

### 01-C — 대량 교체·지원 기물 (7장)

- `eagle` — 알리바바
- `log` — 통나무
- `missionary` — 선교사
- `ordination` — 서품
- `reformation` — 종교 개혁
- `standard-bearer` — 기수
- `windmill` — 풍차

한 기물을 여러 기물로 바꾸거나 복수 기물을 동시에 변환하는 공통 경로를 만든다.

## Part 02 — 오프닝 (25장, 구현 완료)

### 02-A — 단일 특수 기물 배치 (7장)

- `apprentice-knights` — 견습 기사단
- `big-rook` — ROOK
- `checker` — 체커
- `elephant-escape` — 코끼리 탈출
- `guard` — 근위병
- `last-stand` — 집단 광기
- `queen-cavalry` — 퀸의 기병대

게임 시작 배치 변경과 종자·빅룩·체커·알필·근위병·광신도를 추가한다.

### 02-B — 전체 진영 재구성 (5장)

- `dutch` — 네덜란드
- `horde` — 호드
- `knightmate` — 나이트메이트
- `london-system` — 런던 시스템
- `summon-colossus` — 메가체스트론

여러 기물을 동시에 제거·재배치하는 오프닝 트랜잭션을 구현한다.

### 02-C — 지속·왕권 효과 (13장)

- `calling-card` — 예고장
- `democracy` — 민주주의
- `fast-growth` — 성급한 승진
- `fianchetto` — 피앙케토
- `holdout` — 존버
- `iron-monarch` — 철혈군주
- `loyalist` — 충신
- `martyrdom` — 순교
- `merchant-guild` — 상인 조합
- `otherworld` — 이세계
- `pawn-conversion` — 전환
- `queens-gambit` — 퀸즈 갬빗
- `qxe1` — Qxe1!!

대체 패배 조건, 로열 기물 변경, 보호, 지연 소환과 상인 경제를 구현한다.

## Part 03 — 중반 (62장, 구현 완료)

### 03-A — 이동 확장·추가 이동 (13장)

- `backward-knight` — 뒤로 가는 나이트
- `basic-training` — 기초 교습
- `breakthrough-order` — 돌파
- `corner-kick` — 코너킥
- `en-passant-bang` — 앙파상
- `file-surge` — 급수
- `horse-riding` — 승마
- `injury` — 부상
- `leap` — 도약
- `promotion-rush` — 승진 체험
- `radical-charge` — 난폭한 돌진
- `retreat` — 퇴각
- `rook-lift` — 룩 리프트

기존 이동 생성기에 합성 move provider와 추가 이동 trigger를 연결한다.

### 03-B — 제한·보호 상태 (13장)

- `armistice` — 휴전
- `encouragement` — 독려
- `evasion` — 회피
- `exhaustion` — 탈진
- `ghost` — 고스트
- `ice-sheet` — 빙판
- `inertia` — 관성
- `overwhelm` — 압도
- `royal-command` — 왕명
- `royal-shield` — 가호
- `severance` — 절단
- `socialism` — 사회주의
- `taunt` — 도발

긍정·부정 상태의 소유자, 만료 시점과 이동·포획 필터를 일반화한다.

### 03-C — 포획 반응·연쇄 효과 (13장)

- `alekhine-machine-gun` — 알레킨의 머신건
- `chameleon-mutation` — 카멜레온 변이
- `chimera` — 키메라
- `feudal-contract` — 봉건 계약
- `imperial-studies` — 제왕학
- `mad-horse` — 광마
- `mistake-card` — 실수
- `poisoned-pawn` — 독이 든 폰
- `queen-afterimage` — 잔상
- `sacrifice` — 희생
- `snipe` — 저격
- `suicide-bomber` — 자폭병
- `trojan-horse` — 트로이 목마

포획 전·후 hook, 반격, 폭발, 변신과 보호 연쇄의 처리 순서를 고정한다.

### 03-D — 다중 선택·예약 행동 (13장)

- `castling` — 캐슬링 (기본 캐슬링과 별개인 원거리 즉시 캐슬링 카드)
- `emergency-evacuation` — 긴급 피난
- `gale` — 강풍
- `panic` — 패닉
- `pawn-storm` — 폰 스톰
- `premove` — 프리 무브
- `quantum-mechanics` — 양자 역학
- `scarecrow` — 허수아비
- `schrodinger-pawns` — 슈뢰딩거의 폰
- `stealth` — 은신
- `submerge` — 잠복
- `substitution` — 치환
- `twins` — 환상의 콤비

복수 대상 선택, 예약된 다음 턴 행동과 중첩 위치 표현을 구현한다.

### 03-E — 중반 전역·고유 효과 (10장)

- `black-magic` — 흑마법
- `conversion` — 기도
- `fanatical-ritual` — 광신적 의식
- `mongolian-gambit` — 몽골리안 갬빗
- `random-roulette` — 랜덤룰렛
- `religious-victory` — 종교 승리
- `switcheroo` — 바꿔치기
- `trolley` — 트롤리
- `vortex` — 소용돌이
- `witch-trial` — 마녀재판

대량 변환·셔플·특수 승리 조건과 트롤리의 상대 선택 절차를 구현한다.

## Part 04 — 후반 (39장, 구현 완료)

### 04-A — 임시 변신·성장 (13장)

- `baby-bear` — 아기곰
- `bribe` — 삼일천하
- `charge` — 마지막 질주
- `coronation` — 대관식
- `frenzy` — 광란
- `hallucination` — 환각
- `last-resistance` — 마지막 저항
- `moving` — 무빙
- `necromancy` — 빙의
- `reverse-pawns` — 폰 방향 반전
- `stake` — 말뚝
- `underpromotion` — 언더프로모션
- `vip` — 귀빈

원형 기물 복원, 성장 타이머와 표시 전용 상태를 구현한다.

### 04-B — 재배치·보드 구조물 (13장)

- `barricade` — 엄폐물 설치
- `canceling` — 캔슬링
- `chain` — 사슬
- `conscription` — 징집
- `collapse` — 붕괴
- `desperado` — 데스페라도
- `exile` — 유배
- `homecoming` — 귀환
- `lobster` — 랍스터
- `palace` — 궁성
- `portal-gun` — 포탈 건
- `underground-bunker` — 지하벙커
- `zugzwang` — 추크츠방

기물 시작점 이력, 지형, 포탈과 HP를 포함하는 보드 객체 계층을 구현한다.

### 04-C — 승리·제거·전향 (13장)

- `bina-mate` — 더블 체크
- `blue-jeans` — 청바지
- `death-squad` — 결사대
- `final-weapon` — 최종병기
- `fleeting-dream` — 일장춘몽
- `icbm` — ICBM
- `othello` — 오델로
- `prophecy` — 종전
- `racing-king` — 레이싱 킹
- `spy` — 스파이
- `traitor` — 변절자
- `ultimatum` — 최후 통첩
- `vanish` — 소멸

직접 승리 조건, 소유권 변경, 추방·사망 이력과 매 턴 무작위 제거를 구현한다.

## Part 05 — RULE (23장, 구현 완료)

### 05-A — 정적 배치·지형 RULE (11장)

- `black-hole` — 블랙홀
- `chess-344200` — 344200 체스
- `chess-960` — 960 체스
- `crown` — 왕관
- `diagonal-chess` — 대각선 체스
- `football` — 축구공
- `high-ground` — 고지전
- `highway` — 고속도로
- `monochrome-chess` — 단색 체스
- `portal` — 포탈
- `rule-bombs` — 폭탄

드래프트 전 초기 배치, 지형 점유와 RULE 전용 기물·승리 조건을 구현한다.

### 05-B — 지속 동작 RULE (12장)

- `acceleration` — 가속
- `camouflage-color` — 위장색
- `conveyor` — 컨베이어
- `cool-guy` — 매너
- `macho-chess` — 상남자 모드
- `mistake` — 실수
- `monster` — 괴물
- `periodic-collapse` — 붕괴
- `recycling` — 재활용
- `saturation` — 포화
- `transcendence` — 초월
- `winter-kingdom` — 겨울 왕국

매 턴 RULE tick, 전역 이동·포획 제약과 주기적 보드 변형을 구현한다.

## Part 06 — 카드 메타 효과 (6장, 구현 완료)

- `black-box` — 검은 상자
- `white-box` — 하얀 상자
- `rule-ticket` — 규칙 티켓
- `insight` — 통찰
- `clone` — 복제
- `joker` — 조커

전체 카드 레지스트리, 긍정·부정 효과 분류와 사용 이력이 완성된 뒤 구현한다. 이 파트가
끝나면 manifest의 구현 수는 192장이 되지만, 사이트 parity 완료 표시는 전체 교차
검증을 별도로 통과한 후에만 올린다.

## 진행 순서와 Git 정책

Part 01부터 순서대로 진행한다. 각 파트의 커밋은 다음 파트가 의존하는 안정 지점이다.
카드 수를 채우기 위해 미완성 stub을 구현 완료 목록에 넣지 않는다. 사이트 번들 해석이
불명확한 카드는 해당 파트 안에서 보류 상태로 기록하고 다른 파트로 조용히 넘기지 않는다.
6개 파트 완료 후 전체 192장 조합 테스트와 사이트 대국 replay 검증을 별도 통합 단계로
수행한다.
