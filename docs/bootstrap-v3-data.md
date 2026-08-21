# V3 부트스트랩 데이터 생성

## 목표와 범위

7단계는 약한 휴리스틱이나 V2 출력을 정답으로 복사하지 않는다. 첫 V3 모델이 사용할 수
있는 데이터는 다음 두 종류로 제한한다.

- 일반 체스 PGN에서 가져온 `board_pretraining` policy;
- Augmentia 엔진이 제한 깊이 안에서 강제승리를 증명한 policy/value.

모델의 랜덤 초기화는 8단계에서 구현한다. 7단계 생성기는 모델 가중치나 ONNX를 요구하지
않는다. 인터넷에서 PGN을 자동 다운로드하지 않으며 사용자가 지정한 로컬 파일만 읽는다.

## 일반 체스 PGN

`bootstrap_v3_data`는 일반 `.pgn`과 `.pgn.zst`를 mainline 단위로 streaming parsing한다.
압축 파일 전체를 메모리나 별도 평문 파일로 풀지 않는다. 각 착수 전 위치에서
일반 체스와 현재 Augmentia 엔진 양쪽에 존재하는 board move만 후보로 남기고, 실제 PGN
착수를 one-hot behavior distribution으로 기록한다.

이 record는 항상 다음 제한을 가진다.

- `source_domain = standard_chess`;
- `policy.scope = board_pretraining`;
- value target 없음;
- 카드, runtime, 드래프트와 증강체스 승패 학습에 사용 금지.

PGN의 결과, 체크메이트와 value는 복사하지 않는다. 일반 체스의 체크·승패 규칙과
Augmentia의 킹 포획 규칙이 다르기 때문이다. en passant와 castling을 포함해 선택한 착수가
두 도메인의 후보 행동 교집합에 없으면 해당 위치는 건너뛴다. 원본 파일 SHA-256, 게임과
ply 위치, 사용자가 지정한 license 문자열을 provenance에 남긴다.

같은 위치·같은 착수를 단순 중복 제거하지 않는다. 반복 one-hot sample의 cross-entropy
합은 착수 횟수를 합산한 behavior distribution과 같으므로 인간 착수 빈도를 보존하며,
각 sample을 원래 대국 단위 split에 남겨 provenance와 split 경계를 유지한다. PGN 위치에는
강제승리 solver를 실행하지 않는다. 일반 체스 전체 corpus를 Augmentia 전술 탐색에 넣는
비용과 도메인 혼합을 피하고, solver는 아래 tactical input에만 적용한다.

Lichess 입력에는 다음 선택 필터를 조합한다.

- `--min-rating`: 양쪽 플레이어가 모두 만족해야 하는 최소 레이팅;
- `--speeds`: `bullet,blitz,rapid,classical,correspondence` 등의 허용 속도 목록;
- `--rated-only`: PGN Event가 rated game인 대국만 허용;
- `--normal-termination-only`: time forfeit와 비정상 종료 제외;
- `--min-game-plies`: 너무 짧은 대국 제외.

`--max-games`는 필터를 통과한 대국 수이고 `--max-records`는 실제 출력 sample 수다.
출력은 메모리에 모두 쌓지 않고 split별 `--shard-records`개마다 새 shard로 기록한다.

## 증명된 Augmentia 전술

`augment-ai::bootstrap_v3::solve_forced_win_v3`는 `LegalActionSetV3`와 실제
`Engine::apply`만 사용한다. material score, neural value, V2 `ActionIndex`를 사용하지 않는다.

현재 actor 차례에는 하나 이상의 승리 자식이 있으면 되고, 상대 차례에는 모든 합법 응수가
승리로 이어져야만 강제승리로 인정한다. 가장 짧게 이기는 root action들을
`optimal_action_set`으로 저장한다. 제한 깊이나 node budget 안에서 증명하지 못한 상태는
unknown이며 학습 record가 되지 않는다. unknown을 무승부나 패배로 바꾸지 않는다.

board-only tactical input v1은 작은 수작업·회귀 fixture용이다. 카드와 runtime 상태를
표현하지 않으며 coverage report에 이 한계를 명시한다. 이후 카드 전술은 정상적인
`GameRecordV3` replay에서 상태를 가져와 같은 solver로 검증하는 방식으로 확장한다.

## 실행

저장소 루트에서 다음처럼 실행한다.

```powershell
$commit = git rev-parse HEAD
cargo run -p augment-cli --bin bootstrap_v3_data -- `
  --pgn C:\data\lichess_db_standard_rated_2026-07.pgn.zst `
  --expected-pgn-sha256 <official-64-hex-digest> `
  --expected-pgn-bytes <official-byte-count> `
  --out replay-buffers\v3\lichess-pilot `
  --dataset-id lichess-standard-pilot-v1 `
  --created-at-utc 2026-08-21T00:00:00Z `
  --engine-commit $commit `
  --pgn-license CC0-1.0 `
  --min-rating 2000 `
  --speeds rapid,classical `
  --rated-only `
  --normal-termination-only `
  --min-game-plies 20 `
  --max-games 1000 `
  --max-records 100000 `
  --shard-records 10000
```

`--created-at-utc`와 `--engine-commit`은 필수다. `--pgn`을 사용하면 `--pgn-license`,
`--tactics`를 사용하면 `--tactics-license`도 각각 필수다. 전술 입력은 별도 실행에서
`--tactics`, `--tactics-license`, `--tactical-depth`, `--tactical-nodes`를 지정한다.
서로 다른 출처에 하나의 license를
잘못 붙이지 않는다. 출력 디렉터리에 기존 manifest가 있으면 덮어쓰지 않는다.
production corpus에는 `--expected-pgn-sha256`와 `--expected-pgn-bytes`를 함께 지정해,
다운로드가 완전하고 고정한 공식 source와 동일한지 생성기가 직접 확인하게 한다.

PGN importer는 GPL-3.0-or-later인 `pgn-reader 0.28`에 의존한다. 이 CLI를 바이너리로
배포할 계획이라면 배포 전에 프로젝트의 라이선스 정책과 의무를 별도로 검토한다. 이 문구는
법률 판단을 대신하지 않는다.

생성 결과:

```text
replay-buffers/v3/bootstrap-pilot/
|-- manifest.json
|-- coverage.json
`-- shards/<split>/decision-NNNNN.jsonl.zst
```

`manifest.json`과 shard는 6단계 validator를 통과한 뒤에만 완료된다. `coverage.json`은 다음을
보고한다.

- 읽은/허용/필터된 게임, 필터 사유, 위치와 건너뛴 위치 수;
- board-pretraining과 proven tactical record 수;
- 증명 실패와 node budget 초과 수;
- action/phase/piece/card coverage;
- 입력 PGN과 tactical JSON SHA-256;
- 입력/압축 출력 byte, shard 수, 생성 시간과 records/sec;
- 현재 입력 포맷의 알려진 한계.

## 대량 생성 전 gate

처음에는 작은 pilot만 만든다. 다음 조건을 검토하기 전 대규모 PGN을 처리하지 않는다.

- 모든 shard checksum과 schema 검증 통과;
- standard-chess record에 V3 value 또는 `augment_policy`가 0개;
- proven tactical record의 engine solver 재현;
- 중복률과 split 누수 없음;
- action/phase/card coverage의 빈 영역이 명시적으로 보고됨;
- 예상 압축 크기와 positions/sec 기록.

현재 fixture는 파이프라인 회귀 검증용이며 학습 규모의 corpus가 아니다. 실제 corpus의 크기,
출처와 license는 별도 pilot 실행 결과와 함께 기록한다.

## Lichess 실데이터 pilot

2026-08-21에 [Lichess Open Database](https://database.lichess.org/)의
`lichess_db_standard_rated_2026-07.pgn.zst` 앞 128 MiB를 HTTP range로 받아 실제 경로를
검증했다. 부분 파일 SHA-256은
`18ae821fcf490743e04e8b68628e934de271778989562986a4162a10166750b3`이다. 위 필터와
`max_records=50000`, `shard_records=10000`을 사용한 결과는 다음과 같다.

- 102,063 games scanned, 649 accepted, 50,000 board-pretraining records;
- 7 shards, 17,026,083 compressed shard bytes, 모든 manifest/checksum/schema 검증 통과;
- warm release와 로컬 cache 기준 생성 14.9초(3,351 records/sec), 전체 validation 24.4초;
- value, proven tactic, card coverage는 모두 0으로 명시됨;
- 같은 비율이면 5백만 position은 압축 약 1.7 GB이며 생성 약 25분으로 추정됨. 전체 파일
  SHA-256 scan과 실제 저장 장치 속도는 별도 비용이다.

이는 처리량과 데이터 계약을 검증한 pilot이지 학습 corpus가 아니다. 부분 원본과 생성물은
재생성 가능한 대용량 산출물이므로 Git에 넣지 않고 측정 후 삭제한다. 본 corpus를 만들 때는
부분 파일이 아니라 Lichess가 제공한 전체 파일과 공식 SHA-256을 사용한다.

## 첫 production board-pretraining corpus

첫 corpus는 디스크가 큰 standard-rated 월별 덤프보다 Lichess 공식 broadcast 대국을 먼저
사용한다. broadcast는 일반 사용자 대국보다 강한 대국의 비중이 높고, 2026-07 파일 하나가
24,535,400 bytes라서 원본과 V3 shard를 함께 보관해도 로컬 디스크를 압박하지 않는다.
Lichess가 공개한 2026-07 원본 정보는 다음과 같이 고정한다.

- URL: `https://database.lichess.org/broadcast/lichess_db_broadcast_2026-07.pgn.zst`
- SHA-256: `714d0eb99f99fca8d791142038b6c59b5ca6a51b3339bd3891a92f4bdffcbf0c`
- license: `CC-BY-SA-4.0`
- published games: 40,038

이 corpus도 일반 체스의 최종 결과나 Stockfish eval을 value target으로 사용하지 않고,
mainline 착수의 `board_pretraining` policy만 사용한다. 두 선수 레이팅이 모두 2000 이상인
대국을 production 기본값으로 삼고, 표본 수와 필터 탈락률은 생성된 `coverage.json`에서
확인한 뒤 추가 월 또는 rating threshold를 결정한다. CC0인 standard-rated corpus와
라이선스가 다르므로 같은 source license로 합치지 않는다.

2026-08-21 첫 전체 변환에서는 40,038대국 중 12,123대국이 필터를 통과해 1,096,662개의
policy position이 생성됐다. 46개 shard의 압축 크기는 372,549,083 bytes였고 manifest
SHA-256은 `e5d87260944424e2875639ae9278d1f9d56f0272eeb5aa9215c8dd8cdb4778c1`이다. value,
card와 proven-tactical target은 모두 0이었으며 독립 Rust engine replay validator를
통과했다. 이 결과는 목표 5백만 board position의 약 22%이므로, 나머지는 추가 broadcast
월을 동일 조건으로 변환해 채운다.

단, 추가 월 생성은 trainer loader의 bounded-memory 전환 뒤에 진행한다. 현재
`ReplayDatasetV3`는 모든 JSON record를 Python 객체로 materialize한다. 이 corpus의 train
split을 읽는 smoke에서 프로세스 working set이 37.1 GiB까지 증가했는데도 초기화가 끝나지
않아 중단했다. 5백만 position으로 그대로 확대하면 학습 가능한 메모리 범위를 벗어난다.
shard-aware deterministic sampling과 제한된 shard cache를 구현하고, 같은 corpus에서 loader
peak memory와 첫 batch를 통과시키는 것을 다음 data gate로 삼는다.
