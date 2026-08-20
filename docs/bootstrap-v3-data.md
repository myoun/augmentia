# V3 부트스트랩 데이터 생성

## 목표와 범위

7단계는 약한 휴리스틱이나 V2 출력을 정답으로 복사하지 않는다. 첫 V3 모델이 사용할 수
있는 데이터는 다음 두 종류로 제한한다.

- 일반 체스 PGN에서 가져온 `board_pretraining` policy;
- Augmentia 엔진이 제한 깊이 안에서 강제승리를 증명한 policy/value.

모델의 랜덤 초기화는 8단계에서 구현한다. 7단계 생성기는 모델 가중치나 ONNX를 요구하지
않는다. 인터넷에서 PGN을 자동 다운로드하지 않으며 사용자가 지정한 로컬 파일만 읽는다.

## 일반 체스 PGN

`bootstrap_v3_data`는 PGN을 mainline 단위로 streaming parsing한다. 각 착수 전 위치에서
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
  --pgn C:\data\games.pgn `
  --tactics train\v3\fixtures\bootstrap_tactics_v1.json `
  --out replay-buffers\v3\bootstrap-pilot `
  --dataset-id bootstrap-pilot-v1 `
  --created-at-utc 2026-08-21T00:00:00Z `
  --engine-commit $commit `
  --pgn-license "<PGN source license>" `
  --tactics-license "<tactical source license>" `
  --max-games 1000 `
  --max-records 100000 `
  --tactical-depth 3 `
  --tactical-nodes 100000
```

`--created-at-utc`와 `--engine-commit`은 필수다. `--pgn`을 사용하면 `--pgn-license`,
`--tactics`를 사용하면 `--tactics-license`도 각각 필수다. 서로 다른 출처에 하나의 license를
잘못 붙이지 않는다. 출력 디렉터리에 기존 manifest가 있으면 덮어쓰지 않는다.

PGN importer는 GPL-3.0-or-later인 `pgn-reader 0.28`에 의존한다. 이 CLI를 바이너리로
배포할 계획이라면 배포 전에 프로젝트의 라이선스 정책과 의무를 별도로 검토한다. 이 문구는
법률 판단을 대신하지 않는다.

생성 결과:

```text
replay-buffers/v3/bootstrap-pilot/
|-- manifest.json
|-- coverage.json
`-- shards/<split>/decision-00000.jsonl.zst
```

`manifest.json`과 shard는 6단계 validator를 통과한 뒤에만 완료된다. `coverage.json`은 다음을
보고한다.

- 읽은 게임, 위치, 잘못된 게임, 건너뛴 위치와 제거한 중복 수;
- board-pretraining과 proven tactical record 수;
- 증명 실패와 node budget 초과 수;
- action/phase/piece/card coverage;
- 입력 PGN과 tactical JSON SHA-256;
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
