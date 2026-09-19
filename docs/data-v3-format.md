# V3 대국·학습 데이터 포맷

## 결정

V3 데이터는 다음 두 record schema만 사용한다.

- `augmentia.replay.game.v3` (`GameRecordV3`): seed에서 시작하는 전체 대국 정본;
- `augmentia.training.decision.v3` (`DecisionRecordV3`): 원본 전체 대국이 없어도 독립적으로
  사용할 수 있는 한 결정 상태와 자동 생성 target.

`TrainingSelectionV3`는 만들지 않는다. 전체 대국의 오프닝 구간이나 종반 몇 수만 사용할
때는 loader가 `GameRecordV3.steps`를 조건으로 선택한다. 고정된 독립 전술이나 외부 기보
상태만 `DecisionRecordV3`로 물질화한다.

정본 shard는 한 줄에 record 하나인 compact UTF-8 JSONL을 Zstandard로 압축한
`*.jsonl.zst`다. 대형 shard와 manifest는 일반 Git에 넣지 않는다.

```text
replay-buffers/v3/<dataset-id>/
|-- manifest.json
`-- shards/
    |-- train/part-00000.jsonl.zst
    |-- validation/part-00000.jsonl.zst
    `-- holdout/part-00000.jsonl.zst
```

학습 속도를 위한 Arrow 또는 memory-mapped tensor는 8~9단계에서 정한다. 그것은 이
정본에서 다시 만들 수 있는 삭제 가능한 cache이며 데이터의 의미 기준이 아니다.

## GameRecordV3

한 record가 한 대국 전체다. 모든 record는 다음 provenance를 자체 포함한다.

- ruleset ID와 ruleset manifest SHA-256;
- observation/action schema ID;
- run ID, engine Git commit, engine API version, collector ID;
- 양측 agent ID와 선택적 model ID/SHA-256/config ID;
- mode와 64-bit seed.

dirty worktree에서 생성한 record는 재현 가능한 데이터로 인정하지 않고 validator가
거부한다. seed와 상태 fingerprint는 JSON 구현의 53-bit 정수 제한을 피하기 위해 16자리
lowercase hexadecimal string으로 저장한다.

각 `GameStepV3`는 다음을 저장한다.

- 모든 엔진 결정을 세는 0-based `decision_index`;
- 보드 진행을 나타내는 `engine_ply`;
- 절대 색 기준 actor와 actor 관점 `ObservationV3`;
- 상태 fingerprint와 적용 후 fingerprint;
- 순서가 의미의 일부인 전체 `LegalActionSetV3`;
- MCTS 방문 횟수, behavior distribution 또는 명시적인 policy target 없음;
- 실제 선택한 legal action index.

카드, 기물 능력, 드래프트, 강제 이동과 후속 선택은 보드 한 수 사이에도 발생하므로
`decision_index`와 `engine_ply`를 합치지 않는다. MCTS target은 정규화 확률이 아니라
원래 `u32` 방문 횟수를 저장한다. 그래야 합이 완료 simulation 수와 같은지 검증하고,
나중에 정규화나 temperature를 다시 적용할 수 있다.

progressive widening 또는 Gumbel 후보 제한을 사용한 step은 `sampled_mcts_visits` target을
사용한다. 이 target은 전체 합법 행동 폭의 visit과 함께 다음을 저장한다.

- 초기 `proposed_mask`와 최종 `opened_mask`;
- 행동별 proposal inclusion probability;
- 행동별 최초 open simulation;
- `search_semantics_id`와 `candidate_proposal_id`.

닫힌 행동은 visit 0의 부정 정답이 아니다. Python loader는 sampled target을 읽을 때 열린
행동만 후보 단위 학습 case로 만들고 그 안에서 visit distribution을 정규화한다. mask 길이,
양의 inclusion probability, 최초 open 시점 또는 닫힌 행동의 0 visit 계약이 맞지 않으면
record를 거부한다.

step마다 value label을 중복 저장하지 않는다. `engine_terminal` 종료 결과와 각 step의
actor로부터 `+1`, `0`, `-1`을 계산한다. 다음 종료 종류에는 value target이 없다.

- `limit`: decision, wall-clock 또는 resource 제한;
- `aborted`: 사용자나 운영상의 중단;
- `error`: 초기화, 관측, 탐색, 엔진 또는 직렬화 오류.

따라서 제한 초과나 실패한 대국을 무승부로 학습시킬 수 없다.

`trajectory_sha256`은 ruleset, mode, seed, fingerprint chain, 선택된 구조화 action과 종료
정보로 다시 계산한다. 중복 제거에는 관측, 합법 행동과 policy target까지 포함한 별도의
학습 내용 digest를 사용한다. 착수열은 같아도 서로 다른 탐색 방문 분포를 가진 대국은
유용한 별도 supervision이므로 보존한다.

## DecisionRecordV3

`DecisionRecordV3`는 사람이 직접 라벨링한다는 뜻이 아니다. 독립 ObservationV3,
LegalActionSetV3와 다음과 같은 출처가 명확한 자동 target을 저장한다.

- MCTS visits;
- behavior distribution;
- solver가 증명한 optimal action set;
- 정상 종료 대국에서 가져온 value;
- solver가 증명한 value;
- mate distance나 opening family 같은 typed auxiliary target.

모든 record에는 source kind/domain/ID와 `split_group_id`가 필요하다. policy, value,
auxiliary target 중 하나도 없으면 학습 record가 아니므로 거부한다. policy supervision은
`augment_policy` 또는 `board_pretraining` scope를 반드시 가진다. `standard_chess` 출처는
`board_pretraining` policy와 보조 과제만 사용할 수 있고 V3 value나 `augment_policy`를
가질 수 없다. 기보에서 실제로 둔 수는 `behavior_distribution`이며 자동으로 최선 수로
승격하지 않는다.

## Shard와 manifest

하나의 shard에는 하나의 record schema만 들어간다. writer는 대상 파일을 덮어쓰지 않고
먼저 `.partial`에 쓴 다음 Zstd stream을 정상 종료하고 파일을 동기화한 뒤 최종 이름으로
바꾼다. 기본 압축 level은 3이다. 운영 시 권장 회전 기준은 압축 전 512 MiB 또는 512개
record 중 먼저 도달한 조건이다.

`augmentia.dataset.manifest.v3` manifest에는 다음을 기록한다.

- dataset ID와 UTC 생성 시각;
- observation/action schema와 압축 설정;
- split algorithm과 salt;
- shard별 상대 경로, split, record schema, 압축 파일 SHA-256와 byte length;
- record/game/decision/game-step 수;
- policy/value 사용 가능 decision 수;
- 모든 shard 통계의 합계.

절대 경로와 `..`를 포함한 shard 경로는 거부한다. manifest validator는 실제 파일의
크기와 압축된 byte의 SHA-256을 확인하고, shard를 끝까지 읽어 내부 통계와 중복을 다시
검사한다.

## Split과 누수 방지

split은 `sha256_split_group_v1`을 사용하며 기본 비율은 train 9000, validation 500,
holdout 500 basis points다. 대국의 split group은 ruleset, mode와 seed로 결정한다. 같은
seed를 여러 agent나 모델에 재사용해도 같은 split으로 들어간다. 독립 decision은 원본
게임이나 문제 묶음을 나타내는 `split_group_id`를 명시한다.

한 게임의 step을 서로 다른 split으로 나누지 않는다. `GameRecordV3`에서 뽑은
`DecisionRecordV3`도 원본 게임과 같은 split group을 사용해야 한다.

## 검증 명령

```powershell
cargo run -p augment-cli --bin validate_v3_dataset -- `
  --manifest replay-buffers/v3/<dataset-id>/manifest.json
```

Rust validator는 다음을 거부한다.

- 알려지지 않은 record/schema/ruleset 연결;
- 잘못된 ObservationV3 shape, 범위, local slot과 runtime field ID;
- 중복되거나 관측과 맞지 않는 ActionDescriptorV3;
- 범위를 벗어난 played action;
- 합법 행동 수와 길이가 다른 policy target;
- simulation 합과 다른 방문 횟수 또는 잘못된 확률 분포;
- 끊어진 state fingerprint chain과 engine replay drift;
- 중복 record ID/학습 내용, 잘못된 split, checksum과 manifest 통계 불일치.

Rust `GameRecordV3::validate_replay`는 seed에서 엔진을 다시 시작해 매 decision의 actor,
engine ply, ObservationV3, 전체 합법 행동 목록, 적용 후 fingerprint와 최종 종료 의미가
동일한지 확인한다.

## 이 포맷을 선택한 이유

게임 단위 record는 결과를 한 번만 보관하여 중단 대국 오라벨링을 막고, seed와 행동열로
재현하며, split 누수를 방지한다. JSON은 현재 Rust serde 타입과 Python V3 계약을 그대로
표현하고 스키마 drift를 조사하기 쉽다. 반복되는 field name, 0이 많은 dense observation과
인접 상태의 유사성은 Zstd가 압축한다.

Parquet나 Arrow를 정본으로 사용하지 않는 이유는 재귀적인 runtime 값과 가변 action
descriptor를 모델별 tensor schema에 너무 일찍 고정하지 않기 위해서다. 데이터 규모가
커져 JSON parsing이 학습 병목이 되면 정본은 유지하고 모델 입력에 맞춘 binary cache만
교체한다.
