# V3 후보 평가와 승격 gate

13단계는 `python -m v3.promotion`과 Rust `arena_v3`로 구현한다. 이 단계는 candidate를
champion registry에 쓰지 않는다. 모든 필수 gate를 통과한 immutable 보고서만 14단계의
승격 입력이 될 수 있으며, 보고서에는 항상 `registry_mutated: false`가 기록된다.

평가 순서는 다음과 같다.

1. candidate, champion, V2 baseline, ruleset과 V3 schema/hash 검사;
2. engine-only solver가 증명한 봉인 전술의 top-1 policy와 optimal probability mass;
3. 봉인 terminal/proven value의 RMSE, MAE, Brier score와 ECE;
4. phase, ActionV3 종류, 51개 기물 종류, 147개 runtime 필드와 구현 카드 coverage;
5. candidate/champion 동일 simulation 수 arena;
6. candidate/champion 동일 move-time arena;
7. candidate/V2 baseline의 동일 simulation 및 동일 move-time arena.

어느 한 항목이라도 실패하면 `promotion_eligible`은 false다. 오류·누락도 실패로 기록되며
다른 gate의 성공으로 상쇄되지 않는다.

## 봉인 데이터

전술, calibration과 coverage dataset은 `validation` 또는 `holdout` split만 허용한다.
설정에 manifest SHA-256을 직접 넣고, 실행 시 다음을 모두 다시 확인한다.

- manifest와 shard SHA-256 및 크기;
- Rust V3 schema와 전체 engine replay;
- split policy에 따른 실제 record 위치;
- candidate와 동일한 ruleset ID 및 ruleset manifest SHA-256.

전술 gate는 `optimal_action_set`과 `proven` value를 함께 가진 decision만 사용한다. 휴리스틱
라벨이나 일반 체스 결과는 전술 정답이 아니다. calibration은 proven decision 또는 엔진이
정상 종료한 game만 사용하며 제한·오류 종료를 포함하지 않는다.

## coverage 계약

현재 `augment_chess_20260807` ruleset은 공개 192장 중 124장을 구현한다. 평가 설정은 ruleset
manifest의 `implemented_card_count`보다 낮은 카드 threshold를 사용할 수 없다. 또한 다음
전체 계약을 요구한다.

- 51개 PieceKind code;
- `playing`, `forced_move`, `draft`, `awaiting_choice` phase;
- `move`, `piece_ability`, `card`, `draft_pick`, `resolve_choice` action;
- ObservationV3 runtime field ID 1~147.

구현되지 않은 68장을 AI coverage로 꾸며 채우지는 않는다. 이 영역은 엔진 ruleset 확장
문제이며, 현재 모델의 보고서에는 124/192 상태가 그대로 남는다.

## arena 의미

각 seed는 candidate가 백인 게임과 흑인 게임을 한 쌍으로 실행한다. root noise와 temperature는
끄고 최대 방문 수 행동을 선택한다. 제한 종료와 검색 오류는 무승부로 세지 않으며 terminal
rate를 낮춰 gate를 실패시킨다.

동일 simulation arena는 모든 검색의 실제 visit 합이 설정값과 정확히 같은지 확인한다.
동일 시간 arena는 속도를 미리 재서 simulation 수로 환산하지 않고 양쪽 MCTS에 같은 실제
deadline을 준다. ONNX 한 번이나 leaf batch가 deadline을 조금 넘길 수 있으므로 실제 검색
시간과 simulations/search도 보고한다. V2가 ONNX 오류를 heuristic으로 fallback하면 해당
게임은 오류로 처리한다.

승·무·패는 candidate 관점에서 1, 0.5, 0점이다. 95% Wilson interval, 백·흑 score 차이,
paired seed 완전성, terminal rate와 candidate 카드 분포의 색상별 total-variation distance를
함께 기록한다.

## smoke와 정식 평가

`evaluation_mode`는 `smoke` 또는 `promotion`이다. smoke는 연결 검사용으로 모든 개별 수치를
통과하더라도 `promotion_scale_budget` gate가 항상 실패한다. 따라서 적은 게임으로 만든
보고서를 승격에 사용할 수 없다.

정식 promotion mode는 설정을 다음보다 느슨하게 만들 수 없다.

| 항목 | 최소/최대 계약 |
|---|---:|
| 전술 위치 | 최소 100 |
| 전술 top-1 | 최소 90% |
| optimal probability mass | 최소 70% |
| calibration 위치 | 최소 1,000 |
| value RMSE | 최대 0.75 |
| value ECE | 최대 0.15 |
| arena seed pair | 최소 100 |
| simulations/search | 최소 150 |
| move time | 최소 500 ms |
| terminal game | 최소 100 |
| terminal rate | 최소 80% |
| champion/V2 score | 최소 55% |
| score 95% LCB | 최소 50% |
| 백·흑 score 차이 | 최대 0.10 |
| 카드 색상 TV | 최대 0.20 |

필요하면 이보다 엄격하게 설정할 수 있지만 낮은 예산이나 느슨한 threshold는 config 검증에서
거부한다.

## 실행

설정에는 다음 top-level 필드가 필요하다.

```json
{
  "promotion_config_schema_id": "augmentia.promotion-config.v3",
  "evaluation_id": "candidate-20260822-001",
  "created_at_utc": "2026-08-22T00:00:00Z",
  "evaluation_mode": "promotion",
  "ruleset_id": "augment_chess_20260807",
  "ruleset_manifest": "rulesets/augment_chess_20260807/manifest.json",
  "ruleset_manifest_sha256": "<64 hex>",
  "candidate": {"onnx": "candidate.onnx", "metadata": "candidate.metadata.json"},
  "champion": {"kind": "v3", "onnx": "champion.onnx", "metadata": "champion.metadata.json"},
  "v2_baseline": {"onnx": "augment_net_v2_canonical_baseline.onnx", "sha256": "53d3c81cb94f6ae277164d219f8bdae944f2a963fdf60cb09409e9ccf5a4933c"},
  "tactical": {"manifest": "tactics/manifest.json", "manifest_sha256": "<64 hex>", "split": "holdout"},
  "calibration": {"manifest": "calibration/manifest.json", "manifest_sha256": "<64 hex>", "split": "holdout"},
  "coverage": {"datasets": [{"manifest": "coverage/manifest.json", "manifest_sha256": "<64 hex>", "split": "holdout"}]},
  "arena": {
    "pairs": 100,
    "base_seed": 2026082201,
    "max_decisions": 512,
    "simulations": 400,
    "move_time_ms": 1000,
    "max_simulations": 100000,
    "leaf_batch_size": 16
  },
  "thresholds": "위 표보다 같거나 엄격한 전체 threshold 객체"
}
```

실행 명령:

```powershell
cd train
uv run python -m v3.promotion path\to\promotion.json path\to\report.json
```

모든 gate가 통과하면 exit code 0, 하나라도 실패하면 보고서를 원자적으로 쓴 뒤 exit code 2를
반환한다.

## 구현 smoke 결과

2026-08-21 smoke는 랜덤 V3 candidate, 동일 V3 상대와 보존된 277 MB V2 baseline을 대상으로
두 arena 조건을 모두 실제 실행했다. candidate/ruleset/V2 hash와 봉인 holdout replay 검사는
통과했다. 전술 한 위치는 통과했지만 value RMSE는 약 1.0이었고, coverage는 3개 위치에
불과했으며, 2 decision 제한 arena에는 terminal game이 없었다. 보고서는 정상 생성되고 exit
code 2 및 `promotion_eligible: false`를 반환했다. 이는 강한 모델의 성능 주장이 아니라
불충분한 후보가 승격되지 않는다는 경로 검증이다.
