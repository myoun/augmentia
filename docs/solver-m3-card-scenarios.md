# V3 Solver M3 카드 전술 생성기

상태: 구현 완료

## 목적

실제 증강체스 self-play의 보드·카드·runtime 상태를 출발점으로 사용하되, 행동자와 상대에게
카드를 추가한 변형을 만들어 카드 때문에 생기거나 깨지는 얕은 강제승을 수집한다. 라벨은
휴리스틱이 아니라 엔진의 전체 합법수와 AND-OR 강제승 탐색 결과만 사용한다.

## 후보 계약

- 원본 self-play manifest를 먼저 전체 replay 검증한다.
- `PhaseV3::Playing` decision만 카드 합성 후보로 사용한다.
- 원본의 실제 `played_action_index` prefix를 재생하고 목표 fingerprint를 확인한다.
- 각 선택 국면에서 다음 세 변형을 만든다.
  - 행동자 카드 1장
  - 상대 카드 1장
  - 양쪽 카드 각 1장
- 카드 owner는 백/흑이 아니라 목표 국면의 actor/opponent 기준이다.
- 진행 중 상태에 직접 주입 가능한 Middle/End 카드만 허용한다.
- acquisition 선택이 필요한 Opening/Piece 카드와 RULE 카드는 시작 전에 거부한다.
- 액티브 카드는 완성 상태의 card instance로 배치하고, 패시브 카드는 엔진의 grant transition을
  사용해 즉시 runtime 효과까지 반영한다.

기본 행동자 카드:

`disarm, freeze, gale, panic, reposition, vortex`

기본 상대 카드:

`parry, evasion, retreat, emergency-evacuation, sacrifice, suspicious-potion`

## split과 중복 방지

생성 dataset은 원본 self-play의 split policy와 salt를 그대로 상속한다. 파생 레코드의
`split_group_id`도 원본 game key를 유지하므로 같은 대국에서 나온 base/카드 변형이 다른
split으로 갈 수 없다. 출력 position fingerprint 중복도 제거한다.

`--max-candidates`를 사용할 때는 앞부분을 자르지 않고 전체 eligible base에 균등 간격으로
접근한다. 카드 선택 순번은 이 표본 순서를 사용하므로 stride를 사용해도 카드 종류가 빠지지
않는다.

## 출력

- `manifest.json`: 검증 완료 후 마지막에 원자적으로 쓰는 completion marker
- `shards/<split>/decision-00000.jsonl.zst`: solver-proven positive records
- `refuted.jsonl`: `not_proven`, `node_budget_exceeded`, `duplicate_position` 진단
- `scenario-summary.json`: 카드별 candidate/proven 수, generator commit, source manifest hash,
  탐색 budget

`manifest.json`이 없으면 해당 실행은 완료된 dataset으로 취급하지 않는다.

## 실행 예시

```powershell
cargo run --release -p augment-cli --bin solve_v3_tactics -- `
  --selfplay-manifest D:\augmentia\replay-buffers\v3\selfplay\manifest.json `
  --output-dir D:\augmentia\replay-buffers\v3\solver-m3-card-v1 `
  --ruleset-hash <sha256> `
  --run-id solver-m3-card-v1 `
  --created-at-utc 2026-08-27T00:00:00Z `
  --max-depth 3 `
  --max-nodes-per-position 300000 `
  --card-scenarios
```

사용자 지정 목록은 comma-separated `--attacker-cards`, `--defender-cards`로 전달한다.
빠른 검증에는 `--card-scenario-stride`와 `--max-candidates`를 함께 사용할 수 있다.

## 비범위

- Opening/Piece 카드의 획득 선택 과정 합성
- refuted 국면을 곧바로 `value=-1`로 학습하는 것
- depth 5 전체 탐색

`not_proven`은 설정한 깊이 안에서 강제승을 증명하지 못했다는 뜻이지 패배가 증명됐다는 뜻이
아니다. 따라서 현재는 sidecar에만 보존한다.

## 2026-08-27 전체 생성 결과

- generator commit: `8f700bec537651df1eff9679aa97607286b64691`
- 입력 playing positions: 4,041
- 카드 변형 후보: 12,123
- depth: 3
- position당 node budget: 300,000
- proven: 392
- not proven: 11,719
- node budget exceeded: 12
- 계산 시간: 472.33초
- split: train 353 / validation 10 / holdout 29
- manifest SHA-256: `9cc5d553868ac29085dfa7cc8bbaeda63a2b47a6a4fbc030f5281ff8f27d54c0`
- 전체 artifact 크기: 3,660,059 bytes

카드별 후보 수는 1,346~1,348로 균등했고, 12종 모두 proven record가 생성됐다. 카드별 proven
수는 26~66이었다. `scenario-summary.json`의 generator `dirty_worktree`는 false이며 dataset
validator도 통과했다.

validation 10개와 holdout 29개는 정식 promotion 평가에는 여전히 작다. 또한 이 데이터는
기존 104개 self-play 대국에서 파생됐으므로, 이전 M2 학습과 완전히 독립적인 sealed 평가
자료로 사용해서는 안 된다. 다음 확장은 학습에 사용하지 않은 새로운 source game group 또는
별도 합성 board family를 먼저 확보해야 한다.

## 2026-08-27 독립 model-mix 생성 결과

기존 104게임과 겹치지 않는 seed `202608280000..202608280511`로 새 source game group을
생성했다. 일반 baseline과 solver 미세조정 모델을 A/B로 두고 매 게임 좌석을 교대했다.
두 모델을 한 프로세스에 함께 올리는 실제 동시성 측정에서는 worker 2/4/6/8이 각각
0.076/0.118/0.126/0.148 games/s였고 worker 10은 VRAM 12 GiB 포화로 급격히 느려졌다.
따라서 CUDA worker 8을 사용했다.

긴 대국의 512 decision 꼬리는 8개 worker를 10~26분씩 점유했다. M3는 중간 국면을 solver가
다시 라벨링하므로 source 생성 한도를 128 decision으로 고정했다. limit 종료는 계속 value
supervision에서 제외된다. 평가 표본을 충분히 확보하기 위해 split은 80/10/10으로 설정했다.

- engine commit: `35c82fc7caac74e419a28c9a0729ea70448eff56`
- source: `selfplay-20260827-modelmix512-v5`
- games: 512 (train 387 / validation 65 / holdout 60)
- game steps: 26,596, 평균 51.95, 중앙값 42
- engine terminal: 481, decision limit: 31
- 344200 RULE 대국: 12
- 생성 및 root 검증 wall time: 4,307.4초
- source manifest SHA-256:
  `53558847a6f3eefe7f1725bc166a25538e924935a9de0fe4ec07ba7f2782cd68`
- source artifact: 16,299,473 bytes

이 실행 전에 대량 seed가 드러낸 두 엔진 상태 오류도 fail-fast로 수정했다. 344200 RULE은
기존 말을 복사해 stable piece ID를 중복시키던 임시 배치에서, 양쪽 킹을 각 홈 랭크에 두고
나머지 모든 말을 seed 기반 무복원 추첨으로 재배치하는 카탈로그 계약으로 바뀌었다. 후속
효과가 말을 제거하거나 막아 합법수가 0이 된 `ForcedMove`는 continuation을 적용한 뒤에만
엔진 상태로 공개한다.

새 source 전체에서 동일한 12개 카드와 depth 3, position당 node budget 300,000으로 M3를
다시 생성했다.

- dataset: `solver-m3-card-independent-20260827-v1`
- candidates: 72,984
- proven: 1,933
- not proven: 71,031
- node budget exceeded: 20
- solver elapsed: 2,109.97초, 전체 wall time 2,197.1초
- split: train 1,447 / validation 259 / holdout 227
- 카드별 candidate: 8,108~8,110
- 카드별 proven: 148~267, 12종 모두 포함
- manifest SHA-256:
  `185a386c1780ab4850b62bde744f35e5203e34f57e8a51c38ae845de8c41ea2e`
- artifact: 22,432,057 bytes

validation과 holdout 모두 목표 100개를 넘겼다. source game group이 기존 M2/M3와 독립이고
파생 레코드가 source split을 그대로 상속하므로, validation/holdout은 학습에 섞지 않고
봉인 평가에 사용할 수 있다.

## pre-tensor artifact

JSON 원본과 분리해 세 split 모두 chunk 64, worker 8로 pre-tensorize했다.

| split | examples | frames | build | manifest SHA-256 |
|---|---:|---:|---:|---|
| train | 1,447 | 23 | 47.44초 | `a3874c74c45a0825bbb6baccca57be39dfc0093afb7743d09ffd6cfe17941308` |
| validation | 259 | 5 | 8.28초 | `c4541a2baa67e37bc5a9077f776481f43d263ecc7281528430d5574c6f203cc6` |
| holdout | 227 | 4 | 7.37초 | `b2de9b5b140dcfa1aeead33787ea1a8f3082b5877b79c3d0bbed7104bb8a9486` |

train artifact의 warm initialization은 0.0031초, batch 64의 첫 collation은 0.0156초,
20 batch 평균은 0.0117초(5,464 examples/s)였다. validation은 학습 중 모델 선택에만 쓰고,
holdout은 promotion 판단 전까지 열지 않는다.
