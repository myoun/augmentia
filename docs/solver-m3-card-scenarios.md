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
