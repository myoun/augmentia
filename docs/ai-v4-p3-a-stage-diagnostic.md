# V4 P3 A단계 30-game longer-horizon 진단

이 문서는 현재 PC에서 실행한 P3 A단계의 corpus, P0 metric gate, P1
shadow-negative 결과를 기록한다. 이 단계는 longer-horizon과 terminal game을
포함하지만 30-game diagnostic corpus이므로 production/promotion 증거가 아니다.

## 1. 실행 설정

- Run: `phase1-p3-a30-smoke-long64`
- Games: 30
- Base seed: `2026092201`
- Unique seeds: 30
- Simulations: 100
- Chance samples: 8
- Maximum decisions: 64
- CPU workers: 4
- Target/replay: `augmentia.factorized-mcts-visits.v3` / `augmentia.replay.game.v5`
- Corpus generation wall time: `441.39s` (약 7분 21초)

Artifact:

- [manifest.json](../../augmentia-v4-artifacts/phase1-p3-a30-smoke-long64/manifest.json)
- [validation.json](../../augmentia-v4-artifacts/phase1-p3-a30-smoke-long64/validation.json)
- [dataset-index.json](../../augmentia-v4-artifacts/phase1-p3-a30-smoke-long64/dataset-index.json)
- [corpus-audit.json](../../augmentia-v4-artifacts/phase1-p3-a30-smoke-long64/corpus-audit.json)

Corpus audit seal: `8edb77c38dfeba55dc90e3aeab7390f02bee6fd0671827095d12688035261eef`

## 2. Corpus 분포

- Games: **30**
- Decisions: **1,692**
- Terminal games: **11**
- Truncated games: **19**
- Legal actions: **57,620**
- Opened actions: **24,837**
- Closed actions: **32,783**
- Mixed prefix rows: **10,914**
- Decisions with mixed rows: **1,517**
- False-zero closed-child mass: **8,087.13**
- Weighted closed support: **1,057.1745**

이전 20-game/16-step corpus와 달리 terminal value label이 생겼고, decision horizon과
support 분포가 크게 넓어졌다. 따라서 P3 A단계는 P0/P1 결론의 longer-horizon
재현성 확인으로 해석한다.

## 3. P0 metric gate

P0 A/B는 다음 bounded 조건으로 실행했다.

- 1 epoch
- Train: 24 games / 1,325 decisions
- Validation: 6 games / 367 decisions
- 동일 split/init
- `value_weight=0`
- parity/export 생략
- `minimum_game_blocks=8`

Artifact:

- [ab-summary.json](../../augmentia-v4-artifacts/phase1-p3-a30-p0-smoke/ab-summary.json)
- Summary seal: `cf5970afb403bbb678800b0dcc85b0ba7f9f98abe849dd8a98e3c6d5c0c804e8`

### Opened-support primary

| Metric | Improvement | 95% CI | 판정 |
|---|---:|---:|---|
| Conditional KL | `+0.007059` | `[+0.001461, +0.011617]` | 방향은 양수 |
| Tail recall macro | `+0.000391` | `[-0.001316, +0.002052]` | 불확실 |
| Tail recall micro | `+0.000431` | `[-0.001443, +0.002271]` | 불확실 |

Opened-support non-degradation:

- Complete-action soft NLL: 통과
- All-modal top-1: 통과
- All-modal ECE: 통과

그러나 실제 independent validation block은 6개뿐이어서:

```text
outcome = inconclusive
evidence_sufficient = false
promotable = false
```

All-legal secondary metric은 opened-only primary gate에 포함하지 않았다.

## 4. P1 shadow-negative

P1은 동일 longer-horizon corpus에서 다음 두 cell만 실행했다.

- `lambda=0`
- `lambda=0.05`

Artifact:

- [shadow-summary.json](../../augmentia-v4-artifacts/phase1-p3-a30-p1-smoke/shadow-summary.json)
- Summary seal: `760c0128901277eef07efcd52d35a31340c43b61e37ba66c48f1887c6ccdba4d`
- Plan seal: `f9a6972d0b687658a78b9baeeaed723826516941ec48591c8874cbee1b6bee4a`
- `evidence_sufficient=false`

| Metric | λ=0 | λ=0.05 | Δ (λ0−λ0.05) |
|---|---:|---:|---:|
| Conditional KL | `0.741247` | `0.741306` | `-0.000059` |
| Opened-support soft NLL | `2.702602` | `2.702661` | `-0.000059` |
| All-modal top-1 | `0.171662` | `0.171662` | `0.000000` |
| Fractional modal score | `0.088556` | `0.088556` | `0.000000` |
| All-modal ECE | `0.053889` | `0.059948` | `-0.006059` |
| Tail macro | `0.269276` | `0.269472` | `+0.000196` |
| Tail micro | `0.195605` | `0.195821` | `+0.000215` |

20-game/16-step P1에서 관찰된 λ `0.05–0.1`의 KL/NLL 개선은 longer-horizon
30-game corpus에서 재현되지 않았다. λ `0.05`는 conditional KL/NLL과 ECE가
약간 악화되고 tail만 미세하게 개선됐다.

## 5. A단계 해석

A단계가 확인한 것은 다음이다.

1. Longer horizon과 terminal corpus는 정상 생성·검증된다.
2. 30 games 자체는 80–100 held-out games gate를 대체하지 못한다.
3. 30-game hash split은 실제로 6 validation blocks를 만들었고, `minimum_game_blocks=8`을
   충족하지 못했다.
4. Opened-support conditional KL은 양의 improvement를 보였지만 evidence 부족으로
   확정할 수 없다.
5. Mixed-row shadow-negative의 효과는 이전 16-step smoke와 달라 horizon-sensitive하다.
6. 따라서 P3 다음 단계는 capacity/Hybrid가 아니라 80–100 held-out games와
   200+100 game gate-grade corpus 확장이다.

## 6. Promotion 정책

A단계 artifact는 다음 상태를 유지한다.

```text
promotable=false
strength_claim=none
web_default=unchanged
self_play_producer_promotion=forbidden
```

P3 A단계는 성공적으로 실행되었지만 model gate recovery가 완료된 것은 아니다.
