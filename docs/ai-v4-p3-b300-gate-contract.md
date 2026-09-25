# V4 P3 B300 gate 계약 (사전 동결)

B300 corpus 학습 **전에** 동결한 계약이다. B300 결과를 확인한 뒤에 이 계약을
수정하면 사후 기준 변경으로 간주한다.

동결 시점: corpus 생성 직후, 첫 B300 학습 실행 전
동결 근거: P3 A단계 분석의 권장 순서

## 1. Corpus

- Run: `phase1-p3-b300-smoke-long64`
- Games: 300
- Decisions: 16,539
- Terminal: 118 / Truncated: 182
- Max decisions: 64
- Simulations: 100, chance samples: 8
- Unique seeds: 300
- Replay/target: `augmentia.replay.game.v5` / `augmentia.factorized-mcts-visits.v3`

검증 seal:

- Rust validation: 18 checks 통과 (260.8s)
- Dataset index sha256: `39edf72814163369b00702febf0259aece523e2cb107da8b4e94a9b6bbba705b`
- Corpus audit sha256: `b142c71d4d008f24935c79bf626ba07bc542034b11fade6ff41cd7724c444f54`

Corpus 무결성 관찰:

- `closed_positive_visit_actions = 0`
- Legal actions 558,078 / opened 242,955 / closed 315,123
- Mixed prefix rows 104,391 (decisions 14,825 / 16,539 = 89.6%)
- False-zero closed-child mass 79,617.50

## 2. Split

- Algorithm: `sha256(f"{split_seed}:{game_id}")[:8] % 10000 < validation_bp`
- `split_seed = 2026091501`
- `validation_bp = 3333`
- 예상 결과: **train 207 games / validation 93 games**
- 93 blocks는 gate-grade 목표 범위(80–100)에 포함된다.
- 목표는 "정확히 200+100"이 아니라 **최소 80 held-out blocks**이며, 실제
  93으로 기록한다.

## 3. Arm

- Primary arm: **순수 opened-only**, `mixed_row_shadow_lambda = 0`
- Control arm: **all-open experiment** (`legacy-all-open-experiment`)
  - diagnostic control
  - export/resume 금지
  - promotion 후보 금지
- `lambda = 0.05`는 B단계 기본 후보에서 **제외**한다. A단계에서 longer-horizon
  이득이 재현되지 않았고 calibration이 악화됐다.
- `value_weight = 0`로 고정한다. policy gate와 value 실험을 분리한다.

## 4. Metric / margin

Primary (opened-support):

- conditional KL improvement (opened-only − all-open)
- opened-support soft NLL improvement
- tail recall macro / micro improvement

Secondary safety (all-legal runtime, gate 실패를 만들지는 않지만 별도 보고):

- all-legal soft NLL
- all-legal all-modal top-1
- all-modal ECE
- fractional modal ECE

Non-degradation margin은 A단계와 동일한 계약을 사용한다.

## 5. Evidence 조건

- `minimum_game_blocks = 30`
- 93 held-out games → evidence 조건 충족 가능
- bootstrap은 game block 단위로 수행한다.
- parity/export는 B단계 1-epoch run에서 생략하며, promotion 판정은 여전히 금지한다.

## 6. Epoch / seed

- B단계 1차: **1 seed × 1 epoch** (2 arms)
- 4-seed 확장은 1차 결과가 유망할 때만 실행한다.
- 1차 결과가 방향적으로 무너지면 4-seed sweep은 중단한다.

## 7. 판정 규칙

| 조건 | 결과 |
|---|---|
| KL improvement CI 하한 > 0 | primary 방향 통과 |
| tail macro/micro 동시 양의 방향 | tail 통과 |
| non-degradation 실패 | gate 실패 |
| `evidence_sufficient = false` | inconclusive |
| 어떤 조건이든 promotion | **금지** |

B300 1차 통과는 **policy gate 회복의 첫 증거**일 뿐이며, strength·value·
export 승격을 의미하지 않는다. Phase 2 strength는 별도 paired seed/color swap/
fixed simulation gate로 수행한다.
