# V4 P3 B300 gate 결과 (사전 동결 계약 실행)

이 문서는 [ai-v4-p3-b300-gate-contract.md](ai-v4-p3-b300-gate-contract.md)에
**사전에 동결한** 계약을 B300 corpus에서 실행한 결과를 기록한다.

## 1. 한줄 판정

> **A단계에서 관찰한 opened-only의 긍정적 KL 방향이 93 held-out games에서
> 유의미하게 반전되었다. Gate는 evidence 충분 상태에서 `fail`이며, 동결 계약에
> 따라 4-seed 확장은 중단한다.**

A단계는 6 blocks였다. B300은 93 blocks(최소 30 충족)이므로 이 결과는
통계적으로 판정 가능한 첫 번째 결과다.

## 2. Corpus 검증

| 항목 | 결과 |
|---|---|
| Run | `phase1-p3-b300-smoke-long64` |
| Games | 300 |
| Decisions | 16,539 |
| Terminal / Truncated | 118 / 182 |
| Unique seeds | 300 |
| Max decisions | 64 |
| Corpus audit seal | `b142c71d4d008f24935c79bf626ba07bc542034b11fade6ff41cd7724c444f54` |
| Dataset index sha256 | `39edf72814163369b00702febf0259aece523e2cb107da8b4e94a9b6bbba705b` |
| Rust validation | 18 checks 통과, 260.8s |

Opened-mask 무결성:

- `closed_positive_visit_actions = 0`
- Legal 558,078 / opened 242,955 / closed 315,123
- Mixed prefix rows 104,391 (14,825 / 16,539 decisions = 89.6%)
- False-zero closed-child mass 79,617.50

Corpus 품질 자체는 A단계보다 크게 개선되었고 계약 위반은 없다.

## 3. 실행 조건 (동결 준수)

- Split: train 207 games / validation 93 games
- `split_seed=2026091501`, `validation_bp=3333`
- Arm: opened-only `λ=0` vs all-open experiment control
- `value_weight=0`
- 1 seed, 1 epoch
- `minimum_game_blocks=30` → 93 충족
- `initialization_sha256 = 9634e4bb20820d74f6568c41528cf1168cb6f6bd7940b37ca41e018f2f20d8d3`
  (A단계 P0와 동일 초기화)
- Wall time: 26,130.2s (약 7.26시간)
- Train 11,493 examples / 719 batches, validation 5,046 examples / 316 batches

코드 기준선: B300 실행은 `fdbca1c Optimize V4 loss validation` 이후 코드로
수행되었다. 이 커밋은 `assert_visit_conservation`의 반복 호출을
`PrefixBatchV4.validated` 플래그로 대체하는 성능 변경이며, 검증 로직은
`collate_factorized_batch`로 이동했다. 테스트 78줄이 추가되었다. 계산 결과는
동일하므로 metric 수치는 유효하지만, A단계(6 blocks)와의 비교에서 코드 버전이
달라진 점을 감안해야 한다.

Artifact:

- [ab-summary.json](../../augmentia-v4-artifacts/phase1-p3-b300-p0-smoke/ab-summary.json)
- Plan sha256: `e653a8c411be42a085e74149d9345caae07be169f23b01ee27266392b38ab044`
- Summary sha256: `bb9aca2a5314a9e972936c0d867914af6d2cd02262ff56ca1f349d28b9db7a4b`
- Source corpus sha256: `2b807ff64df1889495a9b333cc143095406d2acc50acfd45f8db885ee6cbf9d2`
- Split sha256: `d40b44eae9d504fe45a627c25e86c8c89d103065930d4d2e09df08c2c9e9ad9f`

## 4. Gate 결과

```text
outcome              = fail
evidence_sufficient   = true
game_blocks          = 93  (minimum 30)
inconclusive_reasons = []
primary_passed       = false
non_degradation_passed = true
promotable           = false
```

Primary rule:

```text
opened-support conditional KL CI lower bound > 0,
or both tie-aware tail recall@3 macro/micro CI lower bounds > 0
```

### Opened-support primary

| Metric | Opened − Control | 95% CI | 판정 |
|---|---:|---:|---|
| Conditional KL | `-0.007601` | `[-0.010698, -0.004466]` | **유의한 악화** |
| Tail recall macro | `+0.000241` | `[-0.000361, +0.000883]` | 불확실 |
| Tail recall micro | `+0.000266` | `[-0.000398, +0.000974]` | 불확실 |
| All-modal top-1 | `-0.004360` | `[-0.009894, +0.000406]` | 불확실 |
| All-modal ECE reduction | `-0.005753` | `[-0.017758, +0.000930]` | 불확실 |

Conditional KL은 CI 전체가 음수이므로 opened-only가 all-open보다 **확실히 나쁩니다**.

### Non-degradation (opened-support)

| Check | 값 | 허용 regression | 판정 |
|---|---:|---:|---|
| Complete-action NLL | `+0.007601` | 0.02 | 통과 |
| All-modal top-1 | `-0.004360` | 0.01 | 통과 |
| All-modal ECE | `+0.005753` | 0.01 | 통과 |

### All-legal secondary (gate 효과 없음, 보고 전용)

| Metric | Improvement | 95% CI |
|---|---:|---:|
| Complete-action soft NLL | `-0.037130` | `[-0.041686, -0.032414]` |
| All-modal ECE reduction | `-0.015055` | `[-0.018806, -0.011014]` |
| All-modal top-1 | `+0.000595` | `[-0.002911, +0.004312]` |
| Tail recall macro | `-0.001020` | `[-0.001803, -0.000229]` |
| Tail recall micro | `-0.001125` | `[-0.001994, -0.000252]` |

All-legal NLL은 opened-only가 control보다 약 0.037 나쁩니다. 이는 opened-only가
closed child를 loss denominator에서 제외하기 때문에 구조적으로 발생하는 방향이며,
A단계의 관찰이 그대로 강화되었습니다.

## 5. A단계와의 비교: 방향 반전

| Metric | A단계 (6 blocks) | B300 (93 blocks) |
|---|---:|---:|
| Conditional KL improvement | `+0.007059` | `-0.007601` |
| CI | `[+0.001461, +0.011617]` | `[-0.010698, -0.004466]` |
| Tail macro | `+0.000391` | `+0.000241` |
| All-modal top-1 | `-0.005450` | `-0.004360` |
| ECE reduction | `+0.000930` | `-0.005753` |
| Outcome | `inconclusive` | `fail` |

두 실험은 동일 초기화를 사용했다. 유일한 실질적 차이는 corpus 크기와
held-out game 수다. 따라서 A단계의 양의 KL은 **6-block 표본의 우연**이며,
opened-only가 longer-horizon에서 우세라는 가설은 93 blocks에서 기각된다.

주의: A단계는 `inconclusive`였고, B300은 `fail`이다. A단계의 부호가
93-block 검증과 반대라는 사실 자체가 "6-block CI는 방향 판정에 부적합"이라는
원래 진단의 교훈을 다시 확인한다.

## 6. Shadow-negative 상태

A단계에서 `λ=0.05`는 longer-horizon corpus에서 KL/NLL 이득이 없었고
calibration이 악화됐다. B300은 그 결론을 변경할 근거를 제공하지 않는다.

B300은 λ=0 opened-only만 실행했으므로 shadow-negative에 대한 새 증거는 없다.
동결 계약에 따라 shadow λ는 기본 후보에서 계속 제외한다.

## 7. 동결 계약에 따른 다음 단계

동결 계약 6절:

> B단계 1차: 1 seed × 1 epoch. 4-seed 확장은 1차 결과가 유망할 때만 실행한다.
> 1차 결과가 방향적으로 무너지면 4-seed sweep은 중단한다.

B300 1차 결과는 방향적으로 무너졌다. 따라서:

1. **4-seed sweep을 실행하지 않는다.**
2. 동일 corpus/split/init에서 opened-only를 production 기본 후보로 확정하지 않는다.
3. 모델 gate 회복이 아직 이루어지지 않았다.

### 실제로 남은 선택지

다음은 아직 실행하지 않았고, 순서만 제안한다.

1. **opened-only vs all-open의 의미론 자체를 재검토**
   - B300는 opened-only가 primary metric에서 유의하게 나쁘다는 것을 보였다.
   - 이는 metric 오류가 아니라 supervision 의미론 문제일 수 있다.
   - "closed child를 negative로 두는 것"이 정말 나쁜지, 아니면
     opened-only가 분포를 과도하게 좁게 만드는지 분리해야 한다.

2. **Horizon 혼합 실험**
   - 64-decision 이후 구간에서 all-open이 더 좋아지는지 확인
   - A단계 horizon별 분석에서 late decision에서 opened-only 이득이 관찰됐으나
     이는 6 blocks에 국한된 분석이다. B300 기준으로 재검토한다.

3. **Value 실험 분리**
   - `value_weight=0` 고정이라 value head는 여전히 미검증이다.
   - policy gate를 먼저 해결한 뒤 별도 실험한다.

4. **Gate 계약 자체 재검토**
   - primary를 opened-support로 두는 것이 opened-only arm에 불리하게
     작동할 가능성이 있는지 검토한다.
   - 이 경우 metric contract 변경이 아니라 **arm 설계 변경**으로 다루며
     새 계약을 사전 동결한다.

## 8. Promotion 정책

```text
promotable           = false
strength_claim       = none
web_default          = unchanged
self_play_producer   = unchanged
parity_required      = false
parity_passed        = true
```

B300는 gate를 통과하지 못했다. Phase 2 strength를 시작할 근거가 없다.
