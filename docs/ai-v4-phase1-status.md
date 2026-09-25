# Augmentia V4 Phase 1 결과

Phase 1 실행과 증거 봉인은 **완료**했지만, 현재 코드로 재실행한 baseline
model gate는 **실패**했다. 따라서 strength 승격이나 web default 변경을 의미하지
않으며, 모든 새 checkpoint와 report는 `promotable=false`로 유지한다.

실패 원인의 독립 감사는 [ai-v4-phase1-model-gate-diagnosis.md](ai-v4-phase1-model-gate-diagnosis.md)에
정리했다. canonical gate 계산 오류는 확인되지 않았고, 현재 결과는 작은 smoke
corpus의 낮은 검출력, mixed-row negative supervision 부재, metric support와 modal
tie 정의를 함께 고려해야 한다.

## 1. Legacy quarantine

기존 replay v4 / target v2 artifact 8개 corpus run, 304 games, 74개 checkpoint,
27개 training run, 7개 bundle을 read-only scan해 격리했다.

- registry: [`ai-v4-legacy-quarantine-registry.json`](ai-v4-legacy-quarantine-registry.json)
- inventory digest: `43a968058371d2cc634598eadfe81abe3c1676993ecd958c7f1eb925a07feefe`
- full verify: 645 files, 304 game contents, declared binaries 모두 hash 일치
- 상세 정책: [`ai-v4-legacy-artifact-quarantine.md`](ai-v4-legacy-artifact-quarantine.md)

원본 bytes는 이동·삭제·복사·재작성하지 않았고 sibling artifact root에 idempotent
root marker를 생성했다.

## 2. Clean 0915 native smoke

legacy `m1-v41-0915-native-prod-20260919`와 producer/seed/config가 같은
v5/v3 run을 재생성했다.

- run: `phase1-clean-v3-smoke-20`
- 20 games / 320 decisions / 100 simulations / 8 chance samples
- seed match 20/20
- trajectory hash match 20/20
- complete visit 및 prefix visit parity 320/320
- legal actions 7,694
- opened actions 4,600
- closed actions 3,094
- mixed visited prefix rows 1,626
- closed-positive violations 0
- validation: 20 games, 20 unique seeds, 20 truncated, 0 terminal

모든 game은 16-decision smoke cap에 걸렸으므로 strength evidence가 아니다.
전체 false-zero 영향은
`false_zero_closed_child_mass_total=918.59`,
`weighted_closed_support_fraction_total=124.6052`로 seal했다.

외부 artifact 경로는 저장소 루트 기준이다:

- manifest: `../augmentia-v4-artifacts/phase1-clean-v3-smoke-20/manifest.json`
- dataset index v2: `../augmentia-v4-artifacts/phase1-clean-v3-smoke-20/dataset-index.json`
- corpus audit: `../augmentia-v4-artifacts/phase1-clean-v3-smoke-20/corpus-audit.json`

## 3. Current-code baseline paired A/B

canonical report는 이전 ad-hoc runner 결과가 아니라 현재
[`phase1_ab.py`](../train/v4/phase1_ab.py)로 20-game clean corpus 전체를 다시 실행한
결과다. 동일 corpus의 14-game train / 6-game validation split, 224 / 96 decisions,
동일 baseline 초기 가중치, 동일 3-epoch config를 사용했다. all-open arm은 legacy
record를 재사용하지 않고 동일 v3 record의 mask만 in-memory에서 all-open view로
변환했다.

두 arm의 초기 model SHA-256은
`9634e4bb20820d74f6568c41528cf1168cb6f6bd7940b37ca41e018f2f20d8d3`로 동일하다.

| Held-out opened-only view | Opened-only | All-open experiment | Opened − control |
|---|---:|---:|---:|
| conditional KL ↓ | 0.696721 | 0.689097 | +0.007624 |
| complete-action soft NLL ↓ | 3.146072 | 3.122907 | +0.023165 |
| MCTS-modal top-1 ↑ | 0.020833 | 0.041667 | -0.020833 |
| MCTS-modal ECE ↓ | 0.120693 | 0.081571 | +0.039122 |
| visited-tail recall@3 ↑ | 0.218056 | 0.221528 | -0.003472 |

paired game bootstrap은 6 game blocks / 1000 samples다. conditional-KL improvement
mean은 `-0.007624`, 95% CI는 `[-0.013838, -0.001582]`; tail-recall improvement
mean은 `-0.003472`, 95% CI는 `[-0.011111, +0.003472]`다. 두 primary metric 모두
사전 정의한 일관 개선 조건을 만족하지 못한다.

canonical evidence (저장소 루트 기준):

- `../augmentia-v4-artifacts/phase1-current-paired-smoke-20/ab-plan.json`
- `../augmentia-v4-artifacts/phase1-current-paired-smoke-20/ab-summary.json`
- summary seal: `5615005b097388ab2295d15d4ce24f5bec4d41f6a97fe333056807ebea6326b9`
- split seal: `cd88adac2279e47e564b7c5344fa1123863aa172be10eb29f4f7d22267e36ba0`

이전 `phase1-paired-baseline-smoke-20`의 양(+) report는 policy-target semantics
ID와 experiment export 차단 계약이 없던 구식 ad-hoc 결과이므로 canonical evidence에서
supersede했다.

## 4. Cross-runtime parity and experiment isolation

production semantics을 가진 opened-only checkpoint만 Python eager/ONNX와 Rust V4
runtime으로 검증했다.

- tolerance: `2e-4`
- Python ONNX value max error: `1.34e-7`
- Rust value max error: `1.34e-7`
- Rust complete-action logit max error: `2.38e-7`
- complete-action probability sum: `0.99999994`
- parity receipt seal: `ef4cfcf03002918aefbc83045377a59156c2860d89b19992213d2844445504fd`

legacy all-open experiment checkpoint는
`augmentia.policy-target.legacy-all-open-experiment.v1`로 명시되고 production
export/resume에서 실제로 거부된다. source replay는 all-open으로 재작성되지 않는다.

## 5. Gate 판정

Phase 1 실행·무결성·최소 A/B 요구는 완료했지만 model/evaluation gate는 **fail**이다.
opened-only는 conditional KL과 tail recall을 개선하지 않았고, complete-action NLL,
top-1, ECE의 사전 정의한 non-degradation 한계도 위반했다. 따라서 다음 조건을
유지한다.

- checkpoint promotion 금지
- web default 변경 금지
- self-play producer 승격 금지
- epoch2는 experimental preview 유지
- model enlargement, Hybrid, candidate/search 튜닝을 이 결과의 근거로 사용 금지

Phase 2 strength 실행은 비승격 진단 evidences로만 제한한다. Phase 1 model gate를
먼저 다시 통과하지 않은 checkpoint는 registry/web default 승격 근거가 될 수 없다.
