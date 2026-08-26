# Solver warmup 실험 기록 — 2026-08-27

## 결론

`20260827-finetune-solver-v1`은 solver 전술 정확도는 개선했지만 일반 대국 성능 개선은
확인하지 못했다. 기존 `20260826-finetune-v1`을 일반 성능 기준선으로 유지하고, 새 후보는
전술 학습 실험 artifact로만 보존한다. champion으로 승격하지 않는다.

## 데이터 무결성 수정

최초 solver extractor는 원본 self-play의 `played_action_index`를 재생하지 않고 매 상태의 첫
합법 수를 적용했다. 따라서 생성된 국면은 엔진상 유효해도 원본 대국 provenance와 일치하지
않았다. 커밋 `5afae18`에서 다음을 적용했다.

- 원본 manifest 전체 replay 검증
- 후보별 실제 action prefix 재생
- 목표 decision의 `state_fingerprint` 일치 확인
- 불일치 시 즉시 실패하는 fail-closed 처리

잘못된 solver shard와 이를 사용한 미완성 학습 실행 70,150,728 bytes는
`D:\augmentia-quarantine\20260827-invalid-solver-replay`로 이동했다. 삭제하지 않았으므로
복구 가능하다.

## 수정 후 solver 데이터

- 입력: `selfplay-20260826-cuda104-v1`, 104 games, 4,497 candidate decisions
- 탐색: depth 3, position당 최대 300,000 nodes
- 결과: proven 151, refuted 4,346, budget exceeded 0
- split: train 136 / validation 11 / holdout 4
- 생성 시간: 74.94초
- manifest SHA-256: `79f1f229a47626971da76803a9da26ba9d094ba3d1eb69d9b38aa334db614dbb`

## 학습

- parent: `20260826-finetune-v1`, step 300 checkpoint
- mixture: Lichess 77% / solver 4% / weak self-play 19%
- epoch quota: 15,400 / 800 / 3,800
- CUDA, batch 64, 2,000 steps, 128,000 examples
- 전체 실행 시간: 1,256.7초(20분 56.7초)
- 최종 training-slice loss: 0.0811563
- ONNX SHA-256: `520dab217c37545d717b6934cf97e834163fc9251db1cd4ef550105c18ad05fa`
- checkpoint SHA-256: `fa23ae723ef3db626dccb5f718e59452f21237dd15ac73fe6bd14ec67227359d`

training-slice loss는 이전 iteration과 완전히 동일한 sample set이 아니므로 승격 근거로
사용하지 않는다.

## 미학습 solver split 진단

| 지표 | parent | solver fine-tune |
|---|---:|---:|
| validation top-1 (11) | 3/11 (27.3%) | 9/11 (81.8%) |
| validation optimal mass | 0.230 | 0.681 |
| validation value RMSE | 0.986 | 0.632 |
| holdout top-1 (4) | 1/4 (25.0%) | 4/4 (100.0%) |
| holdout optimal mass | 0.170 | 0.745 |
| holdout value RMSE | 1.144 | 0.693 |

표본이 validation 11개, holdout 4개뿐이므로 전술 신호가 전달됐다는 진단으로만 해석한다.

## 동일 simulation arena 진단

- 20 seed pairs, 색상 교대, 총 40 games
- search당 32 simulations, 최대 128 decisions
- 모든 40 games가 terminal
- solver fine-tune 17승 / parent 21승 / 2무
- solver fine-tune 점수율 45.0%
- 후보가 백일 때 7승, 흑일 때 10승

정식 promotion 기준인 100 pairs보다 작으며 45%의 95% Wilson 구간도 약
30.7%–60.1%로 넓다. 다만 현재 증거로는 일반 성능 개선을 주장할 수 없고 parent를 교체할
이유도 없다.

## 후속 조치

1. 현재 후보는 승격하지 않는다.
2. 서로 독립적인 카드 전술·합성 패턴을 추가해 solver validation/holdout을 확대한다.
3. 다음 학습부터 iteration 자체가 `metrics.json`에 기본 10-step 간격의 loss를 원자적으로
   기록하고, resume 시 checkpoint 이후 기록을 잘라낸 뒤 이어 쓰도록 한다.
4. 독립 전술 set과 일반 대국 smoke가 함께 개선될 때만 100-pair 정식 gate를 실행한다.

이번 실행은 iteration 경로의 metrics journal 연결 전이어서 전체 loss history를 남기지
못했고, 보존된 step 1,750/2,000 checkpoint의 마지막 배치 지표만 복구할 수 있다. 이 제약은
결과 해석에 포함하며 artifact를 사후 수정하지 않는다.
