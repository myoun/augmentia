# V3 학습 및 ONNX 내보내기

9단계는 `train/v3/replay.py`, `trainer.py`, `onnx_export.py`에 구현되어 있다. 학습기는
Rust가 만든 `augmentia.training.decision.v3`와 `augmentia.replay.game.v3` shard를 읽으며,
manifest에 기록된 압축 파일 크기와 SHA-256을 먼저 확인한다. 한 run에는 한 ruleset
artifact만 허용한다. policy는 MCTS visit, behavior distribution, solver optimal set을
지원하고 value는 종료 결과와 proven value를 지원한다. 현재 정의된 보조 head는
`forced_win_distance_decisions` 하나이며 `log1p(distance)`를 회귀한다. 알 수 없는 보조
target을 조용히 버리지 않고 오류로 처리한다.

실행 예:

```powershell
cd train
uv sync --frozen
uv run python -m v3.trainer ..\replay-buffers\v3\manifest.json ..\runs\v3-first `
  --device cuda --batch-size 64 --steps 100000 --checkpoint-interval 500
```

같은 명령에 `--resume`을 추가하면 가장 최근의 정상 checkpoint를 찾는다. checkpoint는
완료된 optimizer step에서만 만들어지며 model, optimizer, cosine scheduler, AMP scaler,
sampler epoch/cursor, Python/NumPy/Torch CPU·CUDA RNG, dataset·shard hash, schema·ruleset,
vocabulary, model/training config, commit, run ID, sample 수와 최근 metric을 저장한다.
임시 파일을 완전히 기록하고 fsync한 뒤 rename하며 별도 SHA-256이 맞는 파일만 정상으로
인정한다. 최신 파일이 손상되면 직전 정상 파일로 돌아간다. dataset, vocabulary, 모델 구조,
seed, batch/accumulation, learning rate, AMP 같은 핵심 설정이 바뀌면 resume은 실패한다.
의도적인 변경은 새 run directory와 `parent_checkpoint` provenance를 사용해야 한다.

기본 보존 한도는 최근 checkpoint 네 개다. `.pt`와 `.onnx`는 Git 대상이 아니며 run/object
storage에 둔다. candidate는 checkpoint에서 export한 immutable ONNX+metadata 쌍이고,
champion 승격은 13단계 평가 gate를 통과한 뒤 별도 registry에서 수행한다. 따라서 export가
기존 champion을 자동 덮어쓰지 않는다.

내보내기 예:

```powershell
uv run python -m v3.onnx_export `
  ..\runs\v3-first\checkpoints\step-000000100000.pt `
  ..\artifacts\models\v3-first-candidate.onnx
```

ONNX와 함께 `.metadata.json`이 원자적으로 생성되며 ONNX/checkpoint hash, opset,
ObservationV3/ActionV3, ruleset artifact, dataset provenance, exact vocabulary와 실제 ONNX
input contract를 기록한다. export 직후 PyTorch와 ONNX Runtime policy/value를 수치 비교한다.

자동 검증은 `uv run python -m unittest discover -s v3 -p "test_*.py"`로 실행한다. 고정 seed
연속 4-step 실행과 2-step 중단·재개의 batch 순서, loss, parameter, optimizer 상태를
bitwise 비교하고, 손상 checkpoint 복구, config mismatch, NaN/불법 policy, ONNX 동등성을
검사한다.

12단계부터 단일 manifest 외에 `augmentia.training-mixture.v3` 설정을 같은 positional
인자로 줄 수 있다. 혼합 설정은 `bootstrap`, `first_champion`, `mature` profile의 정확한
source quota를 만들며 그 identity 전체가 checkpoint와 ONNX metadata에 저장된다. self-play
생성부터 candidate까지의 통합 실행과 보존 정책은 `ai-v3-self-play.md`를 참고한다.
