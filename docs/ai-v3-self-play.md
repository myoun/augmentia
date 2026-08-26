# V3 반복 self-play와 학습

12단계는 `augment-cli self_play_v3`와 `python -m v3.iteration`으로 구현한다. 한 iteration은
입력 모델 두 개, seed, 탐색 설정과 외부 데이터 manifest를 명시적으로 받고 다음 순서를
원자적인 artifact 단위로 실행한다.

1. 두 입력 모델의 좌석을 게임마다 교대해 V3 MCTS self-play를 생성한다.
2. 생성 shard의 checksum, 스키마와 전체 엔진 replay를 검증한다.
3. 선택한 단계별 비율로 immutable source manifest를 섞는다.
4. 정확한 optimizer step부터 재개 가능한 candidate를 학습한다.
5. ONNX를 내보내고 PyTorch/ONNX 동등성과 제한된 loss 진단을 기록한다.
6. 모든 입력·출력 hash를 `iteration-manifest.json`에 연결하고 보존 한도를 적용한다.

이 단계의 평가는 파이프라인 오류를 찾는 진단이며 champion 승격 판정이 아니다. arena,
봉인 holdout과 통계적 승격 gate는 13단계가 담당한다.

## self-play 의미

모든 decision에는 상태·다음 상태 fingerprint, 정확한 `LegalActionSetV3`, MCTS visit 수,
완료 simulation 수, root value, Dirichlet noise seed와 실제 sampling temperature가 저장된다.
승패는 엔진 terminal에서만 가져온다. decision·시간·자원 제한이나 오류로 끝난 대국은
policy에는 사용할 수 있지만 value target은 만들지 않는다.

생성기는 같은 모델끼리 또는 서로 다른 두 모델을 대국시킬 수 있다. 모델 A/B는 게임마다
백·흑을 교대하며 각 대국에 ONNX SHA-256과 탐색 config가 기록된다. dirty worktree에서 만든
데이터는 재현 가능한 학습 자료로 게시되지 않는다.

## 혼합 비율

비율은 파일 크기나 전체 보유 데이터 비율이 아니라 한 virtual epoch의 sample quota다.
정수 quota, source별 shuffle seed와 실제 manifest hash가 checkpoint identity에 포함되므로
중단·재개 후 source와 sample 순서가 같다.

| profile | Lichess | solver | 약한 self-play | champion self-play |
|---|---:|---:|---:|---:|
| `bootstrap` | 50% | 30% | 20% | 0% |
| `first_champion` | 20% | 20% | 10% | 50% |
| `mature` | 10% | 15% | 5% | 70% |

양수 비율의 source manifest가 하나라도 빠지면 실행을 거부한다. 작은 epoch에서 반올림이
필요하면 모든 활성 source에 최소 한 sample을 준 뒤 largest-remainder 방식으로 나머지를
배정하고 확정된 quota를 iteration manifest에 기록한다. Lichess record의 value supervision
mask는 계속 꺼져 있으므로 일반 체스 결과가 증강체스 value로 유입되지 않는다.

## iteration 설정과 실행

설정 예시는 경로와 예산을 실제 artifact에 맞게 바꿔 사용한다.

```json
{
  "iteration_config_schema_id": "augmentia.selfplay-iteration-config.v3",
  "iteration_id": "20260822-bootstrap-001",
  "created_at_utc": "2026-08-22T00:00:00Z",
  "seed": 2026082201,
  "ruleset_manifest_sha256": "<64 hex>",
  "models": [
    {"onnx": "seed.onnx", "metadata": "seed.metadata.json"},
    {"onnx": "seed.onnx", "metadata": "seed.metadata.json"}
  ],
  "selfplay": {
    "source": "weak_selfplay",
    "games": 1000,
    "workers": 4,
    "simulations": 400,
    "leaf_batch_size": 16,
    "execution_provider": "cpu",
    "intra_threads": 1,
    "max_decisions": 512,
    "temperature_decisions": 30,
    "temperature": 1.0,
    "noise_alpha": 0.3,
    "noise_epsilon": 0.25,
    "train_bp": 9000,
    "validation_bp": 500,
    "holdout_bp": 500,
    "compression_level": 3
  },
  "mixture": {
    "profile": "bootstrap",
    "epoch_size": 1000000,
    "sources": {
      "lichess": "lichess/manifest.json",
      "solver": "solver/manifest.json"
    }
  },
  "training": {
    "device": "cuda",
    "batch_size": 64,
    "total_steps": 100000,
    "checkpoint_interval": 500,
    "checkpoint_keep": 4,
    "amp": true,
    "evaluation_examples": 256
  },
  "retention": {"keep_iterations": 6, "max_bytes": 536870912000}
}
```

`weak_selfplay` manifest는 이번 iteration이 생성하므로 bootstrap 설정의 외부 sources에는
Lichess와 solver만 쓴다. 이후 profile에서는 생성 데이터의 분류에 따라
`selfplay.source`를 정하고 나머지 양수 source를 명시한다.

```powershell
cd train
uv run python -m v3.iteration path\to\iteration.json ..\runs\v3-iterations
```

SIGINT 등으로 학습이 멈춘 경우 같은 설정으로 `--resume`을 붙인다. 기존 input config와
mixture가 byte 단위로 다르면 이어가지 않는다. self-play manifest가 이미 완성됐다면 다시
생성하지 않고 replay 검증하며, checkpoint는 마지막 정상 optimizer step부터 이어간다.

self-play는 절대 game index별로 독립 프로세스를 실행하고 `parts/game-NNNNNNNN`에 한 대국씩
checksum manifest를 원자적으로 확정한다. 기본 CPU 프로필은 worker 4개와 worker당 ONNX
thread 1개다. 완료된 part는 `--resume`에서 전체 엔진 replay를 다시 통과한 뒤 재사용하고,
worker 하나가 실패하면 아직 실행 중인 다른 worker를 즉시 종료한다. 모든 part가 검증된 뒤에만
루트 `manifest.json`을 원자적으로 공개하므로 중단 시 완료 대국을 잃거나 불완전 dataset을
학습에 넘기지 않는다.

`execution_provider`는 `cpu` 또는 `cuda`만 허용한다. `cuda`는 CUDA EP 등록 실패를 오류로
처리하며 CPU로 자동 fallback하지 않는다. CUDA 실행에는 `ORT_DYLIB_PATH`가 가리키는 ONNX
Runtime과 설치된 CUDA/cuDNN의 major 버전이 일치해야 한다.

## lineage와 보존

완료 manifest는 입력 config, 부모 ONNX/metadata, 각 source manifest, 생성 self-play,
혼합 identity, checkpoint, candidate ONNX/metadata의 절대 경로·SHA-256·크기를 기록한다.
candidate export는 champion registry를 수정하지 않는다.

보존 정책은 설정한 iteration root의 직계 자식 중 유효한 완료 manifest가 있는 디렉터리만
삭제 후보로 본다. 최신 개수나 총 byte 한도를 넘으면 오래된 unpinned iteration 전체를
제거한다. `PINNED` 파일이 있거나 현재 실행 중인 iteration은 삭제하지 않으며, 한도를 맞출
수 없으면 임의로 보호 artifact를 지우지 않고 실패한다.

## 12단계 smoke 검증

2026-08-21 CPU smoke run은 1게임·2 decision·decision당 8 simulations, bootstrap virtual
epoch 10개와 optimizer 1 step으로 전체 경로를 실행했다. 생성 manifest는 1 shard, 1 game,
2 policy decision으로 엔진 replay 검증을 통과했다. 혼합 quota는 Lichess 5, solver 3,
weak self-play 2로 정확히 기록됐다. candidate의 PyTorch/ONNX 최대 절대 오차는 policy
`5.22e-8`, value `7.45e-8`였다. 동일 설정의 `--resume` 재실행은 기존 artifact hash를
유지하고 추가 optimizer step을 실행하지 않았다.
