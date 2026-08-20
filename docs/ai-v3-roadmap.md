# AI V3 전환 로드맵

이 문서는 V2 AI를 기준선으로 보존한 뒤, 증강체스 규칙을 손실 없이 이해하는
AI로 교체하는 작업 순서와 각 단계의 완료 조건을 고정한다. 구현 중 세부 설계는
바뀔 수 있지만, 앞 단계의 계약과 검증을 통과하지 않은 채 뒤 단계로 넘어가지는
않는다.

현재 진행 상태(2026-08-21): 1~6단계 완료. 7단계 생성기 구현과 회귀 pilot은 완료했고,
실제 학습 규모 corpus는 외부 PGN의 출처·license·용량을 정한 뒤 생성한다. 8단계 모델
구현 전에는 그 pilot의 coverage와 처리량을 검토한다. 1단계는
`ai/v2-baseline-before-observation-v3` 태그로 고정했고, 2단계 계약은
`observation-v3-contract.md`와 `augment-ai::observation_v3`에 고정했다. 3단계 Rust
인코더는 `augment-ai::encoder_v3`에 구현했다.
`ActionDescriptorV3`와 가변 합법 행동 policy는 `augment-ai::action_v3`에 구현했다.
Rust/Python parity는 `train/v3`의 checked-in golden fixture와 validator로 고정했다.
V3 대국·학습 데이터 포맷은 `data-v3-format.md`와 `augment-ai::data_v3`에 고정했다.

## 핵심 원칙

- 리체스 일반 체스 기보는 사전학습이나 보조 과제에는 쓸 수 있지만, 증강체스의
  정책·가치 정답으로 간주하지 않는다.
- 상태와 행동의 의미를 먼저 보존하고, 그 위에 모델과 탐색을 올린다.
- 첫 V3 모델은 디버깅과 반복 속도가 빠른 합성곱/잔차 기반 하이브리드로 만든다.
  Transformer는 데이터 규모와 장거리 상호작용 병목이 실제 측정으로 확인된 뒤
  선택한다.
- 모든 데이터와 모델에는 ruleset, 관측 스키마, 행동 스키마, 생성 주체와 결과를
  기록한다. 중단된 대국을 무승부로 학습시키지 않는다.
- 승격은 같은 시뮬레이션 수와 같은 벽시계 시간 조건을 모두 통과해야 한다.

## 1~14단계

### 1. V2 기준선 동결과 저장소 정리

현재 동작과 결함, 속도, 품질 지표, 활성 모델의 해시를 기록한다. 비교에 필요한
V2 런타임과 최소 학습 코드, 기준 ONNX 한 개만 남기고 생성 데이터, 실패한 모델,
중복 모델, 체크포인트, 가상환경, 빌드 산출물과 일회성 진단 코드를 제거한다. Git의
도달 불가능한 대용량 객체도 정리한다.

완료 조건:

- `docs/ai-v2-baseline.md`에 재현 가능한 기준선과 보존 정책이 있다.
- 기준 ONNX의 경로, 크기, SHA-256과 opset이 확인된다.
- 전체 워크스페이스 테스트와 Git 무결성 검사가 통과한다.
- 정리된 상태가 전용 커밋과 annotated tag로 고정된다.

### 2. ObservationV3 의미 계약 확정

엔진의 공개 상태를 보드 공간 정보, 전역 게임 상태, 플레이어별 자원, 카드 정의,
카드 런타임 상태, 드래프트·후속 선택·강제 행동 상태로 분류한다. 각 필드에 타입,
범위, 관측 가능성, 소유자, 정규화 방식과 마스킹 규칙을 명시한다. 카드 ID나 기물
종류를 작은 해시 버킷으로 압축하지 않는다.

완료 조건:

- 현재 ruleset에서 AI가 볼 수 있는 상태가 누락 없이 목록화된다.
- 동일 의미가 안정된 ID를 가지며, 미지 값과 값 0이 구분된다.
- `observation_schema_id`가 대국과 학습 샘플에 기록된다.

### 3. Rust ObservationV3 인코더 구현

공간 정보는 명시적 보드 채널로, 전역 수치는 별도 벡터로, 가변 길이 카드와 효과는
타입이 있는 토큰/레코드로 인코딩한다. 현재 플레이어 관점 정규화와 대칭 변환은 원래
상태를 복원하거나 의미를 검증할 수 있는 방식으로 구현한다.

완료 조건:

- 대표 상태와 모든 특수 기물·카드 상태를 포함한 단위 테스트가 있다.
- 서로 다른 의미 상태가 의도치 않게 같은 인코딩이 되지 않는다.
- 인코더 출력의 shape, dtype, 범위가 버전 계약으로 고정된다.

### 4. ActionDescriptorV3와 합법 행동 인터페이스 구현

고정된 32,768칸 인덱스에 행동을 우겨 넣는 방식을 제거한다. 이동, 프로모션,
카드 사용, 대상 선택, 드래프트와 후속 선택을 구조화된 descriptor로 표현하고,
각 상태의 합법 행동 목록에 대해 정책 점수를 계산한다. 행동 수가 늘어나도 clamp나
충돌이 없어야 한다.

완료 조건:

- 한 상태의 서로 다른 합법 행동은 항상 서로 다른 descriptor를 가진다.
- encode/decode 왕복과 엔진 적용 결과가 일치한다.
- 모든 현재 카드 행동을 포함한 충돌 검사가 통과한다.

### 5. Rust/Python 계약 일치와 골든 픽스처 구축

Rust가 학습 데이터의 의미 기준이 되며 Python은 같은 스키마를 읽는다. 프로모션을
포함한 행동 매핑, 관점 변환, 마스킹과 값 라벨을 골든 픽스처로 비교한다. 가능하면
학습 입력은 Rust가 완성된 텐서/레코드로 내보내 Python의 규칙 재구현을 줄인다.

완료 조건:

- Rust와 Python의 관측·행동·결과 해석이 바이트 또는 허용 오차 내에서 일치한다.
- CI에서 스키마 불일치와 오래된 데이터 로딩을 즉시 거부한다.

### 6. V3 대국·리플레이 데이터 포맷 확정

샘플에 ruleset ID, observation/action schema ID, 생성 엔진과 모델 ID, 난수 시드,
ply, 합법 행동, 탐색 방문 분포, 최종 결과와 종료 사유를 저장한다. 파일 단위 checksum,
샤딩, 압축, 중복 제거와 train/validation/holdout 분리 규칙도 정한다.

완료 조건:

- 정상 종료, 중단, 오류, 제한 초과가 구별된다.
- 중단 대국은 가치 정답으로 사용되지 않는다.
- 데이터 검증기가 잘못된 스키마와 불법 행동을 차단한다.

### 7. 부트스트랩 데이터 생성

일반 체스 기보는 양쪽 도메인에 공통인 board move만 `board_pretraining` behavior로
가져오며 증강체스 policy/value 정답으로 변환하지 않는다. Augmentia policy/value는
ActionV3와 실제 엔진 전이만 사용하는 bounded solver가 강제승리를 증명한 상태에만
부여한다. 현재 material heuristic, V2 policy/value와 제한 깊이에서 증명하지 못한 결과는
정답으로 저장하지 않는다. 모델의 랜덤 초기화는 8단계에서 다룬다.

완료 조건:

- PGN SHA-256, 원본 위치와 license가 모든 imported record에서 역추적된다.
- standard-chess record는 board-pretraining 범위만 가지며 V3 value가 없다.
- proven tactical record의 최적 행동과 value가 engine-only solver에서 재현된다.
- schema/checksum/split 검증과 action/phase/piece/card coverage 보고서가 생성된다.
- 알려진 coverage 공백을 숨기거나 heuristic label로 채우지 않는다.

### 8. 첫 V3 네트워크 구현

보드는 작은 residual CNN, 전역 상태는 MLP, 카드·효과 레코드는 embedding과 pooling
또는 작은 set encoder로 처리한 뒤 결합한다. 정책은 현재 합법 행동 descriptor를
조건으로 점수화하고, value와 필요한 보조 예측 head를 둔다. 처음부터 대형
Transformer를 도입하지 않는다.

완료 조건:

- 가변 개수의 합법 행동과 카드 레코드를 처리한다.
- 작은 데이터에 의도적으로 overfit하는 smoke test를 통과한다.
- 파라미터 수, 추론 시간과 메모리 기준을 기록한다.

### 9. 학습·내보내기 파이프라인 구축

정책 방문 분포, 최종 결과와 보조 과제를 학습하고 데이터 provenance를 모델
메타데이터에 포함한다. 체크포인트 보존 한도와 candidate/champion 수명주기를
설정하며, ONNX export 뒤 Python과 Rust 출력 동등성을 검사한다.

학습 재개는 마지막 epoch부터 대략 이어가는 방식이 아니라 마지막으로 완료된 optimizer
step부터 재현 가능해야 한다. checkpoint는 임시 파일에 완전히 기록한 뒤 atomic rename으로
공개하며, 기록 도중 중단된 파일은 `latest`로 인정하지 않는다. 각 checkpoint에는 다음을
함께 저장한다.

- model, optimizer, learning-rate scheduler와 AMP gradient scaler 상태;
- global optimizer step, epoch, gradient accumulation 경계와 deterministic sampler cursor;
- Python, NumPy, PyTorch CPU/CUDA RNG 상태와 data-loader shuffle seed;
- dataset manifest ID/hash, shard 목록, ruleset/schema ID, 학습 설정과 코드 commit;
- run ID, 누적 sample 수와 직전 검증 지표.

재개 시 dataset, schema, 모델 구조 또는 핵심 학습 설정이 checkpoint와 다르면 조용히
이어가지 않고 실패한다. 의도적인 변경은 새 run으로 fork하고 부모 checkpoint를 provenance로
남긴다. 정상 종료 요청에는 즉시 checkpoint를 만들고, 강제 종료에서는 마지막 주기 checkpoint
이후의 step만 다시 실행한다.

완료 조건:

- 고정 seed 학습 smoke test가 재현된다.
- 동일 seed에서 연속 학습과 중단·재개 학습의 batch 순서, loss, parameter와 optimizer
  상태가 허용 오차 안에서 일치한다.
- 손상되거나 절반만 기록된 checkpoint를 무시하고 직전 정상 checkpoint로 재개한다.
- checkpoint와 다른 dataset/schema/config로 재개하려 하면 명시적으로 실패한다.
- NaN, 불법 행동 확률, 스키마 불일치를 자동 검출한다.
- ONNX 수치 동등성 테스트가 통과한다.

### 10. 추론 런타임과 성능 경로 교체

요청마다 ONNX session을 다시 만들지 않고 프로세스에서 재사용한다. leaf 평가를
batch 처리하고 복사와 재인코딩을 줄인다. 모델 로딩 실패와 스키마 불일치는 조용히
fallback하지 않고 명시적으로 보고한다.

완료 조건:

- 상태 조회와 한 수 탐색의 p50/p95가 기준선보다 개선된다.
- batch 크기별 처리량과 지연 시간이 기록된다.
- 메모리 증가와 장시간 실행 안정성 검사가 통과한다.

### 11. MCTS 의미와 탐색 효율 수정

정책 prior가 매우 작아도 합법 행동이 탐색 기회를 갖도록 선택식을 검증한다. root
noise와 temperature를 용도에 맞게 분리하고, value 관점과 백업 부호를 테스트한다.
가능하면 수 사이의 tree reuse, transposition과 선택적 quiescence를 측정 후 적용한다.

완료 조건:

- 강제 수와 희생 전술 회귀 테스트가 통과한다.
- 합법 행동 방문 커버리지와 탐색 분포가 정상 범위에 든다.
- 동일 모델에서 시뮬레이션당 처리량이 기록된다.

### 12. 반복 self-play와 학습 루프 운영

초기 네트워크와 다양한 상대를 사용해 self-play를 생성하고, 최신 데이터와 과거
데이터를 적절히 섞어 candidate를 학습한다. 카드·상태 희소성에 따라 curriculum과
샘플링 가중치를 조절하되, 변경 효과를 실험 메타데이터에 남긴다.

완료 조건:

- 생성, 검증, 학습, export, 평가가 한 번의 재현 가능한 run으로 연결된다.
- 디스크 사용 한도와 자동 보존/폐기 정책이 작동한다.
- 데이터·모델 lineage를 역추적할 수 있다.

### 13. 승격 게이트 구축과 후보 평가

스키마 검사, 전술 세트, 봉인된 value calibration, 규칙·카드 커버리지, 같은
시뮬레이션 수 arena, 같은 벽시계 시간 arena 순으로 후보를 평가한다. 통계적 신뢰구간과
색·시드·카드 분포 균형을 포함한다.

완료 조건:

- 하나의 명령으로 후보 평가 보고서를 만들 수 있다.
- 어느 필수 게이트든 실패하면 champion 교체가 차단된다.
- V2 기준선과의 품질·속도 비교가 같은 조건에서 재현된다.

### 14. V3 승격, 호환 제거와 다음 모델 판단

통과한 champion을 웹·CLI 기본 AI로 승격하고 모델과 스키마 호환 검사를 배포 경로에
고정한다. 롤백 절차를 확인한 뒤 V2 코드와 로컬 기준 모델을 제거할 수 있다. 이 시점에
CNN/set-encoder의 오류 분석을 보고 Transformer 도입 여부를 결정한다.

Transformer 전환은 다음 중 하나가 측정될 때 별도 실험으로 진행한다.

- 카드·효과 사이 장거리 상호작용이 현재 표현의 주된 오차 원인이다.
- 데이터가 충분히 커졌고, 작은 모델 대비 검증·arena 성능 향상이 반복된다.
- 같은 벽시계 예산에서 증가한 추론 비용보다 탐색 품질 향상이 크다.

완료 조건:

- V3가 기본 경로이며 실패 시 검증된 이전 champion으로 롤백할 수 있다.
- V2 제거 여부와 Transformer 실험 여부가 측정 결과에 근거해 기록된다.

## 단계 의존성

1단계는 비교와 복구 지점을 만든다. 2~5단계가 AI가 보는 세계와 행동의 계약을
고정하고, 6~7단계가 그 계약으로 신뢰할 수 있는 데이터를 만든다. 8~11단계는 모델,
런타임과 탐색을 교체하며, 12~13단계는 데이터 플라이휠과 승격 판단을 만든다.
14단계에서만 V2를 폐기하거나 더 복잡한 모델 계열로 이동한다.
