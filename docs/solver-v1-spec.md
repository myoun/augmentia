# V3 Solver 데이터 생성기 스펙 (v1)

상태: 초안 — 구현 전 설계 확정용
기준 커밋: 11925a0

## 1. 목적과 비범위

증강체스 엔진(augment-core) 위에서 **증명 가능한 전술 국면**을 탐색해, V3 학습 데이터의
`DecisionValueTargetV3::Proven` 라벨을 생성한다.

비범위:

- 클래식 체스 솔버·종료국 DB의 결과 재사용 (규칙이 달라 무효)
- 확률적 능력이 개입될 수 있는 국면의 강제승 주장
- 장기 강제승(깊은 N) 증명 — v1은 얕은 깊이만 다룬다

## 2. 유일한 진실원

- 합법수 생성과 상태 전이: `Engine::legal_actions`, `Engine::apply`
- 종료 판정: `GameStatus::Won / Draw`
- 카드 부여: `Engine::grant_card`
- 국면 합성: `GameState::from_board` (board + turn + seed)

탐색기는 이 API만 사용한다. 체스 휴리스틱(바쿠레시 등)은 후보 국면 선별의 참고로만 쓰고,
라벨 결정에는 절대 사용하지 않는다.

## 3. 강제승의 정의 (증강체스 버전)

어떤 완전한 상태 S에서 행동자 P가 강제승이라 함은:

1. S는 규칙상 진행 중(Ongoing)이고, P에게 합법수가 존재하며
2. S의 **모든** 관측 변수(보드, 양측 카드 인스턴스와 쿨다운, HP, 별 점수, 드래프트
   진행도, card_runtime, 반복 횟수)가 스냅샷에 고정돼 있고
3. P의 적절한 착수에 대해 **상대의 모든 합법 응답**(카드 사용 포함)에 대해
   재귀적으로 같은 조건이 성립하고
4. 탐색 깊이 내에서 GameStatus::Won{winner:P}에 도달하며
5. 경로상 단 하나의 비결정성(난수 소비)도 발생하지 않는다.

5번의 검증: 탐색 중 DeterministicRng 시퀀스를 기록해, 경로별 소비량이 예상(0)과
다르면 그 가지를 폐기한다.

## 4. 탐색 알고리즘

깊이제한 AND-OR 탐색(alpha-beta 불필요, 논리 구조가 명확):

- OR 노드(행동자 차례): 하나라도 이기는 수가 있으면 PROVEN
- AND 노드(상대 차례): 모든 응답에 대해 PROVEN이어야 한다
- 깊이 한도: v1은 홀수 플라이 d ∈ {1, 3, 5} (d=1: 즉시 킹 포획/승리 착수,
  d=3: 두 수 조합, d=5까지 선택적)
- 가지치기: AND 노드에서 하나라도 반례가 나오면 즉시 REFUTED
- 메모리: 경로만 저장(스택), transposition table 없음(v1)

## 5. 국면 선별 (후보 생성)

세 경로로 합성하고, 출처를 provenance.source_id에 구분 기록:

| 경로 | 방법 | 목표 |
|---|---|---|
| self-play 추출 | 기존 self-play 게임의 실제 국면 중 얕은 강제승 후보 | 현재 모델이 자주 만나는 국면 |
| 체크메이트 패턴 | 클래식 메이트 패턴(백랭 미트 등) 보드를 from_board로 구성 후 카드 무부여 | 일반 체스 직관이 유효한지 검증되는 코어 |
| 카드 시나리오 | 특정 카드 보유 상태를 grant_card로 구성 (예: 번개로 디펜스 말 제거 후 메이트) | 카드-전술 상호작용 커버 |

편향 통제: 세 경로의 비율을 manifest에 기록하고, 동일 board+cards 해시 중복 제거.

## 6. 라벨 스키마

각 국면마다 DecisionRecordV3 1개 생성:

- targets.value = Proven { value: ±1.0, proof_source: "solver_v1:d{depth}" }
- legal_actions: 엔진이 생성한 전체 합법수
- policy target: 강제승 착수 집합이 1개면 optimal_action_set, 여러 개면 생략(v1은
  정확히 증명된 착수만 기록)
- auxiliary:
  - forced_win_distance: 정수 (d)
  - proof_depth_total_nodes: 탐색 노드 수 (디버깅용)
  - nondeterminism_checked: bool

반증된 국면(REFUTED)은 value 라벨 없이 별도 사이드카(JSONL)에 기록해 재검증 큐로 남긴다.

## 7. 검증 계약

1. 생성 직후 같은 프로세스에서 라벨 재계산(2회 독립 탐색, 결과 일치 확인)
2. validate_v3_dataset으로 shard 전체 검증 통과
3. 샘플링 감사: 매 500개마다 1개를 무작위 골라 사람이 읽을 수 있는 PV(착수열) 덤프
4. 결정론성: 동일 입력·커밋·seed로 재실행 시 SHA-256 일치

## 8. 산출물과 CLI

```
cargo run -p augment-cli --bin solve_v3_tactics -- \
  --positions candidates.jsonl \      # 또는 --from-selfplay <manifest>
  --depths 1,3 \
  --max-nodes-per-position 2_000_000 \
  --output replay-buffers/v3/solver-v1 \
  --games-per-shard 1000
```

manifest는 기존 dataset manifest v3 스키마를 그대로 사용해 mixture 소스로 바로 연결.
mixture 프로파일의 solver 슬롯(bootstrap 30%)이 자연스럽게 활성화된다.

## 9. 마일스톤

1. M1: d=1 탐색 + self-play 추출 경로 (엔진 API만으로 완결, 가장 빠름)
2. M2: d=3 AND-OR + 체크메이트 패턴 합성
3. M3: 카드 시나리오 합성 + d=5
4. M4: mixture 연결 및 첫 solver 혼합 학습 실행

예상 난이도: M1은 엔진 API 위 얇은 레이어라 즉시 구현 가능. M2부터 상태 공간
폭발 관리가 본격화된다.

