# V3.1 계층형 표본 PUCT와 확률적 Draft 탐색 제안

상태: H1~H7 구현 후보 완료, arena 채택 검증 전
작성일: 2026-08-29
구현일: 2026-08-29

이 문서는 현재 `MctsV3Tree`를 증강체스의 조합형 행동 공간에 맞게 확장하는 후속 탐색
설계를 정의한다. 제안 이름은 **Hierarchical Sampled PUCT**다. 정확한 게임 전이는 계속
`Engine`이 담당하고, 탐색은 큰 합법 행동 집합을 계층화·표본화·점진 확장해 제한된 계산
예산을 유망한 수에 집중한다. 공개되지 않은 Draft offer와 미래 무작위 효과는 실제 게임
RNG를 미리 보지 않고 탐색 전용 RNG로 소수 표본화한다.

이 문서는 구현 완료 기록이 아니다. 아래 설정값과 승격 기준은 구현 실험으로 검증하기 전까지
제안값이다.

이 문서에서 `V3.1`은 한 번에 켜는 단일 알고리즘 이름이 아니라 후속 탐색 작업 묶음의
문서 버전이다. 각 변경은 다음 세 종류로 나누고 서로 독립적으로 배포·롤백한다.

1. **정보 공정성 수정**: 실제 RNG와 탐색 RNG 분리. 기존 구현의 미래 Draft offer 누출을
   제거하는 정확성 변경이며, 기존 탐색보다 강해야 한다는 조건을 붙이지 않는다.
2. **의미 불변 최적화**: 정확한 후손 subtree 재사용. 탐색 결과의 의미를 바꾸지 않는 범위에서
   별도 채택한다.
3. **정책 개선 실험**: 후보 제안, progressive widening, 루트 Gumbel, 가상 행동 계층.
   각각 feature flag와 고유 `search_semantics_id`를 가지며 단독 ablation을 통과해야 한다.

정보 공정성 수정과 나머지 탐색 실험을 같은 변경 집합으로만 배포해서는 안 된다. 특히 미래
실제 offer를 아는 기존 baseline은 누출 탐지용 비교군일 뿐, 공정성 수정의 승격 상대가 아니다.

## 구현 스냅샷

2026-08-29 구현 후보에는 다음이 들어 있다.

- H1: candidate/opponent별 simulation·벽시계 예산과 search variant를 분리한 arena, wide
  chameleon fixture, 열린 행동/family/tail/chance/reuse 진단;
- H2: `SearchChanceContext`, 실제 RNG 소비 전이 감지와 탐색 seed 재실행, Draft action별
  1~16개 균등 chance outcome;
- H3: 최대 깊이 4의 exact descendant reuse, chance-sampled subtree 재사용 금지, 웹/arena의
  persistent model tree;
- H4: policy head·family coverage·seeded tail proposal, progressive widening, 열린 prior
  재정규화와 `sampled_mcts_visits` provenance;
- H5: 고정 후보 Gumbel ranking, delta-visit sequential halving과 정확한 호출 예산;
- H6: `chameleon-mutation`의 canonical virtual prefix tree와 완성 행동 property test;
- H7: Python state/action scorer 분리, Rust ONNX candidate-only tensorization API와 benchmark.

기본 `MctsV3Config`는 공정한 chance sampling과 exact descendant reuse만 켠다. widening,
Gumbel과 가상 계층은 독립 feature flag이며 arena 승격 전 기본 제품 경로에 켜지 않는다.
`arena_v3 --candidate-search/--opponent-search`와 `benchmark_v3_search --search`에서
`fair_puct`, `reuse`, `widening`, `gumbel`, `virtual_chameleon`,
`widening_virtual_chameleon`을 비교할 수 있다.

아직 채택 완료로 간주하지 않는 범위도 명시한다.

- 최소 100 paired seed arena와 신뢰구간은 실행 전이다.
- chance outcome은 공정하게 균등 표본화하지만 같은 information key의 중복 outcome 통계
  병합은 아직 하지 않는다.
- Gumbel root와 가상 계층의 동시 구성은 별도 결합 semantics가 없어 config validation에서
  거부한다.
- candidate-only runtime은 수치 동등성과 지연 측정까지 완료했지만 MCTS 기본 evaluator는
  기존 full policy head를 유지한다. factorized head와 재학습은 arena gate 뒤의 후속 작업이다.

## 1. 문제 정의

현재 V3 MCTS는 한 상태의 모든 `LegalActionSetV3` 항목을 자식 노드로 만든다. 자식 상태는
선택될 때까지 지연 생성하므로 모든 행동에 `Engine::apply`를 실행하지는 않지만, 다음 비용은
합법 행동 수에 비례한다.

- 모든 행동을 모델 입력으로 tensorize하고 policy logit을 계산한다.
- 모든 행동의 node, action, prior를 트리에 보관한다.
- 루트 합법 행동 수가 simulation 예산보다 크면 일부 행동은 한 번도 방문하지 못한다.

기존 진단에서 조합형 카드가 포함된 상태의 합법 행동 수는 최대 4,538개까지 관측됐다.
300 simulations로는 최대 6.6%만 한 번씩 방문할 수 있다. 현재 `root_min_visits=1`은 예산이
충분할 때 굶는 행동을 막지만 `|A(s)| > simulations`인 상태에서는 높은 prior 행동부터
예산을 소진한다.

현재 탐색량 실험은 다음 방향성을 보였다.

| 동일 champion 비대칭 arena | 결과 | 제한을 무승부로 계산한 점수 |
|---|---:|---:|
| 300 simulations 대 100 | 9승 1패 2제한 | 83.3% |
| 300 simulations 대 150 | 5승 4패 3제한 | 54.2% |

각 실험은 6개 seed의 색을 바꾼 12국 소표본이다. 100 simulations가 부족하다는 강한
방향성은 보이지만 150과 300의 Elo 차이를 확정하기에는 표본이 부족하다. 별도의 50~400
self-play 수렴 실험에서 300 simulations의 400 기준 공유 상태 top-action 일치율은 95.6%,
평균 total variation distance는 0.0422였고 300이 수렴 knee로 선택됐다.

따라서 개선 목표는 무조건 simulation 수를 늘리는 것이 아니다. 같은 simulation 또는 같은
벽시계 예산에서 다음을 달성해야 한다.

1. 수천 개 행동이 생기는 상태에서도 후보 다양성을 보존한다.
2. 조합형 행동의 공통 prefix에 계산을 공유한다.
3. 적은 simulation에서 루트 후보를 더 효율적으로 비교한다.
4. 낮은 prior의 희생·강제 수를 영구적으로 배제하지 않는다.
5. 엔진의 합법성, terminal, 같은 플레이어 연속 phase 의미를 그대로 유지한다.
6. 공개되지 않은 미래 Draft offer를 열거하거나 미리 보지 않고 기대값으로 처리한다.

## 2. 비목표

이 설계는 다음을 하지 않는다.

- 학습된 dynamics model로 `Engine`을 대체하지 않는다.
- 휴리스틱으로 승패나 킹 포획을 만들어 내지 않는다.
- V2 material/capture quiescence를 그대로 복원하지 않는다.
- 근사 fingerprint로 서로 다른 runtime/RNG/phase 상태를 합치지 않는다.
- 탐색 편의를 위해 ruleset의 실제 행동 순서를 변경하지 않는다.
- 실제 게임의 숨겨진 RNG 상태를 탐색용 determinization으로 사용하지 않는다.
- 가능한 Draft offer 조합을 전부 열거하지 않는다.

Sampled MuZero의 복잡한 행동 공간 표본화 원리는 참고하지만, Augmentia에는 정확한 엔진이
있으므로 MuZero의 latent dynamics는 도입하지 않는다.

## 3. 전체 구조

```text
GameState
  │
  ├─ 공개된 Draft phase
  │    ├─ 현재 offer 3개 모두 확장
  │    └─ 미래 offer는 root-sampled search RNG로 표본
  │
  ├─ root candidate proposal
  │    ├─ policy 상위 후보
  │    ├─ 행동 family별 대표 후보
  │    └─ policy tail 표본
  │
  ├─ root: seeded Gumbel sequential halving
  │
  └─ internal nodes: PUCT + progressive widening
       │
       ├─ 단순 행동: 완성된 Engine Action
       └─ 조합 행동: 탐색 전용 virtual selection nodes
            └─ 완성 시점에만 Engine Action materialize/apply
```

루트와 내부 노드는 서로 다른 확장 정책을 쓴다.

- Gumbel feature가 켜진 루트는 적은 예산에서 후보 비교 품질을 높이기 위해 sequential
  halving을 사용한다.
- widening feature가 켜진 내부 노드는 기존 PUCT를 유지하되 방문 수에 따라 열린 자식 수를
  늘린다.
- handler별 가상 계층이 켜진 조합형 행동은 전체 조합을 평평하게 열지 않고 선택 prefix를
  탐색 전용 노드로 표현한다.
- Draft 루트는 분기가 작으므로 Gumbel 표본이나 widening 없이 공개된 offer를 모두 연다.
- 공개되지 않은 다음 offer는 chance 결과를 열거하지 않고 simulation별로 표본화한다.

## 4. 엔진 행동과 가상 선택 노드

가상 선택 노드는 검색 내부 구조이며 `augment_core::Action`이 아니다. 예를 들어 세 개까지
대상을 고르는 카드의 탐색 경로는 다음과 같이 표현할 수 있다.

```text
카드 사용
  → 첫 번째 대상
    → 두 번째 대상 또는 종료
      → 세 번째 대상 또는 종료
        → 완성된 기존 Engine Action
```

가상 노드는 다음 계약을 지켜야 한다.

- `GameRecordV3.played_action_index`나 `LegalActionSetV3`에 직렬화하지 않는다.
- 가상 노드 통과 중에는 실제 `GameState`가 변하지 않는다.
- 완성된 leaf만 기존 local action index와 정확히 대응한다.
- 완성된 행동의 합법성과 전이는 반드시 `Engine`으로 확인한다.
- 같은 완성 행동으로 이어지는 가상 경로가 둘 이상 생기면 오류로 처리한다.
- search 결과와 self-play policy target은 기존 완성 행동 순서로 다시 집계한다.

가상 계층은 handler가 제공하는 canonical prefix tree다. 선택 순서가 게임 의미에 포함되면
그 순서를 보존하고, 순서가 의미 없는 대상 집합이면 handler가 엔진의 canonical ordering 하나만
생성한다. 서로 다른 prefix가 같은 완성 행동을 만들도록 허용한 뒤 마지막에 deduplicate하지
않는다.

단계 A처럼 완성 행동 logit을 이미 계산한 경우 가상 edge의 prior는 하위 완성 행동의 policy
mass로부터 정확히 유도한다.

```text
mass(prefix) = sum(P(a|s) for a compatible with prefix)
P(next | prefix) = mass(prefix + next) / mass(prefix)
```

`stop`은 해당 prefix 자체로 완성되는 실제 엔진 행동이 있을 때만 별도 자식으로 존재한다.
policy mass가 0인 경우 임의의 균등 fallback을 넣지 않고 수치 오류 또는 명시적으로 버전된
최소 확률 처리로 fail-closed한다. 단계 B/C의 조건부 head도 같은 완성 행동 분포와 비교할 수
있어야 한다.

가상 노드는 모델 평가 leaf가 아니다. 탐색은 완성 행동에 도달할 때까지 동일한 실제 state에서
가상 edge를 선택하고, 완성 시점에 `Engine::apply`를 한 번 호출해 다음 실제 node를 만든다.
가상 edge의 visit과 Q는 그 edge를 지난 simulation으로 갱신하지만 가상 edge 사이에서는 turn,
actor 또는 value 관점이 바뀌지 않으므로 backup 부호를 뒤집지 않는다. 완성 행동 visit은 그
행동의 유일한 canonical leaf를 통과한 simulation 수이며 가상 prefix visit을 중복 합산하지
않는다.

따라서 첫 구현에서 observation/action schema, ruleset hash, 기존 리플레이 의미는 바뀌지
않는다. 탐색 구현과 self-play generator provenance만 새 버전으로 기록한다.

모든 카드를 즉시 계층화하지 않는다. 우선 다음 조건을 만족하는 handler만 대상으로 한다.

- 합법 행동 수가 대상 개수의 조합으로 증가한다.
- 서로 다른 완성 행동이 명확한 선택 prefix를 공유한다.
- prefix에서 가능한 다음 선택을 엔진 규칙으로 정확하게 열거할 수 있다.

첫 대상은 `chameleon-mutation`처럼 실제 분기 폭발을 일으킨 handler다. 일반 기물 이동과
분기가 작은 카드는 기존 완성 행동 노드를 유지한다.

handler별 property test는 집합 동일성뿐 아니라 다음도 확인한다.

- 각 완성 행동의 canonical path가 정확히 하나다.
- 모든 prefix의 child mass 합이 parent mass와 허용 오차 안에서 같다.
- 임의 simulation trace에서 prefix visit, 완성 행동 visit과 root visit 보존식이 맞는다.
- 중간 prefix에서 evaluator 또는 `Engine::apply`가 호출되지 않는다.

## 5. 후보 제안

후보 집합은 top-K만으로 구성하지 않는다. policy가 낮게 평가한 희생 전술을 영구적으로
제거할 수 있기 때문이다. 초기 제안 분포는 다음 세 부분의 혼합으로 만든다.

1. **policy head**: 원래 prior가 큰 행동
2. **family coverage**: 일반 이동, 카드 사용, 선택/종료 등 행동 family별 최소 후보
3. **tail exploration**: 나머지 행동에서 seed 기반 표본

family는 표시 문자열이나 카드 설명이 아니라 안정적인 `ActionDescriptorV3` 구조로 정의한다.
같은 입력 state, model hash, search config, seed에서는 후보 순서가 완전히 재현돼야 한다.

후보 제안 결과는 단순한 집합이 아니라 전체 합법 행동의 재현 가능한 순서다. 초기 prefix는
policy head, family coverage, tail exploration의 합집합으로 만들고, 이후 순서는 아직 선택되지
않은 행동의 seeded tail permutation으로 채운다. 따라서 고정 seed에서 뒤쪽 행동이 당장 열리지
않을 수는 있지만 합법 행동이 영구히 순서에서 사라져서는 안 된다. widening은 이 순서의 prefix를
연다.

제안 초기값은 다음과 같다.

```text
top-policy candidates       = 24
family-coverage candidates  = family마다 최대 2
tail candidates             = 8
initial candidate target    = 대략 32~48
```

이는 확정값이 아니다. 실제 family 수와 카드별 분기 분포를 측정한 뒤 설정 파일로 고정한다.

완전한 top-K는 평가에서는 결정적이지만 편향되어 있다. 제안 분포에서 행동 `a`가 앞 `K`개에
포함될 확률을 `q_K(a|s)`로 정의한다. policy head처럼 항상 포함되는 행동은 `q=1`이고, tail을
가중 비복원 추출한다면 그 inclusion probability를 정확히 계산하거나 같은 결과를 재구성할 수
있는 sampler 버전과 원시 가중치를 저장한다. 계산하기 어려운 혼합 sampler를 도입하는 대신
inclusion probability가 명시적인 계층별 비복원 sampler를 우선한다.

동일 state, model hash, config, seed에서는 실제 후보가 결정적이어야 하지만, seed를 고정하기 전
제안 분포에서는 모든 합법 행동이 어떤 유한한 `K`에 대해 양의 inclusion probability를 가져야
한다. evaluation의 고정 seed와 self-play의 proposal seed는 분리한다.

self-play record에는 최소한 다음을 저장한다.

```text
proposal seed/version
각 합법 행동의 proposed/opened 여부
각 widening 경계 K에서의 q_K(a|s), 또는 이를 정확히 재구성하는 입력
행동별 최초 open simulation
최종 열린 집합과 열린 집합 안의 visit count
```

보정이 없는 첫 버전은 원래 model prior를 열린 후보 안에서 재정규화한 **편향된 baseline**이다.
이 baseline의 policy loss는 열린 후보 안에서만 정규화하며, 닫힌 행동의 visit 0을 부정 정답으로
사용하지 않는다. `N(a)/q_K(a|s)` 같은 importance-aware target은 실험할 수 있지만 이것만으로
full-action search policy의 불편 추정량이라고 주장하지 않는다. 보정 방식이 arena와 calibration
검증을 통과하기 전에는 `sampled_policy_target_v1`처럼 별도 target semantics로 기록한다.

## 6. Progressive widening

내부 노드에서 방문 수 `N(s)`에 따라 열린 자식 수를 제한한다.

```text
K_initial(s) = min(|A(s)|, max(K0, mandatory_initial_union_size(s)))
K(s, N) = min(|A(s)|, K_initial(s) + floor(c × N^alpha))
```

초기 제안값은 다음과 같다.

```text
K0    = 32
c     = 4
alpha = 0.5
```

`mandatory_initial_union_size`는 중복 제거한 policy head, 필수 family coverage와 초기 tail의
합집합 크기다. 따라서 family 수가 많은 상태에서는 초기 후보가 32를 넘을 수 있으며 그 값을
진단에 기록한다. `K_initial=32`인 상태는 300회 방문 시 약 101개 후보를 연다. 이 값은 모든
행동을 한 번 방문하는 것을 목표로 하지 않는다. policy 상위 후보를 깊게 검토하면서 family
coverage와 tail sampling으로 탐색 누락 위험을 제한하는 것이 목표다.

새 자식을 여는 순서는 후보 제안 단계가 만든 재현 가능한 순서를 사용한다. 열린 자식 안의
선택은 기존 PUCT를 유지한다.

```text
score(s, a) = Q(s, a)
              - virtual_loss(s, a)
              + c_puct × P_open(a|s) × sqrt(N(s)) / (1 + N(s, a))
```

`root_min_visits`는 전체 합법 행동이 아니라 현재 열린 루트 후보에만 적용한다. widening으로
새 후보가 열리면 최소 한 번의 방문 기회를 부여한다.

이 `root_min_visits` 규칙은 PUCT 루트에 widening을 적용하는 H4 구성의 의미다. Gumbel root가
켜진 H5 구성에서는 search 호출 시작 시 후보 `m`개를 고정하고 sequential-halving schedule이
최소 방문을 담당하므로 같은 호출 중 root widening을 함께 적용하지 않는다.

`P_open(a|s)`는 현재 열린 집합에서 원래 model prior를 재정규화한 값이며 proposal 확률
`q_K(a|s)`와 혼동하지 않는다. proposal 보정을 선택식에 넣는 실험은 별도 semantics ID로
분리한다. 한 simulation에서 widening과 최소 방문 배정이 동시에 발생하면 먼저 새 후보를 열고,
그 후보의 첫 방문을 예약한 뒤 기존 PUCT 선택으로 돌아간다.

`K(s,N)`의 `N`은 해당 search 호출에서 추가된 방문 수가 아니라 재사용된 방문을 포함한 node의
유효 visit count다. 다만 같은 벽시계 실험의 예산 보고에는 이번 호출에서 추가한 delta visit을
별도로 기록한다. subtree 재사용 유무 때문에 후보 수가 달라지는 현상은 진단에 포함한다.

## 7. 루트 Gumbel sequential halving

루트는 PUCT가 모든 후보에 예산을 얕게 흩뿌리는 대신 후보를 단계적으로 절반씩 제거한다.

1. proposal에서 루트 후보를 고른다.
2. fingerprint, model hash, config로부터 Gumbel seed를 유도한다.
3. 각 후보에 초기 simulation을 배정한다.
4. completed Q-value와 policy logit으로 순위를 갱신한다.
5. 하위 절반을 제거하고 남은 후보에 다음 예산을 집중한다.
6. 마지막 후보 또는 개선된 루트 policy를 반환한다.

초기 후보 수 제안은 simulation 예산에 따라 정한다.

```text
150 simulations → 최대 32개
300 simulations → 최대 64개
400 simulations → 최대 64개
```

위 숫자는 탐색 전에 무조건 적용하는 고정값이 아니다. 호출 예산을 `B`, 후보 수를 `m`이라 할
때 `m <= B`여야 하며 모든 후보가 첫 라운드에 적어도 delta visit 1을 받아야 한다. 150/300
simulation에서는 `m=16/32/64`를 ablation하고 후보 수 증가가 선택 안정성이나 승률을 개선하지
않으면 더 작은 값을 택한다.

첫 구현의 sequential-halving schedule은 탐색 시작 전에 결정적으로 만든다.

1. 활성 집합 `S_0`를 proposal 앞 `m`개로 만든다.
2. 남은 라운드 수는 `ceil(log2(|S_r|))`, 남은 예산은 아직 배정하지 않은 delta visit이다.
3. 라운드 시작 전에 `remaining_budget >= |S_r|`를 확인한다. 만족하지 않으면 활성 집합을
   결정적으로 줄이거나 현재 ranking에서 종료한다.
4. `floor(remaining_budget / (|S_r| × remaining_rounds))`를 기본 배정량으로 사용하되 각
   활성 후보에 최소 1을 배정한다. 나머지는 현재 ranking과 local action index로 결정적으로
   배분한다.
5. 라운드 배정이 끝난 뒤에만 ranking을 다시 계산하고 상위 `ceil(|S_r|/2)`를 남긴다.
6. 예산이 다음 라운드의 최소 배정량보다 작으면 후보 수를 줄이거나 마지막 ranking에서
   종료한다. 예산을 암묵적으로 초과해서는 안 된다.

ranking score의 계약은 다음과 같다.

```text
rank(a) = gumbel(a) + log(P_root(a)) + sigma(Q_completed(s, a))
```

`sigma`의 scale, 미방문 행동의 `Q_completed`를 root network value와 방문 행동 통계로 채우는
공식, tie-break와 부동소수점 정밀도는 `root_selection_id`에 버전으로 고정한다. 정확한 공식이
고정되기 전 H5를 구현 완료로 간주하지 않는다. subtree를 재사용할 때 ranking의 Q는 누적
통계를 사용할 수 있지만 라운드 예산은 이번 호출의 delta visit만 사용한다. 동일한 search
history, seed와 예산에서만 동일 결과를 요구하며, 재사용 history가 다른 두 호출을 같은 입력으로
간주하지 않는다. `P_root(a)=0`으로 `log`가 정의되지 않으면 조용히 clamp하지 않고, 명시적으로
버전된 probability floor를 적용해 진단하거나 평가 오류로 중단한다.

evaluation에서도 재현성을 위해 Gumbel seed를 고정한다. self-play 다양성 seed와 arena seed는
서로 분리한다. root Dirichlet noise와 Gumbel ranking을 동시에 무분별하게 적용하지 않고,
다음 세 구성을 ablation한다.

- Gumbel만 사용
- Gumbel 후보 선택 뒤 self-play용 Dirichlet 적용
- 기존 Dirichlet PUCT baseline

첫 구현의 기본값은 Gumbel과 Dirichlet을 동시에 사용하지 않는다. 결합 실험에서는 Dirichlet이
proposal 전 logit, proposal 후 열린 prior, 또는 최종 visit sampling 중 어디에 적용되는지 하나만
선택해 semantics ID에 기록한다.

Gumbel 방식은 적은 simulation에서 정책 개선을 목적으로 하지만 이 프로젝트에서의 우위는
반드시 같은 벽시계 arena로 확인해야 한다.

`GamePhase::Draft`에는 이 루트 규칙을 그대로 적용하지 않는다. 공개된 offer가 보통 3개이므로
세 행동을 모두 열고 최소 방문을 보장한다. Draft의 불확실성은 현재 행동 표본화가 아니라
아직 공개되지 않은 다음 offer의 표본화에서 처리한다.

## 8. 트리 재사용

현재 `prepare_root`는 직전 루트의 직계 자식만 새 실제 state와 비교한다. 웹은 제안마다 새
트리를 만들고, self-play의 좌석별 트리는 보통 상대 응수 뒤 손자 이상의 상태를 만나므로
일반 교대 구간에서 재사용 효과가 제한적이다.

개선안은 다음과 같다.

- 웹 세션은 모델별 `MctsV3Tree`를 game state와 함께 보관한다.
- 다음 실제 state를 제한 깊이의 materialized descendant에서 찾는다.
- 정확한 `GameState` equality와 전체 fingerprint가 모두 맞을 때만 재사용한다.
- 찾은 subtree만 compact하고 이전 형제는 폐기한다.
- model hash, ruleset hash, search semantics version이 바뀌면 즉시 clear한다.

서로 다른 모델의 트리는 공유하지 않는다. transposition table도 별도 측정에서 정확한 중복이
충분히 관측되기 전에는 활성화하지 않는다.

실제 게임에서 관측된 다음 root를 재사용할 때는 full fingerprint로 확인한다. 탐색 중 생성한
chance determinization은 실제 미래 상태처럼 영속화하지 않으며, 다음 실제 관측과
`search_information_key`가 일치하는 공개 통계만 재사용 대상으로 삼는다.

첫 chance 구현에서는 안전성을 위해 sample-specific descendant와 그 아래 Q/visit을 search 호출
종료 시 폐기하고, 실제로 관측된 deterministic descendant만 persistent tree 재사용 대상으로
삼는다. information-set 통계 재사용은 hit rate와 강도 이득이 별도 측정되고, 서로 다른 숨은
표본의 value를 합치는 계약이 검증된 뒤 별도 feature flag로 연다. H3의 deterministic subtree
재사용 통과를 chance subtree 재사용 승인으로 해석하지 않는다.

## 9. 모델 연동 단계

### 단계 A: 현재 policy head 유지

현재 모델이 모든 완성 행동의 logit을 계산한 뒤 탐색 트리에서 일부만 연다. 이 단계는 node
수, state materialization, search 예산 분산을 줄이지만 전체 action tensorization 비용은
줄이지 못한다. 기존 모델과 checkpoint를 그대로 사용할 수 있다.

### 단계 B: 후보 단위 점수화

state embedding을 한 번 계산하고 후보 `ActionDescriptorV3`만 추가 점수화할 수 있도록
runtime API를 분리한다. widening으로 새 후보가 필요할 때 action score를 추가 계산한다.
이 단계부터 큰 행동 집합의 tensorization 비용도 감소한다.

### 단계 C: factorized policy head

충분한 데이터가 모이면 policy를 다음 조건부 head로 분리한다.

```text
action family
  → card/source
    → target 1
      → target 2 / stop
        → target 3 / stop
```

이 단계는 모델 구조와 학습 target 변경이므로 별도 model contract 버전이 필요하다. 그러나
최종 완성 행동을 기존 `ActionDescriptorV3`에 대응시키는 한 ruleset의 실제 action 의미는
유지할 수 있다.

## 10. 확률적 Draft와 root-sampled 탐색

### 현재 의미 문제

`ObservationV3`는 미래 무작위 결과 유출을 막기 위해 RNG seed와 내부 상태를 모델에 넣지
않는다. 그러나 현재 MCTS는 실제 `GameState`를 복제해 `Engine::apply`를 호출한다. 백이
Draft 카드를 가상 선택하면 엔진이 복제된 실제 RNG로 흑의 다음 offer를 즉시 만든다. 모델은
RNG를 직접 보지 않지만 트리는 아직 공개되지 않은 실제 미래 offer를 전이 결과로 확인한다.

이는 확률 분포의 평균이 아니다. 모든 simulation이 같은 숨겨진 실제 RNG를 복제하므로 현재
플레이어가 미래 한 가지를 정확히 아는 것과 같다. V3.1은 실제 RNG와 탐색 RNG를 분리해야
한다.

### 열거하지 않는 원칙

가능한 카드 3장 조합을 모두 chance 자식으로 만들지 않는다. 현재 actor에게 이미 공개된
offer만 전부 열고, 아직 공개되지 않은 offer는 simulation별로 소수 표본화한다.

```text
공개된 카드 A/B/C
  ├─ A 선택 → 탐색 RNG 표본 → 상대 offer 3개 → 상대 최선 선택 → value
  ├─ B 선택 → 탐색 RNG 표본 → 상대 offer 3개 → 상대 최선 선택 → value
  └─ C 선택 → 탐색 RNG 표본 → 상대 offer 3개 → 상대 최선 선택 → value

다음 simulation
  └─ 새로운 탐색 RNG 표본으로 반복
```

초기 제안값은 다음과 같다.

```text
visible draft actions          = 전부, 보통 3
future offer samples/selection = 초기 8, 최대 16
opponent actions/sample        = 전부, 보통 3
post-draft continuation        = 짧은 search 뒤 value 평가
```

`3 × 8 × 3 = 72`는 직관적인 초기 평가 규모일 뿐 독립된 72개 전체 게임을 끝까지 전개한다는
뜻이 아니다. 각 표본은 짧게 탐색한 뒤 value head가 Draft 이후의 장기 카드 시너지와 보드
가치를 요약한다. 8과 16은 arena로 조정할 제안값이다.

### 탐색용 chance context

엔진에 실제 RNG를 바꾸는 임시 setter를 노출하지 않는다. 검색 전용 전이 API 또는 동등한
`SearchChanceContext`를 둔다.

- 실제 게임 `Engine::apply`만 actual state의 RNG를 규칙이 요구하는 횟수만큼 소비한다.
- 검색 전이는 root information key, sample lane, chance event index와 draw index로 seed를
  유도한다. simulation schedule은 sample lane에 결정적으로 매핑한다.
- 현재 공개 offer는 actual state에서 고정하고 미래 chance event부터 다시 표본화한다.
- 각 현재 Draft 선택을 비교할 때 같은 원시 난수열을 사용해 표본 분산을 줄인다.
- 카드 보유/제외/별 균형에 따라 달라지는 실제 offer 가중치 로직은 그대로 사용한다.
- search RNG와 actual RNG가 같은 값인지 비교하거나 actual RNG를 seed derivation에 넣지 않는다.

같은 원시 난수열은 서로 다른 선택에 같은 카드 ID를 강제로 주는 뜻이 아니다. `(root
information key, sample lane, chance event index, draw index)`에서 공통 uniform variate를 만들고,
각 선택 뒤의 실제 가중 분포에 inverse-CDF 또는 동등한 결정적 sampler로 각각 매핑한다. 후보
카드 집합이나 가중치가 달라져도 각 선택의 주변 분포가 엔진 분포와 정확히 같아야 한다.
중복 없는 offer 생성, 제외 카드와 균형 재추첨도 실제 엔진과 동일한 순서로 uniform variate를
소비한다.

각 simulation은 하나의 determinization을 운반하지만 통계는 플레이어가 구분할 수 있는 공개
정보 단위로 합친다. 기존 full fingerprint는 RNG를 포함하므로 정보 집합 key로 사용하지
않는다. `ObservationV3`의 의미와 현재 actor를 바탕으로 별도의 안정적인
`search_information_key`를 정의한다. 단, 서로 다른 공개 runtime/phase/offer를 근사적으로
합치는 것은 금지한다.

`search_information_key`는 단순히 full fingerprint에서 RNG 필드만 지운 값으로 구현하지
않는다. 버전된 `SearchInformationV1` 구조를 정의하고 다음을 canonical serialization에
포함한다.

- 현재 actor와 그 actor가 관측할 수 있는 `ObservationV3` 전체
- 현재 actor에게 공개된 Draft offer와 선택 순서
- 공개 phase/runtime/effect queue와 합법 행동 의미에 영향을 주는 공개 상태
- ruleset, observation/action schema와 chance semantics version

실제 RNG state, 아직 공개되지 않은 offer와 상대만 아는 정보는 제외한다. 같은 information key로
통계를 합치기 전에 actor와 `LegalActionSetV3`가 정확히 같음을 검사하고, 다르면 key 충돌로
fail-closed한다. 표본화된 offer가 상대에게 공개되는 순간 그 offer는 상대 decision node의 key에
포함된다.

decision policy는 오직 해당 node의 information key로 알 수 있는 값에 조건화한다. simulation이
운반하는 determinization ID나 다른 표본의 숨은 상태를 tie-break, proposal 또는 value 입력에
사용하면 strategy fusion으로 간주해 오류 처리한다. chance 뒤의 action availability가 숨은 상태에
따라 달라지는 효과는 첫 Draft 범위에 포함하지 않고 별도 information-set 설계 전까지 탐색
sampling을 비활성화한다.

chance subtree에서는 공유 통계와 simulation별 상태를 구조적으로 분리한다.

```text
InformationNode
  = information key + actor + legal descriptors + edge N/W/Q/prior

SimulationCursor
  = 이번 determinization의 GameState + SearchChanceContext + sample lane
```

`InformationNode`의 `GameState` 하나를 권위 있는 자식 상태로 저장해 다른 sample lane에서
재사용하지 않는다. 각 simulation은 자신의 `SimulationCursor`에 `Engine` 검색 전이를 적용하고,
그 결과에서 다음 information key를 계산해 공유 통계 node를 찾는다. 같은 key의 두 cursor는
`ObservationV3`, actor와 legal descriptors가 바이트 또는 계약상 정확히 같아야 한다. 모델 출력은
`(information key, model hash)`로 cache할 수 있지만 determinized `GameState` 자체는 cache 값으로
공유하지 않는다.

### 일반 확률 효과와 비공개 정보

첫 구현 범위는 Opening/Middle/End Draft offer 생성이다. 같은 문제가 있는 다른 카드 RNG는
chance event inventory를 작성한 뒤 동일한 search context로 확장한다. 상대 비공개 카드처럼
chance가 아닌 숨은 상태가 존재하면 단순 Draft 표본만으로 충분하지 않으며 information-set
탐색 범위를 별도로 늘린다.

2~3수 exact 전술 탐색 중 공개되지 않은 Draft가 끼면 특정 표본 하나의 승리를 강제승으로
부르지 않는다. 가능한 모든 결과를 증명하지 않는 한 확률적 기대값으로 MCTS에 돌려보낸다.

## 11. 데이터와 버전 호환성

단계 A와 B의 검색 변경은 다음을 의미한다.

- 기존 board pretraining과 기존 model checkpoint는 계속 사용할 수 있다.
- 기존 self-play record도 형식상 유효하다.
- 새 탐색의 visit distribution은 달라지므로 generator provenance를 새 버전으로 기록한다.
- 새 모델 학습에서는 새 탐색 데이터를 우선하고 기존 self-play 비율을 점진적으로 줄인다.
- 탐색 변경만으로 기존 학습을 무조건 처음부터 다시 시작하지 않는다.

기존 record의 실제 Draft offer, 선택, 이후 대국 결과는 유효하다. 그러나 기존 MCTS가 실제
미래 offer를 미리 보고 만든 Draft visit target은 공정한 정보 집합 policy target으로
간주하지 않는다. V3.1 Draft 데이터가 충분해질 때까지 기존 record는 다음처럼 사용한다.

- Draft 이전의 일반 board/choice/card action target은 기존 정책대로 유지한다.
- 기존 `draft_pick` policy target은 제외하거나 낮은 가중치를 준다.
- 누출된 탐색이 선택한 Draft 이후의 board/choice/card target은 합법적인 실제 전이 기록이지만
  공정한 정책이 만든 상태 분포는 아니다. `legacy_leaked_draft_descendant` provenance로 분리하고
  새 데이터와의 혼합 비율을 보고하며, 기본값은 낮은 가중치다.
- 기존 게임의 최종 결과는 실제로 일어난 결과이지만 공정한 search policy의 on-policy value
  target이라고 부르지 않는다. value 학습에 쓰면 동일 provenance와 혼합 비율을 기록한다.
- 실제 선택과 최종 결과를 이용한 보조 카드 가치 학습은 별도 auxiliary target provenance 아래
  허용한다.

과거 record에 `search_semantics_id`가 없다면 generator binary/model/run manifest로 누출 가능
구간을 보수적으로 판정한다. 판정할 수 없는 record를 조용히 clean data로 승격하지 않는다.
새 sampled-search record는 전체 합법 행동, proposed/opened mask와 visit을 함께 보존해 닫힌
행동을 visit 0의 부정 정답으로 오해하지 않게 한다.

최소 provenance 필드 또는 동등한 run contract에는 다음이 필요하다.

```text
search_semantics_id
candidate_proposal_id
candidate counts and mixture
proposal inclusion-probability semantics/version
proposed/opened action mask and first-open simulation
progressive_widening: K0, c, alpha
per-node K_initial and mandatory-initial-union version
root_selection_id
gumbel_seed derivation/version
sequential-halving allocation and completed-Q version
chance_sampling_id
search_information_key_version
search_rng_derivation_version
draft_future_offer_samples: initial/max
c_puct, fpu_reduction, virtual_loss
simulation and wall-clock budgets
model/ruleset/action/observation hashes
```

가상 노드를 리플레이 action으로 기록하거나 완성 행동 visit 합계를 잃으면 fail-closed로 해당
게임을 폐기한다.

## 12. 구현 순서

### H1. 측정 기반선과 비대칭 arena

- candidate/opponent에 서로 다른 simulation과 동일 벽시계 예산을 줄 수 있게 한다.
- 루트 합법 행동 수, 열린 행동 수, family coverage, tail 방문률을 기록한다.
- 큰 분기 위치 fixture와 현재 150/300/400 baseline을 봉인한다.
- 미래 실제 offer를 아는 현재 Draft 결과에는 `leaked_actual_rng_baseline` 표식을 붙이고 강도
  승격 기준이 아니라 누출 재현과 회귀 탐지에만 사용한다.

### H2. 실제 RNG와 탐색 chance 분리

- Draft chance event inventory와 정보 공개 계약을 봉인한다.
- `SearchChanceContext`와 `search_information_key`를 구현한다.
- 현재 공개 offer 고정, 미래 offer 재표본, 실제 RNG 불변 테스트를 추가한다.
- 기존 결정론적 Draft 탐색과 1/4/8/16 sample 결과를 비교한다.
- H2는 독립 release boundary다. H3~H7 feature를 켜지 않은 현재 PUCT 위에서도 배포 가능해야
  하며, 누출 baseline보다 승률이 낮다는 이유로 롤백하지 않는다.

### H3. 후손 subtree 재사용

- 웹 persistent tree와 제한 깊이 descendant lookup을 구현한다.
- 재사용 hit rate, 메모리, 수당 지연을 기록한다.
- chance subtree 재사용은 이 단계 범위에서 제외한다.

### H4. Progressive widening

- 현재 완성 행동 policy 위에 widening만 적용한다.
- 낮은 prior 강제승·희생승 회귀 테스트를 추가한다.
- 전체 tensorization 비용이 남는다는 한계를 보고서에 유지한다.
- Gumbel과 가상 계층을 끈 baseline을 먼저 통과한다.

### H5. 루트 Gumbel

- sequential halving과 completed Q-value를 구현한다.
- 기존 PUCT, Gumbel, Gumbel+noise를 같은 모델로 비교한다.
- 후보 수 16/32/64와 subtree 재사용 on/off를 분리해 보고한다.

### H6. 조합 행동 가상 계층

- 분기 폭발 handler 한 개부터 구현한다.
- 완성 행동 집합이 기존 `LegalActionSetV3`와 정확히 일치하는 property test를 통과시킨다.

### H7. 후보 단위 또는 factorized policy

- H1~H6이 실제 arena와 latency를 개선한 뒤에만 모델 head 변경을 진행한다.

각 단계의 기본 비교에서는 한 번에 하나의 feature flag만 바꾼다. 결합 구성은 각 단독 구성이
정확성 gate를 통과한 뒤 실행하며 최소한 다음 행을 같은 모델·seed set으로 남긴다.

| 구성 | chance 분리 | 재사용 | widening | Gumbel | 가상 계층 |
|---|---:|---:|---:|---:|---:|
| 공정한 PUCT 기준선 | on | off | off | off | off |
| 재사용 단독 | on | on | off | off | off |
| widening 단독 | on | off | on | off | off |
| Gumbel 단독 | on | off | off | on | off |
| handler 계층 단독 | on | off | off | off | on |
| 최종 후보 구성 | on | 측정값 | 측정값 | 측정값 | handler별 측정값 |

모든 구성은 고유 `search_semantics_id`를 가지며, 최종 결합이 단독 구성보다 나빠질 수 있음을
전제로 다시 arena를 통과한다.

## 13. 검증 계약

### 정확성

- 열린 모든 완성 행동은 현재 state에서 엔진 합법이다.
- 기존 합법 행동은 후보 생성 가능성이 0이 되어서는 안 된다.
- terminal value는 오직 `GameStatus`에서 얻는다.
- 같은 플레이어가 연속 행동하는 phase에서 backup 부호를 뒤집지 않는다.
- virtual visit/loss는 오류·취소 경로에서도 정확히 회수한다.
- 동일 seed/config/model은 동일 후보, 방문 수, 최종 행동을 만든다.
- sequential-halving의 delta visit 합은 요청 예산을 넘지 않으며 라운드별 활성 후보가 정확히
  감소한다.
- 닫힌 행동은 policy target에서 visit 0의 부정 정답으로 사용되지 않는다.
- 기록한 proposal 정보로 후보 순서와 `q_K(a|s)`를 재구성할 수 있다.

### Draft와 chance

- 현재 actor에게 공개된 offer는 빠짐없이 모두 연다.
- 실제 RNG state를 읽거나 search seed에 섞지 않는다.
- 검색을 수행해도 실제 `GameState.rng_state()`가 바뀌지 않는다.
- 같은 search seed/config는 같은 sampled offer sequence를 만든다.
- 서로 다른 search seed는 실제 offer 분포와 통계적으로 일치한다.
- 공통 uniform variate를 사용해도 각 Draft 선택의 주변 offer 분포는 실제 엔진 분포와
  통계적으로 일치한다.
- 선택별 미래 표본 수와 실제 evaluator 호출 수를 보고한다.
- 실제 미래 RNG 하나를 아는 baseline보다 약해지더라도 공정한 탐색을 필수 의미로 유지한다.
- determinization ID만 바꿔도 동일 information key의 proposal과 decision policy가 바뀌지 않는다.
- 같은 information key인데 actor 또는 합법 행동 집합이 다르면 merge하지 않고 충돌을 보고한다.

### 조합 계층

- 가상 leaf의 완성 행동 집합은 기존 평면 합법 행동 집합과 정확히 같다.
- 중복, 누락, 순서 비결정성이 없어야 한다.
- 완성 행동 prior를 prefix에 합산했다가 leaf로 전개한 결과가 원래 prior와 일치한다.
- 가상 edge에서는 actor와 backup 부호가 바뀌지 않고 `Engine::apply`와 evaluator가 호출되지
  않는다.
- 가상 노드는 리플레이, ONNX metadata, 웹 action index에 노출되지 않는다.

### 성능

다음 두 arena를 모두 실행한다.

1. 같은 simulations: 탐색 자체의 품질과 회귀 확인
2. 같은 벽시계: 실제 제품·self-play 계산 효율 확인

최종 채택 판단은 최소 100 seed pair와 색 교환을 사용한다. 점 추정만으로 승격하지 않고
paired 결과의 신뢰구간을 함께 기록한다. 큰 분기 fixture에서는 다음도 별도로 측정한다.

- action tensorization p50/p95
- peak node count와 working set
- simulation throughput
- 열린 행동 수와 family coverage
- low-prior/tail visit rate
- proposal `q_K` 분포와 solver 최적 행동/강제 수의 proposal recall@K
- candidate 수 16/32/64별 simulation allocation과 최종 선택 안정성
- 선택 행동의 300↔400 안정성
- Draft future-offer sample 1/4/8/16의 선택 안정성과 같은 벽시계 승률
- Draft 선택별 평균·분산·표본 수와 information-key merge 수

### 중단 조건

다음 중 하나라도 발생하면 해당 실험을 즉시 중단한다.

- 불법 행동, 중복 완성 행동 또는 visit 합계 불일치
- 같은 seed의 비결정적 결과
- terminal/value 관점 회귀
- 메모리 또는 지연이 기존 baseline보다 악화됐는데 강도 개선도 없음
- 특정 action family의 proposal 확률이 0
- 닫힌 행동을 visit 0 target으로 학습하거나 proposal provenance를 재구성할 수 없음
- sequential-halving이 simulation 예산을 초과하거나 같은 입력 history에서 schedule이 달라짐
- 실제 RNG와 search RNG의 공유 또는 미래 실제 offer 누출
- 공개 정보가 다른 Draft 상태의 information-key 충돌
- determinization identity가 decision policy에 영향을 주는 strategy fusion

## 14. 채택 판단

첫 실용 목표는 “150 simulations에서 300 simulations 기존 PUCT에 가까운 강도”가 아니다.
같은 150 simulations 또는 같은 웹 수당 지연에서 기존 150 PUCT를 통계적으로 앞서는 것이
우선이다. self-play에서는 같은 계산 시간당 더 좋은 policy target과 더 많은 완료 게임을
만드는지를 평가한다.

권장 채택 순서는 다음과 같다.

1. 실제 RNG와 search RNG 분리는 정보 공정성 수정으로 가장 먼저 독립 채택한다. 누출
   baseline과의 승률은 이 단계의 롤백 조건이 아니다.
2. 후손 재사용은 의미 불변 최적화로 독립 채택한다.
3. progressive widening은 큰 분기 fixture와 arena를 모두 통과하면 채택한다.
4. Gumbel root는 적은 simulation arena에서 우위가 확인될 때 채택한다.
5. 가상 계층은 handler별 완전성 증명과 property test 뒤 활성화한다.
6. factorized policy는 별도 model contract와 재학습 계획으로 진행한다.

3~5번은 서로의 선행 성공을 자동 채택 근거로 삼지 않는다. 예를 들어 widening이 이겼더라도
Gumbel을 결합한 구성이 다시 같은 벽시계 arena를 통과하지 못하면 widening 단독 구성을 유지한다.

## 15. 참고 연구

- Thomas Hubert 외, [Learning and Planning in Complex Action Spaces](https://proceedings.mlr.press/v139/hubert21a.html), ICML 2021. Sampled MuZero와 표본 행동 집합에서의 policy improvement.
- Ivo Danihelka 외, [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO), ICLR 2022. 적은 simulation을 위한 Gumbel AlphaZero/MuZero와 sequential halving.
- Yunhyeok Kwak 외, [Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction](https://arxiv.org/abs/2406.00614), 2024. 조합형/factored action space의 상태 조건부 행동 추상화.
- David Silver, Joel Veness, [Monte-Carlo Planning in Large POMDPs](https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html), NeurIPS 2010. 숨은 상태와 관측을 전부 열거하지 않는 root sampling 기반 Monte-Carlo planning.
