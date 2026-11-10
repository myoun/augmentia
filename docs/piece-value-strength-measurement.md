# 기물 값 교정의 강도 측정

2026-11-11 · 커밋 `cb8ad50` (값 교정) 기준, 측정 시점은 black-box 수정(`f4398fd`) 이후

## 측정 방법

`Evaluation::LegacyMaterial`을 추가해 **같은 엔진에서 기물 값 테이블만 다른 두 탐색**을 대국시켰습니다.
다른 요소는 전부 동일합니다: 같은 빌드, 같은 탐색 코드, 같은 예산, 색 교차.

- `arena --evaluation material --against legacy-material`
- games 40 × 색 2 = **80게임**, max-actions 200, time 100ms, depth 4
- 룰셋 `augment_chess_20261102`

## 결과

| 항목 | 값 |
|---|---:|
| 결판 | **66게임** |
| 새 테이블 승 | **46** |
| 이전 테이블 승 | 20 |
| 무승부 | 4 |
| 수순 캡 도달 | 9 |
| 오류 / 예산 초과 | 0 / 1 |

**새 테이블 승률 0.697 (95% CI 약 ±0.111)** — 결판 게임 기준.

색 교차로 확인:

| 선수 | 새 테이블 | 이전 테이블 |
|---|---:|---:|
| White | 25 | 9 |
| Black | 21 | 11 |

두 색 모두 이깁니다. 선수 이점이 결과를 만든 게 아닙니다.

## 결론

기물 값 교정(82종 중 71종 수정)은 **탐색 강도를 유의미하게 올렸습니다.**
80게임에서 결판 66게임, 승률 0.697. CI 하한이 0.586으로 0.5를 넘으므로
이 표본에서 이득이 우연일 확률은 낮습니다.

평균 수순 57.4수로 실제 결판이 나서, 이전 측정(32수 캡, 전부 무승부)과 달리
강도 비교 자체가 성립했습니다.

## 한계

- 표본 66결판. CI가 넓습니다(±0.111). 0.697이 정확한 강도가 아니라 **방향의 근거**입니다.
- 시간 예산 100ms/수, 깊이 4. 더 긴 예산에서는 결과가 다를 수 있습니다.
- `Evaluation::LegacyMaterial`은 측정 전용 통제군입니다. 실사용 평가가 아닙니다.
- 엔진 내부 비교라 서로 다른 빌드 간 차이(컴파일러 변동)는 배제됐습니다.

## 재현

```
cargo build --release -p augment-search
target/release/augment-search.exe arena --evaluation material --against legacy-material `
  --games 40 --max-actions 200 --time-ms 100 --depth 4 --seed 7
```

원본 산출물: `artifacts/arena-material-vs-legacy.json`
