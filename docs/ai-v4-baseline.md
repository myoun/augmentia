# Augmentia V4 고정 예산 baseline 봉인 기록

이 문서는 V4 factorized search의 고정 예산 성능 기준을 봉인한다.
현재 기준은 **2026-09-19 / 0915 ruleset / lazy prefix scoring 적용 후**이며,
기존 0818 기준은 아래에 역사 기록으로 보존한다.

측정 도구는 crates/augment-cli/src/bin/benchmark_v4_search.rs다.
ruleset, fixture/bundle, executable, 탐색 옵션 중 하나라도 바뀌면 새 baseline으로 취급한다.

## 현재 봉인: 2026-09-19 / augment_chess_20260915

### 봉인 identity

- **ruleset**: augment_chess_20260915
- **manifest SHA-256**: 7b2736f8e9f727db818d8c8f4aaf7742a23a717e2f31d749ea293a3dfc051b13
- **base Git commit**: 118660f62c072b4c204d0e5c5ea42a7bcbbde61f
- 측정 시 working tree는 dirty 상태였다. 따라서 commit만으로 재현성을 주장하지 않고 아래 **release benchmark executable SHA-256**을 실행물 identity로 사용한다.
- **benchmark executable SHA-256**: 800d1c25d79563aa316812950358201bc4a9e94a975999ddfd05b7635b44ced8
- **benchmark source SHA-256**: 354965056585c9e01586aec7cab85d464febd93250668083d3d751a6403d8506

0915 전용 Rust fixture는 기존 0818 fixture와 분리해서 보존한다.

- train/v3/fixtures/0915/observation_action_v3.json
  - SHA-256 7ac80a8ff683614dc7b1b4988a1d3be8cbf313ae55a8ffee4a71641343eea0bb
- train/v3/fixtures/0915/runtime_vocabulary_v3.json
  - SHA-256 78f512beae3a89d9608a9dc744ad604f905809145c338af6c18084ca836399b5

0915 V4 fixture bundle:

- train/v3/fixtures/v4_bundle_0915/v4-model.v4-bundle.metadata.json
  - SHA-256 006b6e4ec44cabc5e3d347edf286faad30502e795ae2234e878ed0dba61b4cf9
- state encoder SHA-256: 49344ae55ba13a92ce0de8ec212f0853a0c6f6c0c1f89ef59937f48c1297eb0f
- factorized scorer SHA-256: bd74d651204503c8b6c302c20009627d2569b74db4cd965c601cdbb8cbedc69e
- text vocabulary size: **475**
- 이 번들은 작은 synthetic regression fixture다. 학습된 실전 모델의 품질을 뜻하지 않는다.

원시 benchmark JSON과 machine-readable seal은 artifacts/v4-baseline-0915/에 저장한다.

### 측정 환경

- CPU: AMD Ryzen 7 5800X 8-Core Processor
- OS: Microsoft Windows NT 10.0.26200.0
- PowerShell: 5.1.26100.9444
- Python: 3.12.8
- PyTorch: 2.11.0+cu128
- ONNX Runtime: 1.26.0
- ONNX execution provider: CPU
- ONNX intra-op threads: 1

### 고정 탐색 조건

- simulations: **200**
- iterations: **8**
- chance_outcome_samples: **8**
- gumbel_candidates cap: **16**
- search_seed: **6000273900112991796**
- gumbel_seed: **5140099496739100212**
- 각 측정 iteration 뒤 같은 runtime으로 fresh-tree warm repeat를 한 번 더 수행한다.
- warm repeat는 선택 행동과 전체 root visit 분포가 모두 동일해야 한다.

### 재현 명령

저장소 루트에서 0915 fixture를 먼저 만든다.

~~~powershell
cargo run -p augment-cli --bin export_v3_fixtures -- --ruleset 20260915 --out train/v3/fixtures/0915/observation_action_v3.json --vocabulary-out train/v3/fixtures/0915/runtime_vocabulary_v3.json
~~~

0915 manifest hash를 실제 파일에서 계산한 뒤 V4 fixture bundle을 만든다.

~~~powershell
$hash = (Get-FileHash rulesets/augment_chess_20260915/manifest.json -Algorithm SHA256).Hash.ToLower()
cd train
uv run python -m v4.export_fixture_bundle --source-fixture v3/fixtures/0915/observation_action_v3.json --runtime-vocabulary v3/fixtures/0915/runtime_vocabulary_v3.json --bundle-dir v3/fixtures/v4_bundle_0915 --ruleset-manifest-sha256 $hash
cd ..
~~~

release benchmark를 빌드하고 ONNX Runtime DLL을 지정한다.

~~~powershell
cargo build --release -p augment-cli --bin benchmark_v4_search
$env:ORT_DYLIB_PATH = "D:/augmentia/train/.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
~~~

core sealed fixtures:

~~~powershell
target/release/benchmark_v4_search.exe --ruleset 20260915 --ruleset-hash $hash --evaluator deterministic --fixture standard --simulations 200 --iterations 8
target/release/benchmark_v4_search.exe --ruleset 20260915 --ruleset-hash $hash --evaluator onnx --bundle train/v3/fixtures/v4_bundle_0915/v4-model.v4-bundle.metadata.json --fixture standard --simulations 200 --iterations 8 --intra-threads 1
target/release/benchmark_v4_search.exe --ruleset 20260915 --ruleset-hash $hash --evaluator deterministic --fixture draft --simulations 200 --iterations 8
target/release/benchmark_v4_search.exe --ruleset 20260915 --ruleset-hash $hash --evaluator onnx --bundle train/v3/fixtures/v4_bundle_0915/v4-model.v4-bundle.metadata.json --fixture draft --simulations 200 --iterations 8 --intra-threads 1
~~~

보조 stress fixture:

~~~powershell
target/release/benchmark_v4_search.exe --ruleset 20260915 --ruleset-hash $hash --evaluator onnx --bundle train/v3/fixtures/v4_bundle_0915/v4-model.v4-bundle.metadata.json --fixture wide_chameleon --simulations 200 --iterations 8 --intra-threads 1
~~~

### 봉인 결과

| fixture | evaluator | root legal | sims/s | mean ms | mean scored prefix rows | mean engine apply | max chance outcomes | depth | root coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | deterministic | 20 | 3,175.74 | 62.98 | 12,197 | 200 | 0 | 4 | 0.800 |
| standard | ONNX | 20 | **678.68** | **294.69** | **260** | 200 | 0 | 5 | 0.800 |
| draft | deterministic | 3 | 2,725.23 | 73.39 | 10,941 | 263 | 96 | 4 | 1.000 |
| draft | ONNX | 3 | **702.00** | **284.90** | **205** | 263 | 96 | 4 | 1.000 |
| wide_chameleon | deterministic | 484 | 5,110.99 | 39.13 | 4,962 | 200 | 0 | 4 | 0.0331 |
| wide_chameleon | ONNX | 484 | **673.66** | **296.88** | **677** | 200 | 0 | 5 | 0.0331 |

모든 6개 조합에서 repeat_top_action_equal = true, repeat_total_visit_delta = 0이었다.
evaluator 호출은 평균 201회 / 200 simulations였다.

방문 분포 SHA-256:

| fixture | evaluator | visit distribution SHA-256 |
|---|---|---|
| standard | deterministic | 9913e8256d71488efb8941a86778b8aad63575322335704d1796ee0d3792c24e |
| standard | ONNX | 241ed80802159199dfc3a24b39a6269ae0ed3c74166191b9eba491ddcafa9334 |
| draft | deterministic | 782482027031d02fa1b48c665612ca1587fe39835ece79a9b49b11ce19a488ff |
| draft | ONNX | dad65cab3e33ba8a0543c75205ef3fb444d664fa73c413e3657a60eeb505ce0c |
| wide_chameleon | deterministic | 6767738d1b8a9865fdc714acec1f66fd101ef5ba06bf5e7f56be2552231ee14b |
| wide_chameleon | ONNX | a4c1716287518a412c978a29c1eb03d69590826504680df1c2b1aa324c1a2907 |

### 해석

현재 ONNX search는 200 simulations 기준 이 머신에서 standard/draft/wide fixture 모두 약 **674~702 simulations/s**, 평균 **285~297 ms/search** 범위다.

lazy prefix scoring은 실제 ONNX scorer row 수를 크게 줄인다. 동일 search diagnostics의 eager complete-policy row 집계와 비교하면:

- standard: 12,197 -> 260, 약 **97.9% 감소**
- draft: 10,941 -> 205, 약 **98.1% 감소**
- wide_chameleon: 4,962 -> 677, 약 **86.4% 감소**

wide_chameleon은 root legal action이 484개라 200 simulations로 전체 root를 방문할 수 없다.
현재 16개 Gumbel 후보만 실제 방문하므로 root coverage는 약 3.31%다.
이 fixture는 정책 품질 기준이 아니라 넓은 factorized action space의 성능/stability stress 기준이다.

아래 0818 최적화 직전 ONNX 수치(standard 362.18, draft 362.95 sims/s)보다 현재 수치는 크게 높지만, **ruleset, fixture vocabulary, graph, search 코드가 동시에 달라졌으므로 직접적인 lazy-only speedup 수치로 해석하면 안 된다.** 현재 수치는 0915 이후 변경의 새 기준점이다.

## 역사 기록: 0818 최초 baseline

최초 baseline은 2026-09-18에 augment_chess_20260818로 봉인했다.

- base commit: 118660f
- 모델: train/v3/fixtures/v4_bundle/ synthetic fixture bundle
- simulations 200
- chance_outcome_samples 8
- gumbel_candidates 16
- 동일 search/gumbel seeds
- CPU, intra_threads 1
- fixture당 8회 + warm repeat

최초 측정:

| fixture | evaluator | sims/s | mean ms | 재실행 |
|---|---|---:|---:|---|
| standard | deterministic | 3,426 | 58.4 | visit delta 0 |
| standard | ONNX | 334 | 598.3 | visit delta 0 |
| draft | deterministic | 3,597 | 55.6 | visit delta 0 |
| draft | ONNX | 397 | 503.8 | visit delta 0 |

lazy scoring 적용 직전 동일 checkout 재측정:

| fixture | evaluator | sims/s | mean ms | 분포 hash |
|---|---|---:|---:|---|
| standard | ONNX | 362.18 | 552.21 | e634e2bd... |
| draft | ONNX | 362.95 | 551.04 | 95d53250... |

당시 구현은 매 시뮬레이션마다 모든 합법 trie prefix를 평가했고, 이 측정이 lazy prefix scoring 최적화의 출발점이었다.

## 아직 하지 않은 것

- V3 대비 paired arena 우위 검증
- 학습된 실전 V4 모델을 사용한 latency/strength baseline
- Hybrid encoder의 학습 후 baseline 대비 A/B
- NN-only exact/beam fast path
- search-risk predictor / adaptive compute controller
