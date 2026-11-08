param(
    [string]$OutputDirectory = '',
    [switch]$SkipBenchmarks
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'artifacts/engine-baseline-audit' }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$output = (Resolve-Path $OutputDirectory).Path
$manifest = Join-Path $root 'Cargo.toml'

& cargo fmt --manifest-path $manifest --all -- --check
if ($LASTEXITCODE -ne 0) { throw 'Workspace formatting failed' }
& cargo test --manifest-path $manifest --workspace
if ($LASTEXITCODE -ne 0) { throw 'Workspace tests failed' }
& cargo clippy --manifest-path $manifest --workspace --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) { throw 'Workspace clippy failed' }
& node --check (Join-Path $root 'crates/augment-app/src/web/static/app.js')
if ($LASTEXITCODE -ne 0) { throw 'Web JavaScript syntax failed' }
foreach ($testScript in @('test-acg-codec.mjs', 'test-acg-export.mjs', 'test-reference-help.mjs', 'test-workspace-ui.mjs', 'test-latest-ruleset.mjs', 'test-october-ports.mjs', 'test-opening-acquisition.mjs', 'test-target-procedure-ports.mjs', 'test-compound-pieces.mjs', 'test-delayed-ports.mjs', 'test-check-objectives.mjs', 'test-rule-ports.mjs', 'test-piece-card-ports.mjs', 'test-snapshot-transitions.mjs')) {
    & node (Join-Path $PSScriptRoot $testScript)
    if ($LASTEXITCODE -ne 0) { throw ('Offline web check failed: ' + $testScript) }
}
& cargo build --release --manifest-path $manifest -p augment-search
if ($LASTEXITCODE -ne 0) { throw 'Search release build failed' }
$exe = Join-Path $root 'target/release/augment-search.exe'
if ($env:CARGO_TARGET_DIR) { $exe = Join-Path $env:CARGO_TARGET_DIR 'release/augment-search.exe' }
if (-not (Test-Path $exe)) { throw ('Release executable not found: {0}' -f $exe) }

& $exe tactics --depth 2 --nodes 100000 | Out-File -Encoding utf8 (Join-Path $output 'tactics.json')
if ($LASTEXITCODE -ne 0) { throw 'Tactical diagnostics failed' }
$tactics = Get-Content -Raw (Join-Path $output 'tactics.json') | ConvertFrom-Json
if ($tactics.reports.Count -ne 2 -or @($tactics.reports | Where-Object { $_.status -ne 'win-in-enumerated-tree' }).Count -ne 0) {
    throw 'Expected tactical fixture wins were not established'
}

& $exe cannon-fixtures | Out-File -Encoding utf8 (Join-Path $output 'cannon-fixtures.json')
if ($LASTEXITCODE -ne 0) { throw 'Cannon fixture export failed' }
& node (Join-Path $PSScriptRoot 'check-cannon-oracle.mjs') (Join-Path $output 'cannon-fixtures.json') |
    Out-File -Encoding utf8 (Join-Path $output 'cannon-oracle.json')
if ($LASTEXITCODE -ne 0) { throw 'Original-source cannon differential smoke failed' }

# Execute measured workloads serially; do not run concurrent training/benchmarks.
if (-not $SkipBenchmarks) {
    foreach ($seed in 1, 2, 3) {
        & $exe bench --seed $seed --time-ms 250 --depth 4 --decisions 24 |
            Out-File -Encoding utf8 (Join-Path $output ('bench-250ms-seed{0}.json' -f $seed))
        if ($LASTEXITCODE -ne 0) { throw ('Bench seed {0} failed' -f $seed) }
    }
    & $exe bench --seed 1 --time-ms 1000 --depth 4 --decisions 24 |
        Out-File -Encoding utf8 (Join-Path $output 'bench-1000ms-seed1.json')
    if ($LASTEXITCODE -ne 0) { throw '1000 ms bench failed' }
    foreach ($seed in 1, 2, 3) {
        & $exe play --seed $seed --time-ms 250 --depth 4 --decisions 32 |
            Out-File -Encoding utf8 (Join-Path $output ('play-250ms-seed{0}.json' -f $seed))
        if ($LASTEXITCODE -ne 0) { throw ('Short play seed {0} failed' -f $seed) }
    }
}
$commit = & git -C $root rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Git provenance query failed' }
$rustVersion = & rustc --version
if ($LASTEXITCODE -ne 0) { throw 'Rust version query failed' }
$nodeVersion = & node --version
if ($LASTEXITCODE -ne 0) { throw 'Node version query failed' }
$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
@{
    baseline_commit = $commit; working_tree_changes = $true; measured_at_utc = (Get-Date).ToUniversalTime().ToString('o');
    rust = $rustVersion; node = $nodeVersion; cpu = $processor.Name; logical_processors = $processor.NumberOfLogicalProcessors;
    execution = 'serial local workload smoke; single sample per position; not playing strength';
    benchmarks_skipped = [bool]$SkipBenchmarks; ruleset = 'augment_chess_20261017'; diagnostic_ruleset = 'augment_chess_20260915'; site_parity_verified = $false
} | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $output 'environment.json')
Write-Output ('Search verification artifacts: {0}' -f $output)
