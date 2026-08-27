#!/usr/bin/env pwsh
# Mandatory pre-commit gate for the V3 Rust/Python model contract.
#
# 1. Regenerate the golden fixture and runtime vocabulary from the current
#    engine, then fail if either checked-in artifact drifted. A ruleset or
#    encoder change without fixture regeneration cannot pass this gate.
# 2. Run every Rust test (engine, action codec, runtime metadata checks).
# 3. Run every Python V3 test (schema parity, tensorizer, trainer, ONNX).

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo "train\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "train venv python not found at $python; create train\.venv first"
}

$fixtureDir = Join-Path $repo "train\v3\fixtures"
$fixtureHash = {
    Get-ChildItem $fixtureDir -File | Get-FileHash -Algorithm SHA256 |
        Sort-Object Path | ForEach-Object { "$($_.Path)=$($_.Hash)" }
}
$beforeHash = & $fixtureHash

cargo run -q -p augment-cli --bin export_v3_fixtures | Out-Null
$afterHash = & $fixtureHash
if (Compare-Object $beforeHash $afterHash) {
    throw (@(
        "V3 fixtures changed during regeneration; review and commit them:";
        "Run 'cargo run -p augment-cli --bin export_v3_fixtures', review, and commit them."
    ) -join [Environment]::NewLine)
}

cargo fmt --all -- --check
cargo test --workspace

$env:PYTHONPATH = "$repo\train"
& $python -m unittest discover -s train\v3 -t train -p "test_*.py"
