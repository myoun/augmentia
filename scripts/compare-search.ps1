#!/usr/bin/env pwsh
# Before/after measurement harness for augment-search engine optimization.
#
# Takes two ALREADY-BUILT augment-search executables and measures them against
# each other, writing raw bench JSON plus summaries into
# artifacts/engine-optimization-audit (override with -OutputDirectory).
#
# Contract this harness keeps:
#   * It builds nothing, tests nothing, trains nothing, and starts no other
#     measurement. The two executables are taken as given; resolving them and
#     running them is all it does.
#   * Everything is serial. No process is started while another is running.
#   * Fixed-work runs (depth 3 under a non-binding cooperative ceiling) must be
#     semantically identical between the two binaries once timing fields are
#     removed. A mismatch throws.
#   * Wall-clock runs (250 ms / 1000 ms) are NEVER asserted equal. They are only
#     summarized statistically; node and depth counts there are search
#     statistics, not playing-strength evidence.
#   * Paired runs alternate AB / BA by run index so machine drift (thermal
#     load, background work) lands on both binaries symmetrically instead of
#     always on whichever binary happened to run second.
#
# Usage (executable paths are resolved from the current directory):
#   pwsh -File scripts/compare-search.ps1 -BeforeExecutable target/before/augment-search.exe -AfterExecutable target/after/augment-search.exe

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$BeforeExecutable,

    [Parameter(Mandatory)]
    [string]$AfterExecutable,

    # Default: <repository>/artifacts/engine-optimization-audit
    [string]$OutputDirectory = '',

    [int]$Repeats = 3
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo 'artifacts/engine-optimization-audit'
}

# ---------------------------------------------------------------------------
# Fixed measurement configuration. Changing these changes the audit contract.
# ---------------------------------------------------------------------------
$jsonDepth = 30              # deep JSON, valid on Windows PowerShell 5.1
$nodes = 100000
$decisions = 24
$fixedDepth = 3              # asserted completed_depth for the fixed workload
$fixedStopReason = 'depth_limit'
$warmupTimeMs = 30000
$warmupDepth = 2
$warmupSeed = 1
$statFields = @('nodes', 'legal_calls', 'apply_calls', 'chance_samples', 'evaluations', 'cutoffs')

$workloads = @(
    [ordered]@{
        key         = 'fixed'
        seeds       = @(1, 2, 3)
        time_ms     = 30000
        depth       = 3
        assert      = $true
        description = 'Fixed work: the cooperative 30 s ceiling does not bind at depth 3, so before and after must be semantically identical once timing is removed.'
    },
    [ordered]@{
        key         = 'timed250'
        seeds       = @(1, 2, 3)
        time_ms     = 250
        depth       = 4
        assert      = $false
        description = 'Wall-clock limited: search stops on the time budget, so results are not required to match and are only summarized.'
    },
    [ordered]@{
        key         = 'timed1000'
        seeds       = @(1)
        time_ms     = 1000
        depth       = 4
        assert      = $false
        description = 'Wall-clock limited (longer budget): summarized only.'
    }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-JsonFile {
    # -InputObject (never the pipeline) so collections are not unrolled.
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][AllowNull()]$Value
    )
    ConvertTo-Json -InputObject $Value -Depth $jsonDepth | Out-File -Encoding utf8 -FilePath $Path
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Label
    )
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw ('{0}: executable path must not be empty.' -f $Label)
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ('{0}: executable not found: {1}' -f $Label, $Path)
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-BenchArgv {
    # argv after the executable itself; identical for before and after.
    param(
        [Parameter(Mandatory)][int]$Seed,
        [Parameter(Mandatory)][int]$TimeMs,
        [Parameter(Mandatory)][int]$Depth
    )
    return @(
        'bench'
        '--seed', $Seed
        '--time-ms', $TimeMs
        '--depth', $Depth
        '--nodes', $nodes
        '--chance-samples', 4
        '--search-seed', 19
        '--decisions', $decisions
    )
}

function Invoke-BenchRun {
    # Runs one bench into a UTF-8 JSON file. Throws immediately on nonzero exit.
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$BenchArgs,
        [Parameter(Mandatory)][string]$OutputPath,
        [Parameter(Mandatory)][string]$Label
    )
    $startedUtc = (Get-Date).ToUniversalTime()
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & $Executable @BenchArgs | Out-File -Encoding utf8 -FilePath $OutputPath
    $exitCode = $LASTEXITCODE
    $watch.Stop()
    $finishedUtc = (Get-Date).ToUniversalTime()
    if ($null -eq $exitCode -or $exitCode -ne 0) {
        throw ('{0}: bench failed (exit code {1}); partial output kept at {2}' -f $Label, $exitCode, $OutputPath)
    }
    return [ordered]@{
        label         = $Label
        executable    = $Executable
        argv          = @($BenchArgs)
        file          = (Split-Path -Leaf $OutputPath)
        started_utc   = $startedUtc.ToString('o')
        finished_utc  = $finishedUtc.ToString('o')
        wall_clock_ms = [math]::Round($watch.Elapsed.TotalMilliseconds, 3)
        exit_code     = $exitCode
    }
}

function Read-BenchDocument {
    param([Parameter(Mandatory)][string]$Path)
    $raw = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw ('Empty bench output: {0}' -f $Path)
    }
    try {
        return ($raw | ConvertFrom-Json)
    } catch {
        throw ('Unparsable bench JSON: {0} :: {1}' -f $Path, $_.Exception.Message)
    }
}

function Get-PropertyOrNull {
    param($Object, [Parameter(Mandatory)][string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
    return $null
}

function Get-RequiredValue {
    # Loud on schema drift instead of silently coercing a missing field to 0.
    param(
        $Object,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Source
    )
    $value = Get-PropertyOrNull $Object $Name
    if ($null -eq $value) {
        throw ('bench JSON is missing required field "{0}" in {1}; the CLI schema changed.' -f $Name, $Source)
    }
    return $value
}

function New-RecordFingerprint {
    # Deterministic projection of one bench record: semantic fields only.
    # Timing fields (elapsed_ms, budget_overrun_ms, stats.legal_ms,
    # stats.apply_ms, stats.evaluation_ms) and the document header are
    # deliberately excluded.
    param([Parameter(Mandatory)]$Record)
    $report = (Get-PropertyOrNull $Record 'report')
    $stats = (Get-PropertyOrNull $report 'stats')
    $statFingerprint = [ordered]@{}
    foreach ($field in $statFields) {
        $statFingerprint[$field] = (Get-PropertyOrNull $stats $field)
    }
    $statFingerprint['known_omitted_target_cards'] = @(
        Get-PropertyOrNull $stats 'known_omitted_target_cards' |
            Where-Object { $null -ne $_ } |
            ForEach-Object { [string]$_ }
    )
    return [ordered]@{
        scenario            = [string](Get-PropertyOrNull $Record 'scenario')
        phase               = [string](Get-PropertyOrNull $Record 'phase')
        cards               = (Get-PropertyOrNull $Record 'cards')
        action              = (Get-PropertyOrNull $report 'action')
        score               = [double](Get-PropertyOrNull $report 'score')
        completed_depth     = (Get-PropertyOrNull $report 'completed_depth')
        root_actions        = (Get-PropertyOrNull $report 'root_actions')
        scored_root_actions = (Get-PropertyOrNull $report 'scored_root_actions')
        stop_reason         = [string](Get-PropertyOrNull $report 'stop_reason')
        stats               = $statFingerprint
    }
}

function Get-DocumentProjection {
    # scenario name -> fingerprint, plus a header fingerprint.
    # 'build' is excluded from the header: a release/debug difference is
    # surfaced as a warning, not treated as a result mismatch.
    param([Parameter(Mandatory)]$Document)
    $positions = [ordered]@{}
    foreach ($record in @(Get-PropertyOrNull $Document 'records')) {
        $fingerprint = New-RecordFingerprint -Record $record
        $positions[$fingerprint['scenario']] = $fingerprint
    }
    $header = [ordered]@{
        ruleset               = (Get-PropertyOrNull $Document 'ruleset')
        semantics             = (Get-PropertyOrNull $Document 'semantics')
        site_parity_verified = (Get-PropertyOrNull $Document 'site_parity_verified')
        seed                  = (Get-PropertyOrNull $Document 'seed')
        config                = (Get-PropertyOrNull $Document 'config')
    }
    return [ordered]@{ header = $header; positions = $positions }
}

function ConvertTo-CanonicalJson {
    param([Parameter(Mandatory)]$Value)
    return (ConvertTo-Json -InputObject $Value -Depth $jsonDepth -Compress)
}

function Assert-FixedRecord {
    # Proves the fixed-work ceiling really was non-binding.
    param(
        [Parameter(Mandatory)]$Record,
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Label
    )
    $report = (Get-PropertyOrNull $Record 'report')
    $scenario = [string](Get-PropertyOrNull $Record 'scenario')
    $depth = (Get-PropertyOrNull $report 'completed_depth')
    $stop = [string](Get-PropertyOrNull $report 'stop_reason')
    if ($null -eq $depth -or [int]$depth -ne $fixedDepth) {
        throw ('{0}: fixed-work record "{1}" in {2} has completed_depth={3}, expected {4}. The fixed workload did not run fixed work.' -f $Label, $scenario, $Source, $depth, $fixedDepth)
    }
    if ($stop -ne $fixedStopReason) {
        throw ('{0}: fixed-work record "{1}" in {2} has stop_reason="{3}", expected "{4}". The fixed workload did not run fixed work.' -f $Label, $scenario, $Source, $stop, $fixedStopReason)
    }
}

function Compare-FixedPair {
    # Throws on any semantic mismatch. Returns a structured equivalence result.
    param(
        [Parameter(Mandatory)]$BeforeDocument,
        [Parameter(Mandatory)][string]$BeforePath,
        [Parameter(Mandatory)]$AfterDocument,
        [Parameter(Mandatory)][string]$AfterPath,
        [Parameter(Mandatory)][string]$Label
    )
    $before = Get-DocumentProjection -Document $BeforeDocument
    $after = Get-DocumentProjection -Document $AfterDocument
    $mismatches = @()

    $beforeHeaderJson = ConvertTo-CanonicalJson -Value $before['header']
    $afterHeaderJson = ConvertTo-CanonicalJson -Value $after['header']
    if ($beforeHeaderJson -ne $afterHeaderJson) {
        $mismatches += ('{0}: header/ruleset/config mismatch between {1} and {2}. before={3} after={4}' -f $Label, $BeforePath, $AfterPath, $beforeHeaderJson, $afterHeaderJson)
    }

    $beforeNames = @($before['positions'].Keys)
    $afterNames = @($after['positions'].Keys)
    # Both projections are keyed by scenario, so a missing or extra position is a mismatch.
    $compared = 0
    foreach ($name in $beforeNames) {
        if (-not ($afterNames -contains $name)) {
            $mismatches += ('{0}: scenario "{1}" is missing from {2}' -f $Label, $name, $AfterPath)
            continue
        }
        $compared++
        $beforeJson = ConvertTo-CanonicalJson -Value $before['positions'][$name]
        $afterJson = ConvertTo-CanonicalJson -Value $after['positions'][$name]
        if ($beforeJson -ne $afterJson) {
            $mismatches += ('{0}: scenario "{1}" differs between {2} and {3}. before={4} after={5}' -f $Label, $name, $BeforePath, $AfterPath, $beforeJson, $afterJson)
        }
    }
    foreach ($name in $afterNames) {
        if (-not ($beforeNames -contains $name)) {
            $mismatches += ('{0}: scenario "{1}" exists only in {2}' -f $Label, $name, $AfterPath)
        }
    }
    if ($mismatches.Count -gt 0) {
        throw ('{0}: fixed-work before/after results are NOT equivalent once timing is removed.' -f $Label + [Environment]::NewLine + ($mismatches -join [Environment]::NewLine))
    }
    return [ordered]@{
        match            = $true
        records_compared = $compared
        header_match     = $true
        mismatches       = @()
    }
}

function Get-Median {
    # Every caller passes @(...)-wrapped arrays so a single sample can never
    # collapse the argument into a scalar.
    param([Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Values)
    $numbers = @($Values | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    if ($numbers.Count -eq 0) { return $null }
    $sorted = @($numbers | Sort-Object)
    $middle = [int][math]::Floor(($sorted.Count - 1) / 2)
    if ($sorted.Count % 2 -eq 1) {
        return [double]$sorted[$middle]
    }
    return (([double]$sorted[$middle] + [double]$sorted[$middle + 1]) / 2.0)
}

function Get-MeasureSummary {
    param([Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Values)
    $numbers = @($Values | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    if ($numbers.Count -eq 0) {
        return [ordered]@{ count = 0; median = $null; min = $null; max = $null; mean = $null }
    }
    $sorted = @($numbers | Sort-Object)
    $total = 0.0
    foreach ($value in $sorted) { $total += $value }
    return [ordered]@{
        count  = $sorted.Count
        median = (Get-Median -Values $sorted)
        min    = [double]$sorted[0]
        max    = [double]$sorted[$sorted.Count - 1]
        mean   = ($total / $sorted.Count)
    }
}

function New-MetricBucket {
    return @{
        samples     = 0
        wall_ms     = @()
        elapsed_ms  = @()
        overrun_ms  = @()
        nodes       = @()
        depth       = @()
        legal_calls = @()
        evaluations = @()
    }
}

function Add-RecordToBucket {
    param(
        [Parameter(Mandatory)]$Bucket,
        [Parameter(Mandatory)]$Record,
        [Parameter(Mandatory)][string]$Source
    )
    $report = (Get-PropertyOrNull $Record 'report')
    $stats = (Get-PropertyOrNull $report 'stats')
    $Bucket['samples'] = [int]$Bucket['samples'] + 1
    $Bucket['elapsed_ms'] = @($Bucket['elapsed_ms']) + @([double](Get-RequiredValue $report 'elapsed_ms' $Source))
    $Bucket['overrun_ms'] = @($Bucket['overrun_ms']) + @([double](Get-RequiredValue $report 'budget_overrun_ms' $Source))
    $Bucket['nodes'] = @($Bucket['nodes']) + @([double](Get-RequiredValue $stats 'nodes' $Source))
    $Bucket['depth'] = @($Bucket['depth']) + @([double](Get-RequiredValue $report 'completed_depth' $Source))
    $Bucket['legal_calls'] = @($Bucket['legal_calls']) + @([double](Get-RequiredValue $stats 'legal_calls' $Source))
    $Bucket['evaluations'] = @($Bucket['evaluations']) + @([double](Get-RequiredValue $stats 'evaluations' $Source))
}

function Get-BinarySummary {
    param([Parameter(Mandatory)]$Bucket)
    return [ordered]@{
        wall_clock_ms     = (Get-MeasureSummary -Values $Bucket['wall_ms'])
        report_elapsed_ms = (Get-MeasureSummary -Values $Bucket['elapsed_ms'])
        nodes             = (Get-MeasureSummary -Values $Bucket['nodes'])
        completed_depth   = (Get-MeasureSummary -Values $Bucket['depth'])
        budget_overrun_ms = (Get-MeasureSummary -Values $Bucket['overrun_ms'])
        legal_calls       = (Get-MeasureSummary -Values $Bucket['legal_calls'])
        evaluations       = (Get-MeasureSummary -Values $Bucket['evaluations'])
    }
}

function Get-PerPositionSummary {
    param([Parameter(Mandatory)][string]$Binary, [Parameter(Mandatory)]$Buckets)
    $out = @()
    foreach ($name in @($Buckets.Keys) | Sort-Object) {
        $bucket = $Buckets[$name]
        $out += [ordered]@{
            scenario          = $name
            binary            = $Binary
            samples           = $bucket['samples']
            nodes             = (Get-MeasureSummary -Values $bucket['nodes'])
            completed_depth   = (Get-MeasureSummary -Values $bucket['depth'])
            elapsed_ms        = (Get-MeasureSummary -Values $bucket['elapsed_ms'])
            budget_overrun_ms = (Get-MeasureSummary -Values $bucket['overrun_ms'])
        }
    }
    return $out
}

function Get-ToolVersion {
    param([Parameter(Mandatory)][string]$Command, [Parameter(Mandatory)][string[]]$ToolArguments)
    try {
        $output = & $Command @ToolArguments 2>&1
        if ($LASTEXITCODE -ne 0) { return ('unavailable (exit {0})' -f $LASTEXITCODE) }
        $text = ($output | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) { return 'unavailable (no output)' }
        return $text
    } catch {
        return ('unavailable ({0})' -f $_.Exception.Message)
    }
}

function Get-UtcStamp {
    return (Get-Date).ToUniversalTime().ToString('o')
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if ($Repeats -lt 1) {
    throw ('Repeats must be greater than 0; got {0}.' -f $Repeats)
}

$beforeExe = Resolve-Executable -Path $BeforeExecutable -Label 'BeforeExecutable'
$afterExe = Resolve-Executable -Path $AfterExecutable -Label 'AfterExecutable'
if ($beforeExe -eq $afterExe) {
    throw ('BeforeExecutable and AfterExecutable resolve to the same file ({0}); two distinct builds are required.' -f $beforeExe)
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path
$comparisonPath = Join-Path $output 'comparison.json'
$environmentPath = Join-Path $output 'environment.json'

$binaries = [ordered]@{
    before = $beforeExe
    after  = $afterExe
}

$otherProcesses = @()
try {
    $otherProcesses = @(Get-Process -Name 'augment-search' -ErrorAction SilentlyContinue | ForEach-Object { '{0}(pid {1})' -f $_.ProcessName, $_.Id })
} catch {
    $otherProcesses = @()
}
if ($otherProcesses.Count -gt 0) {
    Write-Warning ('Another augment-search process is already running ({0}). Serial measurement assumes an otherwise idle machine.' -f ($otherProcesses -join ', '))
}

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
$startedAt = Get-Date
$startedUtc = Get-UtcStamp
$hashes = [ordered]@{}
foreach ($label in @('before', 'after')) {
    $path = $binaries[$label]
    $item = Get-Item -LiteralPath $path
    $hashes[$label] = [ordered]@{
        path           = $item.FullName
        sha256         = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        size_bytes     = [int64]$item.Length
        last_write_utc = $item.LastWriteTimeUtc.ToString('o')
    }
}

$processor = $null
try {
    $processor = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Select-Object -First 1
} catch {
    $processor = $null
}

$environment = [ordered]@{
    schema_version   = 1
    generator        = 'scripts/compare-search.ps1'
    started_utc      = $startedUtc
    finished_utc     = $null
    duration_seconds = $null
    repository       = $repo
    output_directory = $output
    executes         = 'bench only; this script builds nothing, tests nothing, and starts no other measurement.'
    execution_policy = 'strictly serial; one bench process at a time; paired runs alternate AB/BA to spread machine drift across both binaries.'
    binaries         = $hashes
    sha256_algorithm = 'SHA256 (Get-FileHash)'
    toolchain        = [ordered]@{
        rustc = (Get-ToolVersion -Command 'rustc' -ToolArguments @('--version'))
        node  = (Get-ToolVersion -Command 'node' -ToolArguments @('--version'))
        git   = (Get-ToolVersion -Command 'git' -ToolArguments @('-C', $repo, 'rev-parse', 'HEAD'))
    }
    powershell       = [ordered]@{
        version    = $PSVersionTable.PSVersion.ToString()
        edition    = $PSVersionTable.PSEdition
        json_depth = $jsonDepth
    }
    machine          = [ordered]@{
        cpu                = $(if ($null -ne $processor) { [string]$processor.Name } else { 'unavailable' })
        cpu_cores          = $(if ($null -ne $processor) { $processor.NumberOfCores } else { $null })
        logical_processors = $(if ($null -ne $processor) { $processor.NumberOfLogicalProcessors } else { $null })
        os                 = [string][System.Environment]::OSVersion.VersionString
        machine_name       = [string]$env:COMPUTERNAME
    }
    notes            = @(
        'Fixed-work runs must be semantically identical between binaries once timing is removed; a mismatch throws.',
        'Wall-clock runs are never asserted equal; their node and depth counts are search statistics, not playing-strength evidence.',
        'Warmup runs are discarded and never compared.',
        'Reported wall clock is the whole process: start-up and ruleset load included, not just the search.'
    )
    argv_config      = [ordered]@{
        alternation = 'run 1 = AB (before first), run 2 = BA (after first), alternating per paired run'
        nodes       = $nodes
        decisions   = $decisions
        warmup      = [ordered]@{
            note = 'one discarded run per binary, before any measurement'
            argv = (Get-BenchArgv -Seed $warmupSeed -TimeMs $warmupTimeMs -Depth $warmupDepth)
        }
        scenarios   = @(
            foreach ($workload in $workloads) {
                [ordered]@{
                    key           = $workload['key']
                    seeds         = @($workload['seeds'])
                    repeats       = $Repeats
                    paired_runs   = (@($workload['seeds']).Count * $Repeats)
                    bench_runs    = (@($workload['seeds']).Count * $Repeats * 2)
                    time_ms       = $workload['time_ms']
                    depth         = $workload['depth']
                    nodes         = $nodes
                    decisions     = $decisions
                    assert_equal  = $workload['assert']
                    argv_template = @('bench', '--seed', '<seed>', '--time-ms', '<time_ms>', '--depth', '<depth>', '--nodes', [string]$nodes, '--chance-samples', '4', '--search-seed', '19', '--decisions', [string]$decisions)
                    description   = $workload['description']
                }
            }
        )
    }
    warmups          = @()
}

# ---------------------------------------------------------------------------
# Warmup: once per binary, discarded
# ---------------------------------------------------------------------------
Write-Output ('[compare-search] output: {0}' -f $output)
Write-Output ('[compare-search] before: {0}' -f $beforeExe)
Write-Output ('[compare-search] after:  {0}' -f $afterExe)
Write-Output ('[compare-search] repeats: {0}' -f $Repeats)

foreach ($label in @('before', 'after')) {
    $warmupArgv = Get-BenchArgv -Seed $warmupSeed -TimeMs $warmupTimeMs -Depth $warmupDepth
    $warmupPath = Join-Path $output ('warmup-{0}-seed{1}.json' -f $label, $warmupSeed)
    $warmupRun = Invoke-BenchRun -Executable $binaries[$label] -BenchArgs $warmupArgv -OutputPath $warmupPath -Label ('warmup {0}' -f $label)
    $environment['warmups'] = @($environment['warmups']) + @($warmupRun)
}

# ---------------------------------------------------------------------------
# Measurement: serial, alternating AB/BA
# ---------------------------------------------------------------------------
$runs = @()
$scenarios = [ordered]@{}
$stability = [ordered]@{ checked = 0; stable = $true; warnings = @() }
$builds = [ordered]@{ before = $null; after = $null }

foreach ($workload in $workloads) {
    $key = $workload['key']
    $buckets = @{ before = (New-MetricBucket); after = (New-MetricBucket) }
    $positions = @{ before = @{}; after = @{} }
    $wallSamples = @{ before = @(); after = @() }
    $fingerprintCache = @{ before = @{}; after = @{} }
    $equivalence = [ordered]@{
        fixed_pairs_compared          = 0
        fixed_records_compared        = 0
        fingerprint_match            = $true
        semantic_equivalence         = $true
        completed_depth_equals       = $fixedDepth
        completed_depth_assert_holds = $true
        stop_reason_asserted         = $fixedStopReason
        stop_reason_assert_holds     = $true
        mismatches                    = @()
    }

    foreach ($seed in @($workload['seeds'])) {
        for ($runIndex = 1; $runIndex -le $Repeats; $runIndex++) {
            $beforeFirst = (($runIndex % 2) -eq 1)
            $order = $(if ($beforeFirst) { @('before', 'after') } else { @('after', 'before') })
            $pairDocuments = @{}

            for ($slot = 0; $slot -lt $order.Count; $slot++) {
                $label = $order[$slot]
                $fileName = '{0}-{1}-seed{2}-run{3}.json' -f $key, $label, $seed, $runIndex
                $target = Join-Path $output $fileName
                $benchArgv = Get-BenchArgv -Seed $seed -TimeMs $workload['time_ms'] -Depth $workload['depth']
                $runLabel = '{0} {1} seed {2} run {3}' -f $key, $label, $seed, $runIndex
                $run = Invoke-BenchRun -Executable $binaries[$label] -BenchArgs $benchArgv -OutputPath $target -Label $runLabel
                $run['scenario'] = $key
                $run['binary'] = $label
                $run['seed'] = $seed
                $run['run'] = $runIndex
                $run['order_in_pair'] = ($slot + 1)
                $run['order_policy'] = $(if ($beforeFirst) { 'AB' } else { 'BA' })
                $runs += $run

                $document = Read-BenchDocument -Path $target
                $builds[$label] = [string](Get-PropertyOrNull $document 'build')

                if ($workload['assert']) {
                    foreach ($record in @(Get-PropertyOrNull $document 'records')) {
                        Assert-FixedRecord -Record $record -Source $fileName -Label $runLabel
                    }
                }

                $bucket = $buckets[$label]
                $bucket['samples'] = [int]$bucket['samples'] + 1
                $bucket['wall_ms'] = @($bucket['wall_ms']) + @([double]$run['wall_clock_ms'])
                $wallSamples[$label] = @($wallSamples[$label]) + @([double]$run['wall_clock_ms'])

                $projection = Get-DocumentProjection -Document $document
                foreach ($name in @($projection['positions'].Keys)) {
                    $record = @(Get-PropertyOrNull $document 'records' | Where-Object { [string]$_.scenario -eq $name })[0]
                    Add-RecordToBucket -Bucket $bucket -Record $record -Source $fileName
                    if (-not $positions[$label].ContainsKey($name)) {
                        $positions[$label][$name] = New-MetricBucket
                    }
                    Add-RecordToBucket -Bucket $positions[$label][$name] -Record $record -Source $fileName
                }

                if ($workload['assert']) {
                    # Keyed by seed first: different seeds legitimately build
                    # different positions, only repeats of one seed must agree.
                    $seedKey = [string]$seed
                    if (-not $fingerprintCache[$label].ContainsKey($seedKey)) {
                        $fingerprintCache[$label][$seedKey] = @{}
                    }
                    $fingerprintCache[$label][$seedKey][$runIndex] = ConvertTo-CanonicalJson -Value $projection['positions']
                }

                $pairDocuments[$label] = $document
            }

            if ($workload['assert']) {
                $pairArguments = @{
                    BeforeDocument = $pairDocuments['before']
                    BeforePath     = ('{0}-before-seed{1}-run{2}.json' -f $key, $seed, $runIndex)
                    AfterDocument  = $pairDocuments['after']
                    AfterPath      = ('{0}-after-seed{1}-run{2}.json' -f $key, $seed, $runIndex)
                    Label          = ('{0} seed {1} run {2}' -f $key, $seed, $runIndex)
                }
                $pairResult = Compare-FixedPair @pairArguments
                $equivalence['fixed_pairs_compared'] = [int]$equivalence['fixed_pairs_compared'] + 1
                $equivalence['fixed_records_compared'] = [int]$equivalence['fixed_records_compared'] + [int]$pairResult['records_compared']
            }
        }
    }

    if ($workload['assert']) {
        # Same binary, same seed, different repeats must also agree; otherwise
        # the whole comparison rests on a nondeterministic engine. Reported,
        # not fatal: the before/after mismatch above is the hard failure.
        foreach ($label in @('before', 'after')) {
            foreach ($seedKey in @($fingerprintCache[$label].Keys) | Sort-Object) {
                $runMap = $fingerprintCache[$label][$seedKey]
                $runKeys = @($runMap.Keys) | Sort-Object
                if ($runKeys.Count -lt 2) { continue }
                $reference = $runMap[$runKeys[0]]
                foreach ($runKey in $runKeys) {
                    $stability['checked'] = [int]$stability['checked'] + 1
                    if ($runMap[$runKey] -ne $reference) {
                        $stability['stable'] = $false
                        $stability['warnings'] = @($stability['warnings']) + @(('{0}: {1} fixed work differs across repeats for seed {2} run {3}' -f $key, $label, $seedKey, $runKey))
                    }
                }
            }
        }
    }

    $beforeSummary = Get-BinarySummary -Bucket $buckets['before']
    $afterSummary = Get-BinarySummary -Bucket $buckets['after']
    $beforeMedian = $beforeSummary['wall_clock_ms']['median']
    $afterMedian = $afterSummary['wall_clock_ms']['median']
    $ratio = $null
    $percentFaster = $null
    if ($null -ne $beforeMedian -and $null -ne $afterMedian) {
        if ([double]$afterMedian -ne 0) { $ratio = [double]$beforeMedian / [double]$afterMedian }
        if ([double]$beforeMedian -ne 0) { $percentFaster = (([double]$beforeMedian - [double]$afterMedian) / [double]$beforeMedian) * 100.0 }
    }

    $scenarioEntry = [ordered]@{
        key                     = $key
        description             = $workload['description']
        seeds                   = @($workload['seeds'])
        repeats                 = $Repeats
        paired_runs             = (@($workload['seeds']).Count * $Repeats)
        bench_runs              = ([int]$buckets['before']['samples'] + [int]$buckets['after']['samples'])
        record_samples          = [ordered]@{ before = [int]$buckets['before']['samples']; after = [int]$buckets['after']['samples'] }
        argv_template           = @('bench', '--seed', '<seed>', '--time-ms', [string]$workload['time_ms'], '--depth', [string]$workload['depth'], '--nodes', [string]$nodes, '--chance-samples', '4', '--search-seed', '19', '--decisions', [string]$decisions)
        equal_results_asserted  = [bool]$workload['assert']
        before                  = $beforeSummary
        after                   = $afterSummary
        speedup                 = [ordered]@{
            basis                     = 'median process wall clock per bench run (process start and ruleset load included)'
            before_median_ms          = $beforeMedian
            after_median_ms           = $afterMedian
            ratio_before_over_after   = $ratio
            percent_faster            = $percentFaster
            per_run_samples_ms        = [ordered]@{ before = @($wallSamples['before']); after = @($wallSamples['after']) }
        }
        per_position             = [ordered]@{
            before = (Get-PerPositionSummary -Binary 'before' -Buckets $positions['before'])
            after  = (Get-PerPositionSummary -Binary 'after' -Buckets $positions['after'])
        }
        caveats                  = @(
            'nodes, completed_depth, legal_calls and evaluations are search statistics, not playing-strength evidence.',
            'wall-clock runs stop on the time budget, so before/after results are intentionally NOT asserted equal.'
        )
    }
    if ($workload['assert']) {
        $scenarioEntry['fixed_equivalence'] = $equivalence
    }
    $scenarios[$key] = $scenarioEntry

    Write-Output ('[compare-search] scenario {0}: before median {1} ms, after median {2} ms, ratio {3}' -f $key, $beforeMedian, $afterMedian, $ratio)
}

$finishedAt = Get-Date
$finishedUtc = Get-UtcStamp
$environment['finished_utc'] = $finishedUtc
$environment['duration_seconds'] = [math]::Round(($finishedAt - $startedAt).TotalSeconds, 3)

$comparison = [ordered]@{
    schema_version     = 1
    generator          = 'scripts/compare-search.ps1'
    generated_at_utc   = $finishedUtc
    repository         = $repo
    output_directory   = $output
    before             = $hashes['before']
    after              = $hashes['after']
    repeats            = $Repeats
    build_profile      = [ordered]@{ before = $builds['before']; after = $builds['after']; warning = $null }
    measurement_policy = [ordered]@{
        execution   = 'serial; one bench process at a time'
        alternation = 'paired runs alternate AB/BA so drift is spread symmetrically'
        warmup      = 'one discarded run per binary (seed 1, depth 2) before any measurement'
        fixed_work  = 'depth 3 under a non-binding 30 s ceiling; semantic equality asserted'
        timed_work  = 'depth 4 under a binding 250 ms / 1000 ms ceiling; summarized only'
    }
    fixed              = [ordered]@{
        fingerprint_match              = $true
        semantic_equivalence           = $true
        completed_depth_equals         = $fixedDepth
        completed_depth_assert_pass    = $true
        stop_reason_asserted           = $fixedStopReason
        stop_reason_assert_pass        = $true
        self_consistent_across_repeats = [bool]$stability['stable']
        self_consistency_warnings      = @()
        compared_fields = @(
            'scenario', 'phase', 'cards', 'action', 'score', 'completed_depth',
            'root_actions', 'scored_root_actions', 'stop_reason',
            'stats.nodes', 'stats.legal_calls', 'stats.apply_calls', 'stats.chance_samples',
            'stats.evaluations', 'stats.cutoffs', 'stats.known_omitted_target_cards'
        )
        excluded_fields = @('elapsed_ms', 'budget_overrun_ms', 'stats.legal_ms', 'stats.apply_ms', 'stats.evaluation_ms')
    }
    scenarios          = $scenarios
    runs               = $runs
    caveats            = @(
        'This harness measures search work only; it is not a playing-strength or match result.',
        'Node and depth counts under a time budget are search statistics, not evidence of stronger play.',
        'Wall clock includes process start and ruleset load, not just the search.',
        'before/after runs alternate AB/BA, but on a busy machine the medians still carry real noise.'
    )
}
$comparison['fixed']['self_consistency_warnings'] = @($stability['warnings'])

if ([string]$builds['before'] -ne [string]$builds['after']) {
    $warning = ('before is "{0}" and after is "{1}"; a build-profile difference confounds the timing comparison.' -f $builds['before'], $builds['after'])
    $comparison['build_profile']['warning'] = $warning
    $comparison['caveats'] = @($comparison['caveats']) + @($warning)
    Write-Warning $warning
}

Write-JsonFile -Path $comparisonPath -Value $comparison
Write-JsonFile -Path $environmentPath -Value $environment

Write-Output ('[compare-search] comparison: {0}' -f $comparisonPath)
Write-Output ('[compare-search] environment: {0}' -f $environmentPath)
Write-Output ('[compare-search] fixed-work before/after equivalence: PASS')
