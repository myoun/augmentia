use augment_ai::V3OnnxRuntime;
use augment_core::{BUNDLED_RULESET_ID, Engine, GamePhase, GameState, Mode};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize)]
struct BatchReport {
    batch_size: usize,
    p50_ms: f64,
    p95_ms: f64,
    positions_per_second: f64,
}

#[derive(Serialize)]
struct Report {
    value: f32,
    probabilities: Vec<f32>,
    session_instances: usize,
    session_runs: u64,
    batches: Vec<BatchReport>,
    soak_iterations: usize,
    working_set_before_soak_bytes: usize,
    working_set_after_soak_bytes: usize,
    working_set_growth_bytes: isize,
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let model = required_path(&args, "--model");
    let metadata = required_path(&args, "--metadata");
    let ruleset_hash = required(&args, "--ruleset-hash");
    let iterations = optional_usize(&args, "--iterations", 100);
    let soak = optional_usize(&args, "--soak", 1_000);
    let engine = Engine::bundled_20260807().expect("load bundled engine");
    let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 2026082102);
    state.phase = GamePhase::Playing;
    let runtime = V3OnnxRuntime::from_files(model, metadata, BUNDLED_RULESET_ID, &ruleset_hash)
        .expect("load strict V3 runtime");
    let first = runtime.evaluate(&state, &engine).expect("evaluate state");
    let mut reports = Vec::new();
    for batch_size in [1, 8, 32] {
        let states = vec![state.clone(); batch_size];
        for _ in 0..10 {
            runtime.evaluate_batch(&states, &engine).expect("warmup");
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let started = Instant::now();
            runtime.evaluate_batch(&states, &engine).expect("benchmark");
            samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        samples.sort_by(f64::total_cmp);
        let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
        reports.push(BatchReport {
            batch_size,
            p50_ms: percentile(&samples, 0.50),
            p95_ms: percentile(&samples, 0.95),
            positions_per_second: batch_size as f64 / (mean_ms / 1_000.0),
        });
    }
    let memory_before = working_set_bytes();
    for _ in 0..soak {
        runtime.evaluate(&state, &engine).expect("soak inference");
    }
    let memory_after = working_set_bytes();
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            value: first.value,
            probabilities: first.priors.into_iter().map(|item| item.2).collect(),
            session_instances: 1,
            session_runs: runtime.session_runs(),
            batches: reports,
            soak_iterations: soak,
            working_set_before_soak_bytes: memory_before,
            working_set_after_soak_bytes: memory_after,
            working_set_growth_bytes: memory_after as isize - memory_before as isize,
        })
        .unwrap()
    );
}

#[cfg(windows)]
fn working_set_bytes() -> usize {
    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCounters {
        cb: size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    // SAFETY: both functions are process-local Win32 APIs and the initialized structure size
    // is passed exactly as required by GetProcessMemoryInfo.
    let succeeded = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            size_of::<ProcessMemoryCounters>() as u32,
        )
    };
    if succeeded == 0 {
        0
    } else {
        counters.working_set_size
    }
}

#[cfg(not(windows))]
fn working_set_bytes() -> usize {
    0
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * percentile).round() as usize]
}

fn required(args: &[String], flag: &str) -> String {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| panic!("{flag} is required"))
}

fn required_path(args: &[String], flag: &str) -> PathBuf {
    required(args, flag).into()
}

fn optional_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect("numeric argument"))
        .unwrap_or(default)
}
