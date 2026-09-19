use augment_ai::{LegalActionSetV3, ObservationV3, V4OnnxRuntime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const PARITY_SCHEMA_ID: &str = "augmentia.v4-trained-bundle-parity.v1";
const RULESET_ID: &str = "augment_chess_20260915";

#[derive(Debug, Deserialize)]
struct ParitySample {
    schema_id: String,
    ruleset_id: String,
    ruleset_manifest_sha256: String,
    checkpoint_sha256: String,
    bundle_metadata_sha256: String,
    observation: ObservationV3,
    legal_actions: LegalActionSetV3,
    expected_value: f32,
    expected_action_logits: Vec<f32>,
    complete_action_probability_sum: f64,
}

#[derive(Debug, Serialize)]
struct RustParityReport {
    schema_id: &'static str,
    ruleset_id: String,
    checkpoint_sha256: String,
    bundle_metadata_sha256: String,
    value_max_abs_error: f32,
    action_logits_max_abs_error: f32,
    complete_action_probability_sum: f32,
    expected_complete_action_probability_sum: f64,
    action_count: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V4 Rust parity failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let bundle = required_path(&args, "--bundle")?;
    let parity_path = required_path(&args, "--parity")?;
    let tolerance = numeric(&args, "--tolerance", 2e-4f32)?;

    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err("--tolerance must be finite and positive".into());
    }
    let parity: ParitySample = serde_json::from_slice(&std::fs::read(&parity_path)?)?;
    if parity.schema_id != PARITY_SCHEMA_ID || parity.ruleset_id != RULESET_ID {
        return Err("parity sample identity mismatch".into());
    }
    if parity.expected_action_logits.len() != parity.legal_actions.len() {
        return Err("parity expected action logits do not align with legal actions".into());
    }
    let actual_bundle_hash = sha256_file(&bundle)?;
    if actual_bundle_hash != parity.bundle_metadata_sha256 {
        return Err("bundle metadata SHA-256 differs from parity sample".into());
    }

    let runtime = V4OnnxRuntime::from_bundle(&bundle, RULESET_ID, &parity.ruleset_manifest_sha256)?;
    let evaluation =
        runtime.evaluate_observation_factorized(&parity.observation, &parity.legal_actions)?;
    let value_error = (evaluation.value - parity.expected_value).abs();
    let action_error = evaluation
        .action_logits
        .iter()
        .zip(&parity.expected_action_logits)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let probability_sum = evaluation
        .action_logits
        .iter()
        .map(|logit| logit.exp())
        .sum::<f32>();

    if value_error > tolerance {
        return Err(
            format!("value parity error {value_error} exceeds tolerance {tolerance}").into(),
        );
    }
    if action_error > tolerance {
        return Err(format!(
            "action-logit parity error {action_error} exceeds tolerance {tolerance}"
        )
        .into());
    }
    if (probability_sum - 1.0).abs() > tolerance {
        return Err(
            format!("complete-action probability sum {probability_sum} differs from 1").into(),
        );
    }

    let report = RustParityReport {
        schema_id: "augmentia.v4-rust-parity.v1",
        ruleset_id: parity.ruleset_id,
        checkpoint_sha256: parity.checkpoint_sha256,
        bundle_metadata_sha256: parity.bundle_metadata_sha256,
        value_max_abs_error: value_error,
        action_logits_max_abs_error: action_error,
        complete_action_probability_sum: probability_sum,
        expected_complete_action_probability_sum: parity.complete_action_probability_sum,
        action_count: evaluation.action_logits.len(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn required(args: &[String], name: &str) -> Result<String, String> {
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .ok_or_else(|| format!("{name} is required"))
}

fn required_path(args: &[String], name: &str) -> Result<PathBuf, String> {
    required(args, name).map(PathBuf::from)
}

fn numeric<T>(args: &[String], name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .map_or(Ok(default), |value| {
            value.parse().map_err(|error| format!("{name}: {error}"))
        })
}
