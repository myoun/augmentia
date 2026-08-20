use augment_ai::{DatasetManifestV3, validate_manifest, verify_manifest_replays};
use augment_core::Engine;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("V3 dataset validation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = manifest_path(std::env::args().skip(1))?;
    let bytes = std::fs::read(&manifest_path)?;
    let manifest: DatasetManifestV3 = serde_json::from_slice(&bytes)?;
    validate_manifest(&manifest)?;
    let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let engine = Engine::bundled_20260807()?;
    verify_manifest_replays(base_dir, &manifest, &engine)?;
    println!(
        "VALID dataset={} shards={} records={} games={} decisions={} game_steps={}",
        manifest.dataset_id,
        manifest.totals.shards,
        manifest.totals.records,
        manifest.totals.games,
        manifest.totals.decisions,
        manifest.totals.game_steps,
    );
    Ok(())
}

fn manifest_path(args: impl Iterator<Item = String>) -> Result<PathBuf, String> {
    let args = args.collect::<Vec<_>>();
    let value = args
        .windows(2)
        .find(|pair| pair[0] == "--manifest")
        .map(|pair| pair[1].as_str())
        .ok_or_else(|| {
            "usage: validate_v3_dataset --manifest <dataset/manifest.json>".to_owned()
        })?;
    Ok(PathBuf::from(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_argument_is_required() {
        assert_eq!(
            manifest_path(Vec::<String>::new().into_iter()).unwrap_err(),
            "usage: validate_v3_dataset --manifest <dataset/manifest.json>"
        );
        assert_eq!(
            manifest_path(["--manifest".to_owned(), "data/manifest.json".to_owned()].into_iter())
                .unwrap(),
            PathBuf::from("data/manifest.json")
        );
    }
}
