//! Independent verification of a policy against recorded teacher decisions.
//!
//! Two separate questions, kept apart on purpose:
//!
//! 1. **Does the policy reproduce the teacher?** Measured on held-out records the policy never saw.
//!    This is about imitation only.
//! 2. **Does it add search strength?** Measured by the arena, search with the policy's guidance
//!    against the same search without it. A policy can imitate well and still not help.
//!
//! A single blended number would hide the difference, so this module reports both and never folds
//! them together.

use serde::{Deserialize, Serialize};

use crate::features::{ACTION_FEATURES, POSITION_FEATURES};

/// A recorded teacher decision, as read back from a JSONL file.
///
/// Only the fields verification needs are required; unknown fields are ignored so records can grow
/// without breaking older verifiers.
#[derive(Clone, Debug, Deserialize)]
pub struct LearnedRecord {
    pub provenance: RecordProvenance,
    pub position: Vec<f32>,
    pub candidates: Vec<LearnedCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RecordProvenance {
    pub state_fingerprint: u64,
    pub engine: RecordEngine,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RecordEngine {
    pub ruleset_id: String,
    pub catalog_hash: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LearnedCandidate {
    pub index: usize,
    pub chosen: bool,
    pub features: Vec<f32>,
}

impl LearnedRecord {
    /// Whether this record's feature vectors have the expected shape.
    pub fn well_formed(&self) -> bool {
        self.position.len() == POSITION_FEATURES
            && !self.candidates.is_empty()
            && self
                .candidates
                .iter()
                .all(|candidate| candidate.features.len() == ACTION_FEATURES)
            && self
                .candidates
                .iter()
                .filter(|candidate| candidate.chosen)
                .count()
                == 1
    }

    /// The teacher's choice, if the record marks exactly one.
    pub fn teacher_choice(&self) -> Option<&LearnedCandidate> {
        self.candidates.iter().find(|candidate| candidate.chosen)
    }
}

/// A linear policy: score = position · W · action + b · action.
///
/// Deliberately the smallest model that can express a preference. It is inspectable, its training
/// is a least-squares fit, and it needs no framework, which keeps verification reproducible.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearPolicy {
    /// `POSITION_FEATURES * ACTION_FEATURES` weights, row-major over position features.
    pub weights: Vec<f32>,
    /// Per-action-feature bias.
    pub bias: Vec<f32>,
}

impl LinearPolicy {
    /// A zero policy, which prefers the first candidate and shows no learned preference.
    pub fn zeros() -> Self {
        Self {
            weights: vec![0.0; POSITION_FEATURES * ACTION_FEATURES],
            bias: vec![0.0; ACTION_FEATURES],
        }
    }

    /// Score one candidate in one position.
    pub fn score(&self, position: &[f32], action: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for (position_index, position_value) in position.iter().enumerate().take(POSITION_FEATURES)
        {
            if *position_value == 0.0 {
                continue;
            }
            let row = position_index * ACTION_FEATURES;
            for (action_index, action_value) in action.iter().enumerate().take(ACTION_FEATURES) {
                if *action_value == 0.0 {
                    continue;
                }
                total += position_value * self.weights[row + action_index] * action_value;
            }
        }
        for (action_index, action_value) in action.iter().enumerate().take(ACTION_FEATURES) {
            total += self.bias[action_index] * action_value;
        }
        total
    }

    /// The candidate this policy prefers.
    pub fn choose<'a>(&self, record: &'a LearnedRecord) -> Option<&'a LearnedCandidate> {
        record.candidates.iter().max_by(|left, right| {
            self.score(&record.position, &left.features)
                .partial_cmp(&self.score(&record.position, &right.features))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Rank all candidates best first.
    pub fn ranking<'a>(&self, record: &'a LearnedRecord) -> Vec<&'a LearnedCandidate> {
        let mut ranked: Vec<&LearnedCandidate> = record.candidates.iter().collect();
        ranked.sort_by(|left, right| {
            self.score(&record.position, &right.features)
                .partial_cmp(&self.score(&record.position, &left.features))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }
}

/// How well a policy reproduces the teacher on a set of records.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ImitationReport {
    pub records: usize,
    /// Records where the policy's top choice is the teacher's choice.
    pub top1_matches: usize,
    /// Records where the teacher's choice is in the policy's top k.
    pub top_k_matches: usize,
    pub k: usize,
    pub top1_rate: f64,
    pub top_k_rate: f64,
    /// Records skipped because they were malformed.
    pub malformed: usize,
    /// Expected top-1 rate of a policy that picks uniformly at random, computed from the candidate
    /// counts. Without this a rate has no meaning: 0.44 is good against 0.1 and bad against 0.8.
    pub chance_top1: f64,
    /// Same for top-k.
    pub chance_top_k: f64,
}

impl ImitationReport {
    /// Whether the policy reproduced the teacher above a stated bar.
    ///
    /// A bar is required rather than a bare rate, because "beats a zero policy" is not the same
    /// claim as "reproduces the teacher".
    pub fn meets(&self, bar: f64) -> bool {
        self.records > 0 && self.top1_rate >= bar
    }
}

/// Measure a policy against held-out records.
pub fn verify_imitation(
    records: &[LearnedRecord],
    policy: &LinearPolicy,
    k: usize,
) -> ImitationReport {
    let mut report = ImitationReport {
        k,
        ..Default::default()
    };
    for record in records {
        if !record.well_formed() {
            report.malformed += 1;
            continue;
        }
        let Some(teacher) = record.teacher_choice() else {
            report.malformed += 1;
            continue;
        };
        report.records += 1;
        let ranking = policy.ranking(record);
        if ranking
            .first()
            .is_some_and(|candidate| candidate.index == teacher.index)
        {
            report.top1_matches += 1;
        }
        if ranking
            .iter()
            .take(k)
            .any(|candidate| candidate.index == teacher.index)
        {
            report.top_k_matches += 1;
        }
    }
    if report.records > 0 {
        report.top1_rate = report.top1_matches as f64 / report.records as f64;
        report.top_k_rate = report.top_k_matches as f64 / report.records as f64;
    }
    // Chance baselines are averaged over the same records, since candidate counts vary.
    let mut chance1 = 0.0f64;
    let mut chance_k = 0.0f64;
    let mut counted = 0usize;
    for record in records {
        if !record.well_formed() {
            continue;
        }
        let total = record.candidates.len() as f64;
        chance1 += 1.0 / total;
        chance_k += (k as f64 / total).min(1.0);
        counted += 1;
    }
    if counted > 0 {
        report.chance_top1 = chance1 / counted as f64;
        report.chance_top_k = chance_k / counted as f64;
    }
    report
}

/// How a policy was fitted, recorded so a run can be reproduced.
#[derive(Clone, Debug, Serialize)]
pub struct TrainingReport {
    pub records: usize,
    pub candidate_examples: usize,
    pub epochs: usize,
    /// Final mean pairwise ranking loss over the training set.
    pub final_loss: f64,
    /// Agreement with the teacher on the data it was fitted to, which is not a quality measure.
    pub train_top1_rate: f64,
}

/// Fit a linear policy by pairwise ranking on the teacher's choices.
///
/// For each record the teacher's candidate should outscore every alternative. The update is the
/// classic perceptron-style rank step on the feature difference, which needs no matrix inversion and
/// stays stable on the small, correlated feature sets here.
///
/// `epochs` bounds the fit so a run is reproducible. Training-set agreement is returned but is not
/// a quality measure: only held-out agreement is, which `verify_imitation` reports separately.
pub fn train_policy(records: &[LearnedRecord], epochs: usize) -> (LinearPolicy, TrainingReport) {
    let mut policy = LinearPolicy::zeros();
    let mut report = TrainingReport {
        records: 0,
        candidate_examples: 0,
        epochs,
        final_loss: 0.0,
        train_top1_rate: 0.0,
    };
    let usable: Vec<&LearnedRecord> = records
        .iter()
        .filter(|record| record.well_formed())
        .collect();
    report.records = usable.len();
    report.candidate_examples = usable.iter().map(|record| record.candidates.len()).sum();

    // A fixed step keeps the fit deterministic; the features are already normalised to [0, 1].
    const STEP: f32 = 0.05;
    for _ in 0..epochs {
        let mut loss = 0.0f64;
        for record in &usable {
            let Some(teacher) = record.teacher_choice() else {
                continue;
            };
            for alternative in &record.candidates {
                if alternative.index == teacher.index {
                    continue;
                }
                let margin = policy.score(&record.position, &teacher.features)
                    - policy.score(&record.position, &alternative.features);
                // Hinge loss: no penalty once the teacher is ahead by a margin.
                if margin > 0.0 {
                    continue;
                }
                loss += f64::from(-margin);
                add_rank_step(
                    &mut policy,
                    &record.position,
                    &teacher.features,
                    &alternative.features,
                    STEP,
                );
            }
        }
        report.final_loss = loss / report.candidate_examples.max(1) as f64;
    }

    let fit = verify_imitation(
        &usable
            .iter()
            .map(|record| (*record).clone())
            .collect::<Vec<_>>(),
        &policy,
        1,
    );
    report.train_top1_rate = fit.top1_rate;
    (policy, report)
}

/// Move weights so `preferred` outscores `other` under this position.
fn add_rank_step(
    policy: &mut LinearPolicy,
    position: &[f32],
    preferred: &[f32],
    other: &[f32],
    step: f32,
) {
    for (position_index, position_value) in position.iter().enumerate().take(POSITION_FEATURES) {
        if *position_value == 0.0 {
            continue;
        }
        let row = position_index * ACTION_FEATURES;
        for (action_index, action_value) in preferred.iter().enumerate().take(ACTION_FEATURES) {
            let delta = action_value - other.get(action_index).copied().unwrap_or(0.0);
            if delta == 0.0 {
                continue;
            }
            policy.weights[row + action_index] += step * position_value * delta;
        }
    }
    for action_index in 0..ACTION_FEATURES {
        let delta = preferred.get(action_index).copied().unwrap_or(0.0)
            - other.get(action_index).copied().unwrap_or(0.0);
        policy.bias[action_index] += step * delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(position_value: f32, candidate_scores: &[f32], chosen: usize) -> LearnedRecord {
        LearnedRecord {
            provenance: RecordProvenance {
                state_fingerprint: 1,
                engine: RecordEngine {
                    ruleset_id: "test".to_owned(),
                    catalog_hash: "test".to_owned(),
                },
            },
            position: {
                let mut position = vec![0.0; POSITION_FEATURES];
                position[0] = position_value;
                position
            },
            candidates: candidate_scores
                .iter()
                .enumerate()
                .map(|(index, value)| LearnedCandidate {
                    index,
                    chosen: index == chosen,
                    features: {
                        let mut features = vec![0.0; ACTION_FEATURES];
                        features[0] = *value;
                        features
                    },
                })
                .collect(),
        }
    }

    /// A zero policy shows no preference, so the report must say it is at chance rather than let a
    /// rate look like a result.
    #[test]
    fn a_zero_policy_is_at_chance() {
        let records = vec![
            record(1.0, &[1.0, 2.0, 3.0], 2),
            record(1.0, &[3.0, 2.0, 1.0], 0),
        ];
        let report = verify_imitation(&records, &LinearPolicy::zeros(), 1);
        assert_eq!(report.records, 2);
        assert!((report.chance_top1 - 1.0 / 3.0).abs() < 1e-6);
    }

    /// The chance baseline must follow the candidate counts: three candidates and thirty contribute
    /// very different guessing odds.
    #[test]
    fn chance_tracks_candidate_count() {
        let records = vec![record(1.0, &[1.0, 2.0, 3.0], 0), record(1.0, &[1.0; 10], 0)];
        let report = verify_imitation(&records, &LinearPolicy::zeros(), 1);
        let expected = (1.0 / 3.0 + 1.0 / 10.0) / 2.0;
        assert!((report.chance_top1 - expected).abs() < 1e-6);
    }

    /// Training must reproduce the teacher on data it can fit, which shows the fit works at all.
    /// Generalisation is a separate claim measured on held-out records.
    #[test]
    fn training_fits_a_separable_set() {
        let records = vec![
            record(1.0, &[1.0, 5.0], 1),
            record(1.0, &[1.0, 5.0], 1),
            record(1.0, &[5.0, 1.0], 0),
            record(1.0, &[5.0, 1.0], 0),
        ];
        let (policy, training) = train_policy(&records, 20);
        assert!(training.train_top1_rate > 0.9, "{training:?}");
        let report = verify_imitation(&records, &policy, 1);
        assert!(report.top1_rate > 0.9, "{report:?}");
    }

    /// A malformed record must be counted and skipped, not silently scored. A record with two
    /// choices, or with the wrong feature length, is not a usable example.
    #[test]
    fn malformed_records_are_reported_not_scored() {
        let good = record(1.0, &[1.0, 2.0], 1);
        let mut two_chosen = record(1.0, &[1.0, 2.0], 1);
        two_chosen.candidates[0].chosen = true;
        let mut wrong_length = record(1.0, &[1.0, 2.0], 1);
        wrong_length.position.truncate(3);

        let report = verify_imitation(&[good, two_chosen, wrong_length], &LinearPolicy::zeros(), 1);
        assert_eq!(report.records, 1, "only the well-formed record is scored");
        assert_eq!(report.malformed, 2);
    }

    /// A fitted policy must beat chance clearly, and the report must make that comparison possible.
    #[test]
    fn a_fitted_policy_beats_chance_on_separable_data() {
        let mut records = Vec::new();
        for _ in 0..20 {
            records.push(record(1.0, &[1.0, 9.0], 1));
            records.push(record(1.0, &[9.0, 1.0], 0));
        }
        let (policy, _) = train_policy(&records, 20);
        let report = verify_imitation(&records, &policy, 1);
        assert!(report.top1_rate > 0.9, "{report:?}");
        assert!(
            report.top1_rate > report.chance_top1,
            "a fitted policy must beat guessing: {report:?}"
        );
    }

    /// Training on one set and verifying on another must not inflate the result: held-out agreement
    /// is what counts, and the two are reported separately.
    #[test]
    fn held_out_agreement_is_reported_separately_from_training() {
        let mut records = Vec::new();
        for _ in 0..10 {
            records.push(record(1.0, &[1.0, 9.0], 1));
        }
        let (policy, training) = train_policy(&records, 10);
        let held = verify_imitation(&records, &policy, 1);
        assert_eq!(training.records, 10);
        assert_eq!(held.records, 10);
        // Both are high here because the data is separable and identical; the point is that they are
        // separate numbers a caller must choose between.
        assert!(training.train_top1_rate >= held.top1_rate - 1e-9);
    }
}
