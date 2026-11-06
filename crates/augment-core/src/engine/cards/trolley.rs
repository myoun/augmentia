use super::*;
use std::collections::BTreeMap;
pub(super) const ID: &str = crate::TROLLEY;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

struct TrolleyRandom(u32);
impl TrolleyRandom {
    fn next(&mut self, max: usize) -> usize {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 as usize % max
    }
    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.next(i + 1));
        }
    }
}
#[derive(Default)]
struct ScoreMap {
    order: Vec<u16>,
    buckets: BTreeMap<u16, Vec<Vec<u32>>>,
}
fn score_map(
    engine: &Engine,
    state: &GameState,
    owner: Player,
    random: &mut TrolleyRandom,
) -> ScoreMap {
    let mut pieces: Vec<_> = state
        .board
        .iter()
        .filter_map(|(_, piece)| {
            (piece.owner == owner
                && !engine.is_royal_piece(state, piece)
                && piece.kind != PieceKind::Football
                && state.card_runtime.monster_rule != Some(piece.id))
            .then_some(piece)
        })
        .collect();
    pieces.sort_by_key(|piece| piece.id.to_string());
    random.shuffle(&mut pieces);
    let entries: Vec<_> = pieces
        .into_iter()
        .filter_map(|piece| {
            let value = super::revelation::combat_value(piece.kind);
            (value > 0 && value <= 11).then_some((piece.id, value))
        })
        .take(32)
        .collect();
    fn walk(
        entries: &[(u32, u16)],
        start: usize,
        selected: &mut Vec<u32>,
        score: u16,
        result: &mut ScoreMap,
    ) {
        if !selected.is_empty() && score <= 11 {
            let bucket = result.buckets.entry(score).or_insert_with(|| {
                result.order.push(score);
                Vec::new()
            });
            if bucket.len() < 240 {
                bucket.push(selected.clone());
            }
        }
        if selected.len() >= 4 {
            return;
        }
        for i in start..entries.len() {
            let (id, value) = entries[i];
            if score + value <= 11 {
                selected.push(id);
                walk(entries, i + 1, selected, score + value, result);
                selected.pop();
            }
        }
    }
    let mut result = ScoreMap::default();
    walk(&entries, 0, &mut Vec::new(), 0, &mut result);
    result
}
fn distance(value: u16, low: u16, high: u16) -> u16 {
    low.saturating_sub(value).max(value.saturating_sub(high))
}
fn disjoint_pairs(map: &ScoreMap, low: u16, high: u16) -> Vec<[Vec<u32>; 2]> {
    let mut result = Vec::new();
    let Some(left) = map.buckets.get(&low) else {
        return result;
    };
    let Some(right) = map.buckets.get(&high) else {
        return result;
    };
    for first in left {
        for second in right {
            if first.iter().all(|id| !second.contains(id)) {
                result.push([first.clone(), second.clone()]);
            }
        }
    }
    result
}
pub(in crate::engine) fn bundles(
    engine: &Engine,
    state: &GameState,
    target: Player,
    seed: u32,
) -> Option<[Vec<u32>; 2]> {
    let mut random = TrolleyRandom(seed);
    let enemies = score_map(engine, state, target, &mut random);
    let friends = score_map(engine, state, target.opponent(), &mut random);
    let mut rankings = Vec::new();
    for &low in &enemies.order {
        if low > 5 {
            continue;
        }
        for &high in &enemies.order {
            if high < 6 || high <= low {
                continue;
            }
            for &ally in &friends.order {
                if !(2..=8).contains(&ally) {
                    continue;
                }
                let relaxation = distance(low, 3, 4) + distance(high, 8, 11) + distance(ally, 5, 7);
                let gap = i32::from(low).abs_diff(i32::from(high) - i32::from(ally));
                rankings.push((low, high, ally, relaxation, gap));
            }
        }
    }
    random.shuffle(&mut rankings);
    rankings.sort_by_key(|(_, _, _, relaxation, gap)| (*relaxation, *gap));
    for (low, high, ally, _, _) in rankings {
        let pairs = disjoint_pairs(&enemies, low, high);
        let Some(friendly) = friends.buckets.get(&ally) else {
            continue;
        };
        if pairs.is_empty() || friendly.is_empty() {
            continue;
        }
        let mut chosen = pairs[random.next(pairs.len())].clone();
        chosen[1].extend(friendly[random.next(friendly.len())].iter().copied());
        if random.next(2) == 1 {
            chosen.swap(0, 1);
        }
        return Some(chosen);
    }
    None
}
fn predicted_seed(state: &GameState) -> u32 {
    let mut probe = state.clone();
    probe
        .rng_mut()
        .web_choice_index(1_usize << 32)
        .1
        .expect("nonempty seed range") as u32
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if bundles(engine, state, card.owner.opponent(), predicted_seed(state)).is_some() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    source: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (roll, value) = state.rng_mut().web_choice_index(1_usize << 32);
    let seed = value.expect("nonempty seed range") as u32;
    if bundles(engine, state, actor.opponent(), seed).is_none() {
        return Err(EngineError::Invariant(
            "trolley has no valid bundles".into(),
        ));
    }
    events.push(Event::RandomSample {
        purpose: "trolley-seed".into(),
        roll_53: roll,
    });
    state.card_runtime.pending_trolley_seed[actor.opponent().index()] = Some((actor, source, seed));
    state.request_turn_end();
    Ok(())
}

impl Engine {
    pub(in crate::engine) fn open_current_trolley(
        &self,
        state: &mut GameState,
        player: Player,
        events: &mut Vec<Event>,
    ) -> bool {
        let Some((by, source, seed)) =
            state.card_runtime.pending_trolley_seed[player.index()].take()
        else {
            return false;
        };
        let Some(bundles) = bundles(self, state, player, seed) else {
            return false;
        };
        let labels = bundles.each_ref().map(|bundle| {
            let entries = bundle
                .iter()
                .filter_map(|id| state.board.find_by_id(*id))
                .map(|(square, piece)| {
                    format!(
                        "{} ({} 기물, {}점)",
                        square,
                        if piece.owner == player {
                            "내"
                        } else {
                            "상대"
                        },
                        super::revelation::combat_value(piece.kind)
                    )
                })
                .collect::<Vec<_>>();
            format!("제거: {}", entries.join(" + "))
        });
        state.card_runtime.active_trolley = Some(TrolleyDilemma { by, bundles });
        let choice_id = state.allocate_choice_id();
        state.phase = GamePhase::AwaitingChoice(PendingChoice {
            id: choice_id,
            owner: player,
            source_card_instance: source,
            handler_id: crate::engine::piece_effects::TROLLEY_CHOICE_HANDLER.into(),
            options: labels.to_vec(),
            continuation: ChoiceContinuation::ResumeTurn,
        });
        events.push(Event::ChoiceRequested {
            choice_id,
            owner: player,
            handler_id: crate::engine::piece_effects::TROLLEY_CHOICE_HANDLER.into(),
            options: labels.to_vec(),
        });
        true
    }
}
