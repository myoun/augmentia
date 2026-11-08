use augment_core::{Action, Engine, GameState, GameStatus};
use augment_search::{DepthUnit, Evaluation, SearchConfig, material_score, search_with_cancel};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use super::view::{action_view, state_view};

pub const DEFAULT_SEED: u64 = 20_260_807;

#[derive(Clone)]
struct PositionNode {
    state: GameState,
    parent: Option<usize>,
    action: Option<Action>,
    label: String,
    events: Vec<String>,
}

#[derive(Clone)]
pub struct GameSession {
    engine: Arc<Engine>,
    cancellation: Arc<AtomicBool>,
    nodes: Vec<PositionNode>,
    cursor: usize,
    seed: u64,
}

impl GameSession {
    pub fn new(seed: u64) -> Result<Self, String> {
        let engine = Engine::bundled_latest()
            .map_err(|e| e.to_string())?
            .with_checked_draft_eligibility();
        let mut session = Self {
            engine: Arc::new(engine),
            cancellation: Arc::new(AtomicBool::new(false)),
            nodes: Vec::new(),
            cursor: 0,
            seed,
        };
        session.reset(seed)?;
        Ok(session)
    }

    pub fn stop_analysis(&self) {
        self.cancellation.store(true, Ordering::Relaxed);
    }

    pub fn analysis_snapshot(&mut self) -> Self {
        self.stop_analysis();
        self.cancellation = Arc::new(AtomicBool::new(false));
        self.clone()
    }

    pub fn reset(&mut self, seed: u64) -> Result<(), String> {
        self.stop_analysis();
        let transition = self
            .engine
            .new_game_transition(augment_core::Mode::Standard, seed)
            .map_err(|e| e.to_string())?;
        self.nodes = vec![PositionNode {
            state: transition.state,
            parent: None,
            action: None,
            label: "시작 국면".into(),
            events: transition
                .events
                .iter()
                .filter_map(super::view::event_label)
                .collect(),
        }];
        self.cursor = 0;
        self.seed = seed;
        self.cancellation = Arc::new(AtomicBool::new(false));
        Ok(())
    }

    pub(crate) fn state(&self) -> &GameState {
        &self.nodes[self.cursor].state
    }

    fn check_fingerprint(&self, fingerprint: Option<&str>) -> Result<(), String> {
        if fingerprint.is_some_and(|f| f != format!("{:016x}", self.state().fingerprint())) {
            return Err("국면이 변경되었습니다. 최신 상태에서 다시 시도하세요.".into());
        }
        Ok(())
    }

    pub fn view(&self) -> Result<Value, String> {
        let state = self.state();
        let actions = if state.status == GameStatus::Ongoing {
            self.engine
                .legal_actions(state)
                .map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };
        let mut view = state_view(state, self.seed, &self.nodes[self.cursor].events, &actions);
        // A transparent material score, not a neural probability or a win-rate estimate.
        let score = material_score(state, augment_core::Player::White);
        view["eval"] = json!({ "score": score, "whiteRatio": 50.0 + score.clamp(-20.0, 20.0) * 2.0,
            "whitePercent": "백", "blackPercent": "흑", "unit": "material" });
        view["notation"] = json!({
            "cursor": self.cursor,
            "nodes": self.nodes.iter().enumerate().map(|(id, node)| json!({
                "id": id, "parent": node.parent, "label": node.label,
                "turn": format!("{:?}", node.state.turn), "ply": node.state.ply,
            })).collect::<Vec<_>>()
        });
        Ok(view)
    }

    pub fn export_path(&self) -> Result<Value, String> {
        let mut path = Vec::new();
        let mut next = Some(self.cursor);
        while let Some(id) = next {
            if path.len() >= 1000 {
                return Err("한 번에 최대 1,000개 국면을 내보낼 수 있습니다.".into());
            }
            path.push(id);
            next = self.nodes[id].parent;
        }
        path.reverse();
        let nodes = path.iter().map(|id| {
            let node = &self.nodes[*id]; let state = &node.state;
            let parent_state = node.parent.map(|parent| &self.nodes[parent].state).unwrap_or(state);
            let mut board = super::view::board_view(state);
            for cell in &mut board {
                if let Some(piece) = cell["piece"].as_object_mut() {
                    let id = piece["id"].as_u64().unwrap_or(0) as u32;
                    piece.insert("moved".into(), json!(state.card_runtime.moved_piece_ids.contains(&id)));
                    piece.insert("origin".into(), json!(state.card_runtime.piece_origins.get(&id).map(ToString::to_string)));
                }
            }
            let cards = |owner: augment_core::Player| state.cards.iter().filter(|c| c.owner == owner).map(|c| json!({
                "id": c.definition_id, "instanceId": format!("c{}", c.instance_id), "color": format!("{:?}", owner).to_ascii_lowercase(),
                "used": c.charges == 0 && self.engine.card_definition(&c.definition_id).is_some_and(|d| d.activation == augment_core::CardActivation::Active),
                "usesRemaining": c.charges, "passiveApplied": self.engine.card_definition(&c.definition_id).is_some_and(|d| d.activation == augment_core::CardActivation::Passive),
                "nextTurnPending": c.available_after_turn.is_some_and(|turn| state.turns_taken[owner.index()] < turn),
            })).collect::<Vec<_>>();
            let kind = match node.action.as_ref() { Some(Action::Move(_)) => "move", Some(Action::UseCard {..}) => "card", Some(Action::DraftPick{..}) => "gain", Some(Action::ResolveChoice{..}) => "effect", _ => "special" };
            json!({"id":id, "label": node.label, "kind":kind, "actor":format!("{:?}",parent_state.turn).to_ascii_lowercase(),
                "moveNumber": parent_state.ply / 2 + 1,
                "board":board, "turn":format!("{:?}",state.turn).to_ascii_lowercase(), "ply":state.ply,
                "turnsTaken":{"white":state.turns_taken[0],"black":state.turns_taken[1]}, "actionsRemaining":state.actions_remaining,
                "cards":{"white":cards(augment_core::Player::White),"black":cards(augment_core::Player::Black)},
                "rules":state.applied_rules, "winner":match state.status { GameStatus::Won { winner, .. } => Some(format!("{:?}",winner).to_ascii_lowercase()), GameStatus::Draw{..} => Some("draw".into()), GameStatus::Ongoing => None }})
        }).collect::<Vec<_>>();
        Ok(json!({"fingerprint":format!("{:016x}",self.state().fingerprint()), "nodes":nodes}))
    }

    pub fn setup_options(&self, owner: augment_core::Player) -> Value {
        let state = self.state();
        let phase_ready = matches!(state.phase, augment_core::GamePhase::Playing)
            && state.status == GameStatus::Ongoing;
        json!({"fingerprint": format!("{:016x}", state.fingerprint()), "node": self.cursor,
            "cards": augment_core::latest_catalog().iter().map(|definition| {
                let eligibility = augment_core::check_card_eligibility(state, &definition.id, owner, augment_core::AcquisitionContext::Setup);
                let mut card = super::view::card_meta(&definition.id);
                card["eligible"] = json!(phase_ready && eligibility.is_eligible());
                card["reason"] = json!(if !phase_ready { "드래프트/선택을 완료하거나 보드 편집으로 새 국면을 만드세요." } else { eligibility.reason().map_or("", |r| r.message_ko.as_str()) });
                card["reasonCode"] = json!(eligibility.reason().map(|r| format!("{:?}", r.code)));
                card
            }).collect::<Vec<_>>() })
    }

    pub fn acquire_setup(&mut self, payload: &Value) -> Result<Value, String> {
        self.check_fingerprint(payload["fingerprint"].as_str())?;
        let owner = match payload["owner"].as_str() {
            Some("white") => augment_core::Player::White,
            Some("black") => augment_core::Player::Black,
            _ => return Err("카드 소유자를 선택하세요.".into()),
        };
        let id = payload["cardId"].as_str().ok_or("cardId가 필요합니다.")?;
        let transition = self
            .engine
            .acquire_for_setup(self.state(), id, owner)
            .map_err(|e| e.to_string())?;
        self.engine
            .legal_actions(&transition.state)
            .map_err(|e| e.to_string())?;
        self.stop_analysis();
        let name = self
            .engine
            .card_definition(augment_core::canonical_card_id(id))
            .map_or(id, |d| d.name_ko);
        self.nodes.push(PositionNode {
            state: transition.state,
            parent: Some(self.cursor),
            action: None,
            label: format!("수동 설정 · {owner:?} +{name}"),
            events: transition
                .events
                .iter()
                .filter_map(super::view::event_label)
                .collect(),
        });
        self.cursor = self.nodes.len() - 1;
        self.view()
    }

    pub fn edit_board(&mut self, payload: &Value) -> Result<Value, String> {
        self.check_fingerprint(payload["fingerprint"].as_str())?;
        let state = super::setup::edited_state(payload, &self.state().ruleset_id, self.seed)?;
        // Validate the render/action space before committing the new node.
        self.engine
            .legal_actions(&state)
            .map_err(|e| e.to_string())?;
        self.stop_analysis();
        self.nodes.push(PositionNode {
            state,
            parent: Some(self.cursor),
            action: None,
            label: "보드 편집 · 카드/룰 초기화".into(),
            events: vec!["새 국면: 카드·룰·상태·포획 이력 초기화".into()],
        });
        self.cursor = self.nodes.len() - 1;
        self.view()
    }

    pub fn navigate(&mut self, id: usize) -> Result<Value, String> {
        if id >= self.nodes.len() {
            return Err("존재하지 않는 기보 국면입니다.".into());
        }
        self.stop_analysis();
        self.cursor = id;
        self.view()
    }

    pub fn apply_index(
        &mut self,
        index: usize,
        fingerprint: Option<&str>,
    ) -> Result<Value, String> {
        self.check_fingerprint(fingerprint)?;
        self.stop_analysis();
        let actions = self
            .engine
            .legal_actions(self.state())
            .map_err(|e| e.to_string())?;
        let action = actions
            .get(index)
            .cloned()
            .ok_or("현재 존재하지 않는 행동입니다.")?;
        if let Some(id) = self
            .nodes
            .iter()
            .position(|n| n.parent == Some(self.cursor) && n.action.as_ref() == Some(&action))
        {
            return self.navigate(id);
        }
        let label = action_view(self.state(), index, &action)["label"]
            .as_str()
            .unwrap_or("행동")
            .to_owned();
        let transition = self
            .engine
            .apply(self.state(), action.clone())
            .map_err(|e| e.to_string())?;
        let mut events = self.nodes[self.cursor].events.clone();
        events.extend(
            transition
                .events
                .iter()
                .filter_map(super::view::event_label),
        );
        self.nodes.push(PositionNode {
            state: transition.state,
            parent: Some(self.cursor),
            action: Some(action),
            label,
            events,
        });
        self.cursor = self.nodes.len() - 1;
        self.view()
    }

    pub fn analyze(&self, payload: &Value) -> Result<Value, String> {
        self.check_fingerprint(payload.get("fingerprint").and_then(Value::as_str))?;
        let number = |key: &str, default: u64, min: u64, max: u64| -> Result<u64, String> {
            match payload.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .filter(|n| (min..=max).contains(n))
                    .ok_or_else(|| format!("{key}: {min}~{max} 범위의 정수가 필요합니다.")),
            }
        };
        let depth_unit = match payload
            .get("depthUnit")
            .and_then(Value::as_str)
            .unwrap_or("turns")
        {
            "turns" => DepthUnit::Turns,
            "actions" => DepthUnit::Actions,
            _ => return Err("깊이 단위는 turns 또는 actions여야 합니다.".into()),
        };
        let evaluation = match payload
            .get("evaluation")
            .and_then(Value::as_str)
            .unwrap_or("material")
        {
            "material" => Evaluation::Material,
            "positional" => Evaluation::Positional,
            _ => return Err("평가는 material 또는 positional이어야 합니다.".into()),
        };
        let config = SearchConfig {
            max_depth: number("depth", 4, 1, 32)? as u8,
            time_limit: Duration::from_millis(number("timeMs", 1000, 10, 30_000)?),
            max_nodes: number("nodes", 100_000, 1, 10_000_000)?,
            chance_samples: number("chanceSamples", 4, 1, 64)? as usize,
            search_seed: number("searchSeed", 19, 0, u64::MAX)?,
            depth_unit,
            evaluation,
        };
        let state = self.state();
        let report = search_with_cancel(&self.engine, state, config, Some(&self.cancellation))
            .map_err(|e| e.to_string())?;
        let actions = self
            .engine
            .legal_actions(state)
            .map_err(|e| e.to_string())?;
        let recommendation = report.selected_action.as_ref().and_then(|selected| {
            actions
                .iter()
                .position(|a| a == selected)
                .map(|i| action_view(state, i, selected))
        });
        let white_score = report.score
            * if state.turn == augment_core::Player::White {
                1.0
            } else {
                -1.0
            };
        Ok(
            json!({ "fingerprint": format!("{:016x}", state.fingerprint()), "node": self.cursor,
            "report": report, "recommendation": recommendation, "whiteScore": white_score,
            "warning": "기물 가치·승패 기반 평가입니다. 승률이 아니며 카드 시너지 평가는 제한적입니다. 확률은 표본 탐색하며 일부 동적 카드 행동은 누락될 수 있습니다." }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn branches_preserve_original_and_navigation_restores_state() {
        let mut session = GameSession::new(DEFAULT_SEED).unwrap();
        let initial = session.view().unwrap();
        session.apply_index(0, None).unwrap();
        let first = session.view().unwrap();
        session.navigate(0).unwrap();
        session.apply_index(1, None).unwrap();
        assert_eq!(session.nodes.len(), 3);
        assert_eq!(session.nodes[1].parent, Some(0));
        assert_eq!(session.nodes[2].parent, Some(0));
        assert_eq!(
            session.navigate(1).unwrap()["fingerprint"],
            first["fingerprint"]
        );
        assert_eq!(
            session.navigate(0).unwrap()["fingerprint"],
            initial["fingerprint"]
        );
        session.apply_index(0, None).unwrap();
        assert_eq!(session.nodes.len(), 3);
    }
    #[test]
    fn snapshots_cancel_without_blocking_navigation_or_editing() {
        let mut session = GameSession::new(DEFAULT_SEED).unwrap();
        let root = session.view().unwrap();
        let snapshot = session.analysis_snapshot();
        assert!(!snapshot.cancellation.load(Ordering::Relaxed));
        session.navigate(0).unwrap();
        assert!(snapshot.cancellation.load(Ordering::Relaxed));
        let cancelled = snapshot.analyze(&json!({"depth": 4})).unwrap();
        assert_eq!(cancelled["report"]["stop_reason"], "cancelled");
        let edit = session
            .edit_board(&json!({"turn":"black", "pieces":[
            {"square":"e1", "kind":"king", "owner":"white"},
            {"square":"e8", "kind":"king", "owner":"black"}]}))
            .unwrap();
        assert_eq!(edit["notation"]["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(edit["turn"], "black");
        assert_eq!(
            session.navigate(0).unwrap()["fingerprint"],
            root["fingerprint"]
        );
    }
    #[test]
    fn checked_setup_preserves_original_and_rejects_duplicate_or_missing_target() {
        let mut session = GameSession::new(DEFAULT_SEED).unwrap();
        let root = session.view().unwrap();
        assert!(
            session
                .acquire_setup(&json!({"cardId":"amazon","owner":"white"}))
                .is_err()
        );
        let edited = session
            .edit_board(&json!({"turn":"white", "pieces":[
            {"square":"e1","kind":"king","owner":"white"},
            {"square":"e8","kind":"king","owner":"black"},
            {"square":"d1","kind":"queen","owner":"white"},
            {"square":"b1","kind":"knight","owner":"white"}]}))
            .unwrap();
        let result = session
            .acquire_setup(
                &json!({"cardId":"amazon","owner":"white","fingerprint":edited["fingerprint"]}),
            )
            .unwrap();
        assert_eq!(result["cards"]["white"][0]["id"], "amazon");
        let before = session.view().unwrap();
        assert!(
            session
                .acquire_setup(&json!({"cardId":"amazon","owner":"black"}))
                .is_err()
        );
        assert!(
            session
                .acquire_setup(&json!({"cardId":"breakthrough-order","owner":"white"}))
                .is_err()
        );
        assert!(
            session
                .acquire_setup(&json!({"cardId":"platform","owner":"white","fingerprint":"stale"}))
                .is_err()
        );
        assert_eq!(session.view().unwrap(), before);
        let rule = session
            .acquire_setup(&json!({"cardId":"platform","owner":"white"}))
            .unwrap();
        assert!(
            session
                .state()
                .applied_rules
                .iter()
                .any(|r| r == "platform")
        );
        assert!(
            session
                .acquire_setup(&json!({"cardId":"platform","owner":"black"}))
                .is_err()
        );
        assert_eq!(rule["notation"]["nodes"].as_array().unwrap().len(), 4);
        assert_eq!(
            session.navigate(0).unwrap()["fingerprint"],
            root["fingerprint"]
        );
    }

    #[test]
    fn analyzer_drafts_are_checked_and_export_is_selected_path_only() {
        let mut session = GameSession::new(1234).unwrap();
        let root = session.view().unwrap();
        for _ in 0..2 {
            let state = session.state();
            let augment_core::GamePhase::Draft(draft) = &state.phase else {
                panic!("opening draft missing");
            };
            assert!(!draft.offers.is_empty());
            for card in &draft.offers {
                assert!(
                    augment_core::check_card_eligibility(
                        state,
                        card,
                        state.turn,
                        augment_core::AcquisitionContext::Draft
                    )
                    .is_eligible()
                );
            }
            let fingerprint = format!("{:016x}", state.fingerprint());
            session.apply_index(0, Some(&fingerprint)).unwrap();
        }
        let before = session.view().unwrap();
        let path = session.export_path().unwrap();
        assert_eq!(path["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(session.view().unwrap(), before);
        session.navigate(0).unwrap();
        session
            .apply_index(1, root["fingerprint"].as_str())
            .unwrap();
        let branch = session.export_path().unwrap();
        assert_eq!(branch["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(branch["nodes"][1]["id"], 3);
    }

    #[test]
    fn analysis_is_model_free_and_does_not_mutate_position() {
        let session = GameSession::new(DEFAULT_SEED).unwrap();
        let before = session.view().unwrap();
        let result = session
            .analyze(&json!({"depth": 1, "timeMs": 100, "nodes": 1000}))
            .unwrap();
        assert_eq!(result["report"]["depth_unit"], "turns");
        assert_eq!(result["fingerprint"], before["fingerprint"]);
        assert_eq!(session.view().unwrap(), before);
        assert!(session.analyze(&json!({"depth": 0})).is_err());
        assert!(session.analyze(&json!({"fingerprint": "stale"})).is_err());
    }
}
