//! A deliberately small, non-executing JavaScript data evaluator.
//! Unsupported expressions are errors, never silently omitted cards.
use super::*;
use serde_json::{Map, Value};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(super) struct Evaluator<'a> {
    source: &'a str,
    stack: Vec<String>,
}
impl<'a> Evaluator<'a> {
    pub(super) fn new(source: &'a str) -> Self {
        Self {
            source,
            stack: Vec::new(),
        }
    }

    pub(super) fn named(&mut self, name: &str) -> Result<Value> {
        if self.stack.len() > 64 || self.stack.iter().any(|s| s == name) {
            return Err(format!("cyclic or deep declaration {name}").into());
        }
        let marker = format!("const {name} = ");
        let start = self
            .source
            .find(&marker)
            .ok_or_else(|| format!("missing declaration {name}"))?
            + marker.len();
        let tail = &self.source[start..];
        let end = find_top_level(tail, ';').ok_or("unterminated declaration")?;
        self.stack.push(name.to_owned());
        let result = self.eval(&tail[..end], &BTreeMap::new());
        self.stack.pop();
        result
    }

    pub(super) fn eval(
        &mut self,
        expression: &str,
        env: &BTreeMap<String, Value>,
    ) -> Result<Value> {
        let s = expression.trim();
        if let Some(value) = env.get(s) {
            return Ok(value.clone());
        }
        if let Ok(value) = serde_json::from_str::<Value>(s) {
            return Ok(value);
        }
        if s.starts_with('\'') {
            return Ok(Value::String(parse_js_string(s)?));
        }
        if let Some(inner) = s
            .strip_prefix("Object.freeze(")
            .and_then(|s| s.strip_suffix(')'))
        {
            return self.eval(inner, env);
        }
        if s.starts_with('[') {
            let end = matching_delimiter(s, 0, '[', ']').ok_or("unclosed data array")?;
            let mut values = Vec::new();
            for part in split_top_level(&s[1..end], ',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                if let Some(spread) = part.strip_prefix("...") {
                    let value = self.eval(spread, env)?;
                    values.extend(
                        value
                            .as_array()
                            .ok_or("array spread is not an array")?
                            .iter()
                            .cloned(),
                    );
                } else {
                    values.push(self.eval(part, env)?);
                }
            }
            let suffix = s[end + 1..].trim();
            match suffix {
                "" | ".map(Object.freeze)" => {}
                ".map(internalBalanceCard)"
                | ".map(({ passive: passive2, ...card2 }) => internalBalanceCard(card2))" => {
                    self.validate_balance_function()?;
                    let patches = self.named("INTERNAL_BALANCE_UPDATES")?;
                    for value in &mut values {
                        let id = value
                            .get("id")
                            .and_then(Value::as_str)
                            .ok_or("mapped card has no id")?
                            .to_owned();
                        let object = value
                            .as_object_mut()
                            .ok_or("mapped card is not an object")?;
                        if suffix.contains("passive:") {
                            object.remove("passive");
                        }
                        if let Some(patch) = patches.get(&id) {
                            object.extend(
                                patch
                                    .as_object()
                                    .ok_or("balance patch is not an object")?
                                    .clone(),
                            );
                        }
                    }
                }
                _ => return Err(format!("unsupported array suffix {suffix:?}").into()),
            }
            return Ok(Value::Array(values));
        }
        if s.starts_with('{') {
            let inner = strip_collection(s, '{', '}')?;
            let mut object = Map::new();
            for part in split_top_level(inner, ',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                if let Some(spread) = part.strip_prefix("...") {
                    object.extend(
                        self.eval(spread, env)?
                            .as_object()
                            .ok_or("object spread is not an object")?
                            .clone(),
                    );
                } else if let Some(colon) = find_top_level(part, ':') {
                    let key = part[..colon].trim().trim_matches(['\'', '"']);
                    if !identifier(key) && !part[..colon].trim().starts_with(['\'', '"']) {
                        return Err("unsupported property key".into());
                    }
                    object.insert(key.to_owned(), self.eval(&part[colon + 1..], env)?);
                } else if identifier(part) {
                    object.insert(part.to_owned(), self.eval(part, env)?);
                } else {
                    return Err(format!("unsupported object member {part:?}").into());
                }
            }
            return Ok(Value::Object(object));
        }
        if let Some(open) = s.find('(') {
            let name = s[..open].trim();
            if !identifier(name) || !s.ends_with(')') {
                return Err(format!("unsupported call {s:?}").into());
            }
            let args = split_top_level(&s[open + 1..s.len() - 1], ',');
            // Legacy static display-name functions return one literal string.
            if args.len() == 1 && args[0].is_empty() {
                let marker = format!("function {name}()");
                let body = assignment_collection(self.source, &marker, '{', '}')?;
                let body = strip_collection(body, '{', '}')?.trim();
                if let Some(ret) = body
                    .strip_prefix("return ")
                    .and_then(|s| s.strip_suffix(';'))
                {
                    return Ok(Value::String(parse_js_string(ret)?));
                }
                return Err("static name function has unsupported statements".into());
            }
            let marker = format!("const {name} = (");
            let start = self
                .source
                .find(&marker)
                .ok_or_else(|| format!("unsupported factory {name}"))?
                + marker.len()
                - 1;
            let end = matching_delimiter(self.source, start, '(', ')')
                .ok_or("unclosed factory params")?;
            let params = split_top_level(&self.source[start + 1..end], ',');
            if args.len() > params.len() {
                return Err("too many factory arguments".into());
            }
            let tail = self.source[end + 1..]
                .trim_start()
                .strip_prefix("=>")
                .ok_or("factory must be declarative arrow")?
                .trim_start();
            let body_end = find_top_level(tail, ';').ok_or("unterminated factory")?;
            let mut local = BTreeMap::new();
            for (index, param) in params.iter().enumerate() {
                let (name, default) = param
                    .split_once('=')
                    .map(|(n, d)| (n.trim(), Some(d.trim())))
                    .unwrap_or((param.trim(), None));
                if !identifier(name) {
                    return Err("invalid factory parameter".into());
                }
                let value = if let Some(arg) = args.get(index) {
                    self.eval(arg, env)?
                } else {
                    self.eval(default.ok_or("missing factory argument")?, &local)?
                };
                local.insert(name.to_owned(), value);
            }
            if self.stack.len() > 64 || self.stack.iter().any(|s| s == name) {
                return Err("recursive factory".into());
            }
            self.stack.push(name.to_owned());
            let result = self.eval(&tail[..body_end], &local);
            self.stack.pop();
            return result;
        }
        if let Some((name, property)) = s.split_once('.') {
            if identifier(name) && identifier(property) {
                return self
                    .named(name)?
                    .get(property)
                    .cloned()
                    .ok_or_else(|| format!("missing property {s}").into());
            }
        }
        if identifier(s) {
            return self.named(s);
        }
        Err(format!("unsupported declarative expression {s:?}").into())
    }
    /// Recognized post-definition catalog patches. This does not execute gameplay code.
    pub(super) fn apply_metadata_overrides(&mut self, definitions: &mut Value) -> Result<()> {
        if !self.source.contains("const INTERNAL_BALANCE_UPDATES = ") {
            return Ok(());
        }
        self.validate_balance_function()?;
        let updates = self.named("INTERNAL_BALANCE_UPDATES")?;
        let stars = self.named("CARD_DISPLAY_STAR_OVERRIDES")?;
        let cards = definitions
            .as_array_mut()
            .ok_or("CARD_DEFS must be an array")?;
        for card in cards {
            let id = card["id"].as_str().ok_or("card id missing")?.to_owned();
            let object = card.as_object_mut().ok_or("card must be an object")?;
            if let Some(value) = stars.get(&id) {
                object.insert("stars".to_owned(), value.clone());
            }
            if let Some(patch) = updates.get(&id) {
                object.extend(
                    patch
                        .as_object()
                        .ok_or("balance patch must be an object")?
                        .clone(),
                );
            }
        }
        Ok(())
    }

    fn validate_balance_function(&self) -> Result<()> {
        let body =
            assignment_collection(self.source, "function internalBalanceCard(card2)", '{', '}')?;
        let normalized: String = body.chars().filter(|c| !c.is_whitespace()).collect();
        if normalized != "{return{...card2,...INTERNAL_BALANCE_UPDATES[card2.id]};}" {
            return Err("unsupported internalBalanceCard body".into());
        }
        Ok(())
    }
}
fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().enumerate().all(|(i, c)| {
            c == '_' || c == '$' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
        })
}
