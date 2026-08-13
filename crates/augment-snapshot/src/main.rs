use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct CatalogSnapshot {
    ruleset_id: String,
    schema_version: u32,
    source_bundle_sha256: String,
    raw_card_definitions: usize,
    visible_card_count: usize,
    hidden_by_default_ids: Vec<String>,
    cards: Vec<CatalogCard>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct CatalogCard {
    id: String,
    name_ko: String,
    phase: String,
    half_stars: Option<u16>,
    effect: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    text_ko: String,
    art: String,
}

enum Command {
    Extract {
        bundle: PathBuf,
        ruleset_id: String,
        output: PathBuf,
    },
    Diff {
        old: PathBuf,
        new: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    match parse_command(std::env::args().skip(1))? {
        Command::Extract {
            bundle,
            ruleset_id,
            output,
        } => {
            let source = fs::read(&bundle)?;
            let snapshot = extract_catalog(&source, ruleset_id)?;
            if let Some(parent) = output.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent)?;
            }
            let mut json = serde_json::to_string_pretty(&snapshot)?;
            json.push('\n');
            fs::write(&output, json)?;
            println!(
                "extracted {} visible cards ({} raw) -> {}",
                snapshot.visible_card_count,
                snapshot.raw_card_definitions,
                output.display()
            );
        }
        Command::Diff { old, new } => {
            let old_snapshot: CatalogSnapshot = serde_json::from_slice(&fs::read(&old)?)?;
            let new_snapshot: CatalogSnapshot = serde_json::from_slice(&fs::read(&new)?)?;
            print_catalog_diff(&old_snapshot, &new_snapshot);
        }
    }
    Ok(())
}

fn parse_command(mut args: impl Iterator<Item = String>) -> Result<Command, Box<dyn Error>> {
    let Some(command) = args.next() else {
        return Err(usage().into());
    };
    let mut values = BTreeMap::new();
    while let Some(flag) = args.next() {
        if !flag.starts_with("--") {
            return Err(format!("unexpected argument {flag:?}\n{}", usage()).into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        values.insert(flag, value);
    }
    match command.as_str() {
        "extract" => Ok(Command::Extract {
            bundle: required_path(&mut values, "--bundle")?,
            ruleset_id: required(&mut values, "--ruleset-id")?,
            output: required_path(&mut values, "--output")?,
        }),
        "diff" => Ok(Command::Diff {
            old: required_path(&mut values, "--old")?,
            new: required_path(&mut values, "--new")?,
        }),
        _ => Err(format!("unknown command {command:?}\n{}", usage()).into()),
    }
}

fn required(values: &mut BTreeMap<String, String>, name: &str) -> Result<String, Box<dyn Error>> {
    values
        .remove(name)
        .ok_or_else(|| format!("missing {name}\n{}", usage()).into())
}

fn required_path(
    values: &mut BTreeMap<String, String>,
    name: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    required(values, name).map(PathBuf::from)
}

fn usage() -> &'static str {
    "usage:\n  augment-snapshot extract --bundle FILE --ruleset-id ID --output FILE\n  augment-snapshot diff --old FILE --new FILE"
}

fn extract_catalog(source: &[u8], ruleset_id: String) -> Result<CatalogSnapshot, Box<dyn Error>> {
    let javascript = std::str::from_utf8(source)?;
    let definitions = assignment_collection(javascript, "const CARD_DEFS =", '[', ']')?;
    let category_groups = assignment_collection(
        javascript,
        "const CARD_CATEGORY_GROUPS = Object.freeze(",
        '{',
        '}',
    )?;
    let categories = parse_category_groups(category_groups)?;
    let raw_cards = split_top_level(strip_collection(definitions, '[', ']')?, ',')
        .into_iter()
        .filter(|item| item.trim_start().starts_with('{'))
        .map(|object| parse_card(object, &categories))
        .collect::<Result<Vec<_>, _>>()?;

    let hidden_by_default_ids: Vec<_> = raw_cards
        .iter()
        .filter(|card| card.phase == "GUN")
        .map(|card| card.id.clone())
        .collect();
    let mut cards: Vec<_> = raw_cards
        .iter()
        .filter(|card| card.phase != "GUN")
        .cloned()
        .collect();
    cards.sort_by(|left, right| left.id.cmp(&right.id));
    let unique_ids: BTreeSet<_> = cards.iter().map(|card| card.id.as_str()).collect();
    if unique_ids.len() != cards.len() {
        return Err("catalog contains duplicate visible card IDs".into());
    }

    Ok(CatalogSnapshot {
        ruleset_id,
        schema_version: 1,
        source_bundle_sha256: format!("{:x}", Sha256::digest(source)),
        raw_card_definitions: raw_cards.len(),
        visible_card_count: cards.len(),
        hidden_by_default_ids,
        cards,
    })
}

fn parse_card(
    object: &str,
    categories: &BTreeMap<String, String>,
) -> Result<CatalogCard, Box<dyn Error>> {
    let properties = parse_object_properties(object)?;
    let id = required_string_property(&properties, "id")?;
    let source_phase = required_string_property(&properties, "phase")?;
    let phase = categories.get(&id).cloned().unwrap_or(source_phase);
    Ok(CatalogCard {
        name_ko: required_string_property(&properties, "name")?,
        half_stars: parse_half_stars(required_property(&properties, "stars")?)?,
        effect: required_string_property(&properties, "effect")?,
        target: properties
            .get("target")
            .map(|value| parse_js_string(value))
            .transpose()?,
        text_ko: required_string_property(&properties, "text")?,
        art: required_string_property(&properties, "art")?,
        id,
        phase,
    })
}

fn parse_category_groups(source: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut categories = BTreeMap::new();
    for (category, value) in parse_object_properties(source)? {
        let array = strip_collection(value.trim(), '[', ']')?;
        for item in split_top_level(array, ',') {
            if item.trim().is_empty() {
                continue;
            }
            let id = parse_js_string(item.trim())?;
            if categories.insert(id.clone(), category.clone()).is_some() {
                return Err(format!("card {id:?} occurs in multiple category groups").into());
            }
        }
    }
    Ok(categories)
}

fn parse_object_properties(source: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let inner = strip_collection(source.trim(), '{', '}')?;
    let mut properties = BTreeMap::new();
    for property in split_top_level(inner, ',') {
        let property = property.trim();
        if property.is_empty() || property.starts_with("...") {
            continue;
        }
        let Some(colon) = find_top_level(property, ':') else {
            continue;
        };
        let key = property[..colon]
            .trim()
            .trim_matches(['"', '\''])
            .to_owned();
        let value = property[colon + 1..].trim().to_owned();
        properties.insert(key, value);
    }
    Ok(properties)
}

fn required_property<'a>(
    properties: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, Box<dyn Error>> {
    properties
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("card definition is missing property {key:?}").into())
}

fn required_string_property(
    properties: &BTreeMap<String, String>,
    key: &str,
) -> Result<String, Box<dyn Error>> {
    parse_js_string(required_property(properties, key)?)
}

fn parse_half_stars(source: &str) -> Result<Option<u16>, Box<dyn Error>> {
    let value = source.trim();
    if value == "null" {
        return Ok(None);
    }
    let (whole, fraction) = value.split_once('.').unwrap_or((value, "0"));
    let whole: u16 = whole.parse()?;
    let fraction = match fraction {
        "0" | "" => 0,
        "5" => 1,
        _ => return Err(format!("unsupported star precision {value:?}").into()),
    };
    Ok(Some(whole.saturating_mul(2).saturating_add(fraction)))
}

fn parse_js_string(source: &str) -> Result<String, Box<dyn Error>> {
    let value = source.trim();
    if value.starts_with('"') {
        return Ok(serde_json::from_str(value)?);
    }
    if value.starts_with('\'') && value.ends_with('\'') {
        let body = &value[1..value.len() - 1];
        return Ok(body.replace("\\'", "'").replace("\\\\", "\\"));
    }
    Err(format!("expected a JavaScript string, got {value:?}").into())
}

fn assignment_collection<'a>(
    source: &'a str,
    marker: &str,
    open: char,
    close: char,
) -> Result<&'a str, Box<dyn Error>> {
    let marker_index = source
        .find(marker)
        .ok_or_else(|| format!("bundle marker {marker:?} was not found"))?;
    let collection_start = source[marker_index + marker.len()..]
        .find(open)
        .map(|offset| marker_index + marker.len() + offset)
        .ok_or_else(|| format!("opening {open:?} after {marker:?} was not found"))?;
    let collection_end = matching_delimiter(source, collection_start, open, close)
        .ok_or_else(|| format!("closing {close:?} after {marker:?} was not found"))?;
    Ok(&source[collection_start..=collection_end])
}

fn strip_collection(source: &str, open: char, close: char) -> Result<&str, Box<dyn Error>> {
    let source = source.trim();
    if !source.starts_with(open) || !source.ends_with(close) {
        return Err(format!("expected collection delimited by {open}{close}").into());
    }
    Ok(&source[open.len_utf8()..source.len() - close.len_utf8()])
}

fn matching_delimiter(source: &str, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0_u32;
    let mut scanner = Scanner::new(source, start);
    while let Some((index, character)) = scanner.next_code_char() {
        if character == open {
            depth += 1;
        } else if character == close {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn split_top_level(source: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depths = [0_i32; 3];
    let mut scanner = Scanner::new(source, 0);
    while let Some((index, character)) = scanner.next_code_char() {
        match character {
            '{' => depths[0] += 1,
            '}' => depths[0] -= 1,
            '[' => depths[1] += 1,
            ']' => depths[1] -= 1,
            '(' => depths[2] += 1,
            ')' => depths[2] -= 1,
            _ if character == separator && depths == [0, 0, 0] => {
                parts.push(&source[start..index]);
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&source[start..]);
    parts
}

fn find_top_level(source: &str, needle: char) -> Option<usize> {
    let mut depths = [0_i32; 3];
    let mut scanner = Scanner::new(source, 0);
    while let Some((index, character)) = scanner.next_code_char() {
        match character {
            '{' => depths[0] += 1,
            '}' => depths[0] -= 1,
            '[' => depths[1] += 1,
            ']' => depths[1] -= 1,
            '(' => depths[2] += 1,
            ')' => depths[2] -= 1,
            _ if character == needle && depths == [0, 0, 0] => return Some(index),
            _ => {}
        }
    }
    None
}

struct Scanner<'a> {
    source: &'a str,
    offset: usize,
}

impl<'a> Scanner<'a> {
    fn new(source: &'a str, offset: usize) -> Self {
        Self { source, offset }
    }

    fn next_code_char(&mut self) -> Option<(usize, char)> {
        while self.offset < self.source.len() {
            let index = self.offset;
            let character = self.source[index..].chars().next()?;
            if matches!(character, '"' | '\'' | '`') {
                self.skip_string(character);
                continue;
            }
            if character == '/' && self.source[index..].starts_with("//") {
                self.offset += 2;
                while self.offset < self.source.len() {
                    let next = self.source[self.offset..].chars().next()?;
                    self.offset += next.len_utf8();
                    if next == '\n' {
                        break;
                    }
                }
                continue;
            }
            if character == '/' && self.source[index..].starts_with("/*") {
                if let Some(end) = self.source[index + 2..].find("*/") {
                    self.offset = index + 2 + end + 2;
                } else {
                    self.offset = self.source.len();
                }
                continue;
            }
            self.offset += character.len_utf8();
            return Some((index, character));
        }
        None
    }

    fn skip_string(&mut self, quote: char) {
        self.offset += quote.len_utf8();
        let mut escaped = false;
        while self.offset < self.source.len() {
            let character = self.source[self.offset..]
                .chars()
                .next()
                .expect("offset is on a character boundary");
            self.offset += character.len_utf8();
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == quote {
                break;
            }
        }
    }
}

fn print_catalog_diff(old: &CatalogSnapshot, new: &CatalogSnapshot) {
    let old_cards: BTreeMap<_, _> = old.cards.iter().map(|card| (&card.id, card)).collect();
    let new_cards: BTreeMap<_, _> = new.cards.iter().map(|card| (&card.id, card)).collect();
    let old_ids: BTreeSet<_> = old_cards.keys().copied().collect();
    let new_ids: BTreeSet<_> = new_cards.keys().copied().collect();
    let mut changes = 0_usize;

    if old.ruleset_id != new.ruleset_id {
        println!("ruleset_id: {:?} -> {:?}", old.ruleset_id, new.ruleset_id);
    }
    if old.source_bundle_sha256 != new.source_bundle_sha256 {
        println!(
            "source_bundle_sha256: {} -> {}",
            old.source_bundle_sha256, new.source_bundle_sha256
        );
    }

    for id in new_ids.difference(&old_ids) {
        println!("+ {id}: {}", new_cards[id].name_ko);
        changes += 1;
    }
    for id in old_ids.difference(&new_ids) {
        println!("- {id}: {}", old_cards[id].name_ko);
        changes += 1;
    }
    for id in old_ids.intersection(&new_ids) {
        let old_card = old_cards[id];
        let new_card = new_cards[id];
        if old_card != new_card {
            println!("~ {id}");
            print_field_change("name_ko", &old_card.name_ko, &new_card.name_ko);
            print_field_change("phase", &old_card.phase, &new_card.phase);
            print_field_change("half_stars", &old_card.half_stars, &new_card.half_stars);
            print_field_change("effect", &old_card.effect, &new_card.effect);
            print_field_change("target", &old_card.target, &new_card.target);
            print_field_change("text_ko", &old_card.text_ko, &new_card.text_ko);
            print_field_change("art", &old_card.art, &new_card.art);
            changes += 1;
        }
    }
    println!(
        "summary: {changes} changed IDs, {} -> {} visible cards",
        old.visible_card_count, new.visible_card_count
    );
}

fn print_field_change<T: std::fmt::Debug + PartialEq>(name: &str, old: &T, new: &T) {
    if old != new {
        println!("    {name}: {old:?} -> {new:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
const CARD_DEFS = [
  { id: "alpha", name: "알파", phase: "OPENING", stars: 2.5, text: "설명", art: "a", effect: "alphaEffect" },
  { id: "rule-one", name: "규칙", phase: "RULE", stars: null, text: "규칙 설명", art: "r", effect: "ruleEffect", target: "square" },
  { id: "hidden", name: "숨김", phase: "OPENING", stars: 3, text: "숨김", art: "h", effect: "hiddenEffect" }
];
const CARD_CATEGORY_GROUPS = Object.freeze({
  OPENING: ["alpha"],
  RULE: ["rule-one"],
  GUN: ["hidden"]
});
"#;

    #[test]
    fn extracts_normalized_visible_catalog() {
        let snapshot =
            extract_catalog(SAMPLE.as_bytes(), "augment_chess_20990101".to_owned()).unwrap();
        assert_eq!(snapshot.raw_card_definitions, 3);
        assert_eq!(snapshot.visible_card_count, 2);
        assert_eq!(snapshot.hidden_by_default_ids, ["hidden"]);
        assert_eq!(snapshot.cards[0].id, "alpha");
        assert_eq!(snapshot.cards[0].half_stars, Some(5));
        assert_eq!(snapshot.cards[1].phase, "RULE");
        assert_eq!(snapshot.cards[1].half_stars, None);
        assert_eq!(snapshot.cards[1].target.as_deref(), Some("square"));
    }

    #[test]
    fn top_level_split_ignores_nested_commas_and_strings() {
        assert_eq!(
            split_top_level(r#"a, { b: "x,y" }, [1, 2], call(1, 2)"#, ','),
            ["a", " { b: \"x,y\" }", " [1, 2]", " call(1, 2)"]
        );
    }

    #[test]
    fn parses_whole_and_half_stars_only() {
        assert_eq!(parse_half_stars("4").unwrap(), Some(8));
        assert_eq!(parse_half_stars("3.5").unwrap(), Some(7));
        assert_eq!(parse_half_stars("null").unwrap(), None);
        assert!(parse_half_stars("3.25").is_err());
    }
}
