use std::{path::Path, fs::File, io::BufReader, collections::HashMap, fmt::Display};

use chumsky::primitive::todo;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hex_pattern::{HexPattern, HexDir, HexAbsoluteDir};

static REGISTRY: Lazy<Result<(HashMap<String, RegistryEntry>, HashMap<String, RegistryEntry>, HashMap<HexPattern, RegistryEntry>), PatternNameRegistryError>> = Lazy::new(|| get_registry_from_file("registry.json"));

pub fn get_consideration() -> Result<StatOrDynRegistryEntry, &'static PatternNameRegistryError> {
	registry_entry_from_id("escape")
}

fn simple_registry_entry_from_name(name: &str) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	REGISTRY.as_ref().and_then(|(entries_by_name, _, _)| entries_by_name.get(name).map(|entry| StatOrDynRegistryEntry::StatRegistryEntry(entry)).ok_or(&PatternNameRegistryError::NoPatternError))
}

fn numeric_name_handler(name: &str) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	let parts = name.split(':').collect::<Vec<_>>();
	if parts.len() == 2 && parts[0] == "NumericalReflection" && parts[1].parse::<f64>().is_ok() {
		Ok(StatOrDynRegistryEntry::DynRegistryEntry(
			RegistryEntry {
				name: format!("Numerical Reflection: {}", parts[1]),
				id: "number".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: None, // TODO
				args: "\u{2192} number".to_string(),
				url: "https://gamma-delta.github.io/HexMod/#patterns/numbers@Numbers".to_string()
			}
		))
	} else {
		Err(&PatternNameRegistryError::NoPatternError)
	}
}

fn bookkeeper_name_handler(name: &str) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	let parts = name.split(':').collect::<Vec<_>>();
	if parts.len() == 2 && parts[0] == "Bookkeeper'sGambit" && parts[1].chars().all(|c| c == 'v' || c == '-') {
		Ok(StatOrDynRegistryEntry::DynRegistryEntry(
			RegistryEntry {
				name: format!("Bookkeeper's Gambit: {}", parts[1]),
				id: "mask".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: None, // TODO
				args: "many \u{2192} many".to_string(),
				url: "https://gamma-delta.github.io/HexMod/#patterns/stackmanip@hexcasting:mask".to_string()
			}
		))
	} else {
		Err(&PatternNameRegistryError::NoPatternError)
	}
}

pub fn registry_entry_from_name(name: &str) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	numeric_name_handler(name)
		.or(bookkeeper_name_handler(name))
		.or(simple_registry_entry_from_name(name))
}



pub fn registry_entry_from_id(id: &str) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	REGISTRY.as_ref().and_then(|(_, entries_by_id, _)| entries_by_id.get(id).map(|entry| StatOrDynRegistryEntry::StatRegistryEntry(entry)).ok_or(&PatternNameRegistryError::NoPatternError))
}


fn simple_registry_entry_from_pattern(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	REGISTRY.as_ref().and_then(|(_, _, entries_by_pattern)| entries_by_pattern.get(pattern).map(|entry| StatOrDynRegistryEntry::StatRegistryEntry(entry)).ok_or(&PatternNameRegistryError::NoPatternError))
}


fn numeric_pattern_handler(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	Err(&PatternNameRegistryError::NoPatternError) // TODO
}

fn bookkeeper_pattern_handler(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	Err(&PatternNameRegistryError::NoPatternError) // TODO
}

pub fn registry_entry_from_pattern(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, &PatternNameRegistryError> {
	numeric_pattern_handler(pattern)
		.or(bookkeeper_pattern_handler(pattern))
		.or(simple_registry_entry_from_pattern(pattern))
}


#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StatOrDynRegistryEntry {
	StatRegistryEntry(&'static RegistryEntry),
	DynRegistryEntry(RegistryEntry)
}

impl Display for StatOrDynRegistryEntry {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			StatOrDynRegistryEntry::StatRegistryEntry(entry) => write!(f, "{}", entry),
			StatOrDynRegistryEntry::DynRegistryEntry(entry) => write!(f, "{}", entry),
		}
	}
}

impl StatOrDynRegistryEntry {
	pub fn get_id(&self) -> &str {
		match self {
			StatOrDynRegistryEntry::StatRegistryEntry(entry) => &entry.id,
			StatOrDynRegistryEntry::DynRegistryEntry(entry) => &entry.id,
		}
	}
	
	pub fn get_pattern(&self) -> Option<HexPattern> {
		match self {
			StatOrDynRegistryEntry::StatRegistryEntry(entry) => entry.pattern.clone(),
			StatOrDynRegistryEntry::DynRegistryEntry(entry) => entry.pattern.clone(),
		}
	}
}

#[derive(Debug)]
pub enum PatternNameRegistryError {
	IOError(std::io::Error),
	ParseError(serde_json::Error),
	RegistryFormatError,
	InvalidPatternError,
	NoPatternError
}

impl From<std::io::Error> for PatternNameRegistryError {
	fn from(value: std::io::Error) -> Self {
		PatternNameRegistryError::IOError(value)
	}
}

impl From<serde_json::Error> for PatternNameRegistryError {
	fn from(value: serde_json::Error) -> Self {
		PatternNameRegistryError::ParseError(value)
	}
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RegistryEntry {
	name: String,
	id: String,
	mod_name: String,
	pattern: Option<HexPattern>,
	args: String,
	url: String

	// TODO: store an image here, constructed when the entry is made!
}

impl Display for RegistryEntry {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name)
	}
}

impl RegistryEntry {
	fn from_raw(raw: RawRegistryEntry, name: String) -> Result<RegistryEntry, PatternNameRegistryError> {
		let pattern = HexAbsoluteDir::from_str(&raw.direction).map(|dir| HexPattern::new(dir, HexDir::from_str(&raw.pattern)));

		Ok(RegistryEntry { name, id: raw.id, mod_name: raw.mod_name, pattern, args: raw.args, url: raw.url })
	}
}

#[derive(Debug, PartialEq, Deserialize)]
struct RawRegistryEntry {
	#[serde(rename = "name")]
	id: String,
	
	#[serde(rename = "modName")]
	mod_name: String,
	direction: String,
	pattern: String,
	args: String,
	url: String
}

fn get_registry_from_file<P: AsRef<Path>>(path: P) -> Result<(HashMap<String, RegistryEntry>, HashMap<String, RegistryEntry>, HashMap<HexPattern, RegistryEntry>), PatternNameRegistryError> {
	let file = File::open(path)?;
	let reader = BufReader::new(file);
	
	let v = serde_json::from_reader(reader)?;

	get_registry(v)
}

fn get_registry(v: Value) -> Result<(HashMap<String, RegistryEntry>, HashMap<String, RegistryEntry>, HashMap<HexPattern, RegistryEntry>), PatternNameRegistryError> {
	match v {
    Value::Null => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Bool(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Number(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::String(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Array(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Object(inner) => {
			let mut entries_by_name = HashMap::new();
			let mut entries_by_id = HashMap::new();
			let mut entries_by_pattern = HashMap::new();
			
			for (name, value) in inner.into_iter() {
				let raw_entry: RawRegistryEntry = serde_json::from_value(value)?;
				let entry: RegistryEntry = RegistryEntry::from_raw(raw_entry, name.clone())?;

				entries_by_name.insert(name.chars().filter(|&c| c != ' ').collect(), entry.clone());
				entries_by_id.insert(entry.id.clone(), entry.clone());

				if let Some(pattern) = &entry.pattern {
					entries_by_pattern.insert(pattern.clone(), entry);
				}
			}

			Ok((entries_by_name, entries_by_id, entries_by_pattern))
		},
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_get_registry() {
		let temp_data = r#"
		{
			"Mind's Reflection": {
				"name": "get_caster",
				"modName": "Hex Casting",
				"image": {
					"filename": "get_caster.png",
					"height": 272,
					"width": 158
				},
				"direction": "NORTH_EAST",
				"pattern": "qaq",
				"args": "\u2192 entity",
				"url": "https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:get_caster"
			},
			"Compass' Purification": {
				"name": "entity_pos/eye",
				"modName": "Hex Casting",
				"image": {
					"filename": "entity_pos_eye.png",
					"height": 129,
					"width": 136
				},
				"direction": "EAST",
				"pattern": "aa",
				"args": "entity \u2192 vector",
				"url": "https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:entity_pos/eye"
			}
		}"#;

		let v: Value = serde_json::from_str(temp_data).map_err(PatternNameRegistryError::ParseError).unwrap(); // TODO: change to from_reader to read from file

		let (entries_by_name, entries_by_id, entries_by_pattern) = get_registry(v).unwrap();

		let expected_entries_vec = vec![
			RegistryEntry {
				name: "Compass' Purification".to_string(),
				id: "entity_pos/eye".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: Some(HexPattern { start_dir: HexAbsoluteDir::East, pattern_vec: vec![HexDir::A, HexDir::A] }),
				args: "entity \u{2192} vector".to_string(),
				url: "https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:entity_pos/eye".to_string()
			},
			RegistryEntry {
				name: "Mind's Reflection".to_string(),
				id: "get_caster".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: Some(HexPattern { start_dir: HexAbsoluteDir::NorthEast, pattern_vec: vec![HexDir::Q, HexDir::A, HexDir::Q] }),
				args: "\u{2192} entity".to_string(),
				url: "https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:get_caster".to_string()
			},
		];

		assert_eq!(entries_by_name, expected_entries_vec.iter().map(|entry| (entry.name.chars().filter(|&c| c != ' ').collect(), entry.clone())).collect());
		assert_eq!(entries_by_id, expected_entries_vec.iter().map(|entry| (entry.id.clone(), entry.clone())).collect());
		assert_eq!(entries_by_pattern, expected_entries_vec.iter().filter_map(|entry| entry.pattern.clone().map(|pat| (pat, entry.clone()))).collect());
	}
}