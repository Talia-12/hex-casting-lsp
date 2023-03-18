use std::{path::Path, fs::File, io::BufReader, collections::HashMap, fmt::Display};

use chumsky::primitive::todo;
use itertools::Either;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hex_pattern::{HexPattern, HexDir, HexAbsoluteDir};

static REGISTRY: Lazy<Result<(HashMap<String, RegistryEntry>, HashMap<String, RegistryEntry>, HashMap<HexPattern, RegistryEntry>), PatternNameRegistryError>> = Lazy::new(|| get_registry_from_file("registry.json"));

pub fn get_consideration() -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	registry_entry_from_id("escape")
}

fn simple_registry_entry_from_name(name: &str) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	REGISTRY.as_ref().map_err(|err| err.clone()).and_then(|(entries_by_name, _, _)| entries_by_name.get(name).map(|entry| StatOrDynRegistryEntry::StatRegistryEntry(entry)).ok_or(PatternNameRegistryError::NoPatternError(name.to_string())))
}

fn numeric_name_handler(name: &str) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	let parts = name.split(':').map(|s| s.trim()).collect::<Vec<_>>();
	if parts.len() == 2 && parts[0] == "Numerical Reflection" && parts[1].parse::<f64>().is_ok() {
		Ok(StatOrDynRegistryEntry::DynRegistryEntry(
			RegistryEntry {
				name: format!("Numerical Reflection: {}", parts[1]),
				id: "number".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: None, // TODO
				args: Some("\u{2192} number".to_string()),
				url: Some("https://gamma-delta.github.io/HexMod/#patterns/numbers@Numbers".to_string())
			}
		))
	} else {
		Err(PatternNameRegistryError::NoPatternError(name.to_string()))
	}
}

fn bookkeeper_name_handler(name: &str) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	let parts = name.split(':').map(|s| s.trim()).collect::<Vec<_>>();
	if parts.len() == 2 && parts[0] == "Bookkeeper's Gambit" && parts[1].chars().all(|c| c == 'v' || c == '-') {
		Ok(StatOrDynRegistryEntry::DynRegistryEntry(
			RegistryEntry {
				name: format!("Bookkeeper's Gambit: {}", parts[1]),
				id: "mask".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: None, // TODO
				args: Some("many \u{2192} many".to_string()),
				url: Some("https://gamma-delta.github.io/HexMod/#patterns/stackmanip@hexcasting:mask".to_string())
			}
		))
	} else {
		Err(PatternNameRegistryError::NoPatternError(name.to_string()))
	}
}

pub fn registry_entry_from_name(name: &str) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	numeric_name_handler(name)
		.or_else(|_| bookkeeper_name_handler(name))
		.or_else(|_| simple_registry_entry_from_name(name))
}



pub fn registry_entry_from_id(id: &str) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	REGISTRY.as_ref().map_err(|err| err.clone())
		.and_then(|(_, entries_by_id, _)|
			entries_by_id.get(id).map(|entry| StatOrDynRegistryEntry::StatRegistryEntry(entry)).ok_or(PatternNameRegistryError::NoPatternError(id.to_string()))
		)
}


fn simple_registry_entry_from_pattern(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	REGISTRY.as_ref().map_err(|err| err.clone())
		.and_then(|(_, _, entries_by_pattern)|
			entries_by_pattern.get(pattern).map(|entry| StatOrDynRegistryEntry::StatRegistryEntry(entry)).ok_or(PatternNameRegistryError::NoPatternError(pattern.to_string()))
		)
}


fn numeric_pattern_handler(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	Err(PatternNameRegistryError::NoPatternError(pattern.to_string())) // TODO
}

fn bookkeeper_pattern_handler(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
	Err(PatternNameRegistryError::NoPatternError(pattern.to_string())) // TODO
}

pub fn registry_entry_from_pattern(pattern: &HexPattern) -> Result<StatOrDynRegistryEntry, PatternNameRegistryError> {
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
			StatOrDynRegistryEntry::StatRegistryEntry(entry) => write!(f, "{entry}"),
			StatOrDynRegistryEntry::DynRegistryEntry(entry) => write!(f, "{entry}"),
		}
	}
}

impl StatOrDynRegistryEntry {
	pub fn get_name(&self) -> &str {
		&self.get_entry().name
	}

	pub fn get_id(&self) -> &str {
		&self.get_entry().id
	}
	
	pub fn get_pattern(&self) -> Option<HexPattern> {
		self.get_entry().pattern.clone()
	}

	pub fn get_entry(&self) -> &RegistryEntry {
		match self {
			StatOrDynRegistryEntry::StatRegistryEntry(entry) => entry,
			StatOrDynRegistryEntry::DynRegistryEntry(entry) => entry,
		}
	}
}

#[derive(Debug)]
pub enum PatternNameRegistryError {
	IOError(Either<std::io::Error, std::io::ErrorKind>),
	ParseError(Option<serde_json::Error>),
	RegistryFormatError,
	InvalidPatternError,
	NoPatternError(String)
}

impl Clone for PatternNameRegistryError {
	fn clone(&self) -> Self {
		match self {
			Self::IOError(io_error) => match io_error {
				Either::Left(io_error) => Self::IOError(Either::Right(io_error.kind())),
				Either::Right(io_error) => Self::IOError(Either::Right(io_error.clone())),
			},
			Self::ParseError(parse_error) => Self::ParseError(None),
			Self::RegistryFormatError => Self::RegistryFormatError,
			Self::InvalidPatternError => Self::InvalidPatternError,
			Self::NoPatternError(error_string) => Self::NoPatternError(error_string.clone()),
		}
	}
}

impl From<std::io::Error> for PatternNameRegistryError {
	fn from(value: std::io::Error) -> Self {
		PatternNameRegistryError::IOError(Either::Left(value))
	}
}

impl From<serde_json::Error> for PatternNameRegistryError {
	fn from(value: serde_json::Error) -> Self {
		PatternNameRegistryError::ParseError(Some(value))
	}
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RegistryEntry {
	pub name: String,
	pub id: String,
	pub mod_name: String,
	pub pattern: Option<HexPattern>,
	pub args: Option<String>,
	pub url: Option<String>

	// TODO: store an image here, constructed when the entry is made!
}

impl Display for RegistryEntry {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name)
	}
}

impl RegistryEntry {
	fn from_raw(raw: RawRegistryEntry, name: String) -> Result<RegistryEntry, PatternNameRegistryError> {
		let pattern = raw.pattern.clone().and_then(|pattern|
			raw.direction.clone()
				.and_then(|direction|
					HexAbsoluteDir::from_str(&direction)
				).and_then(|dir| {
					HexDir::from_str(&pattern).map(|dirs| HexPattern::new(dir, dirs)).ok()
				})
		);

		Ok(RegistryEntry { name, id: raw.id, mod_name: raw.mod_name, pattern, args: raw.args, url: raw.url })
	}
}

#[derive(Debug, PartialEq, Deserialize)]
struct RawRegistryEntry {
	#[serde(rename = "name")]
	id: String,
	
	#[serde(rename = "modName")]
	mod_name: String,
	direction: Option<String>,
	pattern: Option<String>,
	args: Option<String>,
	url: Option<String>
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

				entries_by_name.insert(name, entry.clone());
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

		let v: Value = serde_json::from_str(temp_data).unwrap();

		let (entries_by_name, entries_by_id, entries_by_pattern) = get_registry(v).unwrap();

		let expected_entries_vec = vec![
			RegistryEntry {
				name: "Compass' Purification".to_string(),
				id: "entity_pos/eye".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: Some(HexPattern { start_dir: HexAbsoluteDir::East, pattern_vec: vec![HexDir::A, HexDir::A] }),
				args: Some("entity \u{2192} vector".to_string()),
				url: Some("https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:entity_pos/eye".to_string())
			},
			RegistryEntry {
				name: "Mind's Reflection".to_string(),
				id: "get_caster".to_string(),
				mod_name: "Hex Casting".to_string(),
				pattern: Some(HexPattern { start_dir: HexAbsoluteDir::NorthEast, pattern_vec: vec![HexDir::Q, HexDir::A, HexDir::Q] }),
				args: Some("\u{2192} entity".to_string()),
				url: Some("https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:get_caster".to_string())
			},
		];

		assert_eq!(entries_by_name, expected_entries_vec.iter().map(|entry| (entry.name.clone(), entry.clone())).collect());
		assert_eq!(entries_by_id, expected_entries_vec.iter().map(|entry| (entry.id.clone(), entry.clone())).collect());
		assert_eq!(entries_by_pattern, expected_entries_vec.iter().filter_map(|entry| entry.pattern.clone().map(|pat| (pat, entry.clone()))).collect());
	}
}