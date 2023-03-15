use once_cell::sync::Lazy;
use serde::{Serialize, Deserialize};
use serde_json::Value;

static REGISTRY: Lazy<Result<Vec<RegistryEntry>, PatternNameRegistryError>> = Lazy::new(|| get_registry());

enum PatternNameRegistryError {
	ParseError(serde_json::Error),
	RegistryFormatError
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct RegistryEntry {

}

fn get_registry() -> Result<Vec<RegistryEntry>, PatternNameRegistryError> {
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

	let v: Value = serde_json::from_str(temp_data).map_err(PatternNameRegistryError::ParseError)?; // TODO: change to from_reader to read from file

	match v {
    Value::Null => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Bool(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Number(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::String(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Array(_) => Err(PatternNameRegistryError::RegistryFormatError),
    Value::Object(inner) => {
			let mut entries = Vec::new();
			
			for (name, value) in inner.into_iter() {
				let entry: RegistryEntry = serde_json::from_value(value).map_err(PatternNameRegistryError::ParseError)?;

				entries.push(entry)
			}

			Ok(entries)
		},
	}
}