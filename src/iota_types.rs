use std::fmt::Display;

use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

use crate::{hex_pattern::{HexPattern, HexDir, HexAbsoluteDir}, pattern_name_registry::{StatOrDynRegistryEntry, registry_entry_from_id, registry_entry_from_pattern}};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Iota {
	Null,
	Num(f64),
	Bool(bool),
	Entity(String),
	List(Vec<Iota>),
	Pattern(HexPatternIota),
	Vec3((f64, f64, f64)),
	Str(String),
	Matrix(DMatrix<f64>),
	IotaType(IotaType),
	EntityType(String),
	ItemType(String),
	Gate(String),
	Mote { id: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub enum HexPatternIota {
	HexPattern(HexPattern),
	RegistryEntry(StatOrDynRegistryEntry)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum SerialisedHexPatternIota {
	HexPattern(HexPattern),
	RegistryEntry(String)
}

impl Display for HexPatternIota {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			HexPatternIota::HexPattern(pattern) => write!(f, "{}", pattern),
			HexPatternIota::RegistryEntry(entry) => write!(f, "{}", entry),
		}
	}
}

impl Serialize for HexPatternIota {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer {
		match self {
			HexPatternIota::HexPattern(pattern) => SerialisedHexPatternIota::HexPattern(pattern.clone()).serialize(serializer),
			HexPatternIota::RegistryEntry(entry) => {
				if let Some(pattern) = entry.get_pattern() {
					SerialisedHexPatternIota::HexPattern(pattern.clone()).serialize(serializer)
				} else {
					SerialisedHexPatternIota::RegistryEntry(entry.get_id().to_string()).serialize(serializer)
				}
			},
		}
	}
}

impl<'de> Deserialize<'de> for HexPatternIota {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de> {
		SerialisedHexPatternIota::deserialize(deserializer).map(|serialised| match serialised {
			SerialisedHexPatternIota::HexPattern(pattern) => registry_entry_from_pattern(&pattern).map(HexPatternIota::RegistryEntry).unwrap_or(HexPatternIota::HexPattern(pattern.clone())),
			SerialisedHexPatternIota::RegistryEntry(id) => registry_entry_from_id(&id).map(HexPatternIota::RegistryEntry).unwrap(),
		})
	}
}

impl From<HexPattern> for HexPatternIota {
	fn from(pattern: HexPattern) -> Self {
		registry_entry_from_pattern(&pattern).map(HexPatternIota::RegistryEntry).unwrap_or_else(|_| HexPatternIota::HexPattern(pattern.clone()))
	}
}

impl HexPatternIota {
	pub fn get_pattern(&self) -> HexPattern {
		match self {
			HexPatternIota::HexPattern(pattern) => pattern.clone(),
			HexPatternIota::RegistryEntry(entry) => entry.get_pattern().unwrap_or(HexPattern { start_dir: HexAbsoluteDir::NorthEast, pattern_vec: vec![HexDir::S, HexDir::Q, HexDir::S, HexDir::Q] }),
		}
	}
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IotaType {
	Null,
	Num,
	Bool,
	Entity,
	List,
	Pattern,
	Vec3,
	Str,
	Matrix,
	IotaType,
	EntityType,
	ItemType,
	Gate,
	Mote
}

impl Iota {
	fn get_type(&self) -> IotaType {
		match self {
			Iota::Null => IotaType::Null,
			Iota::Num(_) => IotaType::Num,
			Iota::Bool(_) => IotaType::Bool,
			Iota::Entity(_) => IotaType::Entity,
			Iota::List(_) => IotaType::List,
			Iota::Pattern(_) => IotaType::Pattern,
			Iota::Vec3(_) => IotaType::Vec3,
			Iota::Str(_) => IotaType::Str,
			Iota::Matrix(_) => IotaType::Matrix,
			Iota::IotaType(_) => IotaType::IotaType,
			Iota::EntityType(_) => IotaType::Entity,
			Iota::ItemType(_) => IotaType::ItemType,
			Iota::Gate(_) => IotaType::Gate,
			Iota::Mote { id } => IotaType::Mote,
		}
	}
}

impl std::fmt::Display for Iota {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Self::Null => write!(f, "null"),
			Self::Num(x) => write!(f, "{}", x),
			Self::Bool(x) => write!(f, "{}", x),
			Self::Entity(_) => todo!(),
			Self::List(xs) => write!(
				f,
				"[{}]",
				xs.iter()
					.map(|x| x.to_string())
					.collect::<Vec<_>>()
					.join(", ")
			),
			Self::Pattern(pattern) => write!(f, "{}", pattern),
			Self::Vec3((x, y, z)) => write!(f, "({}, {}, {})", x, y, z),
			Self::Str(x) => write!(f, "{}", x),
			Self::Matrix(mat) => todo!(),
			Self::IotaType(_) => todo!(),
			Self::EntityType(_) => todo!(),
			Self::ItemType(_) => todo!(),
			Self::Gate(_) => todo!(),
			Self::Mote { id } => todo!(),
		}
	}
}
