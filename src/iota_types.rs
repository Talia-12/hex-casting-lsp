use serde::{Serialize, Deserialize};

use crate::hex_pattern::HexPattern;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Iota {
	Null,
	Num(f64),
	Bool(bool),
	Entity(String),
	List(Vec<Iota>),
	Pattern(HexPattern),
	Vec3((f64, f64, f64)),
	Str(String),
	Matrix(Vec<Vec<f64>>),
	IotaType(IotaType),
	EntityType(String),
	ItemType(String),
	Gate(String),
	Mote { id: usize },
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
