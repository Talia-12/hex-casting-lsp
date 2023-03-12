use serde::{Serialize, Deserialize};

use crate::hex_parsing::{Span, Error};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Iota {
	Null,
	Num(f64),
	Bool(bool),
	Entity(String),
	List(Vec<Iota>),
	Pattern(String),
	Vec3((f64, f64, f64)),
	Str(String),
	Matrix(Vec<Vec<f64>>),
	IotaType(String),
	EntityType(String),
	ItemType(String),
	Gate(String),
	Item(String),
}

impl Iota {
	fn num(self, span: Span) -> Result<f64, Error> {
		if let Iota::Num(x) = self {
			Ok(x)
		} else {
			Err(Error {
				span,
				msg: format!("'{}' is not a number", self),
			})
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
			Self::Pattern(_) => todo!(),
			Self::Vec3(_) => todo!(),
			Self::Str(x) => write!(f, "{}", x),
			Self::Matrix(_) => todo!(),
			Self::IotaType(_) => todo!(),
			Self::EntityType(_) => todo!(),
			Self::ItemType(_) => todo!(),
			Self::Gate(_) => todo!(),
			Self::Item(_) => todo!(),
		}
	}
}
