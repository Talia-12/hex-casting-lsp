use std::{ops::{Add, Sub}, f32::consts::TAU};

use num_derive::{FromPrimitive, ToPrimitive};    
use num_traits::{FromPrimitive, ToPrimitive};

use egui::{Pos2, pos2, Vec2};

#[derive(PartialEq, Debug)]
pub enum HexError {
	Overlap,
	InvalidString
}

#[derive(serde::Deserialize, serde::Serialize, Debug, PartialEq, Clone)]
pub struct HexPattern {
	pub start_dir: HexAbsoluteDir,
	pub pattern_vec: Vec<HexDir>
}

impl HexPattern {
	pub fn to_coords(&self) -> Vec<HexCoord> {
		let mut coords = vec![hex_coord(0, 0)];

		let mut prev_coord = self.start_dir.coord_offset();
		let mut prev_dir = self.start_dir;

		coords.push(prev_coord);

		for rel_dir in &self.pattern_vec {
				(prev_coord, prev_dir) = rel_dir.coord_offset(prev_coord, prev_dir);
				coords.push(prev_coord);
		}

		return coords;
	}

	/// Adds a HexDir from the last node of the current pattern in the passed dir, if that doesn't cause an overlap.
	pub fn add_dir(&mut self, dir: HexAbsoluteDir) {
		let mut prev_coord = self.start_dir.coord_offset();
		let mut prev_dir = self.start_dir;

		for rel_dir in &self.pattern_vec {
				(prev_coord, prev_dir) = rel_dir.coord_offset(prev_coord, prev_dir);
		}

		if let Some(rel_dir) = prev_dir.difference(dir) {
			self.pattern_vec.push(rel_dir);

			if Self::check_for_overlap(&self.to_coords()) { self.pattern_vec.pop(); } // if the added direction causes an overlap, remove it.
		}
	}
	
	pub fn hex_pattern(start_dir: HexAbsoluteDir, pattern_vec: Vec<HexDir>) -> Result<HexPattern, HexError> {
		let pattern = HexPattern { start_dir, pattern_vec };
		
		if HexPattern::check_for_overlap(&pattern.to_coords()) {
			return Err(HexError::Overlap)
		}
	
		return Ok(pattern)
	}
	
	fn check_for_overlap(coords: &Vec<HexCoord>) -> bool {
		let mut visited_edges: Vec<(HexCoord, HexCoord)> = vec![];
	
		for index in 1..coords.len() {
			let start_coord = coords[index - 1];
			let end_coord = coords[index];
	
			if visited_edges.contains(&(end_coord, start_coord)) || visited_edges.contains(&(start_coord, end_coord)) {
				return true
			}
			visited_edges.push((start_coord, end_coord));
		}
	
		return false
	}
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone, Copy)]
pub enum HexDir {
		A,
		Q,
		W,
		E,
		D
}

impl HexDir {
	/// Takes in the absolute direction that the line was going, and returns the next coord as well as the new absolute direction.
	pub fn coord_offset(&self, prev_coord: HexCoord, prev_dir: HexAbsoluteDir) -> (HexCoord, HexAbsoluteDir) {
		let new_dir = match *self {
			HexDir::A => prev_dir.turn(-2),
			HexDir::Q => prev_dir.turn(-1),
			HexDir::W => prev_dir,
			HexDir::E => prev_dir.turn(1),
			HexDir::D => prev_dir.turn(2),
		};

		return (prev_coord + new_dir.coord_offset(), new_dir);
	}
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, ToPrimitive, FromPrimitive, Clone, Copy, Debug)]
pub enum HexAbsoluteDir {
	East,
	SouthEast,
	SouthWest,
	West,
	NorthWest,
	NorthEast
}

impl HexAbsoluteDir {
	pub fn nearest_dir(dir: Vec2) -> HexAbsoluteDir {
		let angle = (-dir.angle() % TAU + TAU) % TAU;

		return if TAU/12.0 <= angle && angle < 3.0*TAU/12.0 { HexAbsoluteDir::NorthEast }
		else if 3.0*TAU/12.0 <= angle && angle < 5.0*TAU/12.0 { HexAbsoluteDir::NorthWest }
		else if 5.0*TAU/12.0 <= angle && angle < 7.0*TAU/12.0 { HexAbsoluteDir::West }
		else if 7.0*TAU/12.0 <= angle && angle < 9.0*TAU/12.0 { HexAbsoluteDir::SouthWest }
		else if 9.0*TAU/12.0 <= angle && angle < 11.0*TAU/12.0 { HexAbsoluteDir::SouthEast }
		else { HexAbsoluteDir::East }
	}

	pub fn coord_offset(&self) -> HexCoord {
		match *self {
			HexAbsoluteDir::East => hex_coord(1, 0),
			HexAbsoluteDir::SouthEast => hex_coord(0, 1),
			HexAbsoluteDir::SouthWest => hex_coord(-1, 1),
			HexAbsoluteDir::West => hex_coord(-1, 0),
			HexAbsoluteDir::NorthWest => hex_coord(0, -1),
			HexAbsoluteDir::NorthEast => hex_coord(1, -1),
		}
	}

	pub fn turn(&self, amount: i16) -> HexAbsoluteDir {
		// cursed code to convert the current HexAbsoluteDir to an int, then add amount and modulo 6.
		match FromPrimitive::from_i16(((*ToPrimitive::to_i16(self).get_or_insert(0) + amount) % 6 + 6) % 6) {
				Some(d2) => d2,
				None => FromPrimitive::from_u8(0).unwrap(),
		}
	}

	pub fn difference(&self, other: HexAbsoluteDir) -> Option<HexDir> {
		match (*self, other) {
			(HexAbsoluteDir::East, HexAbsoluteDir::East) => Some(HexDir::W),
			(HexAbsoluteDir::East, HexAbsoluteDir::SouthEast) => Some(HexDir::E),
			(HexAbsoluteDir::East, HexAbsoluteDir::SouthWest) => Some(HexDir::D),
			(HexAbsoluteDir::East, HexAbsoluteDir::West) => None,
			(HexAbsoluteDir::East, HexAbsoluteDir::NorthWest) => Some(HexDir::A),
			(HexAbsoluteDir::East, HexAbsoluteDir::NorthEast) => Some(HexDir::Q),

			(HexAbsoluteDir::SouthEast, HexAbsoluteDir::East) => Some(HexDir::Q),
			(HexAbsoluteDir::SouthEast, HexAbsoluteDir::SouthEast) => Some(HexDir::W),
			(HexAbsoluteDir::SouthEast, HexAbsoluteDir::SouthWest) => Some(HexDir::E),
			(HexAbsoluteDir::SouthEast, HexAbsoluteDir::West) => Some(HexDir::D),
			(HexAbsoluteDir::SouthEast, HexAbsoluteDir::NorthWest) => None,
			(HexAbsoluteDir::SouthEast, HexAbsoluteDir::NorthEast) => Some(HexDir::A),
			
			(HexAbsoluteDir::SouthWest, HexAbsoluteDir::East) => Some(HexDir::A),
			(HexAbsoluteDir::SouthWest, HexAbsoluteDir::SouthEast) => Some(HexDir::Q),
			(HexAbsoluteDir::SouthWest, HexAbsoluteDir::SouthWest) => Some(HexDir::W),
			(HexAbsoluteDir::SouthWest, HexAbsoluteDir::West) => Some(HexDir::E),
			(HexAbsoluteDir::SouthWest, HexAbsoluteDir::NorthWest) => Some(HexDir::D),
			(HexAbsoluteDir::SouthWest, HexAbsoluteDir::NorthEast) => None,
			
			(HexAbsoluteDir::West, HexAbsoluteDir::East) => None,
			(HexAbsoluteDir::West, HexAbsoluteDir::SouthEast) => Some(HexDir::A),
			(HexAbsoluteDir::West, HexAbsoluteDir::SouthWest) => Some(HexDir::Q),
			(HexAbsoluteDir::West, HexAbsoluteDir::West) => Some(HexDir::W),
			(HexAbsoluteDir::West, HexAbsoluteDir::NorthWest) => Some(HexDir::E),
			(HexAbsoluteDir::West, HexAbsoluteDir::NorthEast) => Some(HexDir::D),
			
			(HexAbsoluteDir::NorthWest, HexAbsoluteDir::East) => Some(HexDir::D),
			(HexAbsoluteDir::NorthWest, HexAbsoluteDir::SouthEast) => None,
			(HexAbsoluteDir::NorthWest, HexAbsoluteDir::SouthWest) => Some(HexDir::A),
			(HexAbsoluteDir::NorthWest, HexAbsoluteDir::West) => Some(HexDir::Q),
			(HexAbsoluteDir::NorthWest, HexAbsoluteDir::NorthWest) => Some(HexDir::W),
			(HexAbsoluteDir::NorthWest, HexAbsoluteDir::NorthEast) => Some(HexDir::E),
			
			(HexAbsoluteDir::NorthEast, HexAbsoluteDir::East) => Some(HexDir::E),
			(HexAbsoluteDir::NorthEast, HexAbsoluteDir::SouthEast) => Some(HexDir::D),
			(HexAbsoluteDir::NorthEast, HexAbsoluteDir::SouthWest) => None,
			(HexAbsoluteDir::NorthEast, HexAbsoluteDir::West) => Some(HexDir::A),
			(HexAbsoluteDir::NorthEast, HexAbsoluteDir::NorthWest) => Some(HexDir::Q),
			(HexAbsoluteDir::NorthEast, HexAbsoluteDir::NorthEast) => Some(HexDir::W),
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HexCoord {
	pub q: i32,
	pub r: i32
}

impl HexCoord {
	pub fn to_cartesian(&self) -> Pos2 {
		let q = self.q as f32;
		let r = self.r as f32;
		return pos2(3.0_f32.sqrt() * q + 3.0_f32.sqrt()/2.0 * r, 3.0/2.0 * r);
	}

	// pub fn dir_to(&self, other: HexCoord) -> Option<HexAbsoluteDir> {
	// 	match other - *self {
	// 		HexCoord{q: 1, r: 0} => Some(HexAbsoluteDir::East),
	// 		HexCoord{q: 0, r: 1} => Some(HexAbsoluteDir::SouthEast),
	// 		HexCoord{q: -1, r: 1} => Some(HexAbsoluteDir::SouthWest),
	// 		HexCoord{q: -1, r: 0} => Some(HexAbsoluteDir::West),
	// 		HexCoord{q: 0, r: -1} => Some(HexAbsoluteDir::NorthWest),
	// 		HexCoord{q: 1, r: -1} => Some(HexAbsoluteDir::NorthEast),
	// 		_ => None
	// 	}
	// }

	pub fn from_cartesian(pos: Pos2) -> HexCoord {
		let fq = 3.0_f32.sqrt()/3.0 * pos.x - 1.0/3.0 * pos.y;
		let fr = 2.0/3.0 * pos.y;
		return HexCoord::round(fq, fr)
	}

	fn round(fq: f32, fr: f32) -> HexCoord {
		let fs = -fq - fr;
		let mut q = fq.round();
		let mut r = fr.round();
		let s = fs.round();

		let q_diff = (q - fq).abs();
		let r_diff = (r - fr).abs();
		let s_diff = (s - fs).abs();

		if q_diff > r_diff && q_diff > s_diff {
			q = -r-s
		} else if r_diff > s_diff {
			r = -q-s
		}

		return hex_coord(q as i32, r as i32)
	}
}

impl Add for HexCoord {
    type Output = HexCoord;

    fn add(self, rhs: Self) -> Self::Output {
        return hex_coord(self.q + rhs.q, self.r + rhs.r)
    }
}

impl Sub for HexCoord {
	type Output = HexCoord;

	fn sub(self, rhs: Self) -> Self::Output {
			return hex_coord(self.q - rhs.q, self.r - rhs.r)
	}
}

pub fn hex_coord(q: i32, r: i32) -> HexCoord {
	return HexCoord{ q, r }
}