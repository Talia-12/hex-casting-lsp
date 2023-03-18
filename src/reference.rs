use std::collections::HashMap;

use chumsky::Span;
use im_rc::Vector;


use crate::hex_parsing::{Expr, Macro, Spanned, AST};
#[derive(Debug, Clone)]
pub enum ReferenceSymbol {
	Founded(Spanned<String>),
	Founding(usize),
}

pub fn get_reference(
	ast: &AST,
	ident_offset: usize,
	include_self: bool,
) -> Vec<Spanned<String>> {
	let mut vector = Vector::new();
	let mut reference_list = vec![];
	// for (_, v) in ast.iter() {
	//     if v.name.1.end < ident_offset {
	//         vector.push_back(v.name.clone());
	//     }
	// }
	let mut kv_list = ast.macros_by_name.iter().collect::<Vec<_>>();
	kv_list.sort_by(|a, b| a.1.name.start().cmp(&b.1.name.start()));
	let mut reference_symbol = ReferenceSymbol::Founding(ident_offset);
	// let mut fn_vector = Vector::new();
	for (_, v) in kv_list {
		let (_, range) = &v.name;
		if ident_offset >= range.start && ident_offset < range.end {
			reference_symbol = ReferenceSymbol::Founded(v.name.clone());
			if include_self {
				reference_list.push(v.name.clone());
			}
		};
		vector.push_back(v.name.clone());
		let args = v
			.args
			.iter()
			.map(|arg| {
				if ident_offset >= arg.1.start && ident_offset < arg.1.end {
					reference_symbol = ReferenceSymbol::Founded(arg.clone());
					if include_self {
						reference_list.push(arg.clone());
					}
				}
				arg.clone()
			})
			.collect::<Vector<_>>();
		get_reference_of_expr(
			&v.body,
			args + vector.clone(),
			reference_symbol.clone(),
			&mut reference_list,
			include_self,
		);
	}
	reference_list
}

pub fn get_reference_of_expr(
	expr: &Spanned<Expr>,
	definition_ass_list: Vector<Spanned<String>>,
	reference_symbol: ReferenceSymbol,
	reference_list: &mut Vec<Spanned<String>>,
	include_self: bool,
) {
	match &expr.0 {
		Expr::Error => {}
		Expr::Value(_) => {},
		Expr::List(_) => {},
		Expr::Consideration(_) => {},
		Expr::IntroRetro(_) => {},
		Expr::ConsideredIntroRetro(_) => {},
	}
}
