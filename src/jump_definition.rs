use std::collections::HashMap;

use im_rc::Vector;
use log::debug;
use tower_lsp::{lsp_types::MessageType, Client};

use crate::hex_parsing::{Expr, Macro, Spanned, AST};
/// return (need_to_continue_search, founded reference)
pub fn get_definition(
    ast: &AST,
    ident_offset: usize,
) -> Option<Spanned<String>> {
    let mut vector = Vector::new();
    for (_, v) in ast.macros_by_name.iter() {
        if v.name.1.start < ident_offset && v.name.1.end > ident_offset {
            return Some(v.name.clone());
        }
        if v.name.1.end < ident_offset {
            vector.push_back(v.name.clone());
        }
    }

    for (_, v) in ast.macros_by_name.iter() {
        let args = v.args.iter().map(|arg| arg.clone()).collect::<Vector<_>>();
        match get_definition_of_expr(&v.body, args + vector.clone(), ident_offset) {
            (_, Some(value)) => {
                return Some(value);
            }
            _ => {}
        }
    }
    None
}

pub fn get_definition_of_expr(
    expr: &Spanned<Expr>,
    definition_ass_list: Vector<Spanned<String>>,
    ident_offset: usize,
) -> (bool, Option<Spanned<String>>) {
    match &expr.0 {
        Expr::Error => (true, None),
        Expr::Value(_) => (true, None),
        Expr::List(exprs) => (true, None),
        Expr::Consideration(_, _) => (true, None),
        Expr::IntroRetro(_) => (true, None),
        Expr::ConsideredIntroRetro(_) => (true, None),
    }
}
