use std::collections::HashMap;

use tower_lsp::lsp_types::{SemanticTokenType};

use crate::hex_parsing::{Expr, Macro, ImCompleteSemanticToken, Spanned, AST};

pub const LEGEND_TYPE: &[SemanticTokenType] = &[
    SemanticTokenType::FUNCTION,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::STRING,
    SemanticTokenType::COMMENT,
    SemanticTokenType::NUMBER,
    SemanticTokenType::KEYWORD,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::PARAMETER,
		SemanticTokenType::ENUM_MEMBER
];

pub fn semantic_token_from_ast(ast: &AST) -> Vec<ImCompleteSemanticToken> {
    let mut semantic_tokens = vec![];

    ast.macros_by_name.iter().for_each(|(_func_name, hex_macro)| {
        hex_macro.args.iter().for_each(|(_, span)| {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::PARAMETER)
                    .unwrap(),
            });
        });
        let (_, span) = &hex_macro.name;
        semantic_tokens.push(ImCompleteSemanticToken {
            start: span.start,
            length: span.len(),
            token_type: LEGEND_TYPE
                .iter()
                .position(|item| item == &SemanticTokenType::FUNCTION)
                .unwrap(),
        });
        semantic_token_from_expr(&hex_macro.body, &mut semantic_tokens);
    });

		if let Some(main) = &ast.main {
			semantic_token_from_expr(main, &mut semantic_tokens)
		}

    semantic_tokens
}

pub fn semantic_token_from_expr(
    expr: &Spanned<Expr>,
    semantic_tokens: &mut Vec<ImCompleteSemanticToken>,
) {
    match &expr.0 {
        Expr::Error => {}
        Expr::Value(_) => {}
        Expr::List(_) => {}
        Expr::Consideration(_) => {},
        Expr::IntroRetro(_) => {},
        Expr::ConsideredIntroRetro(_) => {},
    }
}
