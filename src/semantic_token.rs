use std::collections::HashMap;

use tower_lsp::lsp_types::{SemanticTokenType};

use crate::hex_parsing::{Expr, Macro, ImCompleteSemanticToken, Spanned};

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

pub fn semantic_token_from_ast(ast: &HashMap<String, Macro>) -> Vec<ImCompleteSemanticToken> {
    let mut semantic_tokens = vec![];

    ast.iter().for_each(|(_func_name, function)| {
        function.args.iter().for_each(|(_, span)| {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::PARAMETER)
                    .unwrap(),
            });
        });
        let (_, span) = &function.name;
        semantic_tokens.push(ImCompleteSemanticToken {
            start: span.start,
            length: span.len(),
            token_type: LEGEND_TYPE
                .iter()
                .position(|item| item == &SemanticTokenType::FUNCTION)
                .unwrap(),
        });
        semantic_token_from_expr(&function.body, &mut semantic_tokens);
    });

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
        Expr::Consideration(_) => todo!(),
        Expr::IntroRetro(_) => todo!(),
        Expr::ConsideredIntroRetro(_) => todo!(),
    }
}
