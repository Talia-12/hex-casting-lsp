use chumsky::Parser;
use chumsky::primitive::custom;
use chumsky::text::Padding;
use chumsky::{prelude::*, stream::Stream};
use core::fmt;
use std::{collections::HashMap};
use tower_lsp::lsp_types::{SemanticTokenType};

use crate::iota_types::Iota;
use crate::semantic_token::{LEGEND_TYPE};

/// This is the parser and interpreter for the 'Foo' language. See `tutorial.md` in the repository's root to learn
/// about it.
pub type Span = std::ops::Range<usize>;
#[derive(Debug)]
pub struct ImCompleteSemanticToken {
	pub start: usize,
	pub length: usize,
	pub token_type: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Token {
	Null,
	Bool(bool),
	Num(String),
	Str(String),
	Ctrl(char),
	Ident(String),
	Comment(String),
	Entity,
}

impl fmt::Display for Token {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Token::Null => write!(f, "null"),
			Token::Bool(x) => write!(f, "{}", x),
			Token::Num(n) => write!(f, "{}", n),
			Token::Str(s) => write!(f, "{}", s),
			Token::Ctrl(c) => write!(f, "{}", c),
			Token::Ident(s) => write!(f, "{}", s),
			Token::Comment(s) => write!(f, "{}", s),
    	Token::Entity => write!(f, "Entity"),
		}
	}
}

/// A parser that accepts an identifier.
///
/// An identifier is defined as an ASCII alphabetic character or an underscore followed by any number of alphanumeric
/// characters, underscores, or apostrophes. The regex pattern for it is `[a-zA-Z_][a-zA-Z0-9_]*`.
fn ident() -> impl Parser<char, String, Error = Simple<char>> + Copy + Clone
{
	filter(|c: &char| c.is_ascii_alphabetic() || c == &'_')
		.map(Some)
		.chain::<char, Vec<_>, _>(
			filter(|c: &char| c.is_ascii_alphanumeric() || c == &'_' || c == &'\'').repeated(),
		)
		.collect()
}

fn lexer() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
	// A parser for numbers
	let num = text::int(10)
		.chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
		.collect::<String>()
		.map(Token::Num);

	// A parser for strings
	let str_ = just('"')
		.ignore_then(filter(|c| *c != '"').repeated())
		.then_ignore(just('"'))
		.collect::<String>()
		.map(Token::Str);

	// A parser for control characters (delimiters, colons, etc.)
	let ctrl = one_of("()[]{},:\n").map(|c| Token::Ctrl(c));

	// A parser for identifiers and keywords
	let ident = ident().map(|ident: String| match ident.as_str() {
		"true" => Token::Bool(true),
		"false" => Token::Bool(false),
		"null" => Token::Null,
		"Entity" => Token::Entity,
		_ => Token::Ident(ident),
	});

	let single_comment = just("//").then(none_of("\n").repeated())
		.map(|(comment_start, comment)| {
			comment_start.chars().chain(comment).collect()
		}).map(Token::Comment);
		

	let multi_comment = just("/*").then(take_until(just("*/")))
		.map(|(comment_start, (comment, comment_end))| {
			comment_start.chars().chain(comment).chain(comment_end.chars()).collect()
		}).map(Token::Comment);


	// A single token can be one of the above
	let token = num
		.or(str_)
		.or(ctrl)
		.or(ident)
		.or(single_comment)
		.or(multi_comment)
		.recover_with(skip_then_retry_until([]));


	// a parser for whitespace characters other than newlines.
	let non_newline_whitespace = (just(' ').or(just('\t'))).repeated();

	non_newline_whitespace.ignore_then(
			token.map_with_span(|tok, span| (tok, span))
		)
		.then_ignore(non_newline_whitespace)
		.repeated()
}

pub type Spanned<T> = (T, Span);

// An expression node in the AST. Children are spanned so we can generate useful runtime errors.
#[derive(Debug)]
pub enum Expr {
	Error,
	Value(Iota),
	List(Vec<Spanned<Self>>),
	Consideration(Box<Spanned<Self>>),
	IntroRetro(Vec<Spanned<Self>>),
}

#[allow(unused)]
impl Expr {
	/// Returns `true` if the expr is [`Error`].
	///
	/// [`Error`]: Expr::Error
	fn is_error(&self) -> bool {
		matches!(self, Self::Error)
	}

	/// Returns `true` if the expr is [`Value`].
	///
	/// [`Value`]: Expr::Value
	fn is_value(&self) -> bool {
		matches!(self, Self::Value(..))
	}

	fn try_into_value(self) -> Result<Iota, Self> {
		if let Self::Value(v) = self {
			Ok(v)
		} else {
			Err(self)
		}
	}

	fn as_value(&self) -> Option<&Iota> {
		if let Self::Value(v) = self {
			Some(v)
		} else {
			None
		}
	}
}

// A function node in the AST.
#[derive(Debug)]
pub struct Func {
	pub args: Vec<Spanned<String>>,
	pub body: Spanned<Expr>,
	pub name: Spanned<String>,
	pub span: Span,
}

fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
	recursive(|expr| {
		let raw_expr = recursive(|raw_expr| {
			let val = filter_map(|span, tok| match tok {
				Token::Null => Ok(Expr::Value(Iota::Null)),
				Token::Bool(x) => Ok(Expr::Value(Iota::Bool(x))),
				Token::Num(n) => Ok(Expr::Value(Iota::Num(n.parse().unwrap()))),
				Token::Str(s) => Ok(Expr::Value(Iota::Str(s))),
				_ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
			})
			.labelled("value");

			let ident = filter_map(|span: Span, tok| match tok {
				Token::Ident(ident) => Ok((ident.clone(), span)),
				_ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
			})
			.labelled("identifier");

			// A list of expressions
			let items = expr
				.clone()
				.chain(just(Token::Ctrl(',')).ignore_then(expr.clone()).repeated())
				.then_ignore(just(Token::Ctrl(',')).or_not())
				.or_not()
				.map(|item| item.unwrap_or_else(Vec::new));

			let list = items
				.clone()
				.delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')))
				.map(Expr::List);

			// 'Atoms' are expressions that contain no ambiguity
			let atom = val
				.or(list)
				// In Nano Rust, `print` is just a keyword, just like Python 2, for simplicity
				.map_with_span(|expr, span| (expr, span))
				// Atoms can also just be normal expressions, but surrounded with parentheses
				.or(expr
					.clone()
					.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')'))))
				// Attempt to recover anything that looks like a parenthesised expression but contains errors
				.recover_with(nested_delimiters(
					Token::Ctrl('('),
					Token::Ctrl(')'),
					[
						(Token::Ctrl('['), Token::Ctrl(']')),
						(Token::Ctrl('{'), Token::Ctrl('}')),
					],
					|span| (Expr::Error, span),
				))
				// Attempt to recover anything that looks like a list but contains errors
				.recover_with(nested_delimiters(
					Token::Ctrl('['),
					Token::Ctrl(']'),
					[
						(Token::Ctrl('('), Token::Ctrl(')')),
						(Token::Ctrl('{'), Token::Ctrl('}')),
					],
					|span| (Expr::Error, span),
				));

			atom // TODO: temp bad wrong
		});

		// Blocks are expressions but delimited with braces
		let block = expr
			.clone()
			.delimited_by(just(Token::Ctrl('{')), just(Token::Ctrl('}')))
			// Attempt to recover anything that looks like a block but contains errors
			.recover_with(nested_delimiters(
				Token::Ctrl('{'),
				Token::Ctrl('}'),
				[
					(Token::Ctrl('('), Token::Ctrl(')')),
					(Token::Ctrl('['), Token::Ctrl(']')),
				],
				|span| (Expr::Error, span),
			));

		block  // TODO: temp bad wrong
	})
}

// pub fn funcs_parser() -> impl Parser<Token, HashMap<String, Func>, Error = Simple<Token>> + Clone {

// }

pub(crate) struct Error {
	pub(crate) span: Span,
	pub(crate) msg: String,
}

pub fn parse(
	src: &str,
) -> (
	Option<HashMap<String, Func>>,
	Vec<Simple<String>>,
	Vec<ImCompleteSemanticToken>,
) {
	let (tokens, errs) = lexer().parse_recovery(src);

	let (ast, tokenize_errors, semantic_tokens) = if let Some(tokens) = tokens {
		// info!("Tokens = {:?}", tokens);
		let len = src.chars().count();
		// let (ast, parse_errs) =
		// 	funcs_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));
		
		let semantic_tokens = tokens
			.iter()
			.filter_map(|(token, span)| match token {
				Token::Null => None,
				Token::Bool(_) => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
						.iter()
						.position(|item| item == &SemanticTokenType::ENUM_MEMBER)
						.unwrap(),
				}),

				Token::Num(_) => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
						.iter()
						.position(|item| item == &SemanticTokenType::NUMBER)
						.unwrap(),
				}),
				Token::Str(_) => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
						.iter()
						.position(|item| item == &SemanticTokenType::STRING)
						.unwrap(),
				}),
				Token::Ctrl(_) => None,
				Token::Ident(_) => None,
				Token::Comment(_) => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::COMMENT)
							.unwrap(),
				}),
				Token::Entity => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::KEYWORD)
							.unwrap(),
				}),
			})
			.collect::<Vec<_>>();

		// println!("{:#?}", ast);
		// if let Some(funcs) = ast.filter(|_| errs.len() + parse_errs.len() == 0) {
		//     if let Some(main) = funcs.get("main") {
		//         assert_eq!(main.args.len(), 0);
		//         match eval_expr(&main.body, &funcs, &mut Vec::new()) {
		//             Ok(val) => println!("Return value: {}", val),
		//             Err(e) => errs.push(Simple::custom(e.span, e.msg)),
		//         }
		//     } else {
		//         panic!("No main function!");
		//     }
		// }
		todo!()
		// (ast, parse_errs, semantic_tokens)
	} else {
		(0, 0, 0)
		// (None, Vec::new(), vec![])
	};

	todo!()

	// let parse_errors = errs
	// 	.into_iter()
	// 	.map(|e| e.map(|c| c.to_string()))
	// 	.chain(
	// 		tokenize_errors
	// 			.into_iter()
	// 			.map(|e| e.map(|tok| tok.to_string())),
	// 	)
	// 	.collect::<Vec<_>>();

	// (ast, parse_errors, semantic_tokens)
	// .for_each(|e| {
	//     let report = match e.reason() {
	//         chumsky::error::SimpleReason::Unclosed { span, delimiter } => {}
	//         chumsky::error::SimpleReason::Unexpected => {}
	//         chumsky::error::SimpleReason::Custom(msg) => {}
	//     };
	// });
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_lexer() {
		let test_inputs: Vec<&str> = vec![
"{
	Mind's Reflection
	Compass' Purification
	Mind's Reflection
	Alidade's Purification
	Archer's Distillation
}",
"{
	/*
		This will blink the caster 5 blocks forward.
		Be careful not to end up in a block!
	*/
	Mind's Reflection
	Consideration: 5.0
	Blink
}",
"{
	{
		Reveal
	}
	Consideration: [ // an inserted list
		0.0,
		1.1,
		2.2,
		3.3
	]
	Thoth's Gambit
}"];

		let test_outputs: Vec<Vec<Token>> = vec![
			vec![
				Token::Ctrl('{'), Token::Ctrl('\n'),
					Token::Ident("Mind's".to_string()), Token::Ident("Reflection".to_string()), Token::Ctrl('\n'),
					Token::Ident("Compass'".to_string()), Token::Ident("Purification".to_string()), Token::Ctrl('\n'),
					Token::Ident("Mind's".to_string()), Token::Ident("Reflection".to_string()), Token::Ctrl('\n'),
					Token::Ident("Alidade's".to_string()), Token::Ident("Purification".to_string()), Token::Ctrl('\n'),
					Token::Ident("Archer's".to_string()), Token::Ident("Distillation".to_string()), Token::Ctrl('\n'),
				Token::Ctrl('}')],
			vec![
				Token::Ctrl('{'), Token::Ctrl('\n'),
					Token::Comment(
"/*
		This will blink the caster 5 blocks forward.
		Be careful not to end up in a block!
	*/".to_string()),  Token::Ctrl('\n'),
					Token::Ident("Mind's".to_string()), Token::Ident("Reflection".to_string()), Token::Ctrl('\n'),
					Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Num("5.0".to_string()), Token::Ctrl('\n'),
					Token::Ident("Blink".to_string()), Token::Ctrl('\n'),
				Token::Ctrl('}')],
			vec![
				Token::Ctrl('{'), Token::Ctrl('\n'),
					Token::Ctrl('{'), Token::Ctrl('\n'),
						Token::Ident("Reveal".to_string()), Token::Ctrl('\n'),
					Token::Ctrl('}'), Token::Ctrl('\n'),
					Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Ctrl('['), Token::Comment("// an inserted list".to_string()), Token::Ctrl('\n'),
						Token::Num("0.0".to_string()), Token::Ctrl(','), Token::Ctrl('\n'),
						Token::Num("1.1".to_string()), Token::Ctrl(','), Token::Ctrl('\n'),
						Token::Num("2.2".to_string()), Token::Ctrl(','), Token::Ctrl('\n'),
						Token::Num("3.3".to_string()), Token::Ctrl('\n'),
					Token::Ctrl(']'), Token::Ctrl('\n'),
					Token::Ident("Thoth's".to_string()), Token::Ident("Gambit".to_string()), Token::Ctrl('\n'),
				Token::Ctrl('}')],
		];

		for (test_input, test_output) in test_inputs.iter().zip(test_outputs) {
			assert_eq!(lexer().parse_recovery(*test_input).0.unwrap().iter().map(|(token, _range)| token.clone()).collect::<Vec<_>>(), test_output);
		}
	}
}