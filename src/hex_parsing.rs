use chumsky::Parser;
use chumsky::Stream;
use chumsky::prelude::*;
use chumsky::text::newline;
use core::fmt;
use std::collections::HashMap;
use std::collections::HashSet;
use tower_lsp::lsp_types::SemanticTokenType;
use itertools::Itertools;

use crate::hex_pattern::HexAbsoluteDir;
use crate::hex_pattern::HexDir;
use crate::hex_pattern::HexPattern;
use crate::iota_types::HexPatternIota;
use crate::iota_types::Iota;
use crate::iota_types::IotaType;
use crate::matrix_helpers::vecs_to_dyn_matrix;
use crate::pattern_name_registry;
use crate::semantic_token::LEGEND_TYPE;

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
	HexAbsoluteDir(HexAbsoluteDir),
	Bookkeepers(String),
	Ident(String),
	Comment(String),
	Entity,
	Matrix,
	IotaType,
	EntityType,
	ItemType,
	Gate,
	Mote,
	Import,
	Define,
	Arrow,
}

impl Token {
	fn is_comment(&self) -> bool {
		if let Token::Comment(_) = self {
			true
		} else {
			false
		}
	}
}

impl fmt::Display for Token {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Token::Null => write!(f, "null"),
			Token::Bool(x) => write!(f, "{}", x),
			Token::Num(n) => write!(f, "{}", n),
			Token::Str(s) => write!(f, "{}", s),
			Token::Ctrl(c) => write!(f, "{}", c),
			Token::HexAbsoluteDir(dir) => write!(f, "{}", dir),
			Token::Bookkeepers(s) => write!(f, "{}", s),
			Token::Ident(s) => write!(f, "{}", s),
			Token::Comment(s) => write!(f, "{}", s),
    	Token::Entity => write!(f, "Entity"),
			Token::Matrix => write!(f, "Matrix"),
			Token::IotaType => write!(f, "IotaType"),
			Token::EntityType => write!(f, "EntityType"),
			Token::ItemType => write!(f, "ItemType"),
			Token::Gate => write!(f, "Gate"),
			Token::Mote => write!(f, "Mote"),
			Token::Import => write!(f, "import"),
    	Token::Define => write!(f, "define"),
    	Token::Arrow => write!(f, "\u{2192}"),
		}
	}
}

/// A parser that accepts an identifier.
///
/// An identifier is defined as an ASCII alphabetic character or an underscore followed by any number of alphanumeric
/// characters, underscores, or apostrophes. The regex pattern for it is `[a-zA-Z_][a-zA-Z0-9_]*`.
fn ident() -> impl Parser<char, String, Error = Simple<char>> + Copy + Clone
{
	filter(|c: &char| c.is_ascii_alphabetic() || c == &'_' || c == &'#')
		.map(Some)
		.chain::<char, Vec<_>, _>(
			filter(|c: &char| c.is_ascii_alphanumeric() || c == &'_' || c == &'/' || c == &'\'' || c == &'-').repeated(),
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
	let ctrl = one_of("()[]{},;:=").or(newline().map(|_| '\n')).map(|c| Token::Ctrl(c));

	// A parser for ops
	let ops = just("->").map(|_| Token::Arrow);

	// A parser for bookkeeper's gambit
	let bookkeeper = one_of("-v").repeated().at_least(1).collect().map(Token::Bookkeepers);

	// A parser for identifiers and keywords
	let ident = ident().map(|ident: String| match ident.as_str() {
		"true" => Token::Bool(true),
		"True" => Token::Bool(true),
		"false" => Token::Bool(false),
		"False" => Token::Bool(false),
		"null" => Token::Null,
		"Null" => Token::Null,
		"Entity" => Token::Entity,
		"Matrix" => Token::Matrix,
		"IotaType" => Token::IotaType,
		"EntityType" => Token::EntityType,
		"ItemType" => Token::ItemType,
		"Gate" => Token::Gate,
		"Mote" => Token::Mote,
		"#import" => Token::Import, // #import FILENAME
		"#define" => Token::Define, // #define Pattern Name (DIR aqwed) = a -> b { }
		_ => if let Some(absdir) = HexAbsoluteDir::from_str(&ident) { Token::HexAbsoluteDir(absdir) } else { Token::Ident(ident) },
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
		.or(ops)
		.or(bookkeeper)
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
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
	Error,
	Value(Iota),
	List(Vec<Spanned<Self>>),
	Consideration(Box<Spanned<Self>>, Span),
	IntroRetro(Vec<Spanned<Self>>),
	ConsideredIntroRetro(Vec<Spanned<Self>>),
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

// A function node in the AST. (macro)
#[derive(Debug, Clone, PartialEq)]
pub struct Macro {
	pub name: Spanned<String>,
	pub pattern: Spanned<HexPattern>,
	pub args: Vec<Spanned<String>>,
	pub return_type: Vec<Spanned<String>>,
	pub body: Spanned<Expr>,
	pub span: Span,
}

// string to HexPattern
fn hex_pattern_from_signature() -> impl Parser<Token, HexPatternIota, Error = Simple<Token>> + Clone {
	let pattern = filter_map(|span: Span, tok| match tok {
			Token::HexAbsoluteDir(absdir) => Ok(absdir),
			_ => Err(Simple::custom(span, format!("Expected a valid absolute direction, found {tok}")))
		})
		.then_ignore(just(Token::Ctrl(',')).or_not())
		.then(filter_map(|span, tok: Token| match tok.clone() {
			Token::Ident(dirs) => HexDir::from_str(&dirs).map_err(|err| Simple::custom(span, format!("Error {err:?} getting pattern dirs from {dirs}"))),
			_ => Err(Simple::custom(span, format!("Expected hex dirs (AQWEDS), found {tok}")))
		}))
		.map(|(start_dir, dirs)| HexPattern::new(start_dir, dirs));

	// make it so that it can parse HexPattern(DIR, DIRS) or just DIR, DIRS, or even DIR DIRS
	just(Token::Ident("HexPattern".to_string())).or_not()
		.ignore_then(pattern.clone().delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')'))))
		.or(pattern)
		.map(|pattern| pattern.into())
}

fn pattern_name() -> impl Parser<Token, String, Error = Simple<Token>> + Clone {
	filter_map(|span: Span, tok| match tok {
		Token::Num(n) => Ok(n),
		Token::Ident(name_part) => Ok(name_part),
		Token::Bookkeepers(name_part) => Ok(name_part),
		Token::Ctrl(name_part) => if name_part == ':' { Ok(name_part.to_string()) } else { Err(Simple::custom(span, format!("{name_part} is not a valid character for a pattern name."))) }
		_ => Err(Simple::custom(span, format!("Expected a valid section of a pattern name, found {tok}"))),
	})
	.repeated()
	.at_least(1)
	.labelled("pattern name")
	.map(|strings| strings.iter().fold("".to_string(), |mut acc, str| {
		if acc != "" && str != &":" {
			acc.push_str(" ");
		}
		acc.push_str(str);
		acc
	}))
}

// consumes a pattern name
fn hex_pattern_from_name() -> impl Parser<Token, HexPatternIota, Error = Simple<Token>> + Clone {
	let name = pattern_name();

	name.map(|name| {
		pattern_name_registry::registry_entry_from_name(&name)
			.map(|pattern| HexPatternIota::RegistryEntry(pattern))
			.unwrap_or_else(|_| HexPatternIota::MacroPreprocessed(name))
	})
	
}

fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
	recursive(|expr| {
		let non_braced_expr = recursive(|non_braced_expr| {
			let num = select! {
				Token::Num(n) => n.parse::<f64>().unwrap()
			}.labelled("num");

			let pos_int = select! {
				Token::Num(n) => n.parse::<usize>().unwrap()
			}.labelled("pos_int");

			let simple_iota = select! {
				Token::Null => Expr::Value(Iota::Null),
				Token::Bool(x) => Expr::Value(Iota::Bool(x)),
				Token::Num(x) => Expr::Value(Iota::Num(x.parse::<f64>().unwrap())),
				Token::Str(s) => Expr::Value(Iota::Str(s)),
			}.labelled("simple_iota");

			let vec3 = num.clone().separated_by(just(Token::Ctrl(','))).at_least(3).at_most(3)
				.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
				.map(|coords| Expr::Value(Iota::Vec3(coords[0], coords[1], coords[2])))
				.labelled("vec3");

			let pattern = hex_pattern_from_signature()
				.or(hex_pattern_from_name())
				.then_ignore(just(Token::Ctrl('\n')).or(just(Token::Ctrl(',')).rewind()))
				.map(|pattern| Expr::Value(Iota::Pattern(pattern)))
				.labelled("pattern");

			let entity = just(Token::Entity).ignore_then(select! {
				Token::Ident(name) => Expr::Value(Iota::Entity(name)),
			}.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))).labelled("entity");

			let matrix = just(Token::Matrix).ignore_then(
				num.clone().separated_by(just(Token::Ctrl(','))).separated_by(just(Token::Ctrl(';'))).try_map(|vecvec, span| {
					let maybe_dm = vecs_to_dyn_matrix(vecvec);

					maybe_dm
						.map(|dm| Expr::Value(Iota::Matrix(dm)))
						.ok_or_else(|| Simple::custom(span, "A matrix with each row and column containing the same number of entries."))
				}).delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
			).labelled("matrix");

			let iota_type = just(Token::IotaType).ignore_then(
				filter_map(|span: Span, tok| match &tok {
					Token::Ident(s) =>
						if let Some(i_type) = IotaType::get_type(&s) {
							Ok(Expr::Value(Iota::IotaType(i_type))) }
						else {
							Err(Simple::custom(span, "A valid iota type"))
						},
					_ => Err(Simple::custom(span, "A valid iota type")),
				}).delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
			).labelled("iota_type");

			let entity_type = just(Token::EntityType).ignore_then(
				select! { Token::Ident(s) => s }.repeated().at_least(1)
				.map(|name_sections| name_sections.into_iter().intersperse(' '.to_string()).collect())
				.map(|name| Expr::Value(Iota::EntityType(name)))
				.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
			).labelled("entity_type");

			let item_type = just(Token::ItemType).ignore_then(
				select! { Token::Ident(s) => s }.repeated().at_least(1)
				.map(|name_sections| name_sections.into_iter().intersperse(' '.to_string()).collect())
				.map(|name| Expr::Value(Iota::ItemType(name)))
				.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
			).labelled("item_type");

			let gate = just(Token::Gate).ignore_then(
				pos_int.clone()
					.map(|num| Expr::Value(Iota::Gate(num)))
					.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
			).labelled("item_type");

			let mote = just(Token::Mote).ignore_then(
				pos_int.clone()
					.map(|num| Expr::Value(Iota::Mote(num)))
					.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
			).labelled("item_type");

			let iota = simple_iota
				.or(vec3)
				.or(pattern)
				.or(entity)
				.or(matrix)
				.or(iota_type)
				.or(entity_type)
				.or(item_type)
				.or(gate)
				.or(mote)
				.delimited_by(just(Token::Ctrl('\n')).repeated(), just(Token::Ctrl('\n')).repeated());

			// A list of expressions
			let list = non_braced_expr
				.clone()
				.separated_by(just(Token::Ctrl(',')))
				.or_not()
				.map(|item| item.unwrap_or_else(Vec::new))
				.delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')))
				.map(Expr::List);

			// singular iotas that can be considered // TODO gonna have a separate parser for considered consideration TODOTODO figure out how to make it happy with unbalanced ones actually
			let considerable = iota.clone()
				.or(list.clone())
				.map_with_span(|considerable, span| (considerable, span));

			let consideration = just(vec![Token::Ident("Consideration".to_string()), Token::Ctrl(':')]).try_map(|_, span: Span|
					pattern_name_registry::get_consideration()
						.map(|consideration| (HexPatternIota::RegistryEntry(consideration), span.clone()))
						.map_err(|err| Simple::custom(span, format!("Registry error: {:?}", err)))
				).then(considerable)
				.map(|((_, consideration_span), (expr, expr_span)): (Spanned<HexPatternIota>, Spanned<Expr>)| Expr::Consideration(Box::new((expr, expr_span)), consideration_span));

			// 'Atoms' are expressions that contain no ambiguity
			let atom = consideration
				.or(iota)
				.or(list)
				.map_with_span(|expr, span| (expr, span))
				.then_ignore(just(Token::Ctrl('\n')).repeated())
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

				atom
			});

		// Blocks are expressions but delimited with braces
		let block = non_braced_expr.clone().or(expr.clone()).repeated()
			.delimited_by(
				just(Token::Ctrl('{')).then_ignore(just(Token::Ctrl('\n')).repeated()),
				just(Token::Ctrl('}')).then_ignore(just(Token::Ctrl('\n')).repeated())
			)
			.map_with_span(|exprs, span| (Expr::IntroRetro(exprs), span))
			.or(
				non_braced_expr.clone().or(expr.clone()).repeated()
					.delimited_by(
						just(vec![Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Ctrl('{')]).then_ignore(just(Token::Ctrl('\n')).repeated()),
						just(vec![Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Ctrl('}')])).then_ignore(just(Token::Ctrl('\n')).repeated())
					.map_with_span(|exprs, span| (Expr::ConsideredIntroRetro(exprs), span))
				)

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

		block
	})
}

pub fn macros_parser() -> impl Parser<Token, (HashMap<String, Macro>, HashMap<HexPattern, Macro>), Error = Simple<Token>> + Clone {
	let name = pattern_name().map_with_span(|name, span| (name, span));

	let type_half = select! { Token::Ident(s) => s }.map_with_span(|t, span| (t, span))
		.separated_by(just(Token::Ctrl(',')));

	let type_signature = type_half.clone().then_ignore(just(Token::Arrow)).then(type_half);

	let macro_parser = just(Token::Define)
		.ignore_then(name)
		.then(hex_pattern_from_signature().map_with_span(|pattern, span| (pattern.get_pattern(), span)))
		.then_ignore(just(Token::Ctrl('=')))
		.then(type_signature)
		.then(expr_parser())
		.map_with_span(|(((name, pattern), (args, return_type)), body): (((Spanned<String>, Spanned<HexPattern>), (_, _)), Spanned<Expr>), span: Span| {
			(name.clone(), pattern.clone(), Macro { args, return_type, body, name, pattern, span  })
		});

	just(Token::Ctrl('\n')).repeated()
		.ignore_then(macro_parser)
		.then_ignore(just(Token::Ctrl('\n')).repeated()).repeated().collect::<Vec<_>>().try_map(|ms: Vec<(Spanned<String>, Spanned<HexPattern>, Macro)>, _| {
			let mut macros_by_name = HashMap::new();
			let mut macros_by_pattern = HashMap::new();

			for ((name, name_span), (pattern, pattern_span), m) in ms {
				if macros_by_name.insert(name.clone(), m.clone()).is_some() {
					return Err(Simple::custom(name_span, format!("Macro '{name}' already exists")));
				}
				if macros_by_pattern.insert(pattern.clone(), m).is_some() {
					return Err(Simple::custom(pattern_span, format!("Macro with pattern '{pattern}' already exists")));
				}
			}

			Ok((macros_by_name, macros_by_pattern))
		})
}

#[derive(Debug, PartialEq, Clone)]
pub struct AST {
	pub macros_by_name: HashMap<String, Macro>,
	pub macros_by_pattern: HashMap<HexPattern, Macro>,
	pub main: Option<Spanned<Expr>>
}

impl From<(HashMap<String, Macro>, HashMap<HexPattern, Macro>, Option<Spanned<Expr>>)> for AST {
	fn from((macros_by_name, macros_by_pattern,main): (HashMap<String, Macro>, HashMap<HexPattern, Macro>, Option<Spanned<Expr>>)) -> Self {
		AST { macros_by_name, macros_by_pattern, main }
	}
}

impl AST {
	fn get_macro_ids(&self) -> Vec<(String, HexPattern)> {
		return self.macros_by_name.iter().map(|(name, hex_macro)| (name.clone(), hex_macro.pattern.0.clone())).collect()
	}
}

pub fn main_parser() -> impl Parser<Token, AST, Error = Simple<Token>> + Clone {
	macros_parser()
		.then_ignore(just(Token::Ctrl('\n')).repeated())
		.then(expr_parser().or_not())
		.map(|((macros_by_name, macros_by_pattern), main_body)| (macros_by_name, macros_by_pattern, main_body).into())
}

pub(crate) struct Error {
	pub(crate) span: Span,
	pub(crate) msg: String,
}

/// Takes in the AST with all the macro references represented as HexPatternIota::MacroPreprocessed, and replaces them with HexPatternIota::Macros with the
/// correct name and pattern. Also adds parse_errs for non-existant macro names, and for macro reference cycles.
fn finalise_macro_references(ast: &mut AST, parse_errs: &mut Vec<Simple<Token>>) {
	let mut dependency_map = HashMap::new();

	let macro_ids = ast.get_macro_ids();

	for (name, hex_macro) in &mut ast.macros_by_name {
		match &mut hex_macro.body.0 {
			Expr::Error => { },
			Expr::Value(_) => { },
			Expr::List(exprs) => finalise_macro_references_for_exprs(exprs, Some(name), &macro_ids, &mut dependency_map, parse_errs),
			Expr::Consideration(_, _) => { },
			Expr::IntroRetro(exprs) => finalise_macro_references_for_exprs(exprs, Some(name), &macro_ids, &mut dependency_map, parse_errs),
			Expr::ConsideredIntroRetro(exprs) => finalise_macro_references_for_exprs(exprs, Some(name), &macro_ids, &mut dependency_map, parse_errs),
		}
	}

	// TODO: find cyclic dependencies.

	if let Some(main) = &mut ast.main {
		finalise_macro_references_for_expr(main, None, &macro_ids, &mut dependency_map, parse_errs);
	}
}

fn finalise_macro_references_for_exprs<'a>(exprs: &mut Vec<Spanned<Expr>>, owner_macro_name: Option<&'a str>, macro_ids: &'a Vec<(String, HexPattern)>, dependency_map: &mut HashMap<&'a str, HashSet<&'a str>>, parse_errs: &mut Vec<Simple<Token>>) {
	for sub_expr in exprs {
		finalise_macro_references_for_expr(sub_expr, owner_macro_name, macro_ids, dependency_map, parse_errs)
	}
}

/// Takes in an expression, and finds all references to other macros inside it. For each reference found, replace the reference with a proper HexPatternIota::Macro, and add
/// the reference to the dependency map so that cyclic dependencies can be calculated later. If a reference has no referent, add a parse_err.
fn finalise_macro_references_for_expr<'a>((expr, span): &mut Spanned<Expr>, owner_macro_name: Option<&'a str>, macro_ids: &'a Vec<(String, HexPattern)>, dependency_map: &mut HashMap<&'a str, HashSet<&'a str>>, parse_errs: &mut Vec<Simple<Token>>) {
	match expr {
		Expr::Error => { },
		Expr::Value(value) => if let Iota::Pattern(pattern) = value {
			let to_replace_with = match pattern {
				HexPatternIota::HexPattern(macro_pattern) => Ok(macro_ids.iter().find(|&macro_id| *macro_pattern == macro_id.1)),
				HexPatternIota::RegistryEntry(_) => Ok(None),
				HexPatternIota::MacroPreprocessed(macro_name) => {
					macro_ids.iter()
						.find(|&macro_id| *macro_name == macro_id.0)
						.ok_or_else(|| Simple::<Token>::custom(span.clone(), format!("No macro with the name \"{macro_name}\" exists.")))
						.map(|m| Some(m))
				},
				HexPatternIota::Macro(_, _) => Ok(None),
			};

			match to_replace_with {
				Ok(to_replace_with) => if let Some((to_replace_with_name, to_replace_with_pattern)) = to_replace_with {
					*pattern = HexPatternIota::Macro(to_replace_with_name.clone(), to_replace_with_pattern.clone());
					
					if let Some(owner_macro_name) = owner_macro_name {
						dependency_map.entry(owner_macro_name).and_modify(|e| { e.insert(&to_replace_with_name); }).or_insert(vec![to_replace_with_name.as_str()].into_iter().collect());
					}
				},
				Err(err) => parse_errs.push(err),
			}
		},
		Expr::List(sub_exprs) => finalise_macro_references_for_exprs(sub_exprs, owner_macro_name, macro_ids, dependency_map, parse_errs),
		Expr::Consideration(considered, _) => finalise_macro_references_for_expr(considered, owner_macro_name, macro_ids, dependency_map, parse_errs),
		Expr::IntroRetro(sub_exprs) => finalise_macro_references_for_exprs(sub_exprs, owner_macro_name, macro_ids, dependency_map, parse_errs),
		Expr::ConsideredIntroRetro(sub_exprs) => finalise_macro_references_for_exprs(sub_exprs, owner_macro_name, macro_ids, dependency_map, parse_errs),
	}
}

pub fn parse(
	src: &str,
) -> (
	Option<AST>,
	Vec<Simple<String>>,
	Vec<ImCompleteSemanticToken>,
) {
	let (tokens, errs) = lexer().parse_recovery(src);

	let (ast, tokenize_errors, semantic_tokens) = if let Some(tokens) = tokens {
		// info!("Tokens = {:?}", tokens);
		
		// println!("asdftokens: {:#?}", &tokens);

		let semantic_tokens = tokens
			.iter()
			.filter_map(|(token, span)| match token {
				Token::Null => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
						.iter()
						.position(|item| item == &SemanticTokenType::ENUM_MEMBER)
						.unwrap(),
				}),
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
				Token::HexAbsoluteDir(_) => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
						.iter()
						.position(|item| item == &SemanticTokenType::ENUM_MEMBER)
						.unwrap(),
				}),
				Token::Bookkeepers(_) => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::PARAMETER)
							.unwrap(),
				}),
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
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
				Token::Matrix => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
				Token::IotaType => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
				Token::EntityType => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
				Token::ItemType => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
				Token::Gate => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
				Token::Mote => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::FUNCTION)
							.unwrap(),
				}),
    		Token::Import => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::KEYWORD)
							.unwrap(),
				}),
				Token::Define => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::KEYWORD)
							.unwrap(),
				}),
				Token::Arrow => Some(ImCompleteSemanticToken {
					start: span.start,
					length: span.len(),
					token_type: LEGEND_TYPE
							.iter()
							.position(|item| item == &SemanticTokenType::OPERATOR)
							.unwrap(),
				}),
			})
			.collect::<Vec<_>>();

		let len = src.chars().count();
		let (mut ast, mut parse_errs) =
			main_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter().filter(|(token, _)| !token.is_comment())));

		if let Some(ast) = &mut ast {
			finalise_macro_references(ast, &mut parse_errs)
		}

		// println!("{:#?}", ast);
		// if let Some((macros_by_name, macros_by_pattern, main_body)) = ast.filter(|_| errs.len() + parse_errs.len() == 0) {
		// 	if let Some(main_body) = main_body {
		// 		assert_eq!(main_body.args.len(), 0);
		// 		match eval_expr(&main_body.body, &funcs, &mut Vec::new()) {
		// 			Ok(val) => println!("Return value: {}", val),
		// 			Err(e) => errs.push(Simple::custom(e.span, e.msg)),
		// 		}
		// 	} else {
		// 		panic!("No main function!");
		// 	}
		// }
		(ast, parse_errs, semantic_tokens)
	} else {
		(None, Vec::new(), vec![])
	};

	let parse_errors = errs
		.into_iter()
		.map(|e| e.map(|c| c.to_string()))
		.chain(
			tokenize_errors
				.into_iter()
				.map(|e| e.map(|tok| tok.to_string())),
		)
		.collect::<Vec<_>>();

	(ast, parse_errors, semantic_tokens)
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
	use std::{fs::File, io::{BufWriter, Write}};

use crate::{hex_pattern::{HexAbsoluteDir, HexDir}, pattern_name_registry::{StatOrDynRegistryEntry, RegistryEntry, registry_entry_from_id, registry_entry_from_name}};

	use super::*;

	fn test_inputs() -> Vec<String> {
return vec![
"{
	Mind's Reflection
	Compass' Purification
	Mind's Reflection
	Alidade's Purification
	Archer's Distillation
}".to_string(),
"{
	/*
		This will blink the caster 5 blocks forward.
		Be careful not to end up in a block!
	*/
	Mind's Reflection
	Consideration: 5.0
	Blink
}".to_string(),
"{
	Consideration: {
		Reveal
	Consideration: }
	Consideration: [ // an inserted list
		0.0,
		1.1,
		2.2,
		3.3
	]
	Thoth's Gambit
}".to_string(),
"#define New Distillation (SOUTHEAST aqwed) = int, int -> int {
	Bookkeeper's Gambit: v-
}

{
	Numerical Reflection: 0
	Numerical Reflection: 1
	New Distillation
	Reveal
}".to_string()]
	}

	#[test]
	fn test_lexer() {
		let test_inputs: Vec<String> = test_inputs();

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
					Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Ctrl('{'), Token::Ctrl('\n'),
						Token::Ident("Reveal".to_string()), Token::Ctrl('\n'),
					Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Ctrl('}'), Token::Ctrl('\n'),
					Token::Ident("Consideration".to_string()), Token::Ctrl(':'), Token::Ctrl('['), Token::Comment("// an inserted list".to_string()), Token::Ctrl('\n'),
						Token::Num("0.0".to_string()), Token::Ctrl(','), Token::Ctrl('\n'),
						Token::Num("1.1".to_string()), Token::Ctrl(','), Token::Ctrl('\n'),
						Token::Num("2.2".to_string()), Token::Ctrl(','), Token::Ctrl('\n'),
						Token::Num("3.3".to_string()), Token::Ctrl('\n'),
					Token::Ctrl(']'), Token::Ctrl('\n'),
					Token::Ident("Thoth's".to_string()), Token::Ident("Gambit".to_string()), Token::Ctrl('\n'),
				Token::Ctrl('}')],
			vec![
				Token::Define, Token::Ident("New".to_string()), Token::Ident("Distillation".to_string()),
					Token::Ctrl('('), Token::HexAbsoluteDir(HexAbsoluteDir::SouthEast), Token::Ident("aqwed".to_string()), Token::Ctrl(')'),
					Token::Ctrl('='), Token::Ident("int".to_string()), Token::Ctrl(','), Token::Ident("int".to_string()), Token::Arrow, Token::Ident("int".to_string()),  Token::Ctrl('{'),  Token::Ctrl('\n'),
					Token::Ident("Bookkeeper's".to_string()), Token::Ident("Gambit".to_string()), Token::Ctrl(':'), Token::Bookkeepers("v-".to_string()), Token::Ctrl('\n'),
				Token::Ctrl('}'), Token::Ctrl('\n'),
				Token::Ctrl('\n'),
				Token::Ctrl('{'), Token::Ctrl('\n'),
					Token::Ident("Numerical".to_string()), Token::Ident("Reflection".to_string()), Token::Ctrl(':'), Token::Num("0".to_string()), Token::Ctrl('\n'),
					Token::Ident("Numerical".to_string()), Token::Ident("Reflection".to_string()), Token::Ctrl(':'), Token::Num("1".to_string()), Token::Ctrl('\n'),
					Token::Ident("New".to_string()), Token::Ident("Distillation".to_string()), Token::Ctrl('\n'),
					Token::Ident("Reveal".to_string()), Token::Ctrl('\n'),
				Token::Ctrl('}'),]
		];

		for (test_input, test_output) in test_inputs.iter().zip(test_outputs) {
			assert_eq!(lexer().parse_recovery(test_input.as_str()).0.unwrap().iter().map(|(token, _range)| token.clone()).collect::<Vec<_>>(), test_output);
		}
	}

	#[test]
	fn test_hex_pattern_from_signature() {
		let test_inputs = vec!["SOUTHWEST aqweqa", "North_east, qaq", "HexPattern(NORTHWEST, asd)"];
		let test_outputs = vec![
			HexPattern::new(HexAbsoluteDir::SouthWest, vec![HexDir::A, HexDir::Q, HexDir::W, HexDir::E, HexDir::Q, HexDir::A]),
			HexPattern::new(HexAbsoluteDir::NorthEast, vec![HexDir::Q, HexDir::A, HexDir::Q]),
			HexPattern::new(HexAbsoluteDir::NorthWest, vec![HexDir::A, HexDir::S, HexDir::D])
		];

		for (input, output) in test_inputs.iter().zip(test_outputs) {
			let tokens = lexer().parse(*input).unwrap().into_iter().map(|(token, _span)| token).collect::<Vec<_>>();
			let pattern = hex_pattern_from_signature().parse(tokens).unwrap();

			assert_eq!(pattern.get_pattern(), output);
		}
	}

	#[test]
	fn test_hex_pattern_from_name() {
		let test_inputs = vec!["Mind's Reflection", "Blink", "Bookkeeper's Gambit: v-"];

		let minds_reflection: RegistryEntry = RegistryEntry {
			name: "Mind's Reflection".to_string(),
			id: "get_caster".to_string(),
			mod_name: "Hex Casting".to_string(), 
			pattern: Some(HexPattern::new(HexAbsoluteDir::NorthEast, vec![HexDir::Q, HexDir::A, HexDir::Q])),
			args: Some("\u{2192} entity".to_string()),
			url: Some("https://gamma-delta.github.io/HexMod/#patterns/basics@hexcasting:get_caster".to_string())
		};
		let blink: RegistryEntry = RegistryEntry {
			name: "Blink".to_string(),
			id: "blink".to_string(),
			mod_name: "Hex Casting".to_string(), 
			pattern: Some(HexPattern::new(HexAbsoluteDir::SouthWest, vec![HexDir::A, HexDir::W, HexDir::Q, HexDir::Q, HexDir::Q, HexDir::W, HexDir::A, HexDir::Q])),
			args: Some("entity, number \u{2192}".to_string()),
			url: Some("https://gamma-delta.github.io/HexMod/#patterns/spells/basic@hexcasting:blink".to_string())
		};
		let bookkeepers_gambit_v_ = RegistryEntry {
			name: "Bookkeeper's Gambit: v-".to_string(),
			id: "mask".to_string(),
			mod_name: "Hex Casting".to_string(), 
			pattern: None,
			args: Some("many \u{2192} many".to_string()),
			url: Some("https://gamma-delta.github.io/HexMod/#patterns/stackmanip@hexcasting:mask".to_string())
		};

		// Mind's Reflection
		{
			let tokens = lexer().parse(test_inputs[0]).unwrap().into_iter().map(|(token, _span)| token).collect::<Vec<_>>();
			dbg!(&tokens);
			let pattern = hex_pattern_from_name().parse(tokens).unwrap();

			if let HexPatternIota::RegistryEntry(entry) = pattern {
				if let StatOrDynRegistryEntry::StatRegistryEntry(entry) = entry {
					assert_eq!(*entry, minds_reflection);
				} else {
					panic!()
				}
			} else {
				panic!()
			}
		}

		// Blink
		{
			let tokens = lexer().parse(test_inputs[1]).unwrap().into_iter().map(|(token, _span)| token).collect::<Vec<_>>();
			dbg!(&tokens);
			let pattern = hex_pattern_from_name().parse(tokens).unwrap();

			if let HexPatternIota::RegistryEntry(entry) = pattern {
				if let StatOrDynRegistryEntry::StatRegistryEntry(entry) = entry {
					assert_eq!(*entry, blink);
				} else {
					panic!()
				}
			} else {
				panic!()
			}
		}

		// Bookkeeper's Gambit: v-
		{
			let tokens = lexer().parse(test_inputs[2]).unwrap().into_iter().map(|(token, _span)| token).collect::<Vec<_>>();
			dbg!(&tokens);
			let pattern = hex_pattern_from_name().parse(tokens).unwrap();

			dbg!(&pattern);

			if let HexPatternIota::RegistryEntry(entry) = pattern {
				if let StatOrDynRegistryEntry::DynRegistryEntry(entry) = entry {
					assert_eq!(entry, bookkeepers_gambit_v_);
				} else {
					panic!()
				}
			} else {
				panic!()
			}
		}
	}

	#[test]
	fn test_list_parsing() {
		let test_input = "Consideration: {
	[1, 2, 3]
	[
		1,
		2,
		Mind's Reflection,
		4
	]		
Consideration: }";
	
		let minds_reflection = registry_entry_from_name("Mind's Reflection").unwrap();

		let test_output: AST = (
			HashMap::new(),
			HashMap::new(),
			Some((
				Expr::ConsideredIntroRetro(vec![
					(Expr::List(vec![(Expr::Value(Iota::Num(1.0)), 19..20), (Expr::Value(Iota::Num(2.0)), 22..23), (Expr::Value(Iota::Num(3.0)), 25..26)]), 18..27),
					(Expr::List(vec![(Expr::Value(Iota::Num(1.0)), 30..34), (Expr::Value(Iota::Num(2.0)), 35..39),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(minds_reflection))), 40..60),
					(Expr::Value(Iota::Num(4.0)), 61..66)]), 29..68)
				]), 0..87
			))
		).into();

		let (ast, errs, semantic_tokens) = parse(test_input);
		let ast = ast.unwrap();
			
		// if errs.len() != 0 {
			dbg!(test_input);
			dbg!(errs);
			dbg!(semantic_tokens);
		// }

		assert_eq!(ast, test_output)
	}

	#[test]
	fn test_all_parsing() {
		let test_inputs: Vec<String> = test_inputs();

		let minds_reflection = registry_entry_from_name("Mind's Reflection").unwrap();
		let compass_purification = registry_entry_from_name("Compass' Purification").unwrap();
		let alidades_purification = registry_entry_from_name("Alidade's Purification").unwrap();
		let archers_distillation = registry_entry_from_name("Archer's Distillation").unwrap();
		let blink = registry_entry_from_name("Blink").unwrap();
		let reveal = registry_entry_from_name("Reveal").unwrap();
		let thoths_gambit = registry_entry_from_name("Thoth's Gambit").unwrap();
		let bookkeepers_gambit_v_ = registry_entry_from_name("Bookkeeper's Gambit: v-").unwrap();
		let numerical_reflection_0 = registry_entry_from_name("Numerical Reflection: 0").unwrap();
		let numerical_reflection_1 = registry_entry_from_name("Numerical Reflection: 1").unwrap();

		let test_outputs: Vec<AST> = vec![
			(HashMap::new(), HashMap::new(), Some((
				Expr::IntroRetro(vec![
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(minds_reflection.clone()))), 3..21),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(compass_purification.clone()))), 22..44),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(minds_reflection.clone()))), 45..63),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(alidades_purification.clone()))), 64..87),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(archers_distillation.clone()))), 88..110),
				]),
				0..111
			))).into(),
			(HashMap::new(), HashMap::new(), Some((
				Expr::IntroRetro(vec![
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(minds_reflection.clone()))), 97..115),
					(Expr::Consideration(Box::new((Expr::Value(Iota::Num(5.0)), 131..135)), 116..130), 116..135),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(blink.clone()))), 136..142),
				]),
				0..143
			))).into(),
			(HashMap::new(), HashMap::new(), Some((
				Expr::IntroRetro(vec![
					(Expr::ConsideredIntroRetro(vec![
						(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(reveal.clone()))), 22..29),
					]), 3..47),
					(Expr::Consideration(Box::new((Expr::List(vec![
						(Expr::Value(Iota::Num(0.0)), 84..90),
						(Expr::Value(Iota::Num(1.1)), 91..97),
						(Expr::Value(Iota::Num(2.2)), 98..104),
						(Expr::Value(Iota::Num(3.3)), 105..112),
					]), 63..114)), 48..62), 48..114),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(thoths_gambit.clone()))), 116..131),
				]),
				0..132
			))).into(),
			({
				let mut map = HashMap::new();
				map.insert("New Distillation".to_string(), Macro {
					args: vec![("int".to_string(), 45..48), ("int".to_string(), 50..53)],
					return_type: vec![("int".to_string(), 57..60)],
					body: (
						Expr::IntroRetro(vec![
							(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(bookkeepers_gambit_v_.clone()))), 64..88)
						]),
						61..91
					),
					name: ("New Distillation".to_string(), 8..24),
					pattern: (HexPattern::new(HexAbsoluteDir::SouthEast, vec![HexDir:: A, HexDir::Q, HexDir::W, HexDir::E, HexDir::D]), 25..42),
					span: 0..91
				});
				map
			}, {
				let mut map = HashMap::new();
				map.insert(HexPattern::new(HexAbsoluteDir::SouthEast, vec![HexDir:: A, HexDir::Q, HexDir::W, HexDir::E, HexDir::D]), Macro {
					args: vec![("int".to_string(), 45..48), ("int".to_string(), 50..53)],
					return_type: vec![("int".to_string(), 57..60)],
					body: (
						Expr::IntroRetro(vec![
							(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(bookkeepers_gambit_v_.clone()))), 64..88)
						]),
						61..91
					),
					name: ("New Distillation".to_string(), 8..24),
					pattern: (HexPattern::new(HexAbsoluteDir::SouthEast, vec![HexDir:: A, HexDir::Q, HexDir::W, HexDir::E, HexDir::D]), 25..42),
					span: 0..91
				});
				map
			}, Some((
				Expr::IntroRetro(vec![
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(numerical_reflection_0.clone()))), 94..118),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(numerical_reflection_1.clone()))), 119..143),
					(Expr::Value(Iota::Pattern(HexPatternIota::Macro("New Distillation".to_string(), HexPattern::new(HexAbsoluteDir::SouthEast, vec![HexDir::A, HexDir::Q, HexDir::W, HexDir::E, HexDir::D])))), 144..161),
					(Expr::Value(Iota::Pattern(HexPatternIota::RegistryEntry(reveal.clone()))), 162..169),
				]),
				91..170
			))).into(),
		];

		for (input, output) in test_inputs.iter().zip(test_outputs) {
			let (ast, errs, semantic_tokens) = parse(input);
			let ast = ast.unwrap();
			
			// if errs.len() != 0 {
				dbg!(input);
				dbg!(errs);
				dbg!(semantic_tokens);
			// }

			assert_eq!(ast, output)
		}
	}

	#[test]
	#[ignore]
	fn test_asdf() {
    let source = include_str!("../examples/stronghold_finder.nrs");
		
		{
			let (tokens, errs) = lexer().parse_recovery(source);

			{
				let file = File::create("examples/stronghold_finder_parsed.txt").unwrap();
				let mut writer = BufWriter::new(file);

				writer.write(
					format!("{tokens:#?}").as_bytes()
				).unwrap();
			}
		}
		
		println!("asdf_source: {:?}", &source);
    let (ast, errors, semantic_tokens) = parse(source);
		println!("asdf_errors: {:?}", errors);
    if let Some(ref ast) = ast {
        println!("asdf_ast: {:#?}", ast);
    } else {
        println!("asdf_errors: {:?}", errors);
    }
    println!("asdf_semantic_tokens: {:?}", semantic_tokens);
		panic!()
	}
}