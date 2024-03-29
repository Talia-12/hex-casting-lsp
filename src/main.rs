use std::collections::HashMap;

use chumsky::Span;
use dashmap::DashMap;
use hex_language_server::hex_parsing::{parse, Macro, ImCompleteSemanticToken, AST, Expr, Spanned};
use hex_language_server::completion::completion;
use hex_language_server::jump_definition::get_definition;
use hex_language_server::reference::get_reference;
use hex_language_server::semantic_token::{semantic_token_from_ast, LEGEND_TYPE};
use ropey::Rope;
use serde_json::Value;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
#[derive(Debug)]
struct Backend {
	client: Client,
	ast_map: DashMap<String, AST>,
	document_map: DashMap<String, Rope>,
	semantic_token_map: DashMap<String, Vec<ImCompleteSemanticToken>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
	async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
		Ok(InitializeResult {
			server_info: None,
			offset_encoding: None,
			capabilities: ServerCapabilities {
				inlay_hint_provider: None,
				text_document_sync: Some(TextDocumentSyncCapability::Kind(
					TextDocumentSyncKind::FULL,
				)),
				completion_provider: Some(CompletionOptions {
					resolve_provider: Some(false),
					trigger_characters: Some(vec![".".to_string()]),
					work_done_progress_options: Default::default(),
					all_commit_characters: None,
					completion_item: None,
				}),
				execute_command_provider: Some(ExecuteCommandOptions {
					commands: vec!["dummy.do_something".to_string()],
					work_done_progress_options: Default::default(),
				}),

				workspace: Some(WorkspaceServerCapabilities {
					workspace_folders: Some(WorkspaceFoldersServerCapabilities {
						supported: Some(true),
						change_notifications: Some(OneOf::Left(true)),
					}),
					file_operations: None,
				}),
				semantic_tokens_provider: Some(
					SemanticTokensServerCapabilities::SemanticTokensRegistrationOptions(
						SemanticTokensRegistrationOptions {
							text_document_registration_options: {
								TextDocumentRegistrationOptions {
									document_selector: Some(vec![DocumentFilter {
										language: Some("nrs".to_string()),
										scheme: Some("file".to_string()),
										pattern: None,
									}]),
								}
							},
							semantic_tokens_options: SemanticTokensOptions {
								work_done_progress_options: WorkDoneProgressOptions::default(),
								legend: SemanticTokensLegend {
									token_types: LEGEND_TYPE.into(),
									token_modifiers: vec![],
								},
								range: Some(true),
								full: Some(SemanticTokensFullOptions::Bool(true)),
							},
							static_registration_options: StaticRegistrationOptions::default(),
						},
					),
				),
				// definition: Some(GotoCapability::default()),
				definition_provider: Some(OneOf::Left(true)),
				references_provider: Some(OneOf::Left(true)),
				rename_provider: Some(OneOf::Left(true)),
				hover_provider: Some(HoverProviderCapability::Simple(true)),
				..ServerCapabilities::default()
			},
		})
	}
	async fn initialized(&self, _: InitializedParams) {
		self.client
			.log_message(MessageType::INFO, "initialized!")
			.await;
	}

	async fn shutdown(&self) -> Result<()> {
		Ok(())
	}

	async fn did_open(&self, params: DidOpenTextDocumentParams) {
		self.client
			.log_message(MessageType::INFO, "file opened!")
			.await;
		self.on_change(TextDocumentItem {
			uri: params.text_document.uri,
			text: params.text_document.text,
			version: params.text_document.version,
		})
		.await
	}

	async fn did_change(&self, mut params: DidChangeTextDocumentParams) {
		self.on_change(TextDocumentItem {
			uri: params.text_document.uri,
			text: std::mem::take(&mut params.content_changes[0].text),
			version: params.text_document.version,
		})
		.await
	}

	async fn did_save(&self, _: DidSaveTextDocumentParams) {
		self.client
			.log_message(MessageType::INFO, "file saved!")
			.await;
	}
	async fn did_close(&self, _: DidCloseTextDocumentParams) {
		self.client
			.log_message(MessageType::INFO, "file closed!")
			.await;
	}

	async fn goto_definition(
		&self,
		params: GotoDefinitionParams,
	) -> Result<Option<GotoDefinitionResponse>> {
		let definition = async {
			let uri = params.text_document_position_params.text_document.uri;
			let ast = self.ast_map.get(uri.as_str())?;
			let rope = self.document_map.get(uri.as_str())?;

			let position = params.text_document_position_params.position;
			let char = rope.try_line_to_char(position.line as usize).ok()?;
			let offset = char + position.character as usize;
			// self.client.log_message(MessageType::INFO, &format!("{:#?}, {}", ast.value(), offset)).await;
			let span = get_definition(&ast, offset);
			self.client
				.log_message(MessageType::INFO, &format!("{:?}, ", span))
				.await;
			span.and_then(|(_, range)| {
				let start_position = offset_to_position(range.start, &rope)?;
				let end_position = offset_to_position(range.end, &rope)?;

				let range = Range::new(start_position, end_position);

				Some(GotoDefinitionResponse::Scalar(Location::new(uri, range)))
			})
		}
		.await;
		Ok(definition)
	}
	async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
		let reference_list = || -> Option<Vec<Location>> {
			let uri = params.text_document_position.text_document.uri;
			let ast = self.ast_map.get(&uri.to_string())?;
			let rope = self.document_map.get(&uri.to_string())?;

			let position = params.text_document_position.position;
			let char = rope.try_line_to_char(position.line as usize).ok()?;
			let offset = char + position.character as usize;
			let reference_list = get_reference(&ast, offset, false);
			let ret = reference_list
				.into_iter()
				.filter_map(|(_, range)| {
					let start_position = offset_to_position(range.start, &rope)?;
					let end_position = offset_to_position(range.end, &rope)?;

					let range = Range::new(start_position, end_position);

					Some(Location::new(uri.clone(), range))
				})
				.collect::<Vec<_>>();
			Some(ret)
		}();
		Ok(reference_list)
	}

	async fn semantic_tokens_full(
		&self,
		params: SemanticTokensParams,
	) -> Result<Option<SemanticTokensResult>> {
		let uri = params.text_document.uri.to_string();
		self.client
			.log_message(MessageType::LOG, "semantic_token_full")
			.await;
		let semantic_tokens = || -> Option<Vec<SemanticToken>> {
			let mut im_complete_tokens = self.semantic_token_map.get_mut(&uri)?;
			let rope = self.document_map.get(&uri)?;
			let ast = self.ast_map.get(&uri)?;
			let extends_tokens = semantic_token_from_ast(&ast);
			im_complete_tokens.extend(extends_tokens);
			im_complete_tokens.sort_by(|a, b| a.start.cmp(&b.start));
			let mut pre_line = 0;
			let mut pre_start = 0;
			let semantic_tokens = im_complete_tokens
				.iter()
				.filter_map(|token| {
					let line = rope.try_byte_to_line(token.start as usize).ok()? as u32;
					let first = rope.try_line_to_char(line as usize).ok()? as u32;
					let start = rope.try_byte_to_char(token.start as usize).ok()? as u32 - first;
					let delta_line = line - pre_line;
					let delta_start = if delta_line == 0 {
						start - pre_start
					} else {
						start
					};
					let ret = Some(SemanticToken {
						delta_line,
						delta_start,
						length: token.length as u32,
						token_type: token.token_type as u32,
						token_modifiers_bitset: 0,
					});
					pre_line = line;
					pre_start = start;
					ret
				})
				.collect::<Vec<_>>();
			Some(semantic_tokens)
		}();
		if let Some(semantic_token) = semantic_tokens {
			return Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
				result_id: None,
				data: semantic_token,
			})));
		}
		Ok(None)
	}

	async fn semantic_tokens_range(
		&self,
		params: SemanticTokensRangeParams,
	) -> Result<Option<SemanticTokensRangeResult>> {
		let uri = params.text_document.uri.to_string();
		let semantic_tokens = || -> Option<Vec<SemanticToken>> {
			let im_complete_tokens = self.semantic_token_map.get(&uri)?;
			let rope = self.document_map.get(&uri)?;
			let mut pre_line = 0;
			let mut pre_start = 0;
			let semantic_tokens = im_complete_tokens
				.iter()
				.filter_map(|token| {
					let line = rope.try_byte_to_line(token.start as usize).ok()? as u32;
					let first = rope.try_line_to_char(line as usize).ok()? as u32;
					let start = rope.try_byte_to_char(token.start as usize).ok()? as u32 - first;
					let ret = Some(SemanticToken {
						delta_line: line - pre_line,
						delta_start: if start >= pre_start {
							start - pre_start
						} else {
							start
						},
						length: token.length as u32,
						token_type: token.token_type as u32,
						token_modifiers_bitset: 0,
					});
					pre_line = line;
					pre_start = start;
					ret
				})
				.collect::<Vec<_>>();
			Some(semantic_tokens)
		}();
		if let Some(semantic_token) = semantic_tokens {
			return Ok(Some(SemanticTokensRangeResult::Tokens(SemanticTokens {
				result_id: None,
				data: semantic_token,
			})));
		}
		Ok(None)
	}

	async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
		let uri = params.text_document_position.text_document.uri;
		let position = params.text_document_position.position;
		let completions = || -> Option<Vec<CompletionItem>> {
			let rope = self.document_map.get(&uri.to_string())?;
			let ast = self.ast_map.get(&uri.to_string())?;
			let char = rope.try_line_to_char(position.line as usize).ok()?;
			let offset = char + position.character as usize;
			let completions = completion(&ast, offset);
			let mut ret = Vec::with_capacity(completions.len());
			for (_, item) in completions {
				match item {
					hex_language_server::completion::ImCompleteCompletionItem::Variable(var) => {
						ret.push(CompletionItem {
							label: var.clone(),
							insert_text: Some(var.clone()),
							kind: Some(CompletionItemKind::VARIABLE),
							detail: Some(var),
							..Default::default()
						});
					}
					hex_language_server::completion::ImCompleteCompletionItem::Function(
						name,
						args,
					) => {
						ret.push(CompletionItem {
							label: name.clone(),
							kind: Some(CompletionItemKind::FUNCTION),
							detail: Some(name.clone()),
							insert_text: Some(format!(
								"{}({})",
								name,
								args.iter()
									.enumerate()
									.map(|(index, item)| { format!("${{{}:{}}}", index + 1, item) })
									.collect::<Vec<_>>()
									.join(",")
							)),
							insert_text_format: Some(InsertTextFormat::SNIPPET),
							..Default::default()
						});
					}
				}
			}
			Some(ret)
		}();
		Ok(completions.map(CompletionResponse::Array))
	}

	async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
		self.client
			.log_message(MessageType::INFO, format!("renaming to {}", params.new_name))
			.await;
		
		let workspace_edit = || -> Option<WorkspaceEdit> {
			let uri = params.text_document_position.text_document.uri;
			let ast = self.ast_map.get(&uri.to_string())?;
			let rope = self.document_map.get(&uri.to_string())?;

			let position = params.text_document_position.position;
			let char = rope.try_line_to_char(position.line as usize).ok()?;
			let offset = char + position.character as usize;
			let reference_list = get_reference(&ast, offset, true);
			let new_name = params.new_name;
			if reference_list.len() > 0 {
				let edit_list = reference_list
					.into_iter()
					.filter_map(|(_, range)| {
						let start_position = offset_to_position(range.start, &rope)?;
						let end_position = offset_to_position(range.end, &rope)?;
						Some(TextEdit::new(
							Range::new(start_position, end_position),
							new_name.clone(),
						))
					})
					.collect::<Vec<_>>();
				let mut map = HashMap::new();
				map.insert(uri, edit_list);
				let workspace_edit = WorkspaceEdit::new(map);
				Some(workspace_edit)
			} else {
				None
			}
		}();
		Ok(workspace_edit)
	}

	async fn did_change_configuration(&self, _: DidChangeConfigurationParams) {
		self.client
			.log_message(MessageType::INFO, "configuration changed!")
			.await;
	}

	async fn did_change_workspace_folders(&self, _: DidChangeWorkspaceFoldersParams) {
		self.client
			.log_message(MessageType::INFO, "workspace folders changed!")
			.await;
	}

	async fn did_change_watched_files(&self, _: DidChangeWatchedFilesParams) {
		self.client
			.log_message(MessageType::INFO, "watched files have changed!")
			.await;
	}

	async fn execute_command(&self, _: ExecuteCommandParams) -> Result<Option<Value>> {
		self.client
			.log_message(MessageType::INFO, "command executed!")
			.await;

		match self.client.apply_edit(WorkspaceEdit::default()).await {
			Ok(res) if res.applied => self.client.log_message(MessageType::INFO, "applied").await,
			Ok(_) => self.client.log_message(MessageType::INFO, "rejected").await,
			Err(err) => self.client.log_message(MessageType::ERROR, err).await,
		}

		Ok(None)
	}

	async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
		let hover = async {
			let uri = params.text_document_position_params.text_document.uri;
			let ast = self.ast_map.get(&uri.to_string())?;
			let rope = self.document_map.get(&uri.to_string())?;

			let position = params.text_document_position_params.position;
			let char = rope.try_line_to_char(position.line as usize).ok()?;
			let offset = char + position.character as usize;
			let atom = get_ast_atom(&ast, offset);
			
			Some(Hover { contents: HoverContents::Scalar(MarkedString::String(format!("ast atom: {:?}", atom).to_string())), range: None })
		}.await;
		Ok(hover)
	}
}

fn get_ast_atom(ast: &AST, offset: usize) -> Option<Spanned<String>> {
	let mut kv_list = ast.macros_by_name.iter().collect::<Vec<_>>();
	kv_list.sort_by(|a, b| a.1.name.start().cmp(&b.1.name.start()));
	// let mut fn_vector = Vector::new();
	for (_, v) in kv_list {
		let (_, range) = &v.name;
		if offset >= range.start && offset < range.end {
			return Some(v.name.clone())
		};

		let (pattern, range) = &v.pattern;
		if range.start <= offset && offset <= range.end {
			return Some((pattern.to_string(), range.clone()))
		} 

		for arg in &v.args {
			if offset >= arg.1.start && offset < arg.1.end {
				return Some(arg.clone())
			}
		}

		for return_type in &v.return_type {
			if offset >= return_type.1.start && offset < return_type.1.end {
				return Some(return_type.clone())
			}
		}
		
		if let Some((expr, span)) = get_atom_of_expr(&v.body, offset) {
			return Some((format!("{expr:?}"), span.clone()))
		}
	}

	if let Some(main) = &ast.main {
		if let Some((expr, span)) = get_atom_of_expr(&main, offset) {
			return Some((format!("{expr:?}"), span.clone()))
		}
	}
	
	None
}

fn get_atom_of_expr(exprspan: &Spanned<Expr>, offset: usize) -> Option<&Spanned<Expr>> {
	match &exprspan.0 {
    Expr::Error => None,
    Expr::Value(_) => if exprspan.1.start <= offset && offset < exprspan.1.end { Some(exprspan) } else { None },
    Expr::List(subexprs) => {
			for expr in subexprs {
				if let Some(expr) = get_atom_of_expr(expr, offset) {
					return Some(expr);
				}
			}
			None
		},
    Expr::Consideration(considered, consideration_span) => if consideration_span.start <= offset && offset < consideration_span.end {
			Some(exprspan)
		} else { get_atom_of_expr(&considered, offset) },
    Expr::IntroRetro(intro_retrod) => {
			for expr in intro_retrod {
				if let Some(expr) = get_atom_of_expr(expr, offset) {
					return Some(expr);
				}
			}
			None
		},
    Expr::ConsideredIntroRetro(intro_retrod) => {
			for expr in intro_retrod {
				if let Some(expr) = get_atom_of_expr(expr, offset) {
					return Some(expr);
				}
			}
			None
		},
	}
}

struct TextDocumentItem {
	uri: Url,
	text: String,
	version: i32,
}
impl Backend {
	async fn on_change(&self, params: TextDocumentItem) {
		let rope = ropey::Rope::from_str(&params.text);
		self.document_map
			.insert(params.uri.to_string(), rope.clone());
		let (ast, errors, semantic_tokens) = parse(&params.text);
		// self.client
		//     .log_message(MessageType::INFO, format!("{:?}", errors))
		//     .await;
		let diagnostics = errors
			.into_iter()
			.filter_map(|item| {
				let (message, span) = match item.reason() {
					chumsky::error::SimpleReason::Unclosed { span, delimiter } => {
						(format!("Unclosed delimiter {}", delimiter), span.clone())
					}
					chumsky::error::SimpleReason::Unexpected => (
						format!(
							"{}, expected {}",
							if item.found().is_some() {
								"Unexpected token in input"
							} else {
								"Unexpected end of input"
							},
							if item.expected().len() == 0 {
								"something else".to_string()
							} else {
								item.expected()
									.map(|expected| match expected {
										Some(expected) => expected.to_string(),
										None => "end of input".to_string(),
									})
									.collect::<Vec<_>>()
									.join(", ")
							}
						),
						item.span(),
					),
					chumsky::error::SimpleReason::Custom(msg) => (msg.to_string(), item.span()),
				};

				let diagnostic = || -> Option<Diagnostic> {
					// let start_line = rope.try_char_to_line(span.start)?;
					// let first_char = rope.try_line_to_char(start_line)?;
					// let start_column = span.start - first_char;
					let start_position = offset_to_position(span.start, &rope)?;
					let end_position = offset_to_position(span.end, &rope)?;
					// let end_line = rope.try_char_to_line(span.end)?;
					// let first_char = rope.try_line_to_char(end_line)?;
					// let end_column = span.end - first_char;
					Some(Diagnostic::new_simple(
						Range::new(start_position, end_position),
						message,
					))
				}();
				diagnostic
			})
			.collect::<Vec<_>>();

		self.client
			.publish_diagnostics(params.uri.clone(), diagnostics, Some(params.version))
			.await;

		if let Some(ast) = ast {
			self.ast_map.insert(params.uri.to_string(), ast);
		}
		// self.client
		//     .log_message(MessageType::INFO, &format!("{:?}", semantic_tokens))
		//     .await;
		self.semantic_token_map
			.insert(params.uri.to_string(), semantic_tokens);
	}
}

#[tokio::main]
async fn main() {
	env_logger::init();

	let stdin = tokio::io::stdin();
	let stdout = tokio::io::stdout();

	let (service, socket) = LspService::build(|client| Backend {
		client,
		ast_map: DashMap::new(),
		document_map: DashMap::new(),
		semantic_token_map: DashMap::new(),
	})
	// .custom_method("custom/inlay_hint", Backend::inlay_hint)
	.finish();
	Server::new(stdin, stdout, socket).serve(service).await;
}

fn offset_to_position(offset: usize, rope: &Rope) -> Option<Position> {
	let line = rope.try_char_to_line(offset).ok()?;
	let first_char_of_line = rope.try_line_to_char(line).ok()?;
	let column = offset - first_char_of_line;
	Some(Position::new(line as u32, column as u32))
}
