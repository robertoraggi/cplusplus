// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cxx/lsp/enums.h>

#include <format>
#include <unordered_map>

namespace cxx::lsp {

auto to_string(SemanticTokenTypes value) -> std::string {
  switch (value) {
    case SemanticTokenTypes::kNamespace:
      return "namespace";
    case SemanticTokenTypes::kType:
      return "type";
    case SemanticTokenTypes::kClass:
      return "class";
    case SemanticTokenTypes::kEnum:
      return "enum";
    case SemanticTokenTypes::kInterface:
      return "interface";
    case SemanticTokenTypes::kStruct:
      return "struct";
    case SemanticTokenTypes::kTypeParameter:
      return "typeParameter";
    case SemanticTokenTypes::kParameter:
      return "parameter";
    case SemanticTokenTypes::kVariable:
      return "variable";
    case SemanticTokenTypes::kProperty:
      return "property";
    case SemanticTokenTypes::kEnumMember:
      return "enumMember";
    case SemanticTokenTypes::kEvent:
      return "event";
    case SemanticTokenTypes::kFunction:
      return "function";
    case SemanticTokenTypes::kMethod:
      return "method";
    case SemanticTokenTypes::kMacro:
      return "macro";
    case SemanticTokenTypes::kKeyword:
      return "keyword";
    case SemanticTokenTypes::kModifier:
      return "modifier";
    case SemanticTokenTypes::kComment:
      return "comment";
    case SemanticTokenTypes::kString:
      return "string";
    case SemanticTokenTypes::kNumber:
      return "number";
    case SemanticTokenTypes::kRegexp:
      return "regexp";
    case SemanticTokenTypes::kOperator:
      return "operator";
    case SemanticTokenTypes::kDecorator:
      return "decorator";
    case SemanticTokenTypes::kLabel:
      return "label";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(SemanticTokenModifiers value) -> std::string {
  switch (value) {
    case SemanticTokenModifiers::kDeclaration:
      return "declaration";
    case SemanticTokenModifiers::kDefinition:
      return "definition";
    case SemanticTokenModifiers::kReadonly:
      return "readonly";
    case SemanticTokenModifiers::kStatic:
      return "static";
    case SemanticTokenModifiers::kDeprecated:
      return "deprecated";
    case SemanticTokenModifiers::kAbstract:
      return "abstract";
    case SemanticTokenModifiers::kAsync:
      return "async";
    case SemanticTokenModifiers::kModification:
      return "modification";
    case SemanticTokenModifiers::kDocumentation:
      return "documentation";
    case SemanticTokenModifiers::kDefaultLibrary:
      return "defaultLibrary";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(DocumentDiagnosticReportKind value) -> std::string {
  switch (value) {
    case DocumentDiagnosticReportKind::kFull:
      return "full";
    case DocumentDiagnosticReportKind::kUnchanged:
      return "unchanged";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(ErrorCodes value) -> std::string {
  switch (value) {
    case ErrorCodes::kParseError:
      return "ParseError";
    case ErrorCodes::kInvalidRequest:
      return "InvalidRequest";
    case ErrorCodes::kMethodNotFound:
      return "MethodNotFound";
    case ErrorCodes::kInvalidParams:
      return "InvalidParams";
    case ErrorCodes::kInternalError:
      return "InternalError";
    case ErrorCodes::kServerNotInitialized:
      return "ServerNotInitialized";
    case ErrorCodes::kUnknownErrorCode:
      return "UnknownErrorCode";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(LSPErrorCodes value) -> std::string {
  switch (value) {
    case LSPErrorCodes::kRequestFailed:
      return "RequestFailed";
    case LSPErrorCodes::kServerCancelled:
      return "ServerCancelled";
    case LSPErrorCodes::kContentModified:
      return "ContentModified";
    case LSPErrorCodes::kRequestCancelled:
      return "RequestCancelled";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(FoldingRangeKind value) -> std::string {
  switch (value) {
    case FoldingRangeKind::kComment:
      return "comment";
    case FoldingRangeKind::kImports:
      return "imports";
    case FoldingRangeKind::kRegion:
      return "region";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(SymbolKind value) -> std::string {
  switch (value) {
    case SymbolKind::kFile:
      return "File";
    case SymbolKind::kModule:
      return "Module";
    case SymbolKind::kNamespace:
      return "Namespace";
    case SymbolKind::kPackage:
      return "Package";
    case SymbolKind::kClass:
      return "Class";
    case SymbolKind::kMethod:
      return "Method";
    case SymbolKind::kProperty:
      return "Property";
    case SymbolKind::kField:
      return "Field";
    case SymbolKind::kConstructor:
      return "Constructor";
    case SymbolKind::kEnum:
      return "Enum";
    case SymbolKind::kInterface:
      return "Interface";
    case SymbolKind::kFunction:
      return "Function";
    case SymbolKind::kVariable:
      return "Variable";
    case SymbolKind::kConstant:
      return "Constant";
    case SymbolKind::kString:
      return "String";
    case SymbolKind::kNumber:
      return "Number";
    case SymbolKind::kBoolean:
      return "Boolean";
    case SymbolKind::kArray:
      return "Array";
    case SymbolKind::kObject:
      return "Object";
    case SymbolKind::kKey:
      return "Key";
    case SymbolKind::kNull:
      return "Null";
    case SymbolKind::kEnumMember:
      return "EnumMember";
    case SymbolKind::kStruct:
      return "Struct";
    case SymbolKind::kEvent:
      return "Event";
    case SymbolKind::kOperator:
      return "Operator";
    case SymbolKind::kTypeParameter:
      return "TypeParameter";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(SymbolTag value) -> std::string {
  switch (value) {
    case SymbolTag::kDeprecated:
      return "Deprecated";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(UniquenessLevel value) -> std::string {
  switch (value) {
    case UniquenessLevel::kDocument:
      return "document";
    case UniquenessLevel::kProject:
      return "project";
    case UniquenessLevel::kGroup:
      return "group";
    case UniquenessLevel::kScheme:
      return "scheme";
    case UniquenessLevel::kGlobal:
      return "global";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(MonikerKind value) -> std::string {
  switch (value) {
    case MonikerKind::kImport:
      return "import";
    case MonikerKind::kExport:
      return "export";
    case MonikerKind::kLocal:
      return "local";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(InlayHintKind value) -> std::string {
  switch (value) {
    case InlayHintKind::kType:
      return "Type";
    case InlayHintKind::kParameter:
      return "Parameter";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(MessageType value) -> std::string {
  switch (value) {
    case MessageType::kError:
      return "Error";
    case MessageType::kWarning:
      return "Warning";
    case MessageType::kInfo:
      return "Info";
    case MessageType::kLog:
      return "Log";
    case MessageType::kDebug:
      return "Debug";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(TextDocumentSyncKind value) -> std::string {
  switch (value) {
    case TextDocumentSyncKind::kNone:
      return "None";
    case TextDocumentSyncKind::kFull:
      return "Full";
    case TextDocumentSyncKind::kIncremental:
      return "Incremental";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(TextDocumentSaveReason value) -> std::string {
  switch (value) {
    case TextDocumentSaveReason::kManual:
      return "Manual";
    case TextDocumentSaveReason::kAfterDelay:
      return "AfterDelay";
    case TextDocumentSaveReason::kFocusOut:
      return "FocusOut";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(CompletionItemKind value) -> std::string {
  switch (value) {
    case CompletionItemKind::kText:
      return "Text";
    case CompletionItemKind::kMethod:
      return "Method";
    case CompletionItemKind::kFunction:
      return "Function";
    case CompletionItemKind::kConstructor:
      return "Constructor";
    case CompletionItemKind::kField:
      return "Field";
    case CompletionItemKind::kVariable:
      return "Variable";
    case CompletionItemKind::kClass:
      return "Class";
    case CompletionItemKind::kInterface:
      return "Interface";
    case CompletionItemKind::kModule:
      return "Module";
    case CompletionItemKind::kProperty:
      return "Property";
    case CompletionItemKind::kUnit:
      return "Unit";
    case CompletionItemKind::kValue:
      return "Value";
    case CompletionItemKind::kEnum:
      return "Enum";
    case CompletionItemKind::kKeyword:
      return "Keyword";
    case CompletionItemKind::kSnippet:
      return "Snippet";
    case CompletionItemKind::kColor:
      return "Color";
    case CompletionItemKind::kFile:
      return "File";
    case CompletionItemKind::kReference:
      return "Reference";
    case CompletionItemKind::kFolder:
      return "Folder";
    case CompletionItemKind::kEnumMember:
      return "EnumMember";
    case CompletionItemKind::kConstant:
      return "Constant";
    case CompletionItemKind::kStruct:
      return "Struct";
    case CompletionItemKind::kEvent:
      return "Event";
    case CompletionItemKind::kOperator:
      return "Operator";
    case CompletionItemKind::kTypeParameter:
      return "TypeParameter";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(CompletionItemTag value) -> std::string {
  switch (value) {
    case CompletionItemTag::kDeprecated:
      return "Deprecated";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(InsertTextFormat value) -> std::string {
  switch (value) {
    case InsertTextFormat::kPlainText:
      return "PlainText";
    case InsertTextFormat::kSnippet:
      return "Snippet";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(InsertTextMode value) -> std::string {
  switch (value) {
    case InsertTextMode::kAsIs:
      return "asIs";
    case InsertTextMode::kAdjustIndentation:
      return "adjustIndentation";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(DocumentHighlightKind value) -> std::string {
  switch (value) {
    case DocumentHighlightKind::kText:
      return "Text";
    case DocumentHighlightKind::kRead:
      return "Read";
    case DocumentHighlightKind::kWrite:
      return "Write";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(CodeActionKind value) -> std::string {
  switch (value) {
    case CodeActionKind::kEmpty:
      return "";
    case CodeActionKind::kQuickFix:
      return "quickfix";
    case CodeActionKind::kRefactor:
      return "refactor";
    case CodeActionKind::kRefactorExtract:
      return "refactor.extract";
    case CodeActionKind::kRefactorInline:
      return "refactor.inline";
    case CodeActionKind::kRefactorMove:
      return "refactor.move";
    case CodeActionKind::kRefactorRewrite:
      return "refactor.rewrite";
    case CodeActionKind::kSource:
      return "source";
    case CodeActionKind::kSourceOrganizeImports:
      return "source.organizeImports";
    case CodeActionKind::kSourceFixAll:
      return "source.fixAll";
    case CodeActionKind::kNotebook:
      return "notebook";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(CodeActionTag value) -> std::string {
  switch (value) {
    case CodeActionTag::kLLMGenerated:
      return "LLMGenerated";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(TraceValue value) -> std::string {
  switch (value) {
    case TraceValue::kOff:
      return "off";
    case TraceValue::kMessages:
      return "messages";
    case TraceValue::kVerbose:
      return "verbose";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(MarkupKind value) -> std::string {
  switch (value) {
    case MarkupKind::kPlainText:
      return "plaintext";
    case MarkupKind::kMarkdown:
      return "markdown";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(LanguageKind value) -> std::string {
  switch (value) {
    case LanguageKind::kABAP:
      return "abap";
    case LanguageKind::kWindowsBat:
      return "bat";
    case LanguageKind::kBibTeX:
      return "bibtex";
    case LanguageKind::kClojure:
      return "clojure";
    case LanguageKind::kCoffeescript:
      return "coffeescript";
    case LanguageKind::kC:
      return "c";
    case LanguageKind::kCPP:
      return "cpp";
    case LanguageKind::kCSharp:
      return "csharp";
    case LanguageKind::kCSS:
      return "css";
    case LanguageKind::kD:
      return "d";
    case LanguageKind::kDelphi:
      return "pascal";
    case LanguageKind::kDiff:
      return "diff";
    case LanguageKind::kDart:
      return "dart";
    case LanguageKind::kDockerfile:
      return "dockerfile";
    case LanguageKind::kElixir:
      return "elixir";
    case LanguageKind::kErlang:
      return "erlang";
    case LanguageKind::kFSharp:
      return "fsharp";
    case LanguageKind::kGitCommit:
      return "git-commit";
    case LanguageKind::kGitRebase:
      return "rebase";
    case LanguageKind::kGo:
      return "go";
    case LanguageKind::kGroovy:
      return "groovy";
    case LanguageKind::kHandlebars:
      return "handlebars";
    case LanguageKind::kHaskell:
      return "haskell";
    case LanguageKind::kHTML:
      return "html";
    case LanguageKind::kIni:
      return "ini";
    case LanguageKind::kJava:
      return "java";
    case LanguageKind::kJavaScript:
      return "javascript";
    case LanguageKind::kJavaScriptReact:
      return "javascriptreact";
    case LanguageKind::kJSON:
      return "json";
    case LanguageKind::kLaTeX:
      return "latex";
    case LanguageKind::kLess:
      return "less";
    case LanguageKind::kLua:
      return "lua";
    case LanguageKind::kMakefile:
      return "makefile";
    case LanguageKind::kMarkdown:
      return "markdown";
    case LanguageKind::kObjectiveC:
      return "objective-c";
    case LanguageKind::kObjectiveCPP:
      return "objective-cpp";
    case LanguageKind::kPascal:
      return "pascal";
    case LanguageKind::kPerl:
      return "perl";
    case LanguageKind::kPerl6:
      return "perl6";
    case LanguageKind::kPHP:
      return "php";
    case LanguageKind::kPowershell:
      return "powershell";
    case LanguageKind::kPug:
      return "jade";
    case LanguageKind::kPython:
      return "python";
    case LanguageKind::kR:
      return "r";
    case LanguageKind::kRazor:
      return "razor";
    case LanguageKind::kRuby:
      return "ruby";
    case LanguageKind::kRust:
      return "rust";
    case LanguageKind::kSCSS:
      return "scss";
    case LanguageKind::kSASS:
      return "sass";
    case LanguageKind::kScala:
      return "scala";
    case LanguageKind::kShaderLab:
      return "shaderlab";
    case LanguageKind::kShellScript:
      return "shellscript";
    case LanguageKind::kSQL:
      return "sql";
    case LanguageKind::kSwift:
      return "swift";
    case LanguageKind::kTypeScript:
      return "typescript";
    case LanguageKind::kTypeScriptReact:
      return "typescriptreact";
    case LanguageKind::kTeX:
      return "tex";
    case LanguageKind::kVisualBasic:
      return "vb";
    case LanguageKind::kXML:
      return "xml";
    case LanguageKind::kXSL:
      return "xsl";
    case LanguageKind::kYAML:
      return "yaml";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(InlineCompletionTriggerKind value) -> std::string {
  switch (value) {
    case InlineCompletionTriggerKind::kInvoked:
      return "Invoked";
    case InlineCompletionTriggerKind::kAutomatic:
      return "Automatic";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(PositionEncodingKind value) -> std::string {
  switch (value) {
    case PositionEncodingKind::kUTF8:
      return "utf-8";
    case PositionEncodingKind::kUTF16:
      return "utf-16";
    case PositionEncodingKind::kUTF32:
      return "utf-32";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(FileChangeType value) -> std::string {
  switch (value) {
    case FileChangeType::kCreated:
      return "Created";
    case FileChangeType::kChanged:
      return "Changed";
    case FileChangeType::kDeleted:
      return "Deleted";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(WatchKind value) -> std::string {
  switch (value) {
    case WatchKind::kCreate:
      return "Create";
    case WatchKind::kChange:
      return "Change";
    case WatchKind::kDelete:
      return "Delete";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(DiagnosticSeverity value) -> std::string {
  switch (value) {
    case DiagnosticSeverity::kError:
      return "Error";
    case DiagnosticSeverity::kWarning:
      return "Warning";
    case DiagnosticSeverity::kInformation:
      return "Information";
    case DiagnosticSeverity::kHint:
      return "Hint";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(DiagnosticTag value) -> std::string {
  switch (value) {
    case DiagnosticTag::kUnnecessary:
      return "Unnecessary";
    case DiagnosticTag::kDeprecated:
      return "Deprecated";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(CompletionTriggerKind value) -> std::string {
  switch (value) {
    case CompletionTriggerKind::kInvoked:
      return "Invoked";
    case CompletionTriggerKind::kTriggerCharacter:
      return "TriggerCharacter";
    case CompletionTriggerKind::kTriggerForIncompleteCompletions:
      return "TriggerForIncompleteCompletions";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(ApplyKind value) -> std::string {
  switch (value) {
    case ApplyKind::kReplace:
      return "Replace";
    case ApplyKind::kMerge:
      return "Merge";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(SignatureHelpTriggerKind value) -> std::string {
  switch (value) {
    case SignatureHelpTriggerKind::kInvoked:
      return "Invoked";
    case SignatureHelpTriggerKind::kTriggerCharacter:
      return "TriggerCharacter";
    case SignatureHelpTriggerKind::kContentChange:
      return "ContentChange";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(CodeActionTriggerKind value) -> std::string {
  switch (value) {
    case CodeActionTriggerKind::kInvoked:
      return "Invoked";
    case CodeActionTriggerKind::kAutomatic:
      return "Automatic";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(FileOperationPatternKind value) -> std::string {
  switch (value) {
    case FileOperationPatternKind::kFile:
      return "file";
    case FileOperationPatternKind::kFolder:
      return "folder";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(NotebookCellKind value) -> std::string {
  switch (value) {
    case NotebookCellKind::kMarkup:
      return "Markup";
    case NotebookCellKind::kCode:
      return "Code";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(ResourceOperationKind value) -> std::string {
  switch (value) {
    case ResourceOperationKind::kCreate:
      return "create";
    case ResourceOperationKind::kRename:
      return "rename";
    case ResourceOperationKind::kDelete:
      return "delete";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(FailureHandlingKind value) -> std::string {
  switch (value) {
    case FailureHandlingKind::kAbort:
      return "abort";
    case FailureHandlingKind::kTransactional:
      return "transactional";
    case FailureHandlingKind::kTextOnlyTransactional:
      return "textOnlyTransactional";
    case FailureHandlingKind::kUndo:
      return "undo";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(PrepareSupportDefaultBehavior value) -> std::string {
  switch (value) {
    case PrepareSupportDefaultBehavior::kIdentifier:
      return "Identifier";
  }

  lsp_runtime_error("invalid enumerator value");
}

auto to_string(TokenFormat value) -> std::string {
  switch (value) {
    case TokenFormat::kRelative:
      return "relative";
  }

  lsp_runtime_error("invalid enumerator value");
}

namespace string_enums {

auto parseSemanticTokenTypes(std::string_view name)
    -> std::optional<SemanticTokenTypes> {
  static std::unordered_map<std::string_view, SemanticTokenTypes> map{
      {"namespace", SemanticTokenTypes::kNamespace},
      {"type", SemanticTokenTypes::kType},
      {"class", SemanticTokenTypes::kClass},
      {"enum", SemanticTokenTypes::kEnum},
      {"interface", SemanticTokenTypes::kInterface},
      {"struct", SemanticTokenTypes::kStruct},
      {"typeParameter", SemanticTokenTypes::kTypeParameter},
      {"parameter", SemanticTokenTypes::kParameter},
      {"variable", SemanticTokenTypes::kVariable},
      {"property", SemanticTokenTypes::kProperty},
      {"enumMember", SemanticTokenTypes::kEnumMember},
      {"event", SemanticTokenTypes::kEvent},
      {"function", SemanticTokenTypes::kFunction},
      {"method", SemanticTokenTypes::kMethod},
      {"macro", SemanticTokenTypes::kMacro},
      {"keyword", SemanticTokenTypes::kKeyword},
      {"modifier", SemanticTokenTypes::kModifier},
      {"comment", SemanticTokenTypes::kComment},
      {"string", SemanticTokenTypes::kString},
      {"number", SemanticTokenTypes::kNumber},
      {"regexp", SemanticTokenTypes::kRegexp},
      {"operator", SemanticTokenTypes::kOperator},
      {"decorator", SemanticTokenTypes::kDecorator},
      {"label", SemanticTokenTypes::kLabel},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseSemanticTokenModifiers(std::string_view name)
    -> std::optional<SemanticTokenModifiers> {
  static std::unordered_map<std::string_view, SemanticTokenModifiers> map{
      {"declaration", SemanticTokenModifiers::kDeclaration},
      {"definition", SemanticTokenModifiers::kDefinition},
      {"readonly", SemanticTokenModifiers::kReadonly},
      {"static", SemanticTokenModifiers::kStatic},
      {"deprecated", SemanticTokenModifiers::kDeprecated},
      {"abstract", SemanticTokenModifiers::kAbstract},
      {"async", SemanticTokenModifiers::kAsync},
      {"modification", SemanticTokenModifiers::kModification},
      {"documentation", SemanticTokenModifiers::kDocumentation},
      {"defaultLibrary", SemanticTokenModifiers::kDefaultLibrary},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseDocumentDiagnosticReportKind(std::string_view name)
    -> std::optional<DocumentDiagnosticReportKind> {
  static std::unordered_map<std::string_view, DocumentDiagnosticReportKind> map{
      {"full", DocumentDiagnosticReportKind::kFull},
      {"unchanged", DocumentDiagnosticReportKind::kUnchanged},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseFoldingRangeKind(std::string_view name)
    -> std::optional<FoldingRangeKind> {
  static std::unordered_map<std::string_view, FoldingRangeKind> map{
      {"comment", FoldingRangeKind::kComment},
      {"imports", FoldingRangeKind::kImports},
      {"region", FoldingRangeKind::kRegion},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseUniquenessLevel(std::string_view name)
    -> std::optional<UniquenessLevel> {
  static std::unordered_map<std::string_view, UniquenessLevel> map{
      {"document", UniquenessLevel::kDocument},
      {"project", UniquenessLevel::kProject},
      {"group", UniquenessLevel::kGroup},
      {"scheme", UniquenessLevel::kScheme},
      {"global", UniquenessLevel::kGlobal},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseMonikerKind(std::string_view name) -> std::optional<MonikerKind> {
  static std::unordered_map<std::string_view, MonikerKind> map{
      {"import", MonikerKind::kImport},
      {"export", MonikerKind::kExport},
      {"local", MonikerKind::kLocal},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseCodeActionKind(std::string_view name)
    -> std::optional<CodeActionKind> {
  static std::unordered_map<std::string_view, CodeActionKind> map{
      {"", CodeActionKind::kEmpty},
      {"quickfix", CodeActionKind::kQuickFix},
      {"refactor", CodeActionKind::kRefactor},
      {"refactor.extract", CodeActionKind::kRefactorExtract},
      {"refactor.inline", CodeActionKind::kRefactorInline},
      {"refactor.move", CodeActionKind::kRefactorMove},
      {"refactor.rewrite", CodeActionKind::kRefactorRewrite},
      {"source", CodeActionKind::kSource},
      {"source.organizeImports", CodeActionKind::kSourceOrganizeImports},
      {"source.fixAll", CodeActionKind::kSourceFixAll},
      {"notebook", CodeActionKind::kNotebook},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseTraceValue(std::string_view name) -> std::optional<TraceValue> {
  static std::unordered_map<std::string_view, TraceValue> map{
      {"off", TraceValue::kOff},
      {"messages", TraceValue::kMessages},
      {"verbose", TraceValue::kVerbose},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseMarkupKind(std::string_view name) -> std::optional<MarkupKind> {
  static std::unordered_map<std::string_view, MarkupKind> map{
      {"plaintext", MarkupKind::kPlainText},
      {"markdown", MarkupKind::kMarkdown},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseLanguageKind(std::string_view name) -> std::optional<LanguageKind> {
  static std::unordered_map<std::string_view, LanguageKind> map{
      {"abap", LanguageKind::kABAP},
      {"bat", LanguageKind::kWindowsBat},
      {"bibtex", LanguageKind::kBibTeX},
      {"clojure", LanguageKind::kClojure},
      {"coffeescript", LanguageKind::kCoffeescript},
      {"c", LanguageKind::kC},
      {"cpp", LanguageKind::kCPP},
      {"csharp", LanguageKind::kCSharp},
      {"css", LanguageKind::kCSS},
      {"d", LanguageKind::kD},
      {"pascal", LanguageKind::kDelphi},
      {"diff", LanguageKind::kDiff},
      {"dart", LanguageKind::kDart},
      {"dockerfile", LanguageKind::kDockerfile},
      {"elixir", LanguageKind::kElixir},
      {"erlang", LanguageKind::kErlang},
      {"fsharp", LanguageKind::kFSharp},
      {"git-commit", LanguageKind::kGitCommit},
      {"rebase", LanguageKind::kGitRebase},
      {"go", LanguageKind::kGo},
      {"groovy", LanguageKind::kGroovy},
      {"handlebars", LanguageKind::kHandlebars},
      {"haskell", LanguageKind::kHaskell},
      {"html", LanguageKind::kHTML},
      {"ini", LanguageKind::kIni},
      {"java", LanguageKind::kJava},
      {"javascript", LanguageKind::kJavaScript},
      {"javascriptreact", LanguageKind::kJavaScriptReact},
      {"json", LanguageKind::kJSON},
      {"latex", LanguageKind::kLaTeX},
      {"less", LanguageKind::kLess},
      {"lua", LanguageKind::kLua},
      {"makefile", LanguageKind::kMakefile},
      {"markdown", LanguageKind::kMarkdown},
      {"objective-c", LanguageKind::kObjectiveC},
      {"objective-cpp", LanguageKind::kObjectiveCPP},
      {"pascal", LanguageKind::kPascal},
      {"perl", LanguageKind::kPerl},
      {"perl6", LanguageKind::kPerl6},
      {"php", LanguageKind::kPHP},
      {"powershell", LanguageKind::kPowershell},
      {"jade", LanguageKind::kPug},
      {"python", LanguageKind::kPython},
      {"r", LanguageKind::kR},
      {"razor", LanguageKind::kRazor},
      {"ruby", LanguageKind::kRuby},
      {"rust", LanguageKind::kRust},
      {"scss", LanguageKind::kSCSS},
      {"sass", LanguageKind::kSASS},
      {"scala", LanguageKind::kScala},
      {"shaderlab", LanguageKind::kShaderLab},
      {"shellscript", LanguageKind::kShellScript},
      {"sql", LanguageKind::kSQL},
      {"swift", LanguageKind::kSwift},
      {"typescript", LanguageKind::kTypeScript},
      {"typescriptreact", LanguageKind::kTypeScriptReact},
      {"tex", LanguageKind::kTeX},
      {"vb", LanguageKind::kVisualBasic},
      {"xml", LanguageKind::kXML},
      {"xsl", LanguageKind::kXSL},
      {"yaml", LanguageKind::kYAML},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parsePositionEncodingKind(std::string_view name)
    -> std::optional<PositionEncodingKind> {
  static std::unordered_map<std::string_view, PositionEncodingKind> map{
      {"utf-8", PositionEncodingKind::kUTF8},
      {"utf-16", PositionEncodingKind::kUTF16},
      {"utf-32", PositionEncodingKind::kUTF32},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseFileOperationPatternKind(std::string_view name)
    -> std::optional<FileOperationPatternKind> {
  static std::unordered_map<std::string_view, FileOperationPatternKind> map{
      {"file", FileOperationPatternKind::kFile},
      {"folder", FileOperationPatternKind::kFolder},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseResourceOperationKind(std::string_view name)
    -> std::optional<ResourceOperationKind> {
  static std::unordered_map<std::string_view, ResourceOperationKind> map{
      {"create", ResourceOperationKind::kCreate},
      {"rename", ResourceOperationKind::kRename},
      {"delete", ResourceOperationKind::kDelete},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseFailureHandlingKind(std::string_view name)
    -> std::optional<FailureHandlingKind> {
  static std::unordered_map<std::string_view, FailureHandlingKind> map{
      {"abort", FailureHandlingKind::kAbort},
      {"transactional", FailureHandlingKind::kTransactional},
      {"textOnlyTransactional", FailureHandlingKind::kTextOnlyTransactional},
      {"undo", FailureHandlingKind::kUndo},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

auto parseTokenFormat(std::string_view name) -> std::optional<TokenFormat> {
  static std::unordered_map<std::string_view, TokenFormat> map{
      {"relative", TokenFormat::kRelative},
  };
  const auto it = map.find(name);
  if (it != map.end()) return it->second;
  return std::nullopt;
}

}  // namespace string_enums

}  // namespace cxx::lsp
