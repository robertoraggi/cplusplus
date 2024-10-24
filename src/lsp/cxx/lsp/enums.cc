#include <cxx/lsp/enums.h>

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
}

auto to_string(DocumentDiagnosticReportKind value) -> std::string {
  switch (value) {
    case DocumentDiagnosticReportKind::kFull:
      return "full";
    case DocumentDiagnosticReportKind::kUnchanged:
      return "unchanged";
  }
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
}

auto to_string(SymbolTag value) -> std::string {
  switch (value) {
    case SymbolTag::kDeprecated:
      return "Deprecated";
  }
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
}

auto to_string(InlayHintKind value) -> std::string {
  switch (value) {
    case InlayHintKind::kType:
      return "Type";
    case InlayHintKind::kParameter:
      return "Parameter";
  }
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
}

auto to_string(CompletionItemTag value) -> std::string {
  switch (value) {
    case CompletionItemTag::kDeprecated:
      return "Deprecated";
  }
}

auto to_string(InsertTextFormat value) -> std::string {
  switch (value) {
    case InsertTextFormat::kPlainText:
      return "PlainText";
    case InsertTextFormat::kSnippet:
      return "Snippet";
  }
}

auto to_string(InsertTextMode value) -> std::string {
  switch (value) {
    case InsertTextMode::kAsIs:
      return "asIs";
    case InsertTextMode::kAdjustIndentation:
      return "adjustIndentation";
  }
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
}

auto to_string(CodeActionTag value) -> std::string {
  switch (value) {
    case CodeActionTag::kLLMGenerated:
      return "LLMGenerated";
  }
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
}

auto to_string(MarkupKind value) -> std::string {
  switch (value) {
    case MarkupKind::kPlainText:
      return "plaintext";
    case MarkupKind::kMarkdown:
      return "markdown";
  }
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
}

auto to_string(InlineCompletionTriggerKind value) -> std::string {
  switch (value) {
    case InlineCompletionTriggerKind::kInvoked:
      return "Invoked";
    case InlineCompletionTriggerKind::kAutomatic:
      return "Automatic";
  }
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
}

auto to_string(DiagnosticTag value) -> std::string {
  switch (value) {
    case DiagnosticTag::kUnnecessary:
      return "Unnecessary";
    case DiagnosticTag::kDeprecated:
      return "Deprecated";
  }
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
}

auto to_string(ApplyKind value) -> std::string {
  switch (value) {
    case ApplyKind::kReplace:
      return "Replace";
    case ApplyKind::kMerge:
      return "Merge";
  }
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
}

auto to_string(CodeActionTriggerKind value) -> std::string {
  switch (value) {
    case CodeActionTriggerKind::kInvoked:
      return "Invoked";
    case CodeActionTriggerKind::kAutomatic:
      return "Automatic";
  }
}

auto to_string(FileOperationPatternKind value) -> std::string {
  switch (value) {
    case FileOperationPatternKind::kFile:
      return "file";
    case FileOperationPatternKind::kFolder:
      return "folder";
  }
}

auto to_string(NotebookCellKind value) -> std::string {
  switch (value) {
    case NotebookCellKind::kMarkup:
      return "Markup";
    case NotebookCellKind::kCode:
      return "Code";
  }
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
}

auto to_string(PrepareSupportDefaultBehavior value) -> std::string {
  switch (value) {
    case PrepareSupportDefaultBehavior::kIdentifier:
      return "Identifier";
  }
}

auto to_string(TokenFormat value) -> std::string {
  switch (value) {
    case TokenFormat::kRelative:
      return "relative";
  }
}

}  // namespace cxx::lsp
