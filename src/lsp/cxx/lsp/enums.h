#pragma once

#include <string>

namespace cxx::lsp {

enum class SemanticTokenTypes {
  kNamespace,
  kType,
  kClass,
  kEnum,
  kInterface,
  kStruct,
  kTypeParameter,
  kParameter,
  kVariable,
  kProperty,
  kEnumMember,
  kEvent,
  kFunction,
  kMethod,
  kMacro,
  kKeyword,
  kModifier,
  kComment,
  kString,
  kNumber,
  kRegexp,
  kOperator,
  kDecorator,
  kLabel,
};

enum class SemanticTokenModifiers {
  kDeclaration,
  kDefinition,
  kReadonly,
  kStatic,
  kDeprecated,
  kAbstract,
  kAsync,
  kModification,
  kDocumentation,
  kDefaultLibrary,
};

enum class DocumentDiagnosticReportKind {
  kFull,
  kUnchanged,
};

enum class ErrorCodes : int {
  kParseError = -32700,
  kInvalidRequest = -32600,
  kMethodNotFound = -32601,
  kInvalidParams = -32602,
  kInternalError = -32603,
  kServerNotInitialized = -32002,
  kUnknownErrorCode = -32001,
};

enum class LSPErrorCodes : int {
  kRequestFailed = -32803,
  kServerCancelled = -32802,
  kContentModified = -32801,
  kRequestCancelled = -32800,
};

enum class FoldingRangeKind {
  kComment,
  kImports,
  kRegion,
};

enum class SymbolKind : unsigned int {
  kFile = 1,
  kModule = 2,
  kNamespace = 3,
  kPackage = 4,
  kClass = 5,
  kMethod = 6,
  kProperty = 7,
  kField = 8,
  kConstructor = 9,
  kEnum = 10,
  kInterface = 11,
  kFunction = 12,
  kVariable = 13,
  kConstant = 14,
  kString = 15,
  kNumber = 16,
  kBoolean = 17,
  kArray = 18,
  kObject = 19,
  kKey = 20,
  kNull = 21,
  kEnumMember = 22,
  kStruct = 23,
  kEvent = 24,
  kOperator = 25,
  kTypeParameter = 26,
};

enum class SymbolTag : unsigned int {
  kDeprecated = 1,
};

enum class UniquenessLevel {
  kDocument,
  kProject,
  kGroup,
  kScheme,
  kGlobal,
};

enum class MonikerKind {
  kImport,
  kExport,
  kLocal,
};

enum class InlayHintKind : unsigned int {
  kType = 1,
  kParameter = 2,
};

enum class MessageType : unsigned int {
  kError = 1,
  kWarning = 2,
  kInfo = 3,
  kLog = 4,
  kDebug = 5,
};

enum class TextDocumentSyncKind : unsigned int {
  kNone = 0,
  kFull = 1,
  kIncremental = 2,
};

enum class TextDocumentSaveReason : unsigned int {
  kManual = 1,
  kAfterDelay = 2,
  kFocusOut = 3,
};

enum class CompletionItemKind : unsigned int {
  kText = 1,
  kMethod = 2,
  kFunction = 3,
  kConstructor = 4,
  kField = 5,
  kVariable = 6,
  kClass = 7,
  kInterface = 8,
  kModule = 9,
  kProperty = 10,
  kUnit = 11,
  kValue = 12,
  kEnum = 13,
  kKeyword = 14,
  kSnippet = 15,
  kColor = 16,
  kFile = 17,
  kReference = 18,
  kFolder = 19,
  kEnumMember = 20,
  kConstant = 21,
  kStruct = 22,
  kEvent = 23,
  kOperator = 24,
  kTypeParameter = 25,
};

enum class CompletionItemTag : unsigned int {
  kDeprecated = 1,
};

enum class InsertTextFormat : unsigned int {
  kPlainText = 1,
  kSnippet = 2,
};

enum class InsertTextMode : unsigned int {
  kAsIs = 1,
  kAdjustIndentation = 2,
};

enum class DocumentHighlightKind : unsigned int {
  kText = 1,
  kRead = 2,
  kWrite = 3,
};

enum class CodeActionKind {
  kEmpty,
  kQuickFix,
  kRefactor,
  kRefactorExtract,
  kRefactorInline,
  kRefactorMove,
  kRefactorRewrite,
  kSource,
  kSourceOrganizeImports,
  kSourceFixAll,
  kNotebook,
};

enum class CodeActionTag : unsigned int {
  kLLMGenerated = 1,
};

enum class TraceValue {
  kOff,
  kMessages,
  kVerbose,
};

enum class MarkupKind {
  kPlainText,
  kMarkdown,
};

enum class LanguageKind {
  kABAP,
  kWindowsBat,
  kBibTeX,
  kClojure,
  kCoffeescript,
  kC,
  kCPP,
  kCSharp,
  kCSS,
  kD,
  kDelphi,
  kDiff,
  kDart,
  kDockerfile,
  kElixir,
  kErlang,
  kFSharp,
  kGitCommit,
  kGitRebase,
  kGo,
  kGroovy,
  kHandlebars,
  kHaskell,
  kHTML,
  kIni,
  kJava,
  kJavaScript,
  kJavaScriptReact,
  kJSON,
  kLaTeX,
  kLess,
  kLua,
  kMakefile,
  kMarkdown,
  kObjectiveC,
  kObjectiveCPP,
  kPascal,
  kPerl,
  kPerl6,
  kPHP,
  kPowershell,
  kPug,
  kPython,
  kR,
  kRazor,
  kRuby,
  kRust,
  kSCSS,
  kSASS,
  kScala,
  kShaderLab,
  kShellScript,
  kSQL,
  kSwift,
  kTypeScript,
  kTypeScriptReact,
  kTeX,
  kVisualBasic,
  kXML,
  kXSL,
  kYAML,
};

enum class InlineCompletionTriggerKind : unsigned int {
  kInvoked = 1,
  kAutomatic = 2,
};

enum class PositionEncodingKind {
  kUTF8,
  kUTF16,
  kUTF32,
};

enum class FileChangeType : unsigned int {
  kCreated = 1,
  kChanged = 2,
  kDeleted = 3,
};

enum class WatchKind : unsigned int {
  kCreate = 1,
  kChange = 2,
  kDelete = 4,
};

enum class DiagnosticSeverity : unsigned int {
  kError = 1,
  kWarning = 2,
  kInformation = 3,
  kHint = 4,
};

enum class DiagnosticTag : unsigned int {
  kUnnecessary = 1,
  kDeprecated = 2,
};

enum class CompletionTriggerKind : unsigned int {
  kInvoked = 1,
  kTriggerCharacter = 2,
  kTriggerForIncompleteCompletions = 3,
};

enum class ApplyKind : unsigned int {
  kReplace = 1,
  kMerge = 2,
};

enum class SignatureHelpTriggerKind : unsigned int {
  kInvoked = 1,
  kTriggerCharacter = 2,
  kContentChange = 3,
};

enum class CodeActionTriggerKind : unsigned int {
  kInvoked = 1,
  kAutomatic = 2,
};

enum class FileOperationPatternKind {
  kFile,
  kFolder,
};

enum class NotebookCellKind : unsigned int {
  kMarkup = 1,
  kCode = 2,
};

enum class ResourceOperationKind {
  kCreate,
  kRename,
  kDelete,
};

enum class FailureHandlingKind {
  kAbort,
  kTransactional,
  kTextOnlyTransactional,
  kUndo,
};

enum class PrepareSupportDefaultBehavior : unsigned int {
  kIdentifier = 1,
};

enum class TokenFormat {
  kRelative,
};

auto to_string(SemanticTokenTypes value) -> std::string;
auto to_string(SemanticTokenModifiers value) -> std::string;
auto to_string(DocumentDiagnosticReportKind value) -> std::string;
auto to_string(ErrorCodes value) -> std::string;
auto to_string(LSPErrorCodes value) -> std::string;
auto to_string(FoldingRangeKind value) -> std::string;
auto to_string(SymbolKind value) -> std::string;
auto to_string(SymbolTag value) -> std::string;
auto to_string(UniquenessLevel value) -> std::string;
auto to_string(MonikerKind value) -> std::string;
auto to_string(InlayHintKind value) -> std::string;
auto to_string(MessageType value) -> std::string;
auto to_string(TextDocumentSyncKind value) -> std::string;
auto to_string(TextDocumentSaveReason value) -> std::string;
auto to_string(CompletionItemKind value) -> std::string;
auto to_string(CompletionItemTag value) -> std::string;
auto to_string(InsertTextFormat value) -> std::string;
auto to_string(InsertTextMode value) -> std::string;
auto to_string(DocumentHighlightKind value) -> std::string;
auto to_string(CodeActionKind value) -> std::string;
auto to_string(CodeActionTag value) -> std::string;
auto to_string(TraceValue value) -> std::string;
auto to_string(MarkupKind value) -> std::string;
auto to_string(LanguageKind value) -> std::string;
auto to_string(InlineCompletionTriggerKind value) -> std::string;
auto to_string(PositionEncodingKind value) -> std::string;
auto to_string(FileChangeType value) -> std::string;
auto to_string(WatchKind value) -> std::string;
auto to_string(DiagnosticSeverity value) -> std::string;
auto to_string(DiagnosticTag value) -> std::string;
auto to_string(CompletionTriggerKind value) -> std::string;
auto to_string(ApplyKind value) -> std::string;
auto to_string(SignatureHelpTriggerKind value) -> std::string;
auto to_string(CodeActionTriggerKind value) -> std::string;
auto to_string(FileOperationPatternKind value) -> std::string;
auto to_string(NotebookCellKind value) -> std::string;
auto to_string(ResourceOperationKind value) -> std::string;
auto to_string(FailureHandlingKind value) -> std::string;
auto to_string(PrepareSupportDefaultBehavior value) -> std::string;
auto to_string(TokenFormat value) -> std::string;

}  // namespace cxx::lsp
