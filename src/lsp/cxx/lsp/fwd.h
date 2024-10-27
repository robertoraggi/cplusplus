// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#pragma once

#include <concepts>
#include <nlohmann/json.hpp>
#include <variant>

namespace cxx::lsp {

using json = nlohmann::json;

enum class SemanticTokenTypes;
enum class SemanticTokenModifiers;
enum class DocumentDiagnosticReportKind;
enum class ErrorCodes : int;
enum class LSPErrorCodes : int;
enum class FoldingRangeKind;
enum class SymbolKind : long;
enum class SymbolTag : long;
enum class UniquenessLevel;
enum class MonikerKind;
enum class InlayHintKind : long;
enum class MessageType : long;
enum class TextDocumentSyncKind : long;
enum class TextDocumentSaveReason : long;
enum class CompletionItemKind : long;
enum class CompletionItemTag : long;
enum class InsertTextFormat : long;
enum class InsertTextMode : long;
enum class DocumentHighlightKind : long;
enum class CodeActionKind;
enum class CodeActionTag : long;
enum class TraceValue;
enum class MarkupKind;
enum class LanguageKind;
enum class InlineCompletionTriggerKind : long;
enum class PositionEncodingKind;
enum class FileChangeType : long;
enum class WatchKind : long;
enum class DiagnosticSeverity : long;
enum class DiagnosticTag : long;
enum class CompletionTriggerKind : long;
enum class ApplyKind : long;
enum class SignatureHelpTriggerKind : long;
enum class CodeActionTriggerKind : long;
enum class FileOperationPatternKind;
enum class NotebookCellKind : long;
enum class ResourceOperationKind;
enum class FailureHandlingKind;
enum class PrepareSupportDefaultBehavior : long;
enum class TokenFormat;

// structures
class ImplementationParams;
class Location;
class ImplementationRegistrationOptions;
class TypeDefinitionParams;
class TypeDefinitionRegistrationOptions;
class WorkspaceFolder;
class DidChangeWorkspaceFoldersParams;
class ConfigurationParams;
class DocumentColorParams;
class ColorInformation;
class DocumentColorRegistrationOptions;
class ColorPresentationParams;
class ColorPresentation;
class WorkDoneProgressOptions;
class TextDocumentRegistrationOptions;
class FoldingRangeParams;
class FoldingRange;
class FoldingRangeRegistrationOptions;
class DeclarationParams;
class DeclarationRegistrationOptions;
class SelectionRangeParams;
class SelectionRange;
class SelectionRangeRegistrationOptions;
class WorkDoneProgressCreateParams;
class WorkDoneProgressCancelParams;
class CallHierarchyPrepareParams;
class CallHierarchyItem;
class CallHierarchyRegistrationOptions;
class CallHierarchyIncomingCallsParams;
class CallHierarchyIncomingCall;
class CallHierarchyOutgoingCallsParams;
class CallHierarchyOutgoingCall;
class SemanticTokensParams;
class SemanticTokens;
class SemanticTokensPartialResult;
class SemanticTokensRegistrationOptions;
class SemanticTokensDeltaParams;
class SemanticTokensDelta;
class SemanticTokensDeltaPartialResult;
class SemanticTokensRangeParams;
class ShowDocumentParams;
class ShowDocumentResult;
class LinkedEditingRangeParams;
class LinkedEditingRanges;
class LinkedEditingRangeRegistrationOptions;
class CreateFilesParams;
class WorkspaceEdit;
class FileOperationRegistrationOptions;
class RenameFilesParams;
class DeleteFilesParams;
class MonikerParams;
class Moniker;
class MonikerRegistrationOptions;
class TypeHierarchyPrepareParams;
class TypeHierarchyItem;
class TypeHierarchyRegistrationOptions;
class TypeHierarchySupertypesParams;
class TypeHierarchySubtypesParams;
class InlineValueParams;
class InlineValueRegistrationOptions;
class InlayHintParams;
class InlayHint;
class InlayHintRegistrationOptions;
class DocumentDiagnosticParams;
class DocumentDiagnosticReportPartialResult;
class DiagnosticServerCancellationData;
class DiagnosticRegistrationOptions;
class WorkspaceDiagnosticParams;
class WorkspaceDiagnosticReport;
class WorkspaceDiagnosticReportPartialResult;
class DidOpenNotebookDocumentParams;
class NotebookDocumentSyncRegistrationOptions;
class DidChangeNotebookDocumentParams;
class DidSaveNotebookDocumentParams;
class DidCloseNotebookDocumentParams;
class InlineCompletionParams;
class InlineCompletionList;
class InlineCompletionItem;
class InlineCompletionRegistrationOptions;
class TextDocumentContentParams;
class TextDocumentContentResult;
class TextDocumentContentRegistrationOptions;
class TextDocumentContentRefreshParams;
class RegistrationParams;
class UnregistrationParams;
class InitializeParams;
class InitializeResult;
class InitializeError;
class InitializedParams;
class DidChangeConfigurationParams;
class DidChangeConfigurationRegistrationOptions;
class ShowMessageParams;
class ShowMessageRequestParams;
class MessageActionItem;
class LogMessageParams;
class DidOpenTextDocumentParams;
class DidChangeTextDocumentParams;
class TextDocumentChangeRegistrationOptions;
class DidCloseTextDocumentParams;
class DidSaveTextDocumentParams;
class TextDocumentSaveRegistrationOptions;
class WillSaveTextDocumentParams;
class TextEdit;
class DidChangeWatchedFilesParams;
class DidChangeWatchedFilesRegistrationOptions;
class PublishDiagnosticsParams;
class CompletionParams;
class CompletionItem;
class CompletionList;
class CompletionRegistrationOptions;
class HoverParams;
class Hover;
class HoverRegistrationOptions;
class SignatureHelpParams;
class SignatureHelp;
class SignatureHelpRegistrationOptions;
class DefinitionParams;
class DefinitionRegistrationOptions;
class ReferenceParams;
class ReferenceRegistrationOptions;
class DocumentHighlightParams;
class DocumentHighlight;
class DocumentHighlightRegistrationOptions;
class DocumentSymbolParams;
class SymbolInformation;
class DocumentSymbol;
class DocumentSymbolRegistrationOptions;
class CodeActionParams;
class Command;
class CodeAction;
class CodeActionRegistrationOptions;
class WorkspaceSymbolParams;
class WorkspaceSymbol;
class WorkspaceSymbolRegistrationOptions;
class CodeLensParams;
class CodeLens;
class CodeLensRegistrationOptions;
class DocumentLinkParams;
class DocumentLink;
class DocumentLinkRegistrationOptions;
class DocumentFormattingParams;
class DocumentFormattingRegistrationOptions;
class DocumentRangeFormattingParams;
class DocumentRangeFormattingRegistrationOptions;
class DocumentRangesFormattingParams;
class DocumentOnTypeFormattingParams;
class DocumentOnTypeFormattingRegistrationOptions;
class RenameParams;
class RenameRegistrationOptions;
class PrepareRenameParams;
class ExecuteCommandParams;
class ExecuteCommandRegistrationOptions;
class ApplyWorkspaceEditParams;
class ApplyWorkspaceEditResult;
class WorkDoneProgressBegin;
class WorkDoneProgressReport;
class WorkDoneProgressEnd;
class SetTraceParams;
class LogTraceParams;
class CancelParams;
class ProgressParams;
class TextDocumentPositionParams;
class WorkDoneProgressParams;
class PartialResultParams;
class LocationLink;
class Range;
class ImplementationOptions;
class StaticRegistrationOptions;
class TypeDefinitionOptions;
class WorkspaceFoldersChangeEvent;
class ConfigurationItem;
class TextDocumentIdentifier;
class Color;
class DocumentColorOptions;
class FoldingRangeOptions;
class DeclarationOptions;
class Position;
class SelectionRangeOptions;
class CallHierarchyOptions;
class SemanticTokensOptions;
class SemanticTokensEdit;
class LinkedEditingRangeOptions;
class FileCreate;
class TextDocumentEdit;
class CreateFile;
class RenameFile;
class DeleteFile;
class ChangeAnnotation;
class FileOperationFilter;
class FileRename;
class FileDelete;
class MonikerOptions;
class TypeHierarchyOptions;
class InlineValueContext;
class InlineValueText;
class InlineValueVariableLookup;
class InlineValueEvaluatableExpression;
class InlineValueOptions;
class InlayHintLabelPart;
class MarkupContent;
class InlayHintOptions;
class RelatedFullDocumentDiagnosticReport;
class RelatedUnchangedDocumentDiagnosticReport;
class FullDocumentDiagnosticReport;
class UnchangedDocumentDiagnosticReport;
class DiagnosticOptions;
class PreviousResultId;
class NotebookDocument;
class TextDocumentItem;
class NotebookDocumentSyncOptions;
class VersionedNotebookDocumentIdentifier;
class NotebookDocumentChangeEvent;
class NotebookDocumentIdentifier;
class InlineCompletionContext;
class StringValue;
class InlineCompletionOptions;
class TextDocumentContentOptions;
class Registration;
class Unregistration;
class _InitializeParams;
class WorkspaceFoldersInitializeParams;
class ServerCapabilities;
class ServerInfo;
class VersionedTextDocumentIdentifier;
class SaveOptions;
class FileEvent;
class FileSystemWatcher;
class Diagnostic;
class CompletionContext;
class CompletionItemLabelDetails;
class InsertReplaceEdit;
class CompletionItemDefaults;
class CompletionItemApplyKinds;
class CompletionOptions;
class HoverOptions;
class SignatureHelpContext;
class SignatureInformation;
class SignatureHelpOptions;
class DefinitionOptions;
class ReferenceContext;
class ReferenceOptions;
class DocumentHighlightOptions;
class BaseSymbolInformation;
class DocumentSymbolOptions;
class CodeActionContext;
class CodeActionDisabled;
class CodeActionOptions;
class LocationUriOnly;
class WorkspaceSymbolOptions;
class CodeLensOptions;
class DocumentLinkOptions;
class FormattingOptions;
class DocumentFormattingOptions;
class DocumentRangeFormattingOptions;
class DocumentOnTypeFormattingOptions;
class RenameOptions;
class PrepareRenamePlaceholder;
class PrepareRenameDefaultBehavior;
class ExecuteCommandOptions;
class WorkspaceEditMetadata;
class SemanticTokensLegend;
class SemanticTokensFullDelta;
class OptionalVersionedTextDocumentIdentifier;
class AnnotatedTextEdit;
class SnippetTextEdit;
class ResourceOperation;
class CreateFileOptions;
class RenameFileOptions;
class DeleteFileOptions;
class FileOperationPattern;
class WorkspaceFullDocumentDiagnosticReport;
class WorkspaceUnchangedDocumentDiagnosticReport;
class NotebookCell;
class NotebookDocumentFilterWithNotebook;
class NotebookDocumentFilterWithCells;
class NotebookDocumentCellChanges;
class SelectedCompletionInfo;
class ClientInfo;
class ClientCapabilities;
class TextDocumentSyncOptions;
class WorkspaceOptions;
class TextDocumentContentChangePartial;
class TextDocumentContentChangeWholeDocument;
class CodeDescription;
class DiagnosticRelatedInformation;
class EditRangeWithInsertReplace;
class ServerCompletionItemOptions;
class MarkedStringWithLanguage;
class ParameterInformation;
class CodeActionKindDocumentation;
class NotebookCellTextDocumentFilter;
class FileOperationPatternOptions;
class ExecutionSummary;
class NotebookCellLanguage;
class NotebookDocumentCellChangeStructure;
class NotebookDocumentCellContentChanges;
class WorkspaceClientCapabilities;
class TextDocumentClientCapabilities;
class NotebookDocumentClientCapabilities;
class WindowClientCapabilities;
class GeneralClientCapabilities;
class WorkspaceFoldersServerCapabilities;
class FileOperationOptions;
class RelativePattern;
class TextDocumentFilterLanguage;
class TextDocumentFilterScheme;
class TextDocumentFilterPattern;
class NotebookDocumentFilterNotebookType;
class NotebookDocumentFilterScheme;
class NotebookDocumentFilterPattern;
class NotebookCellArrayChange;
class WorkspaceEditClientCapabilities;
class DidChangeConfigurationClientCapabilities;
class DidChangeWatchedFilesClientCapabilities;
class WorkspaceSymbolClientCapabilities;
class ExecuteCommandClientCapabilities;
class SemanticTokensWorkspaceClientCapabilities;
class CodeLensWorkspaceClientCapabilities;
class FileOperationClientCapabilities;
class InlineValueWorkspaceClientCapabilities;
class InlayHintWorkspaceClientCapabilities;
class DiagnosticWorkspaceClientCapabilities;
class FoldingRangeWorkspaceClientCapabilities;
class TextDocumentContentClientCapabilities;
class TextDocumentSyncClientCapabilities;
class TextDocumentFilterClientCapabilities;
class CompletionClientCapabilities;
class HoverClientCapabilities;
class SignatureHelpClientCapabilities;
class DeclarationClientCapabilities;
class DefinitionClientCapabilities;
class TypeDefinitionClientCapabilities;
class ImplementationClientCapabilities;
class ReferenceClientCapabilities;
class DocumentHighlightClientCapabilities;
class DocumentSymbolClientCapabilities;
class CodeActionClientCapabilities;
class CodeLensClientCapabilities;
class DocumentLinkClientCapabilities;
class DocumentColorClientCapabilities;
class DocumentFormattingClientCapabilities;
class DocumentRangeFormattingClientCapabilities;
class DocumentOnTypeFormattingClientCapabilities;
class RenameClientCapabilities;
class FoldingRangeClientCapabilities;
class SelectionRangeClientCapabilities;
class PublishDiagnosticsClientCapabilities;
class CallHierarchyClientCapabilities;
class SemanticTokensClientCapabilities;
class LinkedEditingRangeClientCapabilities;
class MonikerClientCapabilities;
class TypeHierarchyClientCapabilities;
class InlineValueClientCapabilities;
class InlayHintClientCapabilities;
class DiagnosticClientCapabilities;
class InlineCompletionClientCapabilities;
class NotebookDocumentSyncClientCapabilities;
class ShowMessageRequestClientCapabilities;
class ShowDocumentClientCapabilities;
class StaleRequestSupportOptions;
class RegularExpressionsClientCapabilities;
class MarkdownClientCapabilities;
class ChangeAnnotationsSupportOptions;
class ClientSymbolKindOptions;
class ClientSymbolTagOptions;
class ClientSymbolResolveOptions;
class ClientCompletionItemOptions;
class ClientCompletionItemOptionsKind;
class CompletionListCapabilities;
class ClientSignatureInformationOptions;
class ClientCodeActionLiteralOptions;
class ClientCodeActionResolveOptions;
class CodeActionTagOptions;
class ClientCodeLensResolveOptions;
class ClientFoldingRangeKindOptions;
class ClientFoldingRangeOptions;
class DiagnosticsCapabilities;
class ClientSemanticTokensRequestOptions;
class ClientInlayHintResolveOptions;
class ClientShowMessageActionItemOptions;
class CompletionItemTagOptions;
class ClientCompletionItemResolveOptions;
class ClientCompletionItemInsertTextModeOptions;
class ClientSignatureParameterInformationOptions;
class ClientCodeActionKindOptions;
class ClientDiagnosticsTagOptions;
class ClientSemanticTokensRequestFullDelta;
// requests
class ImplementationRequest;
class ImplementationResponse;
class TypeDefinitionRequest;
class TypeDefinitionResponse;
class WorkspaceFoldersRequest;
class WorkspaceFoldersResponse;
class ConfigurationRequest;
class ConfigurationResponse;
class DocumentColorRequest;
class DocumentColorResponse;
class ColorPresentationRequest;
class ColorPresentationResponse;
class FoldingRangeRequest;
class FoldingRangeResponse;
class FoldingRangeRefreshRequest;
class FoldingRangeRefreshResponse;
class DeclarationRequest;
class DeclarationResponse;
class SelectionRangeRequest;
class SelectionRangeResponse;
class WorkDoneProgressCreateRequest;
class WorkDoneProgressCreateResponse;
class CallHierarchyPrepareRequest;
class CallHierarchyPrepareResponse;
class CallHierarchyIncomingCallsRequest;
class CallHierarchyIncomingCallsResponse;
class CallHierarchyOutgoingCallsRequest;
class CallHierarchyOutgoingCallsResponse;
class SemanticTokensRequest;
class SemanticTokensResponse;
class SemanticTokensDeltaRequest;
class SemanticTokensDeltaResponse;
class SemanticTokensRangeRequest;
class SemanticTokensRangeResponse;
class SemanticTokensRefreshRequest;
class SemanticTokensRefreshResponse;
class ShowDocumentRequest;
class ShowDocumentResponse;
class LinkedEditingRangeRequest;
class LinkedEditingRangeResponse;
class WillCreateFilesRequest;
class WillCreateFilesResponse;
class WillRenameFilesRequest;
class WillRenameFilesResponse;
class WillDeleteFilesRequest;
class WillDeleteFilesResponse;
class MonikerRequest;
class MonikerResponse;
class TypeHierarchyPrepareRequest;
class TypeHierarchyPrepareResponse;
class TypeHierarchySupertypesRequest;
class TypeHierarchySupertypesResponse;
class TypeHierarchySubtypesRequest;
class TypeHierarchySubtypesResponse;
class InlineValueRequest;
class InlineValueResponse;
class InlineValueRefreshRequest;
class InlineValueRefreshResponse;
class InlayHintRequest;
class InlayHintResponse;
class InlayHintResolveRequest;
class InlayHintResolveResponse;
class InlayHintRefreshRequest;
class InlayHintRefreshResponse;
class DocumentDiagnosticRequest;
class DocumentDiagnosticResponse;
class WorkspaceDiagnosticRequest;
class WorkspaceDiagnosticResponse;
class DiagnosticRefreshRequest;
class DiagnosticRefreshResponse;
class InlineCompletionRequest;
class InlineCompletionResponse;
class TextDocumentContentRequest;
class TextDocumentContentResponse;
class TextDocumentContentRefreshRequest;
class TextDocumentContentRefreshResponse;
class RegistrationRequest;
class RegistrationResponse;
class UnregistrationRequest;
class UnregistrationResponse;
class InitializeRequest;
class InitializeResponse;
class ShutdownRequest;
class ShutdownResponse;
class ShowMessageRequest;
class ShowMessageResponse;
class WillSaveTextDocumentWaitUntilRequest;
class WillSaveTextDocumentWaitUntilResponse;
class CompletionRequest;
class CompletionResponse;
class CompletionResolveRequest;
class CompletionResolveResponse;
class HoverRequest;
class HoverResponse;
class SignatureHelpRequest;
class SignatureHelpResponse;
class DefinitionRequest;
class DefinitionResponse;
class ReferencesRequest;
class ReferencesResponse;
class DocumentHighlightRequest;
class DocumentHighlightResponse;
class DocumentSymbolRequest;
class DocumentSymbolResponse;
class CodeActionRequest;
class CodeActionResponse;
class CodeActionResolveRequest;
class CodeActionResolveResponse;
class WorkspaceSymbolRequest;
class WorkspaceSymbolResponse;
class WorkspaceSymbolResolveRequest;
class WorkspaceSymbolResolveResponse;
class CodeLensRequest;
class CodeLensResponse;
class CodeLensResolveRequest;
class CodeLensResolveResponse;
class CodeLensRefreshRequest;
class CodeLensRefreshResponse;
class DocumentLinkRequest;
class DocumentLinkResponse;
class DocumentLinkResolveRequest;
class DocumentLinkResolveResponse;
class DocumentFormattingRequest;
class DocumentFormattingResponse;
class DocumentRangeFormattingRequest;
class DocumentRangeFormattingResponse;
class DocumentRangesFormattingRequest;
class DocumentRangesFormattingResponse;
class DocumentOnTypeFormattingRequest;
class DocumentOnTypeFormattingResponse;
class RenameRequest;
class RenameResponse;
class PrepareRenameRequest;
class PrepareRenameResponse;
class ExecuteCommandRequest;
class ExecuteCommandResponse;
class ApplyWorkspaceEditRequest;
class ApplyWorkspaceEditResponse;
class DidChangeWorkspaceFoldersNotification;
class WorkDoneProgressCancelNotification;
class DidCreateFilesNotification;
class DidRenameFilesNotification;
class DidDeleteFilesNotification;
class DidOpenNotebookDocumentNotification;
class DidChangeNotebookDocumentNotification;
class DidSaveNotebookDocumentNotification;
class DidCloseNotebookDocumentNotification;
class InitializedNotification;
class ExitNotification;
class DidChangeConfigurationNotification;
class ShowMessageNotification;
class LogMessageNotification;
class TelemetryEventNotification;
class DidOpenTextDocumentNotification;
class DidChangeTextDocumentNotification;
class DidCloseTextDocumentNotification;
class DidSaveTextDocumentNotification;
class WillSaveTextDocumentNotification;
class DidChangeWatchedFilesNotification;
class PublishDiagnosticsNotification;
class SetTraceNotification;
class LogTraceNotification;
class CancelNotification;
class ProgressNotification;

class LSPObject;

using LSPAny = json;
using Pattern = std::string;

[[nodiscard]] auto withUnsafeJson(auto block) { return block(json()); }
[[noreturn]] void lsp_runtime_error(const std::string& msg);

class LSPObject {
 public:
  LSPObject() = default;
  explicit LSPObject(json& repr) : repr_(&repr) {}

  [[nodiscard]] explicit operator bool() const { return repr_ != nullptr; }
  [[nodiscard]] operator const json&() const { return *repr_; }
  [[nodiscard]] auto get() const -> json& { return *repr_; }

 protected:
  json* repr_{nullptr};
};

class LSPRequest : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] auto id() const
      -> std::optional<std::variant<long, std::string>>;
  [[nodiscard]] auto method() const -> std::string;
};

class LSPResponse : public LSPObject {
 public:
  using LSPObject::LSPObject;
};

template <typename T>
class Vector final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const {
    return repr_ && repr_->is_array();
  }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(int index) const -> T { return T(repr_->at(index)); }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    repr_->emplace_back(std::forward<Args>(args)...);
  }
};

namespace details {

template <typename T>
struct TryEmplace;

template <std::derived_from<LSPObject> T>
struct TryEmplace<T> {
  auto operator()(auto& result, json& value) const -> bool {
    auto obj = T{value};
    if (!obj) return false;
    result.template emplace<T>(std::move(obj));
    return true;
  }
};

template <typename... Ts>
auto try_emplace(std::variant<Ts...>& result, json& value) -> bool {
  return (details::TryEmplace<Ts>{}(result, value) || ...);
}

template <>
struct TryEmplace<std::nullptr_t> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_null()) return false;
    result.template emplace<std::nullptr_t>(nullptr);
    return true;
  }
};

template <>
struct TryEmplace<bool> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_boolean()) return false;
    result.template emplace<bool>(value);
    return true;
  }
};

template <>
struct TryEmplace<int> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_integer()) return false;
    result.template emplace<int>(value);
    return true;
  }
};

template <>
struct TryEmplace<long> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_integer()) return false;
    result.template emplace<long>(value);
    return true;
  }
};

template <>
struct TryEmplace<double> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_float()) return false;
    result.template emplace<double>(value);
    return true;
  }
};

template <>
struct TryEmplace<std::string> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_string()) return false;
    result.template emplace<std::string>(value);
    return true;
  }
};

template <typename... Ts>
struct TryEmplace<std::variant<Ts...>> {
  auto operator()(auto& result, json& value) const -> bool {
    return try_emplace(result, value);
  }
};

template <typename... Ts>
struct TryEmplace<std::tuple<Ts...>> {
  auto operator()(auto& result, json& value) const -> bool {
    lsp_runtime_error("todo: TryEmplace<std::tuple<Ts...>>");
    return false;
  }
};

template <>
struct TryEmplace<json> {
  auto operator()(auto& result, json& value) const -> bool {
    result = value;
    return true;
  }
};

template <>
struct TryEmplace<TextDocumentSyncKind> {
  auto operator()(auto& result, json& value) const -> bool {
    if (!value.is_number_integer()) return false;
    result = TextDocumentSyncKind(value.get<int>());
    return true;
  }
};

}  // namespace details

template <>
class Vector<std::string> final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const {
    return repr_ && repr_->is_array();
  }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(int index) const -> std::string {
    return repr_->at(index);
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    repr_->emplace_back(std::forward<Args>(args)...);
  }
};

template <std::derived_from<LSPObject> T>
class Vector<T> final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const {
    return repr_ && repr_->is_array();
  }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(int index) const -> T { return T(repr_->at(index)); }

  template <typename... Args>
  auto emplace_back(Args&&... args) -> T {
    return T(repr_->emplace_back(std::forward<Args>(args)...));
  }
};

template <typename... Ts>
class Vector<std::variant<Ts...>> final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const {
    return repr_ && repr_->is_array();
  }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(int index) const -> std::variant<Ts...> {
    std::variant<Ts...> result;
    details::try_emplace(result, repr_->at(index));
    return result;
  }

  template <typename T>
  [[nodiscard]] auto emplace_back() -> std::variant<Ts...> {
    std::variant<Ts...> result;
    details::TryEmplace<T>{}(result, repr_->emplace_back());
    return result;
  }

  template <std::derived_from<LSPObject> T>
  [[nodiscard]] auto emplace_back() -> T {
    return T(repr_->emplace_back());
  }
};

template <typename Key, typename Value>
class Map final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const {
    return repr_ && repr_->is_object();
  }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }
  [[nodiscard]] auto at(const Key& key) const -> const Value& {
    return repr_->at(key);
  }
};

template <typename Key, typename... Ts>
class Map<Key, std::variant<Ts...>> final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  [[nodiscard]] explicit operator bool() const {
    return repr_ && repr_->is_object();
  }
  [[nodiscard]] auto size() const -> std::size_t { return repr_->size(); }
  [[nodiscard]] auto empty() const -> bool { return repr_->empty(); }

  [[nodiscard]] auto at(const Key& key) const -> std::variant<Ts...> {
    std::variant<Ts...> result;
    details::try_emplace(result, repr_->at(key));
    return result;
  }
};

using RegularExpressionEngineKind = std::string;

using NotebookDocumentFilter =
    std::variant<NotebookDocumentFilterNotebookType,
                 NotebookDocumentFilterScheme, NotebookDocumentFilterPattern>;

using TextDocumentFilter =
    std::variant<TextDocumentFilterLanguage, TextDocumentFilterScheme,
                 TextDocumentFilterPattern>;

using GlobPattern = std::variant<Pattern, RelativePattern>;

using DocumentFilter =
    std::variant<TextDocumentFilter, NotebookCellTextDocumentFilter>;

using MarkedString = std::variant<std::string, MarkedStringWithLanguage>;

using TextDocumentContentChangeEvent =
    std::variant<TextDocumentContentChangePartial,
                 TextDocumentContentChangeWholeDocument>;

using WorkspaceDocumentDiagnosticReport =
    std::variant<WorkspaceFullDocumentDiagnosticReport,
                 WorkspaceUnchangedDocumentDiagnosticReport>;

using ChangeAnnotationIdentifier = std::string;

using ProgressToken = std::variant<int, std::string>;

using DocumentSelector = Vector<DocumentFilter>;

using PrepareRenameResult =
    std::variant<Range, PrepareRenamePlaceholder, PrepareRenameDefaultBehavior>;

using DocumentDiagnosticReport =
    std::variant<RelatedFullDocumentDiagnosticReport,
                 RelatedUnchangedDocumentDiagnosticReport>;

using InlineValue = std::variant<InlineValueText, InlineValueVariableLookup,
                                 InlineValueEvaluatableExpression>;

using DeclarationLink = LocationLink;

using Declaration = std::variant<Location, Vector<Location>>;

using DefinitionLink = LocationLink;

using Definition = std::variant<Location, Vector<Location>>;

}  // namespace cxx::lsp
