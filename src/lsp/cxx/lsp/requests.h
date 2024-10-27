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

#include <cxx/lsp/fwd.h>

namespace cxx::lsp {

#define FOR_EACH_LSP_REQUEST_TYPE(V)                                     \
  V(Implementation, "textDocument/implementation")                       \
  V(TypeDefinition, "textDocument/typeDefinition")                       \
  V(WorkspaceFolders, "workspace/workspaceFolders")                      \
  V(Configuration, "workspace/configuration")                            \
  V(DocumentColor, "textDocument/documentColor")                         \
  V(ColorPresentation, "textDocument/colorPresentation")                 \
  V(FoldingRange, "textDocument/foldingRange")                           \
  V(FoldingRangeRefresh, "workspace/foldingRange/refresh")               \
  V(Declaration, "textDocument/declaration")                             \
  V(SelectionRange, "textDocument/selectionRange")                       \
  V(WorkDoneProgressCreate, "window/workDoneProgress/create")            \
  V(CallHierarchyPrepare, "textDocument/prepareCallHierarchy")           \
  V(CallHierarchyIncomingCalls, "callHierarchy/incomingCalls")           \
  V(CallHierarchyOutgoingCalls, "callHierarchy/outgoingCalls")           \
  V(SemanticTokens, "textDocument/semanticTokens/full")                  \
  V(SemanticTokensDelta, "textDocument/semanticTokens/full/delta")       \
  V(SemanticTokensRange, "textDocument/semanticTokens/range")            \
  V(SemanticTokensRefresh, "workspace/semanticTokens/refresh")           \
  V(ShowDocument, "window/showDocument")                                 \
  V(LinkedEditingRange, "textDocument/linkedEditingRange")               \
  V(WillCreateFiles, "workspace/willCreateFiles")                        \
  V(WillRenameFiles, "workspace/willRenameFiles")                        \
  V(WillDeleteFiles, "workspace/willDeleteFiles")                        \
  V(Moniker, "textDocument/moniker")                                     \
  V(TypeHierarchyPrepare, "textDocument/prepareTypeHierarchy")           \
  V(TypeHierarchySupertypes, "typeHierarchy/supertypes")                 \
  V(TypeHierarchySubtypes, "typeHierarchy/subtypes")                     \
  V(InlineValue, "textDocument/inlineValue")                             \
  V(InlineValueRefresh, "workspace/inlineValue/refresh")                 \
  V(InlayHint, "textDocument/inlayHint")                                 \
  V(InlayHintResolve, "inlayHint/resolve")                               \
  V(InlayHintRefresh, "workspace/inlayHint/refresh")                     \
  V(DocumentDiagnostic, "textDocument/diagnostic")                       \
  V(WorkspaceDiagnostic, "workspace/diagnostic")                         \
  V(DiagnosticRefresh, "workspace/diagnostic/refresh")                   \
  V(InlineCompletion, "textDocument/inlineCompletion")                   \
  V(TextDocumentContent, "workspace/textDocumentContent")                \
  V(TextDocumentContentRefresh, "workspace/textDocumentContent/refresh") \
  V(Registration, "client/registerCapability")                           \
  V(Unregistration, "client/unregisterCapability")                       \
  V(Initialize, "initialize")                                            \
  V(Shutdown, "shutdown")                                                \
  V(ShowMessage, "window/showMessageRequest")                            \
  V(WillSaveTextDocumentWaitUntil, "textDocument/willSaveWaitUntil")     \
  V(Completion, "textDocument/completion")                               \
  V(CompletionResolve, "completionItem/resolve")                         \
  V(Hover, "textDocument/hover")                                         \
  V(SignatureHelp, "textDocument/signatureHelp")                         \
  V(Definition, "textDocument/definition")                               \
  V(References, "textDocument/references")                               \
  V(DocumentHighlight, "textDocument/documentHighlight")                 \
  V(DocumentSymbol, "textDocument/documentSymbol")                       \
  V(CodeAction, "textDocument/codeAction")                               \
  V(CodeActionResolve, "codeAction/resolve")                             \
  V(WorkspaceSymbol, "workspace/symbol")                                 \
  V(WorkspaceSymbolResolve, "workspaceSymbol/resolve")                   \
  V(CodeLens, "textDocument/codeLens")                                   \
  V(CodeLensResolve, "codeLens/resolve")                                 \
  V(CodeLensRefresh, "workspace/codeLens/refresh")                       \
  V(DocumentLink, "textDocument/documentLink")                           \
  V(DocumentLinkResolve, "documentLink/resolve")                         \
  V(DocumentFormatting, "textDocument/formatting")                       \
  V(DocumentRangeFormatting, "textDocument/rangeFormatting")             \
  V(DocumentRangesFormatting, "textDocument/rangesFormatting")           \
  V(DocumentOnTypeFormatting, "textDocument/onTypeFormatting")           \
  V(Rename, "textDocument/rename")                                       \
  V(PrepareRename, "textDocument/prepareRename")                         \
  V(ExecuteCommand, "workspace/executeCommand")                          \
  V(ApplyWorkspaceEdit, "workspace/applyEdit")

#define FOR_EACH_LSP_NOTIFICATION_TYPE(V)                             \
  V(DidChangeWorkspaceFolders, "workspace/didChangeWorkspaceFolders") \
  V(WorkDoneProgressCancel, "window/workDoneProgress/cancel")         \
  V(DidCreateFiles, "workspace/didCreateFiles")                       \
  V(DidRenameFiles, "workspace/didRenameFiles")                       \
  V(DidDeleteFiles, "workspace/didDeleteFiles")                       \
  V(DidOpenNotebookDocument, "notebookDocument/didOpen")              \
  V(DidChangeNotebookDocument, "notebookDocument/didChange")          \
  V(DidSaveNotebookDocument, "notebookDocument/didSave")              \
  V(DidCloseNotebookDocument, "notebookDocument/didClose")            \
  V(Initialized, "initialized")                                       \
  V(Exit, "exit")                                                     \
  V(DidChangeConfiguration, "workspace/didChangeConfiguration")       \
  V(ShowMessage, "window/showMessage")                                \
  V(LogMessage, "window/logMessage")                                  \
  V(TelemetryEvent, "telemetry/event")                                \
  V(DidOpenTextDocument, "textDocument/didOpen")                      \
  V(DidChangeTextDocument, "textDocument/didChange")                  \
  V(DidCloseTextDocument, "textDocument/didClose")                    \
  V(DidSaveTextDocument, "textDocument/didSave")                      \
  V(WillSaveTextDocument, "textDocument/willSave")                    \
  V(DidChangeWatchedFiles, "workspace/didChangeWatchedFiles")         \
  V(PublishDiagnostics, "textDocument/publishDiagnostics")            \
  V(SetTrace, "$/setTrace")                                           \
  V(LogTrace, "$/logTrace")                                           \
  V(Cancel, "$/cancelRequest")                                        \
  V(Progress, "$/progress")

class ImplementationRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ImplementationRequest&;
  auto id(std::variant<long, std::string> id) -> ImplementationRequest&;

  [[nodiscard]] auto params() const -> ImplementationParams;
  auto params(ImplementationParams result) -> ImplementationRequest&;
};

class ImplementationResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ImplementationResponse&;
  auto id(std::string id) -> ImplementationResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t>;

  auto result(
      std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result)
      -> ImplementationResponse&;
};

class TypeDefinitionRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TypeDefinitionRequest&;
  auto id(std::variant<long, std::string> id) -> TypeDefinitionRequest&;

  [[nodiscard]] auto params() const -> TypeDefinitionParams;
  auto params(TypeDefinitionParams result) -> TypeDefinitionRequest&;
};

class TypeDefinitionResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> TypeDefinitionResponse&;
  auto id(std::string id) -> TypeDefinitionResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t>;

  auto result(
      std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result)
      -> TypeDefinitionResponse&;
};

class WorkspaceFoldersRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WorkspaceFoldersRequest&;
  auto id(std::variant<long, std::string> id) -> WorkspaceFoldersRequest&;
};

class WorkspaceFoldersResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WorkspaceFoldersResponse&;
  auto id(std::string id) -> WorkspaceFoldersResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<WorkspaceFolder>, std::nullptr_t>;

  auto result(std::variant<Vector<WorkspaceFolder>, std::nullptr_t> result)
      -> WorkspaceFoldersResponse&;
};

class ConfigurationRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ConfigurationRequest&;
  auto id(std::variant<long, std::string> id) -> ConfigurationRequest&;

  [[nodiscard]] auto params() const -> ConfigurationParams;
  auto params(ConfigurationParams result) -> ConfigurationRequest&;
};

class ConfigurationResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ConfigurationResponse&;
  auto id(std::string id) -> ConfigurationResponse&;

  [[nodiscard]] auto result() const -> Vector<LSPAny>;

  auto result(Vector<LSPAny> result) -> ConfigurationResponse&;
};

class DocumentColorRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentColorRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentColorRequest&;

  [[nodiscard]] auto params() const -> DocumentColorParams;
  auto params(DocumentColorParams result) -> DocumentColorRequest&;
};

class DocumentColorResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentColorResponse&;
  auto id(std::string id) -> DocumentColorResponse&;

  [[nodiscard]] auto result() const -> Vector<ColorInformation>;

  auto result(Vector<ColorInformation> result) -> DocumentColorResponse&;
};

class ColorPresentationRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ColorPresentationRequest&;
  auto id(std::variant<long, std::string> id) -> ColorPresentationRequest&;

  [[nodiscard]] auto params() const -> ColorPresentationParams;
  auto params(ColorPresentationParams result) -> ColorPresentationRequest&;
};

class ColorPresentationResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ColorPresentationResponse&;
  auto id(std::string id) -> ColorPresentationResponse&;

  [[nodiscard]] auto result() const -> Vector<ColorPresentation>;

  auto result(Vector<ColorPresentation> result) -> ColorPresentationResponse&;
};

class FoldingRangeRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> FoldingRangeRequest&;
  auto id(std::variant<long, std::string> id) -> FoldingRangeRequest&;

  [[nodiscard]] auto params() const -> FoldingRangeParams;
  auto params(FoldingRangeParams result) -> FoldingRangeRequest&;
};

class FoldingRangeResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> FoldingRangeResponse&;
  auto id(std::string id) -> FoldingRangeResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<FoldingRange>, std::nullptr_t>;

  auto result(std::variant<Vector<FoldingRange>, std::nullptr_t> result)
      -> FoldingRangeResponse&;
};

class FoldingRangeRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> FoldingRangeRefreshRequest&;
  auto id(std::variant<long, std::string> id) -> FoldingRangeRefreshRequest&;
};

class FoldingRangeRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> FoldingRangeRefreshResponse&;
  auto id(std::string id) -> FoldingRangeRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> FoldingRangeRefreshResponse&;
};

class DeclarationRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DeclarationRequest&;
  auto id(std::variant<long, std::string> id) -> DeclarationRequest&;

  [[nodiscard]] auto params() const -> DeclarationParams;
  auto params(DeclarationParams result) -> DeclarationRequest&;
};

class DeclarationResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DeclarationResponse&;
  auto id(std::string id) -> DeclarationResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Declaration, Vector<DeclarationLink>, std::nullptr_t>;

  auto result(
      std::variant<Declaration, Vector<DeclarationLink>, std::nullptr_t> result)
      -> DeclarationResponse&;
};

class SelectionRangeRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SelectionRangeRequest&;
  auto id(std::variant<long, std::string> id) -> SelectionRangeRequest&;

  [[nodiscard]] auto params() const -> SelectionRangeParams;
  auto params(SelectionRangeParams result) -> SelectionRangeRequest&;
};

class SelectionRangeResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> SelectionRangeResponse&;
  auto id(std::string id) -> SelectionRangeResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<SelectionRange>, std::nullptr_t>;

  auto result(std::variant<Vector<SelectionRange>, std::nullptr_t> result)
      -> SelectionRangeResponse&;
};

class WorkDoneProgressCreateRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WorkDoneProgressCreateRequest&;
  auto id(std::variant<long, std::string> id) -> WorkDoneProgressCreateRequest&;

  [[nodiscard]] auto params() const -> WorkDoneProgressCreateParams;
  auto params(WorkDoneProgressCreateParams result)
      -> WorkDoneProgressCreateRequest&;
};

class WorkDoneProgressCreateResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WorkDoneProgressCreateResponse&;
  auto id(std::string id) -> WorkDoneProgressCreateResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> WorkDoneProgressCreateResponse&;
};

class CallHierarchyPrepareRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CallHierarchyPrepareRequest&;
  auto id(std::variant<long, std::string> id) -> CallHierarchyPrepareRequest&;

  [[nodiscard]] auto params() const -> CallHierarchyPrepareParams;
  auto params(CallHierarchyPrepareParams result)
      -> CallHierarchyPrepareRequest&;
};

class CallHierarchyPrepareResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CallHierarchyPrepareResponse&;
  auto id(std::string id) -> CallHierarchyPrepareResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<CallHierarchyItem>, std::nullptr_t>;

  auto result(std::variant<Vector<CallHierarchyItem>, std::nullptr_t> result)
      -> CallHierarchyPrepareResponse&;
};

class CallHierarchyIncomingCallsRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CallHierarchyIncomingCallsRequest&;
  auto id(std::variant<long, std::string> id)
      -> CallHierarchyIncomingCallsRequest&;

  [[nodiscard]] auto params() const -> CallHierarchyIncomingCallsParams;
  auto params(CallHierarchyIncomingCallsParams result)
      -> CallHierarchyIncomingCallsRequest&;
};

class CallHierarchyIncomingCallsResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CallHierarchyIncomingCallsResponse&;
  auto id(std::string id) -> CallHierarchyIncomingCallsResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<CallHierarchyIncomingCall>, std::nullptr_t>;

  auto result(
      std::variant<Vector<CallHierarchyIncomingCall>, std::nullptr_t> result)
      -> CallHierarchyIncomingCallsResponse&;
};

class CallHierarchyOutgoingCallsRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CallHierarchyOutgoingCallsRequest&;
  auto id(std::variant<long, std::string> id)
      -> CallHierarchyOutgoingCallsRequest&;

  [[nodiscard]] auto params() const -> CallHierarchyOutgoingCallsParams;
  auto params(CallHierarchyOutgoingCallsParams result)
      -> CallHierarchyOutgoingCallsRequest&;
};

class CallHierarchyOutgoingCallsResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CallHierarchyOutgoingCallsResponse&;
  auto id(std::string id) -> CallHierarchyOutgoingCallsResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<CallHierarchyOutgoingCall>, std::nullptr_t>;

  auto result(
      std::variant<Vector<CallHierarchyOutgoingCall>, std::nullptr_t> result)
      -> CallHierarchyOutgoingCallsResponse&;
};

class SemanticTokensRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SemanticTokensRequest&;
  auto id(std::variant<long, std::string> id) -> SemanticTokensRequest&;

  [[nodiscard]] auto params() const -> SemanticTokensParams;
  auto params(SemanticTokensParams result) -> SemanticTokensRequest&;
};

class SemanticTokensResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> SemanticTokensResponse&;
  auto id(std::string id) -> SemanticTokensResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<SemanticTokens, std::nullptr_t>;

  auto result(std::variant<SemanticTokens, std::nullptr_t> result)
      -> SemanticTokensResponse&;
};

class SemanticTokensDeltaRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SemanticTokensDeltaRequest&;
  auto id(std::variant<long, std::string> id) -> SemanticTokensDeltaRequest&;

  [[nodiscard]] auto params() const -> SemanticTokensDeltaParams;
  auto params(SemanticTokensDeltaParams result) -> SemanticTokensDeltaRequest&;
};

class SemanticTokensDeltaResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> SemanticTokensDeltaResponse&;
  auto id(std::string id) -> SemanticTokensDeltaResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<SemanticTokens, SemanticTokensDelta, std::nullptr_t>;

  auto result(
      std::variant<SemanticTokens, SemanticTokensDelta, std::nullptr_t> result)
      -> SemanticTokensDeltaResponse&;
};

class SemanticTokensRangeRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SemanticTokensRangeRequest&;
  auto id(std::variant<long, std::string> id) -> SemanticTokensRangeRequest&;

  [[nodiscard]] auto params() const -> SemanticTokensRangeParams;
  auto params(SemanticTokensRangeParams result) -> SemanticTokensRangeRequest&;
};

class SemanticTokensRangeResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> SemanticTokensRangeResponse&;
  auto id(std::string id) -> SemanticTokensRangeResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<SemanticTokens, std::nullptr_t>;

  auto result(std::variant<SemanticTokens, std::nullptr_t> result)
      -> SemanticTokensRangeResponse&;
};

class SemanticTokensRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SemanticTokensRefreshRequest&;
  auto id(std::variant<long, std::string> id) -> SemanticTokensRefreshRequest&;
};

class SemanticTokensRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> SemanticTokensRefreshResponse&;
  auto id(std::string id) -> SemanticTokensRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> SemanticTokensRefreshResponse&;
};

class ShowDocumentRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ShowDocumentRequest&;
  auto id(std::variant<long, std::string> id) -> ShowDocumentRequest&;

  [[nodiscard]] auto params() const -> ShowDocumentParams;
  auto params(ShowDocumentParams result) -> ShowDocumentRequest&;
};

class ShowDocumentResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ShowDocumentResponse&;
  auto id(std::string id) -> ShowDocumentResponse&;

  [[nodiscard]] auto result() const -> ShowDocumentResult;

  auto result(ShowDocumentResult result) -> ShowDocumentResponse&;
};

class LinkedEditingRangeRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> LinkedEditingRangeRequest&;
  auto id(std::variant<long, std::string> id) -> LinkedEditingRangeRequest&;

  [[nodiscard]] auto params() const -> LinkedEditingRangeParams;
  auto params(LinkedEditingRangeParams result) -> LinkedEditingRangeRequest&;
};

class LinkedEditingRangeResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> LinkedEditingRangeResponse&;
  auto id(std::string id) -> LinkedEditingRangeResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<LinkedEditingRanges, std::nullptr_t>;

  auto result(std::variant<LinkedEditingRanges, std::nullptr_t> result)
      -> LinkedEditingRangeResponse&;
};

class WillCreateFilesRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WillCreateFilesRequest&;
  auto id(std::variant<long, std::string> id) -> WillCreateFilesRequest&;

  [[nodiscard]] auto params() const -> CreateFilesParams;
  auto params(CreateFilesParams result) -> WillCreateFilesRequest&;
};

class WillCreateFilesResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WillCreateFilesResponse&;
  auto id(std::string id) -> WillCreateFilesResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<WorkspaceEdit, std::nullptr_t>;

  auto result(std::variant<WorkspaceEdit, std::nullptr_t> result)
      -> WillCreateFilesResponse&;
};

class WillRenameFilesRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WillRenameFilesRequest&;
  auto id(std::variant<long, std::string> id) -> WillRenameFilesRequest&;

  [[nodiscard]] auto params() const -> RenameFilesParams;
  auto params(RenameFilesParams result) -> WillRenameFilesRequest&;
};

class WillRenameFilesResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WillRenameFilesResponse&;
  auto id(std::string id) -> WillRenameFilesResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<WorkspaceEdit, std::nullptr_t>;

  auto result(std::variant<WorkspaceEdit, std::nullptr_t> result)
      -> WillRenameFilesResponse&;
};

class WillDeleteFilesRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WillDeleteFilesRequest&;
  auto id(std::variant<long, std::string> id) -> WillDeleteFilesRequest&;

  [[nodiscard]] auto params() const -> DeleteFilesParams;
  auto params(DeleteFilesParams result) -> WillDeleteFilesRequest&;
};

class WillDeleteFilesResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WillDeleteFilesResponse&;
  auto id(std::string id) -> WillDeleteFilesResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<WorkspaceEdit, std::nullptr_t>;

  auto result(std::variant<WorkspaceEdit, std::nullptr_t> result)
      -> WillDeleteFilesResponse&;
};

class MonikerRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> MonikerRequest&;
  auto id(std::variant<long, std::string> id) -> MonikerRequest&;

  [[nodiscard]] auto params() const -> MonikerParams;
  auto params(MonikerParams result) -> MonikerRequest&;
};

class MonikerResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> MonikerResponse&;
  auto id(std::string id) -> MonikerResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<Moniker>, std::nullptr_t>;

  auto result(std::variant<Vector<Moniker>, std::nullptr_t> result)
      -> MonikerResponse&;
};

class TypeHierarchyPrepareRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TypeHierarchyPrepareRequest&;
  auto id(std::variant<long, std::string> id) -> TypeHierarchyPrepareRequest&;

  [[nodiscard]] auto params() const -> TypeHierarchyPrepareParams;
  auto params(TypeHierarchyPrepareParams result)
      -> TypeHierarchyPrepareRequest&;
};

class TypeHierarchyPrepareResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> TypeHierarchyPrepareResponse&;
  auto id(std::string id) -> TypeHierarchyPrepareResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t>;

  auto result(std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result)
      -> TypeHierarchyPrepareResponse&;
};

class TypeHierarchySupertypesRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TypeHierarchySupertypesRequest&;
  auto id(std::variant<long, std::string> id)
      -> TypeHierarchySupertypesRequest&;

  [[nodiscard]] auto params() const -> TypeHierarchySupertypesParams;
  auto params(TypeHierarchySupertypesParams result)
      -> TypeHierarchySupertypesRequest&;
};

class TypeHierarchySupertypesResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> TypeHierarchySupertypesResponse&;
  auto id(std::string id) -> TypeHierarchySupertypesResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t>;

  auto result(std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result)
      -> TypeHierarchySupertypesResponse&;
};

class TypeHierarchySubtypesRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TypeHierarchySubtypesRequest&;
  auto id(std::variant<long, std::string> id) -> TypeHierarchySubtypesRequest&;

  [[nodiscard]] auto params() const -> TypeHierarchySubtypesParams;
  auto params(TypeHierarchySubtypesParams result)
      -> TypeHierarchySubtypesRequest&;
};

class TypeHierarchySubtypesResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> TypeHierarchySubtypesResponse&;
  auto id(std::string id) -> TypeHierarchySubtypesResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t>;

  auto result(std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result)
      -> TypeHierarchySubtypesResponse&;
};

class InlineValueRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InlineValueRequest&;
  auto id(std::variant<long, std::string> id) -> InlineValueRequest&;

  [[nodiscard]] auto params() const -> InlineValueParams;
  auto params(InlineValueParams result) -> InlineValueRequest&;
};

class InlineValueResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InlineValueResponse&;
  auto id(std::string id) -> InlineValueResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<InlineValue>, std::nullptr_t>;

  auto result(std::variant<Vector<InlineValue>, std::nullptr_t> result)
      -> InlineValueResponse&;
};

class InlineValueRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InlineValueRefreshRequest&;
  auto id(std::variant<long, std::string> id) -> InlineValueRefreshRequest&;
};

class InlineValueRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InlineValueRefreshResponse&;
  auto id(std::string id) -> InlineValueRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> InlineValueRefreshResponse&;
};

class InlayHintRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InlayHintRequest&;
  auto id(std::variant<long, std::string> id) -> InlayHintRequest&;

  [[nodiscard]] auto params() const -> InlayHintParams;
  auto params(InlayHintParams result) -> InlayHintRequest&;
};

class InlayHintResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InlayHintResponse&;
  auto id(std::string id) -> InlayHintResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<InlayHint>, std::nullptr_t>;

  auto result(std::variant<Vector<InlayHint>, std::nullptr_t> result)
      -> InlayHintResponse&;
};

class InlayHintResolveRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InlayHintResolveRequest&;
  auto id(std::variant<long, std::string> id) -> InlayHintResolveRequest&;

  [[nodiscard]] auto params() const -> InlayHint;
  auto params(InlayHint result) -> InlayHintResolveRequest&;
};

class InlayHintResolveResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InlayHintResolveResponse&;
  auto id(std::string id) -> InlayHintResolveResponse&;

  [[nodiscard]] auto result() const -> InlayHint;

  auto result(InlayHint result) -> InlayHintResolveResponse&;
};

class InlayHintRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InlayHintRefreshRequest&;
  auto id(std::variant<long, std::string> id) -> InlayHintRefreshRequest&;
};

class InlayHintRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InlayHintRefreshResponse&;
  auto id(std::string id) -> InlayHintRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> InlayHintRefreshResponse&;
};

class DocumentDiagnosticRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentDiagnosticRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentDiagnosticRequest&;

  [[nodiscard]] auto params() const -> DocumentDiagnosticParams;
  auto params(DocumentDiagnosticParams result) -> DocumentDiagnosticRequest&;
};

class DocumentDiagnosticResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentDiagnosticResponse&;
  auto id(std::string id) -> DocumentDiagnosticResponse&;

  [[nodiscard]] auto result() const -> DocumentDiagnosticReport;

  auto result(DocumentDiagnosticReport result) -> DocumentDiagnosticResponse&;
};

class WorkspaceDiagnosticRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WorkspaceDiagnosticRequest&;
  auto id(std::variant<long, std::string> id) -> WorkspaceDiagnosticRequest&;

  [[nodiscard]] auto params() const -> WorkspaceDiagnosticParams;
  auto params(WorkspaceDiagnosticParams result) -> WorkspaceDiagnosticRequest&;
};

class WorkspaceDiagnosticResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WorkspaceDiagnosticResponse&;
  auto id(std::string id) -> WorkspaceDiagnosticResponse&;

  [[nodiscard]] auto result() const -> WorkspaceDiagnosticReport;

  auto result(WorkspaceDiagnosticReport result) -> WorkspaceDiagnosticResponse&;
};

class DiagnosticRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DiagnosticRefreshRequest&;
  auto id(std::variant<long, std::string> id) -> DiagnosticRefreshRequest&;
};

class DiagnosticRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DiagnosticRefreshResponse&;
  auto id(std::string id) -> DiagnosticRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> DiagnosticRefreshResponse&;
};

class InlineCompletionRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InlineCompletionRequest&;
  auto id(std::variant<long, std::string> id) -> InlineCompletionRequest&;

  [[nodiscard]] auto params() const -> InlineCompletionParams;
  auto params(InlineCompletionParams result) -> InlineCompletionRequest&;
};

class InlineCompletionResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InlineCompletionResponse&;
  auto id(std::string id) -> InlineCompletionResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<InlineCompletionList, Vector<InlineCompletionItem>,
                      std::nullptr_t>;

  auto result(std::variant<InlineCompletionList, Vector<InlineCompletionItem>,
                           std::nullptr_t>
                  result) -> InlineCompletionResponse&;
};

class TextDocumentContentRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TextDocumentContentRequest&;
  auto id(std::variant<long, std::string> id) -> TextDocumentContentRequest&;

  [[nodiscard]] auto params() const -> TextDocumentContentParams;
  auto params(TextDocumentContentParams result) -> TextDocumentContentRequest&;
};

class TextDocumentContentResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> TextDocumentContentResponse&;
  auto id(std::string id) -> TextDocumentContentResponse&;

  [[nodiscard]] auto result() const -> TextDocumentContentResult;

  auto result(TextDocumentContentResult result) -> TextDocumentContentResponse&;
};

class TextDocumentContentRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TextDocumentContentRefreshRequest&;
  auto id(std::variant<long, std::string> id)
      -> TextDocumentContentRefreshRequest&;

  [[nodiscard]] auto params() const -> TextDocumentContentRefreshParams;
  auto params(TextDocumentContentRefreshParams result)
      -> TextDocumentContentRefreshRequest&;
};

class TextDocumentContentRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> TextDocumentContentRefreshResponse&;
  auto id(std::string id) -> TextDocumentContentRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> TextDocumentContentRefreshResponse&;
};

class RegistrationRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> RegistrationRequest&;
  auto id(std::variant<long, std::string> id) -> RegistrationRequest&;

  [[nodiscard]] auto params() const -> RegistrationParams;
  auto params(RegistrationParams result) -> RegistrationRequest&;
};

class RegistrationResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> RegistrationResponse&;
  auto id(std::string id) -> RegistrationResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> RegistrationResponse&;
};

class UnregistrationRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> UnregistrationRequest&;
  auto id(std::variant<long, std::string> id) -> UnregistrationRequest&;

  [[nodiscard]] auto params() const -> UnregistrationParams;
  auto params(UnregistrationParams result) -> UnregistrationRequest&;
};

class UnregistrationResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> UnregistrationResponse&;
  auto id(std::string id) -> UnregistrationResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> UnregistrationResponse&;
};

class InitializeRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InitializeRequest&;
  auto id(std::variant<long, std::string> id) -> InitializeRequest&;

  [[nodiscard]] auto params() const -> InitializeParams;
  auto params(InitializeParams result) -> InitializeRequest&;
};

class InitializeResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> InitializeResponse&;
  auto id(std::string id) -> InitializeResponse&;

  [[nodiscard]] auto result() const -> InitializeResult;

  auto result(InitializeResult result) -> InitializeResponse&;
};

class ShutdownRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ShutdownRequest&;
  auto id(std::variant<long, std::string> id) -> ShutdownRequest&;
};

class ShutdownResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ShutdownResponse&;
  auto id(std::string id) -> ShutdownResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> ShutdownResponse&;
};

class ShowMessageRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ShowMessageRequest&;
  auto id(std::variant<long, std::string> id) -> ShowMessageRequest&;

  [[nodiscard]] auto params() const -> ShowMessageRequestParams;
  auto params(ShowMessageRequestParams result) -> ShowMessageRequest&;
};

class ShowMessageResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ShowMessageResponse&;
  auto id(std::string id) -> ShowMessageResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<MessageActionItem, std::nullptr_t>;

  auto result(std::variant<MessageActionItem, std::nullptr_t> result)
      -> ShowMessageResponse&;
};

class WillSaveTextDocumentWaitUntilRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WillSaveTextDocumentWaitUntilRequest&;
  auto id(std::variant<long, std::string> id)
      -> WillSaveTextDocumentWaitUntilRequest&;

  [[nodiscard]] auto params() const -> WillSaveTextDocumentParams;
  auto params(WillSaveTextDocumentParams result)
      -> WillSaveTextDocumentWaitUntilRequest&;
};

class WillSaveTextDocumentWaitUntilResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WillSaveTextDocumentWaitUntilResponse&;
  auto id(std::string id) -> WillSaveTextDocumentWaitUntilResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TextEdit>, std::nullptr_t>;

  auto result(std::variant<Vector<TextEdit>, std::nullptr_t> result)
      -> WillSaveTextDocumentWaitUntilResponse&;
};

class CompletionRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CompletionRequest&;
  auto id(std::variant<long, std::string> id) -> CompletionRequest&;

  [[nodiscard]] auto params() const -> CompletionParams;
  auto params(CompletionParams result) -> CompletionRequest&;
};

class CompletionResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CompletionResponse&;
  auto id(std::string id) -> CompletionResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<CompletionItem>, CompletionList, std::nullptr_t>;

  auto result(
      std::variant<Vector<CompletionItem>, CompletionList, std::nullptr_t>
          result) -> CompletionResponse&;
};

class CompletionResolveRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CompletionResolveRequest&;
  auto id(std::variant<long, std::string> id) -> CompletionResolveRequest&;

  [[nodiscard]] auto params() const -> CompletionItem;
  auto params(CompletionItem result) -> CompletionResolveRequest&;
};

class CompletionResolveResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CompletionResolveResponse&;
  auto id(std::string id) -> CompletionResolveResponse&;

  [[nodiscard]] auto result() const -> CompletionItem;

  auto result(CompletionItem result) -> CompletionResolveResponse&;
};

class HoverRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> HoverRequest&;
  auto id(std::variant<long, std::string> id) -> HoverRequest&;

  [[nodiscard]] auto params() const -> HoverParams;
  auto params(HoverParams result) -> HoverRequest&;
};

class HoverResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> HoverResponse&;
  auto id(std::string id) -> HoverResponse&;

  [[nodiscard]] auto result() const -> std::variant<Hover, std::nullptr_t>;

  auto result(std::variant<Hover, std::nullptr_t> result) -> HoverResponse&;
};

class SignatureHelpRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SignatureHelpRequest&;
  auto id(std::variant<long, std::string> id) -> SignatureHelpRequest&;

  [[nodiscard]] auto params() const -> SignatureHelpParams;
  auto params(SignatureHelpParams result) -> SignatureHelpRequest&;
};

class SignatureHelpResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> SignatureHelpResponse&;
  auto id(std::string id) -> SignatureHelpResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<SignatureHelp, std::nullptr_t>;

  auto result(std::variant<SignatureHelp, std::nullptr_t> result)
      -> SignatureHelpResponse&;
};

class DefinitionRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DefinitionRequest&;
  auto id(std::variant<long, std::string> id) -> DefinitionRequest&;

  [[nodiscard]] auto params() const -> DefinitionParams;
  auto params(DefinitionParams result) -> DefinitionRequest&;
};

class DefinitionResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DefinitionResponse&;
  auto id(std::string id) -> DefinitionResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t>;

  auto result(
      std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result)
      -> DefinitionResponse&;
};

class ReferencesRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ReferencesRequest&;
  auto id(std::variant<long, std::string> id) -> ReferencesRequest&;

  [[nodiscard]] auto params() const -> ReferenceParams;
  auto params(ReferenceParams result) -> ReferencesRequest&;
};

class ReferencesResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ReferencesResponse&;
  auto id(std::string id) -> ReferencesResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<Location>, std::nullptr_t>;

  auto result(std::variant<Vector<Location>, std::nullptr_t> result)
      -> ReferencesResponse&;
};

class DocumentHighlightRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentHighlightRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentHighlightRequest&;

  [[nodiscard]] auto params() const -> DocumentHighlightParams;
  auto params(DocumentHighlightParams result) -> DocumentHighlightRequest&;
};

class DocumentHighlightResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentHighlightResponse&;
  auto id(std::string id) -> DocumentHighlightResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<DocumentHighlight>, std::nullptr_t>;

  auto result(std::variant<Vector<DocumentHighlight>, std::nullptr_t> result)
      -> DocumentHighlightResponse&;
};

class DocumentSymbolRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentSymbolRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentSymbolRequest&;

  [[nodiscard]] auto params() const -> DocumentSymbolParams;
  auto params(DocumentSymbolParams result) -> DocumentSymbolRequest&;
};

class DocumentSymbolResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentSymbolResponse&;
  auto id(std::string id) -> DocumentSymbolResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<SymbolInformation>, Vector<DocumentSymbol>,
                      std::nullptr_t>;

  auto result(std::variant<Vector<SymbolInformation>, Vector<DocumentSymbol>,
                           std::nullptr_t>
                  result) -> DocumentSymbolResponse&;
};

class CodeActionRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CodeActionRequest&;
  auto id(std::variant<long, std::string> id) -> CodeActionRequest&;

  [[nodiscard]] auto params() const -> CodeActionParams;
  auto params(CodeActionParams result) -> CodeActionRequest&;
};

class CodeActionResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CodeActionResponse&;
  auto id(std::string id) -> CodeActionResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<std::variant<Command, CodeAction>>,
                      std::nullptr_t>;

  auto result(
      std::variant<Vector<std::variant<Command, CodeAction>>, std::nullptr_t>
          result) -> CodeActionResponse&;
};

class CodeActionResolveRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CodeActionResolveRequest&;
  auto id(std::variant<long, std::string> id) -> CodeActionResolveRequest&;

  [[nodiscard]] auto params() const -> CodeAction;
  auto params(CodeAction result) -> CodeActionResolveRequest&;
};

class CodeActionResolveResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CodeActionResolveResponse&;
  auto id(std::string id) -> CodeActionResolveResponse&;

  [[nodiscard]] auto result() const -> CodeAction;

  auto result(CodeAction result) -> CodeActionResolveResponse&;
};

class WorkspaceSymbolRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WorkspaceSymbolRequest&;
  auto id(std::variant<long, std::string> id) -> WorkspaceSymbolRequest&;

  [[nodiscard]] auto params() const -> WorkspaceSymbolParams;
  auto params(WorkspaceSymbolParams result) -> WorkspaceSymbolRequest&;
};

class WorkspaceSymbolResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WorkspaceSymbolResponse&;
  auto id(std::string id) -> WorkspaceSymbolResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<SymbolInformation>, Vector<WorkspaceSymbol>,
                      std::nullptr_t>;

  auto result(std::variant<Vector<SymbolInformation>, Vector<WorkspaceSymbol>,
                           std::nullptr_t>
                  result) -> WorkspaceSymbolResponse&;
};

class WorkspaceSymbolResolveRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WorkspaceSymbolResolveRequest&;
  auto id(std::variant<long, std::string> id) -> WorkspaceSymbolResolveRequest&;

  [[nodiscard]] auto params() const -> WorkspaceSymbol;
  auto params(WorkspaceSymbol result) -> WorkspaceSymbolResolveRequest&;
};

class WorkspaceSymbolResolveResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> WorkspaceSymbolResolveResponse&;
  auto id(std::string id) -> WorkspaceSymbolResolveResponse&;

  [[nodiscard]] auto result() const -> WorkspaceSymbol;

  auto result(WorkspaceSymbol result) -> WorkspaceSymbolResolveResponse&;
};

class CodeLensRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CodeLensRequest&;
  auto id(std::variant<long, std::string> id) -> CodeLensRequest&;

  [[nodiscard]] auto params() const -> CodeLensParams;
  auto params(CodeLensParams result) -> CodeLensRequest&;
};

class CodeLensResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CodeLensResponse&;
  auto id(std::string id) -> CodeLensResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<CodeLens>, std::nullptr_t>;

  auto result(std::variant<Vector<CodeLens>, std::nullptr_t> result)
      -> CodeLensResponse&;
};

class CodeLensResolveRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CodeLensResolveRequest&;
  auto id(std::variant<long, std::string> id) -> CodeLensResolveRequest&;

  [[nodiscard]] auto params() const -> CodeLens;
  auto params(CodeLens result) -> CodeLensResolveRequest&;
};

class CodeLensResolveResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CodeLensResolveResponse&;
  auto id(std::string id) -> CodeLensResolveResponse&;

  [[nodiscard]] auto result() const -> CodeLens;

  auto result(CodeLens result) -> CodeLensResolveResponse&;
};

class CodeLensRefreshRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CodeLensRefreshRequest&;
  auto id(std::variant<long, std::string> id) -> CodeLensRefreshRequest&;
};

class CodeLensRefreshResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> CodeLensRefreshResponse&;
  auto id(std::string id) -> CodeLensRefreshResponse&;

  [[nodiscard]] auto result() const -> std::nullptr_t;

  auto result(std::nullptr_t result) -> CodeLensRefreshResponse&;
};

class DocumentLinkRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentLinkRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentLinkRequest&;

  [[nodiscard]] auto params() const -> DocumentLinkParams;
  auto params(DocumentLinkParams result) -> DocumentLinkRequest&;
};

class DocumentLinkResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentLinkResponse&;
  auto id(std::string id) -> DocumentLinkResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<DocumentLink>, std::nullptr_t>;

  auto result(std::variant<Vector<DocumentLink>, std::nullptr_t> result)
      -> DocumentLinkResponse&;
};

class DocumentLinkResolveRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentLinkResolveRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentLinkResolveRequest&;

  [[nodiscard]] auto params() const -> DocumentLink;
  auto params(DocumentLink result) -> DocumentLinkResolveRequest&;
};

class DocumentLinkResolveResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentLinkResolveResponse&;
  auto id(std::string id) -> DocumentLinkResolveResponse&;

  [[nodiscard]] auto result() const -> DocumentLink;

  auto result(DocumentLink result) -> DocumentLinkResolveResponse&;
};

class DocumentFormattingRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentFormattingRequest&;
  auto id(std::variant<long, std::string> id) -> DocumentFormattingRequest&;

  [[nodiscard]] auto params() const -> DocumentFormattingParams;
  auto params(DocumentFormattingParams result) -> DocumentFormattingRequest&;
};

class DocumentFormattingResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentFormattingResponse&;
  auto id(std::string id) -> DocumentFormattingResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TextEdit>, std::nullptr_t>;

  auto result(std::variant<Vector<TextEdit>, std::nullptr_t> result)
      -> DocumentFormattingResponse&;
};

class DocumentRangeFormattingRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentRangeFormattingRequest&;
  auto id(std::variant<long, std::string> id)
      -> DocumentRangeFormattingRequest&;

  [[nodiscard]] auto params() const -> DocumentRangeFormattingParams;
  auto params(DocumentRangeFormattingParams result)
      -> DocumentRangeFormattingRequest&;
};

class DocumentRangeFormattingResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentRangeFormattingResponse&;
  auto id(std::string id) -> DocumentRangeFormattingResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TextEdit>, std::nullptr_t>;

  auto result(std::variant<Vector<TextEdit>, std::nullptr_t> result)
      -> DocumentRangeFormattingResponse&;
};

class DocumentRangesFormattingRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentRangesFormattingRequest&;
  auto id(std::variant<long, std::string> id)
      -> DocumentRangesFormattingRequest&;

  [[nodiscard]] auto params() const -> DocumentRangesFormattingParams;
  auto params(DocumentRangesFormattingParams result)
      -> DocumentRangesFormattingRequest&;
};

class DocumentRangesFormattingResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentRangesFormattingResponse&;
  auto id(std::string id) -> DocumentRangesFormattingResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TextEdit>, std::nullptr_t>;

  auto result(std::variant<Vector<TextEdit>, std::nullptr_t> result)
      -> DocumentRangesFormattingResponse&;
};

class DocumentOnTypeFormattingRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DocumentOnTypeFormattingRequest&;
  auto id(std::variant<long, std::string> id)
      -> DocumentOnTypeFormattingRequest&;

  [[nodiscard]] auto params() const -> DocumentOnTypeFormattingParams;
  auto params(DocumentOnTypeFormattingParams result)
      -> DocumentOnTypeFormattingRequest&;
};

class DocumentOnTypeFormattingResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> DocumentOnTypeFormattingResponse&;
  auto id(std::string id) -> DocumentOnTypeFormattingResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<Vector<TextEdit>, std::nullptr_t>;

  auto result(std::variant<Vector<TextEdit>, std::nullptr_t> result)
      -> DocumentOnTypeFormattingResponse&;
};

class RenameRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> RenameRequest&;
  auto id(std::variant<long, std::string> id) -> RenameRequest&;

  [[nodiscard]] auto params() const -> RenameParams;
  auto params(RenameParams result) -> RenameRequest&;
};

class RenameResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> RenameResponse&;
  auto id(std::string id) -> RenameResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<WorkspaceEdit, std::nullptr_t>;

  auto result(std::variant<WorkspaceEdit, std::nullptr_t> result)
      -> RenameResponse&;
};

class PrepareRenameRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> PrepareRenameRequest&;
  auto id(std::variant<long, std::string> id) -> PrepareRenameRequest&;

  [[nodiscard]] auto params() const -> PrepareRenameParams;
  auto params(PrepareRenameParams result) -> PrepareRenameRequest&;
};

class PrepareRenameResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> PrepareRenameResponse&;
  auto id(std::string id) -> PrepareRenameResponse&;

  [[nodiscard]] auto result() const
      -> std::variant<PrepareRenameResult, std::nullptr_t>;

  auto result(std::variant<PrepareRenameResult, std::nullptr_t> result)
      -> PrepareRenameResponse&;
};

class ExecuteCommandRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ExecuteCommandRequest&;
  auto id(std::variant<long, std::string> id) -> ExecuteCommandRequest&;

  [[nodiscard]] auto params() const -> ExecuteCommandParams;
  auto params(ExecuteCommandParams result) -> ExecuteCommandRequest&;
};

class ExecuteCommandResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ExecuteCommandResponse&;
  auto id(std::string id) -> ExecuteCommandResponse&;

  [[nodiscard]] auto result() const -> std::variant<LSPAny, std::nullptr_t>;

  auto result(std::variant<LSPAny, std::nullptr_t> result)
      -> ExecuteCommandResponse&;
};

class ApplyWorkspaceEditRequest final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ApplyWorkspaceEditRequest&;
  auto id(std::variant<long, std::string> id) -> ApplyWorkspaceEditRequest&;

  [[nodiscard]] auto params() const -> ApplyWorkspaceEditParams;
  auto params(ApplyWorkspaceEditParams result) -> ApplyWorkspaceEditRequest&;
};

class ApplyWorkspaceEditResponse final : public LSPResponse {
 public:
  using LSPResponse::LSPResponse;

  [[nodiscard]] auto id() const -> std::variant<long, std::string>;
  auto id(long id) -> ApplyWorkspaceEditResponse&;
  auto id(std::string id) -> ApplyWorkspaceEditResponse&;

  [[nodiscard]] auto result() const -> ApplyWorkspaceEditResult;

  auto result(ApplyWorkspaceEditResult result) -> ApplyWorkspaceEditResponse&;
};

class DidChangeWorkspaceFoldersNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidChangeWorkspaceFoldersNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidChangeWorkspaceFoldersNotification&;

  [[nodiscard]] auto params() const -> DidChangeWorkspaceFoldersParams;
  auto params(DidChangeWorkspaceFoldersParams result)
      -> DidChangeWorkspaceFoldersNotification&;
};

class WorkDoneProgressCancelNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WorkDoneProgressCancelNotification&;
  auto id(std::variant<long, std::string> id)
      -> WorkDoneProgressCancelNotification&;

  [[nodiscard]] auto params() const -> WorkDoneProgressCancelParams;
  auto params(WorkDoneProgressCancelParams result)
      -> WorkDoneProgressCancelNotification&;
};

class DidCreateFilesNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidCreateFilesNotification&;
  auto id(std::variant<long, std::string> id) -> DidCreateFilesNotification&;

  [[nodiscard]] auto params() const -> CreateFilesParams;
  auto params(CreateFilesParams result) -> DidCreateFilesNotification&;
};

class DidRenameFilesNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidRenameFilesNotification&;
  auto id(std::variant<long, std::string> id) -> DidRenameFilesNotification&;

  [[nodiscard]] auto params() const -> RenameFilesParams;
  auto params(RenameFilesParams result) -> DidRenameFilesNotification&;
};

class DidDeleteFilesNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidDeleteFilesNotification&;
  auto id(std::variant<long, std::string> id) -> DidDeleteFilesNotification&;

  [[nodiscard]] auto params() const -> DeleteFilesParams;
  auto params(DeleteFilesParams result) -> DidDeleteFilesNotification&;
};

class DidOpenNotebookDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidOpenNotebookDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidOpenNotebookDocumentNotification&;

  [[nodiscard]] auto params() const -> DidOpenNotebookDocumentParams;
  auto params(DidOpenNotebookDocumentParams result)
      -> DidOpenNotebookDocumentNotification&;
};

class DidChangeNotebookDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidChangeNotebookDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidChangeNotebookDocumentNotification&;

  [[nodiscard]] auto params() const -> DidChangeNotebookDocumentParams;
  auto params(DidChangeNotebookDocumentParams result)
      -> DidChangeNotebookDocumentNotification&;
};

class DidSaveNotebookDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidSaveNotebookDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidSaveNotebookDocumentNotification&;

  [[nodiscard]] auto params() const -> DidSaveNotebookDocumentParams;
  auto params(DidSaveNotebookDocumentParams result)
      -> DidSaveNotebookDocumentNotification&;
};

class DidCloseNotebookDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidCloseNotebookDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidCloseNotebookDocumentNotification&;

  [[nodiscard]] auto params() const -> DidCloseNotebookDocumentParams;
  auto params(DidCloseNotebookDocumentParams result)
      -> DidCloseNotebookDocumentNotification&;
};

class InitializedNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> InitializedNotification&;
  auto id(std::variant<long, std::string> id) -> InitializedNotification&;

  [[nodiscard]] auto params() const -> InitializedParams;
  auto params(InitializedParams result) -> InitializedNotification&;
};

class ExitNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ExitNotification&;
  auto id(std::variant<long, std::string> id) -> ExitNotification&;
};

class DidChangeConfigurationNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidChangeConfigurationNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidChangeConfigurationNotification&;

  [[nodiscard]] auto params() const -> DidChangeConfigurationParams;
  auto params(DidChangeConfigurationParams result)
      -> DidChangeConfigurationNotification&;
};

class ShowMessageNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ShowMessageNotification&;
  auto id(std::variant<long, std::string> id) -> ShowMessageNotification&;

  [[nodiscard]] auto params() const -> ShowMessageParams;
  auto params(ShowMessageParams result) -> ShowMessageNotification&;
};

class LogMessageNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> LogMessageNotification&;
  auto id(std::variant<long, std::string> id) -> LogMessageNotification&;

  [[nodiscard]] auto params() const -> LogMessageParams;
  auto params(LogMessageParams result) -> LogMessageNotification&;
};

class TelemetryEventNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> TelemetryEventNotification&;
  auto id(std::variant<long, std::string> id) -> TelemetryEventNotification&;

  [[nodiscard]] auto params() const -> LSPAny;
  auto params(LSPAny result) -> TelemetryEventNotification&;
};

class DidOpenTextDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidOpenTextDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidOpenTextDocumentNotification&;

  [[nodiscard]] auto params() const -> DidOpenTextDocumentParams;
  auto params(DidOpenTextDocumentParams result)
      -> DidOpenTextDocumentNotification&;
};

class DidChangeTextDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidChangeTextDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidChangeTextDocumentNotification&;

  [[nodiscard]] auto params() const -> DidChangeTextDocumentParams;
  auto params(DidChangeTextDocumentParams result)
      -> DidChangeTextDocumentNotification&;
};

class DidCloseTextDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidCloseTextDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidCloseTextDocumentNotification&;

  [[nodiscard]] auto params() const -> DidCloseTextDocumentParams;
  auto params(DidCloseTextDocumentParams result)
      -> DidCloseTextDocumentNotification&;
};

class DidSaveTextDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidSaveTextDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidSaveTextDocumentNotification&;

  [[nodiscard]] auto params() const -> DidSaveTextDocumentParams;
  auto params(DidSaveTextDocumentParams result)
      -> DidSaveTextDocumentNotification&;
};

class WillSaveTextDocumentNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> WillSaveTextDocumentNotification&;
  auto id(std::variant<long, std::string> id)
      -> WillSaveTextDocumentNotification&;

  [[nodiscard]] auto params() const -> WillSaveTextDocumentParams;
  auto params(WillSaveTextDocumentParams result)
      -> WillSaveTextDocumentNotification&;
};

class DidChangeWatchedFilesNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> DidChangeWatchedFilesNotification&;
  auto id(std::variant<long, std::string> id)
      -> DidChangeWatchedFilesNotification&;

  [[nodiscard]] auto params() const -> DidChangeWatchedFilesParams;
  auto params(DidChangeWatchedFilesParams result)
      -> DidChangeWatchedFilesNotification&;
};

class PublishDiagnosticsNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> PublishDiagnosticsNotification&;
  auto id(std::variant<long, std::string> id)
      -> PublishDiagnosticsNotification&;

  [[nodiscard]] auto params() const -> PublishDiagnosticsParams;
  auto params(PublishDiagnosticsParams result)
      -> PublishDiagnosticsNotification&;
};

class SetTraceNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> SetTraceNotification&;
  auto id(std::variant<long, std::string> id) -> SetTraceNotification&;

  [[nodiscard]] auto params() const -> SetTraceParams;
  auto params(SetTraceParams result) -> SetTraceNotification&;
};

class LogTraceNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> LogTraceNotification&;
  auto id(std::variant<long, std::string> id) -> LogTraceNotification&;

  [[nodiscard]] auto params() const -> LogTraceParams;
  auto params(LogTraceParams result) -> LogTraceNotification&;
};

class CancelNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> CancelNotification&;
  auto id(std::variant<long, std::string> id) -> CancelNotification&;

  [[nodiscard]] auto params() const -> CancelParams;
  auto params(CancelParams result) -> CancelNotification&;
};

class ProgressNotification final : public LSPRequest {
 public:
  using LSPRequest::id;
  using LSPRequest::LSPRequest;
  using LSPRequest::method;

  auto method(std::string method) -> ProgressNotification&;
  auto id(std::variant<long, std::string> id) -> ProgressNotification&;

  [[nodiscard]] auto params() const -> ProgressParams;
  auto params(ProgressParams result) -> ProgressNotification&;
};

template <typename Visitor>
auto visit(Visitor&& visitor, const LSPRequest& request) -> void {
#define PROCESS_REQUEST_TYPE(NAME, METHOD) \
  if (request.method() == METHOD)          \
    return visitor(static_cast<const NAME##Request&>(request));

#define PROCESS_NOTIFICATION_TYPE(NAME, METHOD) \
  if (request.method() == METHOD)               \
    return visitor(static_cast<const NAME##Notification&>(request));

  FOR_EACH_LSP_REQUEST_TYPE(PROCESS_REQUEST_TYPE)
  FOR_EACH_LSP_NOTIFICATION_TYPE(PROCESS_NOTIFICATION_TYPE)

#undef PROCESS_REQUEST_TYPE
#undef PROCESS_NOTIFICATION_TYPE
}

}  // namespace cxx::lsp
