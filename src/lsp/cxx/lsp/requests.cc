// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/lsp/requests.h>
#include <cxx/lsp/types.h>

namespace cxx::lsp {

auto LSPRequest::id() const -> std::optional<std::variant<long, std::string>> {
  if (!repr_->contains("id")) return std::nullopt;
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto LSPRequest::id(std::optional<std::variant<long, std::string>> id)
    -> LSPRequest& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  if (std::holds_alternative<long>(*id)) {
    (*repr_)["id"] = std::get<long>(*id);
  } else {
    (*repr_)["id"] = std::get<std::string>(*id);
  }
  return *this;
}

auto LSPRequest::method() const -> std::string { return repr_->at("method"); }

auto LSPResponse::id() const -> std::optional<std::variant<long, std::string>> {
  if (!repr_->contains("id")) return std::nullopt;
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto LSPResponse::id(std::optional<std::variant<long, std::string>> id)
    -> LSPResponse& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  if (std::holds_alternative<long>(*id)) {
    (*repr_)["id"] = std::get<long>(*id);
  } else {
    (*repr_)["id"] = std::get<std::string>(*id);
  }
  return *this;
}

auto ImplementationRequest::method(std::string method)
    -> ImplementationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ImplementationRequest::id(std::variant<long, std::string> id)
    -> ImplementationRequest& {
  return static_cast<ImplementationRequest&>(LSPRequest::id(std::move(id)));
}

auto ImplementationRequest::params() const -> ImplementationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ImplementationParams(repr_->at("params"));
}

auto ImplementationRequest::params(ImplementationParams params)
    -> ImplementationRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ImplementationResponse::id(std::variant<long, std::string> id)
    -> ImplementationResponse& {
  return static_cast<ImplementationResponse&>(LSPResponse::id(std::move(id)));
}

auto ImplementationResponse::result() const
    -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionRequest::method(std::string method)
    -> TypeDefinitionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeDefinitionRequest::id(std::variant<long, std::string> id)
    -> TypeDefinitionRequest& {
  return static_cast<TypeDefinitionRequest&>(LSPRequest::id(std::move(id)));
}

auto TypeDefinitionRequest::params() const -> TypeDefinitionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return TypeDefinitionParams(repr_->at("params"));
}

auto TypeDefinitionRequest::params(TypeDefinitionParams params)
    -> TypeDefinitionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TypeDefinitionResponse::id(std::variant<long, std::string> id)
    -> TypeDefinitionResponse& {
  return static_cast<TypeDefinitionResponse&>(LSPResponse::id(std::move(id)));
}

auto TypeDefinitionResponse::result() const
    -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFoldersRequest::method(std::string method)
    -> WorkspaceFoldersRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceFoldersRequest::id(std::variant<long, std::string> id)
    -> WorkspaceFoldersRequest& {
  return static_cast<WorkspaceFoldersRequest&>(LSPRequest::id(std::move(id)));
}

auto WorkspaceFoldersResponse::id(std::variant<long, std::string> id)
    -> WorkspaceFoldersResponse& {
  return static_cast<WorkspaceFoldersResponse&>(LSPResponse::id(std::move(id)));
}

auto WorkspaceFoldersResponse::result() const
    -> std::variant<Vector<WorkspaceFolder>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<WorkspaceFolder>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ConfigurationRequest::method(std::string method) -> ConfigurationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ConfigurationRequest::id(std::variant<long, std::string> id)
    -> ConfigurationRequest& {
  return static_cast<ConfigurationRequest&>(LSPRequest::id(std::move(id)));
}

auto ConfigurationRequest::params() const -> ConfigurationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ConfigurationParams(repr_->at("params"));
}

auto ConfigurationRequest::params(ConfigurationParams params)
    -> ConfigurationRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ConfigurationResponse::id(std::variant<long, std::string> id)
    -> ConfigurationResponse& {
  return static_cast<ConfigurationResponse&>(LSPResponse::id(std::move(id)));
}

auto ConfigurationResponse::result() const -> Vector<LSPAny> {
  auto& value = (*repr_)["result"];

  if (value.is_null()) value = json::array();
  return Vector<LSPAny>(value);
}

auto DocumentColorRequest::method(std::string method) -> DocumentColorRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentColorRequest::id(std::variant<long, std::string> id)
    -> DocumentColorRequest& {
  return static_cast<DocumentColorRequest&>(LSPRequest::id(std::move(id)));
}

auto DocumentColorRequest::params() const -> DocumentColorParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentColorParams(repr_->at("params"));
}

auto DocumentColorRequest::params(DocumentColorParams params)
    -> DocumentColorRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentColorResponse::id(std::variant<long, std::string> id)
    -> DocumentColorResponse& {
  return static_cast<DocumentColorResponse&>(LSPResponse::id(std::move(id)));
}

auto DocumentColorResponse::result() const -> Vector<ColorInformation> {
  auto& value = (*repr_)["result"];

  if (value.is_null()) value = json::array();
  return Vector<ColorInformation>(value);
}

auto ColorPresentationRequest::method(std::string method)
    -> ColorPresentationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ColorPresentationRequest::id(std::variant<long, std::string> id)
    -> ColorPresentationRequest& {
  return static_cast<ColorPresentationRequest&>(LSPRequest::id(std::move(id)));
}

auto ColorPresentationRequest::params() const -> ColorPresentationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ColorPresentationParams(repr_->at("params"));
}

auto ColorPresentationRequest::params(ColorPresentationParams params)
    -> ColorPresentationRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ColorPresentationResponse::id(std::variant<long, std::string> id)
    -> ColorPresentationResponse& {
  return static_cast<ColorPresentationResponse&>(
      LSPResponse::id(std::move(id)));
}

auto ColorPresentationResponse::result() const -> Vector<ColorPresentation> {
  auto& value = (*repr_)["result"];

  if (value.is_null()) value = json::array();
  return Vector<ColorPresentation>(value);
}

auto FoldingRangeRequest::method(std::string method) -> FoldingRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto FoldingRangeRequest::id(std::variant<long, std::string> id)
    -> FoldingRangeRequest& {
  return static_cast<FoldingRangeRequest&>(LSPRequest::id(std::move(id)));
}

auto FoldingRangeRequest::params() const -> FoldingRangeParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return FoldingRangeParams(repr_->at("params"));
}

auto FoldingRangeRequest::params(FoldingRangeParams params)
    -> FoldingRangeRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto FoldingRangeResponse::id(std::variant<long, std::string> id)
    -> FoldingRangeResponse& {
  return static_cast<FoldingRangeResponse&>(LSPResponse::id(std::move(id)));
}

auto FoldingRangeResponse::result() const
    -> std::variant<Vector<FoldingRange>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<FoldingRange>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeRefreshRequest::method(std::string method)
    -> FoldingRangeRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto FoldingRangeRefreshRequest::id(std::variant<long, std::string> id)
    -> FoldingRangeRefreshRequest& {
  return static_cast<FoldingRangeRefreshRequest&>(
      LSPRequest::id(std::move(id)));
}

auto FoldingRangeRefreshResponse::id(std::variant<long, std::string> id)
    -> FoldingRangeRefreshResponse& {
  return static_cast<FoldingRangeRefreshResponse&>(
      LSPResponse::id(std::move(id)));
}

auto FoldingRangeRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto DeclarationRequest::method(std::string method) -> DeclarationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DeclarationRequest::id(std::variant<long, std::string> id)
    -> DeclarationRequest& {
  return static_cast<DeclarationRequest&>(LSPRequest::id(std::move(id)));
}

auto DeclarationRequest::params() const -> DeclarationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DeclarationParams(repr_->at("params"));
}

auto DeclarationRequest::params(DeclarationParams params)
    -> DeclarationRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DeclarationResponse::id(std::variant<long, std::string> id)
    -> DeclarationResponse& {
  return static_cast<DeclarationResponse&>(LSPResponse::id(std::move(id)));
}

auto DeclarationResponse::result() const
    -> std::variant<Declaration, Vector<DeclarationLink>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Declaration, Vector<DeclarationLink>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeRequest::method(std::string method)
    -> SelectionRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SelectionRangeRequest::id(std::variant<long, std::string> id)
    -> SelectionRangeRequest& {
  return static_cast<SelectionRangeRequest&>(LSPRequest::id(std::move(id)));
}

auto SelectionRangeRequest::params() const -> SelectionRangeParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return SelectionRangeParams(repr_->at("params"));
}

auto SelectionRangeRequest::params(SelectionRangeParams params)
    -> SelectionRangeRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto SelectionRangeResponse::id(std::variant<long, std::string> id)
    -> SelectionRangeResponse& {
  return static_cast<SelectionRangeResponse&>(LSPResponse::id(std::move(id)));
}

auto SelectionRangeResponse::result() const
    -> std::variant<Vector<SelectionRange>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<SelectionRange>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressCreateRequest::method(std::string method)
    -> WorkDoneProgressCreateRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkDoneProgressCreateRequest::id(std::variant<long, std::string> id)
    -> WorkDoneProgressCreateRequest& {
  return static_cast<WorkDoneProgressCreateRequest&>(
      LSPRequest::id(std::move(id)));
}

auto WorkDoneProgressCreateRequest::params() const
    -> WorkDoneProgressCreateParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WorkDoneProgressCreateParams(repr_->at("params"));
}

auto WorkDoneProgressCreateRequest::params(WorkDoneProgressCreateParams params)
    -> WorkDoneProgressCreateRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WorkDoneProgressCreateResponse::id(std::variant<long, std::string> id)
    -> WorkDoneProgressCreateResponse& {
  return static_cast<WorkDoneProgressCreateResponse&>(
      LSPResponse::id(std::move(id)));
}

auto WorkDoneProgressCreateResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto CallHierarchyPrepareRequest::method(std::string method)
    -> CallHierarchyPrepareRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyPrepareRequest::id(std::variant<long, std::string> id)
    -> CallHierarchyPrepareRequest& {
  return static_cast<CallHierarchyPrepareRequest&>(
      LSPRequest::id(std::move(id)));
}

auto CallHierarchyPrepareRequest::params() const -> CallHierarchyPrepareParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CallHierarchyPrepareParams(repr_->at("params"));
}

auto CallHierarchyPrepareRequest::params(CallHierarchyPrepareParams params)
    -> CallHierarchyPrepareRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CallHierarchyPrepareResponse::id(std::variant<long, std::string> id)
    -> CallHierarchyPrepareResponse& {
  return static_cast<CallHierarchyPrepareResponse&>(
      LSPResponse::id(std::move(id)));
}

auto CallHierarchyPrepareResponse::result() const
    -> std::variant<Vector<CallHierarchyItem>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<CallHierarchyItem>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyIncomingCallsRequest::method(std::string method)
    -> CallHierarchyIncomingCallsRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyIncomingCallsRequest::id(std::variant<long, std::string> id)
    -> CallHierarchyIncomingCallsRequest& {
  return static_cast<CallHierarchyIncomingCallsRequest&>(
      LSPRequest::id(std::move(id)));
}

auto CallHierarchyIncomingCallsRequest::params() const
    -> CallHierarchyIncomingCallsParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CallHierarchyIncomingCallsParams(repr_->at("params"));
}

auto CallHierarchyIncomingCallsRequest::params(
    CallHierarchyIncomingCallsParams params)
    -> CallHierarchyIncomingCallsRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CallHierarchyIncomingCallsResponse::id(std::variant<long, std::string> id)
    -> CallHierarchyIncomingCallsResponse& {
  return static_cast<CallHierarchyIncomingCallsResponse&>(
      LSPResponse::id(std::move(id)));
}

auto CallHierarchyIncomingCallsResponse::result() const
    -> std::variant<Vector<CallHierarchyIncomingCall>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<CallHierarchyIncomingCall>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyOutgoingCallsRequest::method(std::string method)
    -> CallHierarchyOutgoingCallsRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyOutgoingCallsRequest::id(std::variant<long, std::string> id)
    -> CallHierarchyOutgoingCallsRequest& {
  return static_cast<CallHierarchyOutgoingCallsRequest&>(
      LSPRequest::id(std::move(id)));
}

auto CallHierarchyOutgoingCallsRequest::params() const
    -> CallHierarchyOutgoingCallsParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CallHierarchyOutgoingCallsParams(repr_->at("params"));
}

auto CallHierarchyOutgoingCallsRequest::params(
    CallHierarchyOutgoingCallsParams params)
    -> CallHierarchyOutgoingCallsRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CallHierarchyOutgoingCallsResponse::id(std::variant<long, std::string> id)
    -> CallHierarchyOutgoingCallsResponse& {
  return static_cast<CallHierarchyOutgoingCallsResponse&>(
      LSPResponse::id(std::move(id)));
}

auto CallHierarchyOutgoingCallsResponse::result() const
    -> std::variant<Vector<CallHierarchyOutgoingCall>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<CallHierarchyOutgoingCall>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRequest::method(std::string method)
    -> SemanticTokensRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensRequest& {
  return static_cast<SemanticTokensRequest&>(LSPRequest::id(std::move(id)));
}

auto SemanticTokensRequest::params() const -> SemanticTokensParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return SemanticTokensParams(repr_->at("params"));
}

auto SemanticTokensRequest::params(SemanticTokensParams params)
    -> SemanticTokensRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto SemanticTokensResponse::id(std::variant<long, std::string> id)
    -> SemanticTokensResponse& {
  return static_cast<SemanticTokensResponse&>(LSPResponse::id(std::move(id)));
}

auto SemanticTokensResponse::result() const
    -> std::variant<SemanticTokens, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<SemanticTokens, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensDeltaRequest::method(std::string method)
    -> SemanticTokensDeltaRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensDeltaRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensDeltaRequest& {
  return static_cast<SemanticTokensDeltaRequest&>(
      LSPRequest::id(std::move(id)));
}

auto SemanticTokensDeltaRequest::params() const -> SemanticTokensDeltaParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return SemanticTokensDeltaParams(repr_->at("params"));
}

auto SemanticTokensDeltaRequest::params(SemanticTokensDeltaParams params)
    -> SemanticTokensDeltaRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto SemanticTokensDeltaResponse::id(std::variant<long, std::string> id)
    -> SemanticTokensDeltaResponse& {
  return static_cast<SemanticTokensDeltaResponse&>(
      LSPResponse::id(std::move(id)));
}

auto SemanticTokensDeltaResponse::result() const
    -> std::variant<SemanticTokens, SemanticTokensDelta, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<SemanticTokens, SemanticTokensDelta, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRangeRequest::method(std::string method)
    -> SemanticTokensRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRangeRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensRangeRequest& {
  return static_cast<SemanticTokensRangeRequest&>(
      LSPRequest::id(std::move(id)));
}

auto SemanticTokensRangeRequest::params() const -> SemanticTokensRangeParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return SemanticTokensRangeParams(repr_->at("params"));
}

auto SemanticTokensRangeRequest::params(SemanticTokensRangeParams params)
    -> SemanticTokensRangeRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto SemanticTokensRangeResponse::id(std::variant<long, std::string> id)
    -> SemanticTokensRangeResponse& {
  return static_cast<SemanticTokensRangeResponse&>(
      LSPResponse::id(std::move(id)));
}

auto SemanticTokensRangeResponse::result() const
    -> std::variant<SemanticTokens, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<SemanticTokens, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRefreshRequest::method(std::string method)
    -> SemanticTokensRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRefreshRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensRefreshRequest& {
  return static_cast<SemanticTokensRefreshRequest&>(
      LSPRequest::id(std::move(id)));
}

auto SemanticTokensRefreshResponse::id(std::variant<long, std::string> id)
    -> SemanticTokensRefreshResponse& {
  return static_cast<SemanticTokensRefreshResponse&>(
      LSPResponse::id(std::move(id)));
}

auto SemanticTokensRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto ShowDocumentRequest::method(std::string method) -> ShowDocumentRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowDocumentRequest::id(std::variant<long, std::string> id)
    -> ShowDocumentRequest& {
  return static_cast<ShowDocumentRequest&>(LSPRequest::id(std::move(id)));
}

auto ShowDocumentRequest::params() const -> ShowDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ShowDocumentParams(repr_->at("params"));
}

auto ShowDocumentRequest::params(ShowDocumentParams params)
    -> ShowDocumentRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ShowDocumentResponse::id(std::variant<long, std::string> id)
    -> ShowDocumentResponse& {
  return static_cast<ShowDocumentResponse&>(LSPResponse::id(std::move(id)));
}

auto ShowDocumentResponse::result() const -> ShowDocumentResult {
  auto& value = (*repr_)["result"];

  return ShowDocumentResult(value);
}

auto LinkedEditingRangeRequest::method(std::string method)
    -> LinkedEditingRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LinkedEditingRangeRequest::id(std::variant<long, std::string> id)
    -> LinkedEditingRangeRequest& {
  return static_cast<LinkedEditingRangeRequest&>(LSPRequest::id(std::move(id)));
}

auto LinkedEditingRangeRequest::params() const -> LinkedEditingRangeParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return LinkedEditingRangeParams(repr_->at("params"));
}

auto LinkedEditingRangeRequest::params(LinkedEditingRangeParams params)
    -> LinkedEditingRangeRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto LinkedEditingRangeResponse::id(std::variant<long, std::string> id)
    -> LinkedEditingRangeResponse& {
  return static_cast<LinkedEditingRangeResponse&>(
      LSPResponse::id(std::move(id)));
}

auto LinkedEditingRangeResponse::result() const
    -> std::variant<LinkedEditingRanges, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<LinkedEditingRanges, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WillCreateFilesRequest::method(std::string method)
    -> WillCreateFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillCreateFilesRequest::id(std::variant<long, std::string> id)
    -> WillCreateFilesRequest& {
  return static_cast<WillCreateFilesRequest&>(LSPRequest::id(std::move(id)));
}

auto WillCreateFilesRequest::params() const -> CreateFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CreateFilesParams(repr_->at("params"));
}

auto WillCreateFilesRequest::params(CreateFilesParams params)
    -> WillCreateFilesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WillCreateFilesResponse::id(std::variant<long, std::string> id)
    -> WillCreateFilesResponse& {
  return static_cast<WillCreateFilesResponse&>(LSPResponse::id(std::move(id)));
}

auto WillCreateFilesResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<WorkspaceEdit, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WillRenameFilesRequest::method(std::string method)
    -> WillRenameFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillRenameFilesRequest::id(std::variant<long, std::string> id)
    -> WillRenameFilesRequest& {
  return static_cast<WillRenameFilesRequest&>(LSPRequest::id(std::move(id)));
}

auto WillRenameFilesRequest::params() const -> RenameFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return RenameFilesParams(repr_->at("params"));
}

auto WillRenameFilesRequest::params(RenameFilesParams params)
    -> WillRenameFilesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WillRenameFilesResponse::id(std::variant<long, std::string> id)
    -> WillRenameFilesResponse& {
  return static_cast<WillRenameFilesResponse&>(LSPResponse::id(std::move(id)));
}

auto WillRenameFilesResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<WorkspaceEdit, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WillDeleteFilesRequest::method(std::string method)
    -> WillDeleteFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillDeleteFilesRequest::id(std::variant<long, std::string> id)
    -> WillDeleteFilesRequest& {
  return static_cast<WillDeleteFilesRequest&>(LSPRequest::id(std::move(id)));
}

auto WillDeleteFilesRequest::params() const -> DeleteFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DeleteFilesParams(repr_->at("params"));
}

auto WillDeleteFilesRequest::params(DeleteFilesParams params)
    -> WillDeleteFilesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WillDeleteFilesResponse::id(std::variant<long, std::string> id)
    -> WillDeleteFilesResponse& {
  return static_cast<WillDeleteFilesResponse&>(LSPResponse::id(std::move(id)));
}

auto WillDeleteFilesResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<WorkspaceEdit, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerRequest::method(std::string method) -> MonikerRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto MonikerRequest::id(std::variant<long, std::string> id) -> MonikerRequest& {
  return static_cast<MonikerRequest&>(LSPRequest::id(std::move(id)));
}

auto MonikerRequest::params() const -> MonikerParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return MonikerParams(repr_->at("params"));
}

auto MonikerRequest::params(MonikerParams params) -> MonikerRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto MonikerResponse::id(std::variant<long, std::string> id)
    -> MonikerResponse& {
  return static_cast<MonikerResponse&>(LSPResponse::id(std::move(id)));
}

auto MonikerResponse::result() const
    -> std::variant<Vector<Moniker>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<Moniker>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchyPrepareRequest::method(std::string method)
    -> TypeHierarchyPrepareRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchyPrepareRequest::id(std::variant<long, std::string> id)
    -> TypeHierarchyPrepareRequest& {
  return static_cast<TypeHierarchyPrepareRequest&>(
      LSPRequest::id(std::move(id)));
}

auto TypeHierarchyPrepareRequest::params() const -> TypeHierarchyPrepareParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return TypeHierarchyPrepareParams(repr_->at("params"));
}

auto TypeHierarchyPrepareRequest::params(TypeHierarchyPrepareParams params)
    -> TypeHierarchyPrepareRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TypeHierarchyPrepareResponse::id(std::variant<long, std::string> id)
    -> TypeHierarchyPrepareResponse& {
  return static_cast<TypeHierarchyPrepareResponse&>(
      LSPResponse::id(std::move(id)));
}

auto TypeHierarchyPrepareResponse::result() const
    -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySupertypesRequest::method(std::string method)
    -> TypeHierarchySupertypesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchySupertypesRequest::id(std::variant<long, std::string> id)
    -> TypeHierarchySupertypesRequest& {
  return static_cast<TypeHierarchySupertypesRequest&>(
      LSPRequest::id(std::move(id)));
}

auto TypeHierarchySupertypesRequest::params() const
    -> TypeHierarchySupertypesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return TypeHierarchySupertypesParams(repr_->at("params"));
}

auto TypeHierarchySupertypesRequest::params(
    TypeHierarchySupertypesParams params) -> TypeHierarchySupertypesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TypeHierarchySupertypesResponse::id(std::variant<long, std::string> id)
    -> TypeHierarchySupertypesResponse& {
  return static_cast<TypeHierarchySupertypesResponse&>(
      LSPResponse::id(std::move(id)));
}

auto TypeHierarchySupertypesResponse::result() const
    -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySubtypesRequest::method(std::string method)
    -> TypeHierarchySubtypesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchySubtypesRequest::id(std::variant<long, std::string> id)
    -> TypeHierarchySubtypesRequest& {
  return static_cast<TypeHierarchySubtypesRequest&>(
      LSPRequest::id(std::move(id)));
}

auto TypeHierarchySubtypesRequest::params() const
    -> TypeHierarchySubtypesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return TypeHierarchySubtypesParams(repr_->at("params"));
}

auto TypeHierarchySubtypesRequest::params(TypeHierarchySubtypesParams params)
    -> TypeHierarchySubtypesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TypeHierarchySubtypesResponse::id(std::variant<long, std::string> id)
    -> TypeHierarchySubtypesResponse& {
  return static_cast<TypeHierarchySubtypesResponse&>(
      LSPResponse::id(std::move(id)));
}

auto TypeHierarchySubtypesResponse::result() const
    -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineValueRequest::method(std::string method) -> InlineValueRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineValueRequest::id(std::variant<long, std::string> id)
    -> InlineValueRequest& {
  return static_cast<InlineValueRequest&>(LSPRequest::id(std::move(id)));
}

auto InlineValueRequest::params() const -> InlineValueParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InlineValueParams(repr_->at("params"));
}

auto InlineValueRequest::params(InlineValueParams params)
    -> InlineValueRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InlineValueResponse::id(std::variant<long, std::string> id)
    -> InlineValueResponse& {
  return static_cast<InlineValueResponse&>(LSPResponse::id(std::move(id)));
}

auto InlineValueResponse::result() const
    -> std::variant<Vector<InlineValue>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<InlineValue>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineValueRefreshRequest::method(std::string method)
    -> InlineValueRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineValueRefreshRequest::id(std::variant<long, std::string> id)
    -> InlineValueRefreshRequest& {
  return static_cast<InlineValueRefreshRequest&>(LSPRequest::id(std::move(id)));
}

auto InlineValueRefreshResponse::id(std::variant<long, std::string> id)
    -> InlineValueRefreshResponse& {
  return static_cast<InlineValueRefreshResponse&>(
      LSPResponse::id(std::move(id)));
}

auto InlineValueRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto InlayHintRequest::method(std::string method) -> InlayHintRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintRequest::id(std::variant<long, std::string> id)
    -> InlayHintRequest& {
  return static_cast<InlayHintRequest&>(LSPRequest::id(std::move(id)));
}

auto InlayHintRequest::params() const -> InlayHintParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InlayHintParams(repr_->at("params"));
}

auto InlayHintRequest::params(InlayHintParams params) -> InlayHintRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InlayHintResponse::id(std::variant<long, std::string> id)
    -> InlayHintResponse& {
  return static_cast<InlayHintResponse&>(LSPResponse::id(std::move(id)));
}

auto InlayHintResponse::result() const
    -> std::variant<Vector<InlayHint>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<InlayHint>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintResolveRequest::method(std::string method)
    -> InlayHintResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintResolveRequest::id(std::variant<long, std::string> id)
    -> InlayHintResolveRequest& {
  return static_cast<InlayHintResolveRequest&>(LSPRequest::id(std::move(id)));
}

auto InlayHintResolveRequest::params() const -> InlayHint {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InlayHint(repr_->at("params"));
}

auto InlayHintResolveRequest::params(InlayHint params)
    -> InlayHintResolveRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InlayHintResolveResponse::id(std::variant<long, std::string> id)
    -> InlayHintResolveResponse& {
  return static_cast<InlayHintResolveResponse&>(LSPResponse::id(std::move(id)));
}

auto InlayHintResolveResponse::result() const -> InlayHint {
  auto& value = (*repr_)["result"];

  return InlayHint(value);
}

auto InlayHintRefreshRequest::method(std::string method)
    -> InlayHintRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintRefreshRequest::id(std::variant<long, std::string> id)
    -> InlayHintRefreshRequest& {
  return static_cast<InlayHintRefreshRequest&>(LSPRequest::id(std::move(id)));
}

auto InlayHintRefreshResponse::id(std::variant<long, std::string> id)
    -> InlayHintRefreshResponse& {
  return static_cast<InlayHintRefreshResponse&>(LSPResponse::id(std::move(id)));
}

auto InlayHintRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto DocumentDiagnosticRequest::method(std::string method)
    -> DocumentDiagnosticRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentDiagnosticRequest::id(std::variant<long, std::string> id)
    -> DocumentDiagnosticRequest& {
  return static_cast<DocumentDiagnosticRequest&>(LSPRequest::id(std::move(id)));
}

auto DocumentDiagnosticRequest::params() const -> DocumentDiagnosticParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentDiagnosticParams(repr_->at("params"));
}

auto DocumentDiagnosticRequest::params(DocumentDiagnosticParams params)
    -> DocumentDiagnosticRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentDiagnosticResponse::id(std::variant<long, std::string> id)
    -> DocumentDiagnosticResponse& {
  return static_cast<DocumentDiagnosticResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentDiagnosticResponse::result() const -> DocumentDiagnosticReport {
  auto& value = (*repr_)["result"];

  DocumentDiagnosticReport result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceDiagnosticRequest::method(std::string method)
    -> WorkspaceDiagnosticRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceDiagnosticRequest::id(std::variant<long, std::string> id)
    -> WorkspaceDiagnosticRequest& {
  return static_cast<WorkspaceDiagnosticRequest&>(
      LSPRequest::id(std::move(id)));
}

auto WorkspaceDiagnosticRequest::params() const -> WorkspaceDiagnosticParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WorkspaceDiagnosticParams(repr_->at("params"));
}

auto WorkspaceDiagnosticRequest::params(WorkspaceDiagnosticParams params)
    -> WorkspaceDiagnosticRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WorkspaceDiagnosticResponse::id(std::variant<long, std::string> id)
    -> WorkspaceDiagnosticResponse& {
  return static_cast<WorkspaceDiagnosticResponse&>(
      LSPResponse::id(std::move(id)));
}

auto WorkspaceDiagnosticResponse::result() const -> WorkspaceDiagnosticReport {
  auto& value = (*repr_)["result"];

  return WorkspaceDiagnosticReport(value);
}

auto DiagnosticRefreshRequest::method(std::string method)
    -> DiagnosticRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DiagnosticRefreshRequest::id(std::variant<long, std::string> id)
    -> DiagnosticRefreshRequest& {
  return static_cast<DiagnosticRefreshRequest&>(LSPRequest::id(std::move(id)));
}

auto DiagnosticRefreshResponse::id(std::variant<long, std::string> id)
    -> DiagnosticRefreshResponse& {
  return static_cast<DiagnosticRefreshResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DiagnosticRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto InlineCompletionRequest::method(std::string method)
    -> InlineCompletionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineCompletionRequest::id(std::variant<long, std::string> id)
    -> InlineCompletionRequest& {
  return static_cast<InlineCompletionRequest&>(LSPRequest::id(std::move(id)));
}

auto InlineCompletionRequest::params() const -> InlineCompletionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InlineCompletionParams(repr_->at("params"));
}

auto InlineCompletionRequest::params(InlineCompletionParams params)
    -> InlineCompletionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InlineCompletionResponse::id(std::variant<long, std::string> id)
    -> InlineCompletionResponse& {
  return static_cast<InlineCompletionResponse&>(LSPResponse::id(std::move(id)));
}

auto InlineCompletionResponse::result() const
    -> std::variant<InlineCompletionList, Vector<InlineCompletionItem>,
                    std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<InlineCompletionList, Vector<InlineCompletionItem>,
               std::nullptr_t>
      result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentContentRequest::method(std::string method)
    -> TextDocumentContentRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TextDocumentContentRequest::id(std::variant<long, std::string> id)
    -> TextDocumentContentRequest& {
  return static_cast<TextDocumentContentRequest&>(
      LSPRequest::id(std::move(id)));
}

auto TextDocumentContentRequest::params() const -> TextDocumentContentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return TextDocumentContentParams(repr_->at("params"));
}

auto TextDocumentContentRequest::params(TextDocumentContentParams params)
    -> TextDocumentContentRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TextDocumentContentResponse::id(std::variant<long, std::string> id)
    -> TextDocumentContentResponse& {
  return static_cast<TextDocumentContentResponse&>(
      LSPResponse::id(std::move(id)));
}

auto TextDocumentContentResponse::result() const -> TextDocumentContentResult {
  auto& value = (*repr_)["result"];

  return TextDocumentContentResult(value);
}

auto TextDocumentContentRefreshRequest::method(std::string method)
    -> TextDocumentContentRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TextDocumentContentRefreshRequest::id(std::variant<long, std::string> id)
    -> TextDocumentContentRefreshRequest& {
  return static_cast<TextDocumentContentRefreshRequest&>(
      LSPRequest::id(std::move(id)));
}

auto TextDocumentContentRefreshRequest::params() const
    -> TextDocumentContentRefreshParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return TextDocumentContentRefreshParams(repr_->at("params"));
}

auto TextDocumentContentRefreshRequest::params(
    TextDocumentContentRefreshParams params)
    -> TextDocumentContentRefreshRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TextDocumentContentRefreshResponse::id(std::variant<long, std::string> id)
    -> TextDocumentContentRefreshResponse& {
  return static_cast<TextDocumentContentRefreshResponse&>(
      LSPResponse::id(std::move(id)));
}

auto TextDocumentContentRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto RegistrationRequest::method(std::string method) -> RegistrationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto RegistrationRequest::id(std::variant<long, std::string> id)
    -> RegistrationRequest& {
  return static_cast<RegistrationRequest&>(LSPRequest::id(std::move(id)));
}

auto RegistrationRequest::params() const -> RegistrationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return RegistrationParams(repr_->at("params"));
}

auto RegistrationRequest::params(RegistrationParams params)
    -> RegistrationRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto RegistrationResponse::id(std::variant<long, std::string> id)
    -> RegistrationResponse& {
  return static_cast<RegistrationResponse&>(LSPResponse::id(std::move(id)));
}

auto RegistrationResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto UnregistrationRequest::method(std::string method)
    -> UnregistrationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto UnregistrationRequest::id(std::variant<long, std::string> id)
    -> UnregistrationRequest& {
  return static_cast<UnregistrationRequest&>(LSPRequest::id(std::move(id)));
}

auto UnregistrationRequest::params() const -> UnregistrationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return UnregistrationParams(repr_->at("params"));
}

auto UnregistrationRequest::params(UnregistrationParams params)
    -> UnregistrationRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto UnregistrationResponse::id(std::variant<long, std::string> id)
    -> UnregistrationResponse& {
  return static_cast<UnregistrationResponse&>(LSPResponse::id(std::move(id)));
}

auto UnregistrationResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto InitializeRequest::method(std::string method) -> InitializeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InitializeRequest::id(std::variant<long, std::string> id)
    -> InitializeRequest& {
  return static_cast<InitializeRequest&>(LSPRequest::id(std::move(id)));
}

auto InitializeRequest::params() const -> InitializeParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InitializeParams(repr_->at("params"));
}

auto InitializeRequest::params(InitializeParams params) -> InitializeRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InitializeResponse::id(std::variant<long, std::string> id)
    -> InitializeResponse& {
  return static_cast<InitializeResponse&>(LSPResponse::id(std::move(id)));
}

auto InitializeResponse::result() const -> InitializeResult {
  auto& value = (*repr_)["result"];

  return InitializeResult(value);
}

auto ShutdownRequest::method(std::string method) -> ShutdownRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShutdownRequest::id(std::variant<long, std::string> id)
    -> ShutdownRequest& {
  return static_cast<ShutdownRequest&>(LSPRequest::id(std::move(id)));
}

auto ShutdownResponse::id(std::variant<long, std::string> id)
    -> ShutdownResponse& {
  return static_cast<ShutdownResponse&>(LSPResponse::id(std::move(id)));
}

auto ShutdownResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto ShowMessageRequest::method(std::string method) -> ShowMessageRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowMessageRequest::id(std::variant<long, std::string> id)
    -> ShowMessageRequest& {
  return static_cast<ShowMessageRequest&>(LSPRequest::id(std::move(id)));
}

auto ShowMessageRequest::params() const -> ShowMessageRequestParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ShowMessageRequestParams(repr_->at("params"));
}

auto ShowMessageRequest::params(ShowMessageRequestParams params)
    -> ShowMessageRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ShowMessageResponse::id(std::variant<long, std::string> id)
    -> ShowMessageResponse& {
  return static_cast<ShowMessageResponse&>(LSPResponse::id(std::move(id)));
}

auto ShowMessageResponse::result() const
    -> std::variant<MessageActionItem, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<MessageActionItem, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WillSaveTextDocumentWaitUntilRequest::method(std::string method)
    -> WillSaveTextDocumentWaitUntilRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillSaveTextDocumentWaitUntilRequest::id(
    std::variant<long, std::string> id)
    -> WillSaveTextDocumentWaitUntilRequest& {
  return static_cast<WillSaveTextDocumentWaitUntilRequest&>(
      LSPRequest::id(std::move(id)));
}

auto WillSaveTextDocumentWaitUntilRequest::params() const
    -> WillSaveTextDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WillSaveTextDocumentParams(repr_->at("params"));
}

auto WillSaveTextDocumentWaitUntilRequest::params(
    WillSaveTextDocumentParams params)
    -> WillSaveTextDocumentWaitUntilRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WillSaveTextDocumentWaitUntilResponse::id(
    std::variant<long, std::string> id)
    -> WillSaveTextDocumentWaitUntilResponse& {
  return static_cast<WillSaveTextDocumentWaitUntilResponse&>(
      LSPResponse::id(std::move(id)));
}

auto WillSaveTextDocumentWaitUntilResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TextEdit>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionRequest::method(std::string method) -> CompletionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CompletionRequest::id(std::variant<long, std::string> id)
    -> CompletionRequest& {
  return static_cast<CompletionRequest&>(LSPRequest::id(std::move(id)));
}

auto CompletionRequest::params() const -> CompletionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CompletionParams(repr_->at("params"));
}

auto CompletionRequest::params(CompletionParams params) -> CompletionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CompletionResponse::id(std::variant<long, std::string> id)
    -> CompletionResponse& {
  return static_cast<CompletionResponse&>(LSPResponse::id(std::move(id)));
}

auto CompletionResponse::result() const
    -> std::variant<Vector<CompletionItem>, CompletionList, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<CompletionItem>, CompletionList, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionResolveRequest::method(std::string method)
    -> CompletionResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CompletionResolveRequest::id(std::variant<long, std::string> id)
    -> CompletionResolveRequest& {
  return static_cast<CompletionResolveRequest&>(LSPRequest::id(std::move(id)));
}

auto CompletionResolveRequest::params() const -> CompletionItem {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CompletionItem(repr_->at("params"));
}

auto CompletionResolveRequest::params(CompletionItem params)
    -> CompletionResolveRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CompletionResolveResponse::id(std::variant<long, std::string> id)
    -> CompletionResolveResponse& {
  return static_cast<CompletionResolveResponse&>(
      LSPResponse::id(std::move(id)));
}

auto CompletionResolveResponse::result() const -> CompletionItem {
  auto& value = (*repr_)["result"];

  return CompletionItem(value);
}

auto HoverRequest::method(std::string method) -> HoverRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto HoverRequest::id(std::variant<long, std::string> id) -> HoverRequest& {
  return static_cast<HoverRequest&>(LSPRequest::id(std::move(id)));
}

auto HoverRequest::params() const -> HoverParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return HoverParams(repr_->at("params"));
}

auto HoverRequest::params(HoverParams params) -> HoverRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto HoverResponse::id(std::variant<long, std::string> id) -> HoverResponse& {
  return static_cast<HoverResponse&>(LSPResponse::id(std::move(id)));
}

auto HoverResponse::result() const -> std::variant<Hover, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Hover, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelpRequest::method(std::string method) -> SignatureHelpRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SignatureHelpRequest::id(std::variant<long, std::string> id)
    -> SignatureHelpRequest& {
  return static_cast<SignatureHelpRequest&>(LSPRequest::id(std::move(id)));
}

auto SignatureHelpRequest::params() const -> SignatureHelpParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return SignatureHelpParams(repr_->at("params"));
}

auto SignatureHelpRequest::params(SignatureHelpParams params)
    -> SignatureHelpRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto SignatureHelpResponse::id(std::variant<long, std::string> id)
    -> SignatureHelpResponse& {
  return static_cast<SignatureHelpResponse&>(LSPResponse::id(std::move(id)));
}

auto SignatureHelpResponse::result() const
    -> std::variant<SignatureHelp, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<SignatureHelp, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionRequest::method(std::string method) -> DefinitionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DefinitionRequest::id(std::variant<long, std::string> id)
    -> DefinitionRequest& {
  return static_cast<DefinitionRequest&>(LSPRequest::id(std::move(id)));
}

auto DefinitionRequest::params() const -> DefinitionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DefinitionParams(repr_->at("params"));
}

auto DefinitionRequest::params(DefinitionParams params) -> DefinitionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DefinitionResponse::id(std::variant<long, std::string> id)
    -> DefinitionResponse& {
  return static_cast<DefinitionResponse&>(LSPResponse::id(std::move(id)));
}

auto DefinitionResponse::result() const
    -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ReferencesRequest::method(std::string method) -> ReferencesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ReferencesRequest::id(std::variant<long, std::string> id)
    -> ReferencesRequest& {
  return static_cast<ReferencesRequest&>(LSPRequest::id(std::move(id)));
}

auto ReferencesRequest::params() const -> ReferenceParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ReferenceParams(repr_->at("params"));
}

auto ReferencesRequest::params(ReferenceParams params) -> ReferencesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ReferencesResponse::id(std::variant<long, std::string> id)
    -> ReferencesResponse& {
  return static_cast<ReferencesResponse&>(LSPResponse::id(std::move(id)));
}

auto ReferencesResponse::result() const
    -> std::variant<Vector<Location>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<Location>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightRequest::method(std::string method)
    -> DocumentHighlightRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentHighlightRequest::id(std::variant<long, std::string> id)
    -> DocumentHighlightRequest& {
  return static_cast<DocumentHighlightRequest&>(LSPRequest::id(std::move(id)));
}

auto DocumentHighlightRequest::params() const -> DocumentHighlightParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentHighlightParams(repr_->at("params"));
}

auto DocumentHighlightRequest::params(DocumentHighlightParams params)
    -> DocumentHighlightRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentHighlightResponse::id(std::variant<long, std::string> id)
    -> DocumentHighlightResponse& {
  return static_cast<DocumentHighlightResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentHighlightResponse::result() const
    -> std::variant<Vector<DocumentHighlight>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<DocumentHighlight>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolRequest::method(std::string method)
    -> DocumentSymbolRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentSymbolRequest::id(std::variant<long, std::string> id)
    -> DocumentSymbolRequest& {
  return static_cast<DocumentSymbolRequest&>(LSPRequest::id(std::move(id)));
}

auto DocumentSymbolRequest::params() const -> DocumentSymbolParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentSymbolParams(repr_->at("params"));
}

auto DocumentSymbolRequest::params(DocumentSymbolParams params)
    -> DocumentSymbolRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentSymbolResponse::id(std::variant<long, std::string> id)
    -> DocumentSymbolResponse& {
  return static_cast<DocumentSymbolResponse&>(LSPResponse::id(std::move(id)));
}

auto DocumentSymbolResponse::result() const
    -> std::variant<Vector<SymbolInformation>, Vector<DocumentSymbol>,
                    std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<SymbolInformation>, Vector<DocumentSymbol>,
               std::nullptr_t>
      result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionRequest::method(std::string method) -> CodeActionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeActionRequest::id(std::variant<long, std::string> id)
    -> CodeActionRequest& {
  return static_cast<CodeActionRequest&>(LSPRequest::id(std::move(id)));
}

auto CodeActionRequest::params() const -> CodeActionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CodeActionParams(repr_->at("params"));
}

auto CodeActionRequest::params(CodeActionParams params) -> CodeActionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CodeActionResponse::id(std::variant<long, std::string> id)
    -> CodeActionResponse& {
  return static_cast<CodeActionResponse&>(LSPResponse::id(std::move(id)));
}

auto CodeActionResponse::result() const
    -> std::variant<Vector<std::variant<Command, CodeAction>>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<std::variant<Command, CodeAction>>, std::nullptr_t>
      result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionResolveRequest::method(std::string method)
    -> CodeActionResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeActionResolveRequest::id(std::variant<long, std::string> id)
    -> CodeActionResolveRequest& {
  return static_cast<CodeActionResolveRequest&>(LSPRequest::id(std::move(id)));
}

auto CodeActionResolveRequest::params() const -> CodeAction {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CodeAction(repr_->at("params"));
}

auto CodeActionResolveRequest::params(CodeAction params)
    -> CodeActionResolveRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CodeActionResolveResponse::id(std::variant<long, std::string> id)
    -> CodeActionResolveResponse& {
  return static_cast<CodeActionResolveResponse&>(
      LSPResponse::id(std::move(id)));
}

auto CodeActionResolveResponse::result() const -> CodeAction {
  auto& value = (*repr_)["result"];

  return CodeAction(value);
}

auto WorkspaceSymbolRequest::method(std::string method)
    -> WorkspaceSymbolRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceSymbolRequest::id(std::variant<long, std::string> id)
    -> WorkspaceSymbolRequest& {
  return static_cast<WorkspaceSymbolRequest&>(LSPRequest::id(std::move(id)));
}

auto WorkspaceSymbolRequest::params() const -> WorkspaceSymbolParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WorkspaceSymbolParams(repr_->at("params"));
}

auto WorkspaceSymbolRequest::params(WorkspaceSymbolParams params)
    -> WorkspaceSymbolRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WorkspaceSymbolResponse::id(std::variant<long, std::string> id)
    -> WorkspaceSymbolResponse& {
  return static_cast<WorkspaceSymbolResponse&>(LSPResponse::id(std::move(id)));
}

auto WorkspaceSymbolResponse::result() const
    -> std::variant<Vector<SymbolInformation>, Vector<WorkspaceSymbol>,
                    std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<SymbolInformation>, Vector<WorkspaceSymbol>,
               std::nullptr_t>
      result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbolResolveRequest::method(std::string method)
    -> WorkspaceSymbolResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceSymbolResolveRequest::id(std::variant<long, std::string> id)
    -> WorkspaceSymbolResolveRequest& {
  return static_cast<WorkspaceSymbolResolveRequest&>(
      LSPRequest::id(std::move(id)));
}

auto WorkspaceSymbolResolveRequest::params() const -> WorkspaceSymbol {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WorkspaceSymbol(repr_->at("params"));
}

auto WorkspaceSymbolResolveRequest::params(WorkspaceSymbol params)
    -> WorkspaceSymbolResolveRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WorkspaceSymbolResolveResponse::id(std::variant<long, std::string> id)
    -> WorkspaceSymbolResolveResponse& {
  return static_cast<WorkspaceSymbolResolveResponse&>(
      LSPResponse::id(std::move(id)));
}

auto WorkspaceSymbolResolveResponse::result() const -> WorkspaceSymbol {
  auto& value = (*repr_)["result"];

  return WorkspaceSymbol(value);
}

auto CodeLensRequest::method(std::string method) -> CodeLensRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensRequest::id(std::variant<long, std::string> id)
    -> CodeLensRequest& {
  return static_cast<CodeLensRequest&>(LSPRequest::id(std::move(id)));
}

auto CodeLensRequest::params() const -> CodeLensParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CodeLensParams(repr_->at("params"));
}

auto CodeLensRequest::params(CodeLensParams params) -> CodeLensRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CodeLensResponse::id(std::variant<long, std::string> id)
    -> CodeLensResponse& {
  return static_cast<CodeLensResponse&>(LSPResponse::id(std::move(id)));
}

auto CodeLensResponse::result() const
    -> std::variant<Vector<CodeLens>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<CodeLens>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensResolveRequest::method(std::string method)
    -> CodeLensResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensResolveRequest::id(std::variant<long, std::string> id)
    -> CodeLensResolveRequest& {
  return static_cast<CodeLensResolveRequest&>(LSPRequest::id(std::move(id)));
}

auto CodeLensResolveRequest::params() const -> CodeLens {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CodeLens(repr_->at("params"));
}

auto CodeLensResolveRequest::params(CodeLens params)
    -> CodeLensResolveRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CodeLensResolveResponse::id(std::variant<long, std::string> id)
    -> CodeLensResolveResponse& {
  return static_cast<CodeLensResolveResponse&>(LSPResponse::id(std::move(id)));
}

auto CodeLensResolveResponse::result() const -> CodeLens {
  auto& value = (*repr_)["result"];

  return CodeLens(value);
}

auto CodeLensRefreshRequest::method(std::string method)
    -> CodeLensRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensRefreshRequest::id(std::variant<long, std::string> id)
    -> CodeLensRefreshRequest& {
  return static_cast<CodeLensRefreshRequest&>(LSPRequest::id(std::move(id)));
}

auto CodeLensRefreshResponse::id(std::variant<long, std::string> id)
    -> CodeLensRefreshResponse& {
  return static_cast<CodeLensRefreshResponse&>(LSPResponse::id(std::move(id)));
}

auto CodeLensRefreshResponse::result() const -> std::nullptr_t {
  auto& value = (*repr_)["result"];

  assert(value.is_null());
  return nullptr;
}

auto DocumentLinkRequest::method(std::string method) -> DocumentLinkRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentLinkRequest::id(std::variant<long, std::string> id)
    -> DocumentLinkRequest& {
  return static_cast<DocumentLinkRequest&>(LSPRequest::id(std::move(id)));
}

auto DocumentLinkRequest::params() const -> DocumentLinkParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentLinkParams(repr_->at("params"));
}

auto DocumentLinkRequest::params(DocumentLinkParams params)
    -> DocumentLinkRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentLinkResponse::id(std::variant<long, std::string> id)
    -> DocumentLinkResponse& {
  return static_cast<DocumentLinkResponse&>(LSPResponse::id(std::move(id)));
}

auto DocumentLinkResponse::result() const
    -> std::variant<Vector<DocumentLink>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<DocumentLink>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkResolveRequest::method(std::string method)
    -> DocumentLinkResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentLinkResolveRequest::id(std::variant<long, std::string> id)
    -> DocumentLinkResolveRequest& {
  return static_cast<DocumentLinkResolveRequest&>(
      LSPRequest::id(std::move(id)));
}

auto DocumentLinkResolveRequest::params() const -> DocumentLink {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentLink(repr_->at("params"));
}

auto DocumentLinkResolveRequest::params(DocumentLink params)
    -> DocumentLinkResolveRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentLinkResolveResponse::id(std::variant<long, std::string> id)
    -> DocumentLinkResolveResponse& {
  return static_cast<DocumentLinkResolveResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentLinkResolveResponse::result() const -> DocumentLink {
  auto& value = (*repr_)["result"];

  return DocumentLink(value);
}

auto DocumentFormattingRequest::method(std::string method)
    -> DocumentFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentFormattingRequest& {
  return static_cast<DocumentFormattingRequest&>(LSPRequest::id(std::move(id)));
}

auto DocumentFormattingRequest::params() const -> DocumentFormattingParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentFormattingParams(repr_->at("params"));
}

auto DocumentFormattingRequest::params(DocumentFormattingParams params)
    -> DocumentFormattingRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentFormattingResponse::id(std::variant<long, std::string> id)
    -> DocumentFormattingResponse& {
  return static_cast<DocumentFormattingResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TextEdit>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangeFormattingRequest::method(std::string method)
    -> DocumentRangeFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentRangeFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentRangeFormattingRequest& {
  return static_cast<DocumentRangeFormattingRequest&>(
      LSPRequest::id(std::move(id)));
}

auto DocumentRangeFormattingRequest::params() const
    -> DocumentRangeFormattingParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentRangeFormattingParams(repr_->at("params"));
}

auto DocumentRangeFormattingRequest::params(
    DocumentRangeFormattingParams params) -> DocumentRangeFormattingRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentRangeFormattingResponse::id(std::variant<long, std::string> id)
    -> DocumentRangeFormattingResponse& {
  return static_cast<DocumentRangeFormattingResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentRangeFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TextEdit>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangesFormattingRequest::method(std::string method)
    -> DocumentRangesFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentRangesFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentRangesFormattingRequest& {
  return static_cast<DocumentRangesFormattingRequest&>(
      LSPRequest::id(std::move(id)));
}

auto DocumentRangesFormattingRequest::params() const
    -> DocumentRangesFormattingParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentRangesFormattingParams(repr_->at("params"));
}

auto DocumentRangesFormattingRequest::params(
    DocumentRangesFormattingParams params) -> DocumentRangesFormattingRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentRangesFormattingResponse::id(std::variant<long, std::string> id)
    -> DocumentRangesFormattingResponse& {
  return static_cast<DocumentRangesFormattingResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentRangesFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TextEdit>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentOnTypeFormattingRequest::method(std::string method)
    -> DocumentOnTypeFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentOnTypeFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentOnTypeFormattingRequest& {
  return static_cast<DocumentOnTypeFormattingRequest&>(
      LSPRequest::id(std::move(id)));
}

auto DocumentOnTypeFormattingRequest::params() const
    -> DocumentOnTypeFormattingParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DocumentOnTypeFormattingParams(repr_->at("params"));
}

auto DocumentOnTypeFormattingRequest::params(
    DocumentOnTypeFormattingParams params) -> DocumentOnTypeFormattingRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DocumentOnTypeFormattingResponse::id(std::variant<long, std::string> id)
    -> DocumentOnTypeFormattingResponse& {
  return static_cast<DocumentOnTypeFormattingResponse&>(
      LSPResponse::id(std::move(id)));
}

auto DocumentOnTypeFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<Vector<TextEdit>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto RenameRequest::method(std::string method) -> RenameRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto RenameRequest::id(std::variant<long, std::string> id) -> RenameRequest& {
  return static_cast<RenameRequest&>(LSPRequest::id(std::move(id)));
}

auto RenameRequest::params() const -> RenameParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return RenameParams(repr_->at("params"));
}

auto RenameRequest::params(RenameParams params) -> RenameRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto RenameResponse::id(std::variant<long, std::string> id) -> RenameResponse& {
  return static_cast<RenameResponse&>(LSPResponse::id(std::move(id)));
}

auto RenameResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<WorkspaceEdit, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto PrepareRenameRequest::method(std::string method) -> PrepareRenameRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto PrepareRenameRequest::id(std::variant<long, std::string> id)
    -> PrepareRenameRequest& {
  return static_cast<PrepareRenameRequest&>(LSPRequest::id(std::move(id)));
}

auto PrepareRenameRequest::params() const -> PrepareRenameParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return PrepareRenameParams(repr_->at("params"));
}

auto PrepareRenameRequest::params(PrepareRenameParams params)
    -> PrepareRenameRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto PrepareRenameResponse::id(std::variant<long, std::string> id)
    -> PrepareRenameResponse& {
  return static_cast<PrepareRenameResponse&>(LSPResponse::id(std::move(id)));
}

auto PrepareRenameResponse::result() const
    -> std::variant<PrepareRenameResult, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<PrepareRenameResult, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ExecuteCommandRequest::method(std::string method)
    -> ExecuteCommandRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ExecuteCommandRequest::id(std::variant<long, std::string> id)
    -> ExecuteCommandRequest& {
  return static_cast<ExecuteCommandRequest&>(LSPRequest::id(std::move(id)));
}

auto ExecuteCommandRequest::params() const -> ExecuteCommandParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ExecuteCommandParams(repr_->at("params"));
}

auto ExecuteCommandRequest::params(ExecuteCommandParams params)
    -> ExecuteCommandRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ExecuteCommandResponse::id(std::variant<long, std::string> id)
    -> ExecuteCommandResponse& {
  return static_cast<ExecuteCommandResponse&>(LSPResponse::id(std::move(id)));
}

auto ExecuteCommandResponse::result() const
    -> std::variant<LSPAny, std::nullptr_t> {
  auto& value = (*repr_)["result"];

  std::variant<LSPAny, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ApplyWorkspaceEditRequest::method(std::string method)
    -> ApplyWorkspaceEditRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ApplyWorkspaceEditRequest::id(std::variant<long, std::string> id)
    -> ApplyWorkspaceEditRequest& {
  return static_cast<ApplyWorkspaceEditRequest&>(LSPRequest::id(std::move(id)));
}

auto ApplyWorkspaceEditRequest::params() const -> ApplyWorkspaceEditParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ApplyWorkspaceEditParams(repr_->at("params"));
}

auto ApplyWorkspaceEditRequest::params(ApplyWorkspaceEditParams params)
    -> ApplyWorkspaceEditRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ApplyWorkspaceEditResponse::id(std::variant<long, std::string> id)
    -> ApplyWorkspaceEditResponse& {
  return static_cast<ApplyWorkspaceEditResponse&>(
      LSPResponse::id(std::move(id)));
}

auto ApplyWorkspaceEditResponse::result() const -> ApplyWorkspaceEditResult {
  auto& value = (*repr_)["result"];

  return ApplyWorkspaceEditResult(value);
}

auto DidChangeWorkspaceFoldersNotification::method(std::string method)
    -> DidChangeWorkspaceFoldersNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeWorkspaceFoldersNotification::id(
    std::variant<long, std::string> id)
    -> DidChangeWorkspaceFoldersNotification& {
  return static_cast<DidChangeWorkspaceFoldersNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidChangeWorkspaceFoldersNotification::params() const
    -> DidChangeWorkspaceFoldersParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidChangeWorkspaceFoldersParams(repr_->at("params"));
}

auto DidChangeWorkspaceFoldersNotification::params(
    DidChangeWorkspaceFoldersParams params)
    -> DidChangeWorkspaceFoldersNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WorkDoneProgressCancelNotification::method(std::string method)
    -> WorkDoneProgressCancelNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkDoneProgressCancelNotification::id(std::variant<long, std::string> id)
    -> WorkDoneProgressCancelNotification& {
  return static_cast<WorkDoneProgressCancelNotification&>(
      LSPRequest::id(std::move(id)));
}

auto WorkDoneProgressCancelNotification::params() const
    -> WorkDoneProgressCancelParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WorkDoneProgressCancelParams(repr_->at("params"));
}

auto WorkDoneProgressCancelNotification::params(
    WorkDoneProgressCancelParams params)
    -> WorkDoneProgressCancelNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidCreateFilesNotification::method(std::string method)
    -> DidCreateFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidCreateFilesNotification::id(std::variant<long, std::string> id)
    -> DidCreateFilesNotification& {
  return static_cast<DidCreateFilesNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidCreateFilesNotification::params() const -> CreateFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CreateFilesParams(repr_->at("params"));
}

auto DidCreateFilesNotification::params(CreateFilesParams params)
    -> DidCreateFilesNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidRenameFilesNotification::method(std::string method)
    -> DidRenameFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidRenameFilesNotification::id(std::variant<long, std::string> id)
    -> DidRenameFilesNotification& {
  return static_cast<DidRenameFilesNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidRenameFilesNotification::params() const -> RenameFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return RenameFilesParams(repr_->at("params"));
}

auto DidRenameFilesNotification::params(RenameFilesParams params)
    -> DidRenameFilesNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidDeleteFilesNotification::method(std::string method)
    -> DidDeleteFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidDeleteFilesNotification::id(std::variant<long, std::string> id)
    -> DidDeleteFilesNotification& {
  return static_cast<DidDeleteFilesNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidDeleteFilesNotification::params() const -> DeleteFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DeleteFilesParams(repr_->at("params"));
}

auto DidDeleteFilesNotification::params(DeleteFilesParams params)
    -> DidDeleteFilesNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidOpenNotebookDocumentNotification::method(std::string method)
    -> DidOpenNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidOpenNotebookDocumentNotification::id(std::variant<long, std::string> id)
    -> DidOpenNotebookDocumentNotification& {
  return static_cast<DidOpenNotebookDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidOpenNotebookDocumentNotification::params() const
    -> DidOpenNotebookDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidOpenNotebookDocumentParams(repr_->at("params"));
}

auto DidOpenNotebookDocumentNotification::params(
    DidOpenNotebookDocumentParams params)
    -> DidOpenNotebookDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidChangeNotebookDocumentNotification::method(std::string method)
    -> DidChangeNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeNotebookDocumentNotification::id(
    std::variant<long, std::string> id)
    -> DidChangeNotebookDocumentNotification& {
  return static_cast<DidChangeNotebookDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidChangeNotebookDocumentNotification::params() const
    -> DidChangeNotebookDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidChangeNotebookDocumentParams(repr_->at("params"));
}

auto DidChangeNotebookDocumentNotification::params(
    DidChangeNotebookDocumentParams params)
    -> DidChangeNotebookDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidSaveNotebookDocumentNotification::method(std::string method)
    -> DidSaveNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidSaveNotebookDocumentNotification::id(std::variant<long, std::string> id)
    -> DidSaveNotebookDocumentNotification& {
  return static_cast<DidSaveNotebookDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidSaveNotebookDocumentNotification::params() const
    -> DidSaveNotebookDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidSaveNotebookDocumentParams(repr_->at("params"));
}

auto DidSaveNotebookDocumentNotification::params(
    DidSaveNotebookDocumentParams params)
    -> DidSaveNotebookDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidCloseNotebookDocumentNotification::method(std::string method)
    -> DidCloseNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidCloseNotebookDocumentNotification::id(
    std::variant<long, std::string> id)
    -> DidCloseNotebookDocumentNotification& {
  return static_cast<DidCloseNotebookDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidCloseNotebookDocumentNotification::params() const
    -> DidCloseNotebookDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidCloseNotebookDocumentParams(repr_->at("params"));
}

auto DidCloseNotebookDocumentNotification::params(
    DidCloseNotebookDocumentParams params)
    -> DidCloseNotebookDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InitializedNotification::method(std::string method)
    -> InitializedNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InitializedNotification::id(std::variant<long, std::string> id)
    -> InitializedNotification& {
  return static_cast<InitializedNotification&>(LSPRequest::id(std::move(id)));
}

auto InitializedNotification::params() const -> InitializedParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InitializedParams(repr_->at("params"));
}

auto InitializedNotification::params(InitializedParams params)
    -> InitializedNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ExitNotification::method(std::string method) -> ExitNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ExitNotification::id(std::variant<long, std::string> id)
    -> ExitNotification& {
  return static_cast<ExitNotification&>(LSPRequest::id(std::move(id)));
}

auto DidChangeConfigurationNotification::method(std::string method)
    -> DidChangeConfigurationNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeConfigurationNotification::id(std::variant<long, std::string> id)
    -> DidChangeConfigurationNotification& {
  return static_cast<DidChangeConfigurationNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidChangeConfigurationNotification::params() const
    -> DidChangeConfigurationParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidChangeConfigurationParams(repr_->at("params"));
}

auto DidChangeConfigurationNotification::params(
    DidChangeConfigurationParams params)
    -> DidChangeConfigurationNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ShowMessageNotification::method(std::string method)
    -> ShowMessageNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowMessageNotification::id(std::variant<long, std::string> id)
    -> ShowMessageNotification& {
  return static_cast<ShowMessageNotification&>(LSPRequest::id(std::move(id)));
}

auto ShowMessageNotification::params() const -> ShowMessageParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ShowMessageParams(repr_->at("params"));
}

auto ShowMessageNotification::params(ShowMessageParams params)
    -> ShowMessageNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto LogMessageNotification::method(std::string method)
    -> LogMessageNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LogMessageNotification::id(std::variant<long, std::string> id)
    -> LogMessageNotification& {
  return static_cast<LogMessageNotification&>(LSPRequest::id(std::move(id)));
}

auto LogMessageNotification::params() const -> LogMessageParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return LogMessageParams(repr_->at("params"));
}

auto LogMessageNotification::params(LogMessageParams params)
    -> LogMessageNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto TelemetryEventNotification::method(std::string method)
    -> TelemetryEventNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TelemetryEventNotification::id(std::variant<long, std::string> id)
    -> TelemetryEventNotification& {
  return static_cast<TelemetryEventNotification&>(
      LSPRequest::id(std::move(id)));
}

auto TelemetryEventNotification::params() const -> LSPAny {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return LSPAny(repr_->at("params"));
}

auto TelemetryEventNotification::params(LSPAny params)
    -> TelemetryEventNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidOpenTextDocumentNotification::method(std::string method)
    -> DidOpenTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidOpenTextDocumentNotification::id(std::variant<long, std::string> id)
    -> DidOpenTextDocumentNotification& {
  return static_cast<DidOpenTextDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidOpenTextDocumentNotification::params() const
    -> DidOpenTextDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidOpenTextDocumentParams(repr_->at("params"));
}

auto DidOpenTextDocumentNotification::params(DidOpenTextDocumentParams params)
    -> DidOpenTextDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidChangeTextDocumentNotification::method(std::string method)
    -> DidChangeTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeTextDocumentNotification::id(std::variant<long, std::string> id)
    -> DidChangeTextDocumentNotification& {
  return static_cast<DidChangeTextDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidChangeTextDocumentNotification::params() const
    -> DidChangeTextDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidChangeTextDocumentParams(repr_->at("params"));
}

auto DidChangeTextDocumentNotification::params(
    DidChangeTextDocumentParams params) -> DidChangeTextDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidCloseTextDocumentNotification::method(std::string method)
    -> DidCloseTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidCloseTextDocumentNotification::id(std::variant<long, std::string> id)
    -> DidCloseTextDocumentNotification& {
  return static_cast<DidCloseTextDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidCloseTextDocumentNotification::params() const
    -> DidCloseTextDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidCloseTextDocumentParams(repr_->at("params"));
}

auto DidCloseTextDocumentNotification::params(DidCloseTextDocumentParams params)
    -> DidCloseTextDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidSaveTextDocumentNotification::method(std::string method)
    -> DidSaveTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidSaveTextDocumentNotification::id(std::variant<long, std::string> id)
    -> DidSaveTextDocumentNotification& {
  return static_cast<DidSaveTextDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidSaveTextDocumentNotification::params() const
    -> DidSaveTextDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidSaveTextDocumentParams(repr_->at("params"));
}

auto DidSaveTextDocumentNotification::params(DidSaveTextDocumentParams params)
    -> DidSaveTextDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto WillSaveTextDocumentNotification::method(std::string method)
    -> WillSaveTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillSaveTextDocumentNotification::id(std::variant<long, std::string> id)
    -> WillSaveTextDocumentNotification& {
  return static_cast<WillSaveTextDocumentNotification&>(
      LSPRequest::id(std::move(id)));
}

auto WillSaveTextDocumentNotification::params() const
    -> WillSaveTextDocumentParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return WillSaveTextDocumentParams(repr_->at("params"));
}

auto WillSaveTextDocumentNotification::params(WillSaveTextDocumentParams params)
    -> WillSaveTextDocumentNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DidChangeWatchedFilesNotification::method(std::string method)
    -> DidChangeWatchedFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeWatchedFilesNotification::id(std::variant<long, std::string> id)
    -> DidChangeWatchedFilesNotification& {
  return static_cast<DidChangeWatchedFilesNotification&>(
      LSPRequest::id(std::move(id)));
}

auto DidChangeWatchedFilesNotification::params() const
    -> DidChangeWatchedFilesParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DidChangeWatchedFilesParams(repr_->at("params"));
}

auto DidChangeWatchedFilesNotification::params(
    DidChangeWatchedFilesParams params) -> DidChangeWatchedFilesNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto PublishDiagnosticsNotification::method(std::string method)
    -> PublishDiagnosticsNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto PublishDiagnosticsNotification::id(std::variant<long, std::string> id)
    -> PublishDiagnosticsNotification& {
  return static_cast<PublishDiagnosticsNotification&>(
      LSPRequest::id(std::move(id)));
}

auto PublishDiagnosticsNotification::params() const
    -> PublishDiagnosticsParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return PublishDiagnosticsParams(repr_->at("params"));
}

auto PublishDiagnosticsNotification::params(PublishDiagnosticsParams params)
    -> PublishDiagnosticsNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto SetTraceNotification::method(std::string method) -> SetTraceNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SetTraceNotification::id(std::variant<long, std::string> id)
    -> SetTraceNotification& {
  return static_cast<SetTraceNotification&>(LSPRequest::id(std::move(id)));
}

auto SetTraceNotification::params() const -> SetTraceParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return SetTraceParams(repr_->at("params"));
}

auto SetTraceNotification::params(SetTraceParams params)
    -> SetTraceNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto LogTraceNotification::method(std::string method) -> LogTraceNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LogTraceNotification::id(std::variant<long, std::string> id)
    -> LogTraceNotification& {
  return static_cast<LogTraceNotification&>(LSPRequest::id(std::move(id)));
}

auto LogTraceNotification::params() const -> LogTraceParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return LogTraceParams(repr_->at("params"));
}

auto LogTraceNotification::params(LogTraceParams params)
    -> LogTraceNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CancelNotification::method(std::string method) -> CancelNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CancelNotification::id(std::variant<long, std::string> id)
    -> CancelNotification& {
  return static_cast<CancelNotification&>(LSPRequest::id(std::move(id)));
}

auto CancelNotification::params() const -> CancelParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CancelParams(repr_->at("params"));
}

auto CancelNotification::params(CancelParams params) -> CancelNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ProgressNotification::method(std::string method) -> ProgressNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ProgressNotification::id(std::variant<long, std::string> id)
    -> ProgressNotification& {
  return static_cast<ProgressNotification&>(LSPRequest::id(std::move(id)));
}

auto ProgressNotification::params() const -> ProgressParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ProgressParams(repr_->at("params"));
}

auto ProgressNotification::params(ProgressParams params)
    -> ProgressNotification& {
  (*repr_)["params"] = std::move(params);
  return *this;
}
}  // namespace cxx::lsp
