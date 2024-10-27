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

auto LSPRequest::method() const -> std::string { return repr_->at("method"); }

auto ImplementationRequest::method(std::string method)
    -> ImplementationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ImplementationRequest::id(std::variant<long, std::string> id)
    -> ImplementationRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ImplementationResponse::id(long id) -> ImplementationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ImplementationResponse::id(std::string id) -> ImplementationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ImplementationResponse::result() const
    -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> {
  lsp_runtime_error("ImplementationResponse::result() - not implemented yet");
}

auto ImplementationResponse::result(
    std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result)
    -> ImplementationResponse& {
  lsp_runtime_error("ImplementationResponse::result() - not implemented yet");
  return *this;
}

auto TypeDefinitionRequest::method(std::string method)
    -> TypeDefinitionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeDefinitionRequest::id(std::variant<long, std::string> id)
    -> TypeDefinitionRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto TypeDefinitionResponse::id(long id) -> TypeDefinitionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeDefinitionResponse::id(std::string id) -> TypeDefinitionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TypeDefinitionResponse::result() const
    -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> {
  lsp_runtime_error("TypeDefinitionResponse::result() - not implemented yet");
}

auto TypeDefinitionResponse::result(
    std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result)
    -> TypeDefinitionResponse& {
  lsp_runtime_error("TypeDefinitionResponse::result() - not implemented yet");
  return *this;
}

auto WorkspaceFoldersRequest::method(std::string method)
    -> WorkspaceFoldersRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceFoldersRequest::id(std::variant<long, std::string> id)
    -> WorkspaceFoldersRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto WorkspaceFoldersResponse::id(long id) -> WorkspaceFoldersResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceFoldersResponse::id(std::string id) -> WorkspaceFoldersResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceFoldersResponse::result() const
    -> std::variant<Vector<WorkspaceFolder>, std::nullptr_t> {
  lsp_runtime_error("WorkspaceFoldersResponse::result() - not implemented yet");
}

auto WorkspaceFoldersResponse::result(
    std::variant<Vector<WorkspaceFolder>, std::nullptr_t> result)
    -> WorkspaceFoldersResponse& {
  lsp_runtime_error("WorkspaceFoldersResponse::result() - not implemented yet");
  return *this;
}

auto ConfigurationRequest::method(std::string method) -> ConfigurationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ConfigurationRequest::id(std::variant<long, std::string> id)
    -> ConfigurationRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ConfigurationResponse::id(long id) -> ConfigurationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ConfigurationResponse::id(std::string id) -> ConfigurationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ConfigurationResponse::result() const -> Vector<LSPAny> {
  lsp_runtime_error("ConfigurationResponse::result() - not implemented yet");
}

auto ConfigurationResponse::result(Vector<LSPAny> result)
    -> ConfigurationResponse& {
  lsp_runtime_error("ConfigurationResponse::result() - not implemented yet");
  return *this;
}

auto DocumentColorRequest::method(std::string method) -> DocumentColorRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentColorRequest::id(std::variant<long, std::string> id)
    -> DocumentColorRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentColorResponse::id(long id) -> DocumentColorResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentColorResponse::id(std::string id) -> DocumentColorResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentColorResponse::result() const -> Vector<ColorInformation> {
  lsp_runtime_error("DocumentColorResponse::result() - not implemented yet");
}

auto DocumentColorResponse::result(Vector<ColorInformation> result)
    -> DocumentColorResponse& {
  lsp_runtime_error("DocumentColorResponse::result() - not implemented yet");
  return *this;
}

auto ColorPresentationRequest::method(std::string method)
    -> ColorPresentationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ColorPresentationRequest::id(std::variant<long, std::string> id)
    -> ColorPresentationRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ColorPresentationResponse::id(long id) -> ColorPresentationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ColorPresentationResponse::id(std::string id)
    -> ColorPresentationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ColorPresentationResponse::result() const -> Vector<ColorPresentation> {
  lsp_runtime_error(
      "ColorPresentationResponse::result() - not implemented yet");
}

auto ColorPresentationResponse::result(Vector<ColorPresentation> result)
    -> ColorPresentationResponse& {
  lsp_runtime_error(
      "ColorPresentationResponse::result() - not implemented yet");
  return *this;
}

auto FoldingRangeRequest::method(std::string method) -> FoldingRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto FoldingRangeRequest::id(std::variant<long, std::string> id)
    -> FoldingRangeRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto FoldingRangeResponse::id(long id) -> FoldingRangeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto FoldingRangeResponse::id(std::string id) -> FoldingRangeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto FoldingRangeResponse::result() const
    -> std::variant<Vector<FoldingRange>, std::nullptr_t> {
  lsp_runtime_error("FoldingRangeResponse::result() - not implemented yet");
}

auto FoldingRangeResponse::result(
    std::variant<Vector<FoldingRange>, std::nullptr_t> result)
    -> FoldingRangeResponse& {
  lsp_runtime_error("FoldingRangeResponse::result() - not implemented yet");
  return *this;
}

auto FoldingRangeRefreshRequest::method(std::string method)
    -> FoldingRangeRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto FoldingRangeRefreshRequest::id(std::variant<long, std::string> id)
    -> FoldingRangeRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto FoldingRangeRefreshResponse::id(long id) -> FoldingRangeRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto FoldingRangeRefreshResponse::id(std::string id)
    -> FoldingRangeRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto FoldingRangeRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto FoldingRangeRefreshResponse::result(std::nullptr_t result)
    -> FoldingRangeRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto DeclarationRequest::method(std::string method) -> DeclarationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DeclarationRequest::id(std::variant<long, std::string> id)
    -> DeclarationRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DeclarationResponse::id(long id) -> DeclarationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DeclarationResponse::id(std::string id) -> DeclarationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DeclarationResponse::result() const
    -> std::variant<Declaration, Vector<DeclarationLink>, std::nullptr_t> {
  lsp_runtime_error("DeclarationResponse::result() - not implemented yet");
}

auto DeclarationResponse::result(
    std::variant<Declaration, Vector<DeclarationLink>, std::nullptr_t> result)
    -> DeclarationResponse& {
  lsp_runtime_error("DeclarationResponse::result() - not implemented yet");
  return *this;
}

auto SelectionRangeRequest::method(std::string method)
    -> SelectionRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SelectionRangeRequest::id(std::variant<long, std::string> id)
    -> SelectionRangeRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto SelectionRangeResponse::id(long id) -> SelectionRangeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SelectionRangeResponse::id(std::string id) -> SelectionRangeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SelectionRangeResponse::result() const
    -> std::variant<Vector<SelectionRange>, std::nullptr_t> {
  lsp_runtime_error("SelectionRangeResponse::result() - not implemented yet");
}

auto SelectionRangeResponse::result(
    std::variant<Vector<SelectionRange>, std::nullptr_t> result)
    -> SelectionRangeResponse& {
  lsp_runtime_error("SelectionRangeResponse::result() - not implemented yet");
  return *this;
}

auto WorkDoneProgressCreateRequest::method(std::string method)
    -> WorkDoneProgressCreateRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkDoneProgressCreateRequest::id(std::variant<long, std::string> id)
    -> WorkDoneProgressCreateRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WorkDoneProgressCreateResponse::id(long id)
    -> WorkDoneProgressCreateResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkDoneProgressCreateResponse::id(std::string id)
    -> WorkDoneProgressCreateResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkDoneProgressCreateResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto WorkDoneProgressCreateResponse::result(std::nullptr_t result)
    -> WorkDoneProgressCreateResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto CallHierarchyPrepareRequest::method(std::string method)
    -> CallHierarchyPrepareRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyPrepareRequest::id(std::variant<long, std::string> id)
    -> CallHierarchyPrepareRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto CallHierarchyPrepareResponse::id(long id)
    -> CallHierarchyPrepareResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CallHierarchyPrepareResponse::id(std::string id)
    -> CallHierarchyPrepareResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CallHierarchyPrepareResponse::result() const
    -> std::variant<Vector<CallHierarchyItem>, std::nullptr_t> {
  lsp_runtime_error(
      "CallHierarchyPrepareResponse::result() - not implemented yet");
}

auto CallHierarchyPrepareResponse::result(
    std::variant<Vector<CallHierarchyItem>, std::nullptr_t> result)
    -> CallHierarchyPrepareResponse& {
  lsp_runtime_error(
      "CallHierarchyPrepareResponse::result() - not implemented yet");
  return *this;
}

auto CallHierarchyIncomingCallsRequest::method(std::string method)
    -> CallHierarchyIncomingCallsRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyIncomingCallsRequest::id(std::variant<long, std::string> id)
    -> CallHierarchyIncomingCallsRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto CallHierarchyIncomingCallsResponse::id(long id)
    -> CallHierarchyIncomingCallsResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CallHierarchyIncomingCallsResponse::id(std::string id)
    -> CallHierarchyIncomingCallsResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CallHierarchyIncomingCallsResponse::result() const
    -> std::variant<Vector<CallHierarchyIncomingCall>, std::nullptr_t> {
  lsp_runtime_error(
      "CallHierarchyIncomingCallsResponse::result() - not implemented yet");
}

auto CallHierarchyIncomingCallsResponse::result(
    std::variant<Vector<CallHierarchyIncomingCall>, std::nullptr_t> result)
    -> CallHierarchyIncomingCallsResponse& {
  lsp_runtime_error(
      "CallHierarchyIncomingCallsResponse::result() - not implemented yet");
  return *this;
}

auto CallHierarchyOutgoingCallsRequest::method(std::string method)
    -> CallHierarchyOutgoingCallsRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyOutgoingCallsRequest::id(std::variant<long, std::string> id)
    -> CallHierarchyOutgoingCallsRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto CallHierarchyOutgoingCallsResponse::id(long id)
    -> CallHierarchyOutgoingCallsResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CallHierarchyOutgoingCallsResponse::id(std::string id)
    -> CallHierarchyOutgoingCallsResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CallHierarchyOutgoingCallsResponse::result() const
    -> std::variant<Vector<CallHierarchyOutgoingCall>, std::nullptr_t> {
  lsp_runtime_error(
      "CallHierarchyOutgoingCallsResponse::result() - not implemented yet");
}

auto CallHierarchyOutgoingCallsResponse::result(
    std::variant<Vector<CallHierarchyOutgoingCall>, std::nullptr_t> result)
    -> CallHierarchyOutgoingCallsResponse& {
  lsp_runtime_error(
      "CallHierarchyOutgoingCallsResponse::result() - not implemented yet");
  return *this;
}

auto SemanticTokensRequest::method(std::string method)
    -> SemanticTokensRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto SemanticTokensResponse::id(long id) -> SemanticTokensResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensResponse::id(std::string id) -> SemanticTokensResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SemanticTokensResponse::result() const
    -> std::variant<SemanticTokens, std::nullptr_t> {
  lsp_runtime_error("SemanticTokensResponse::result() - not implemented yet");
}

auto SemanticTokensResponse::result(
    std::variant<SemanticTokens, std::nullptr_t> result)
    -> SemanticTokensResponse& {
  lsp_runtime_error("SemanticTokensResponse::result() - not implemented yet");
  return *this;
}

auto SemanticTokensDeltaRequest::method(std::string method)
    -> SemanticTokensDeltaRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensDeltaRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensDeltaRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto SemanticTokensDeltaResponse::id(long id) -> SemanticTokensDeltaResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensDeltaResponse::id(std::string id)
    -> SemanticTokensDeltaResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SemanticTokensDeltaResponse::result() const
    -> std::variant<SemanticTokens, SemanticTokensDelta, std::nullptr_t> {
  lsp_runtime_error(
      "SemanticTokensDeltaResponse::result() - not implemented yet");
}

auto SemanticTokensDeltaResponse::result(
    std::variant<SemanticTokens, SemanticTokensDelta, std::nullptr_t> result)
    -> SemanticTokensDeltaResponse& {
  lsp_runtime_error(
      "SemanticTokensDeltaResponse::result() - not implemented yet");
  return *this;
}

auto SemanticTokensRangeRequest::method(std::string method)
    -> SemanticTokensRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRangeRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensRangeRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto SemanticTokensRangeResponse::id(long id) -> SemanticTokensRangeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensRangeResponse::id(std::string id)
    -> SemanticTokensRangeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SemanticTokensRangeResponse::result() const
    -> std::variant<SemanticTokens, std::nullptr_t> {
  lsp_runtime_error(
      "SemanticTokensRangeResponse::result() - not implemented yet");
}

auto SemanticTokensRangeResponse::result(
    std::variant<SemanticTokens, std::nullptr_t> result)
    -> SemanticTokensRangeResponse& {
  lsp_runtime_error(
      "SemanticTokensRangeResponse::result() - not implemented yet");
  return *this;
}

auto SemanticTokensRefreshRequest::method(std::string method)
    -> SemanticTokensRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRefreshRequest::id(std::variant<long, std::string> id)
    -> SemanticTokensRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto SemanticTokensRefreshResponse::id(long id)
    -> SemanticTokensRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensRefreshResponse::id(std::string id)
    -> SemanticTokensRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SemanticTokensRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto SemanticTokensRefreshResponse::result(std::nullptr_t result)
    -> SemanticTokensRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto ShowDocumentRequest::method(std::string method) -> ShowDocumentRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowDocumentRequest::id(std::variant<long, std::string> id)
    -> ShowDocumentRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ShowDocumentResponse::id(long id) -> ShowDocumentResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowDocumentResponse::id(std::string id) -> ShowDocumentResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ShowDocumentResponse::result() const -> ShowDocumentResult {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return ShowDocumentResult(repr_->at("result"));  // reference
}

auto ShowDocumentResponse::result(ShowDocumentResult result)
    -> ShowDocumentResponse& {
  lsp_runtime_error("ShowDocumentResponse::result() - not implemented yet");
  return *this;
}

auto LinkedEditingRangeRequest::method(std::string method)
    -> LinkedEditingRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LinkedEditingRangeRequest::id(std::variant<long, std::string> id)
    -> LinkedEditingRangeRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto LinkedEditingRangeResponse::id(long id) -> LinkedEditingRangeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto LinkedEditingRangeResponse::id(std::string id)
    -> LinkedEditingRangeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto LinkedEditingRangeResponse::result() const
    -> std::variant<LinkedEditingRanges, std::nullptr_t> {
  lsp_runtime_error(
      "LinkedEditingRangeResponse::result() - not implemented yet");
}

auto LinkedEditingRangeResponse::result(
    std::variant<LinkedEditingRanges, std::nullptr_t> result)
    -> LinkedEditingRangeResponse& {
  lsp_runtime_error(
      "LinkedEditingRangeResponse::result() - not implemented yet");
  return *this;
}

auto WillCreateFilesRequest::method(std::string method)
    -> WillCreateFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillCreateFilesRequest::id(std::variant<long, std::string> id)
    -> WillCreateFilesRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WillCreateFilesResponse::id(long id) -> WillCreateFilesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillCreateFilesResponse::id(std::string id) -> WillCreateFilesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillCreateFilesResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  lsp_runtime_error("WillCreateFilesResponse::result() - not implemented yet");
}

auto WillCreateFilesResponse::result(
    std::variant<WorkspaceEdit, std::nullptr_t> result)
    -> WillCreateFilesResponse& {
  lsp_runtime_error("WillCreateFilesResponse::result() - not implemented yet");
  return *this;
}

auto WillRenameFilesRequest::method(std::string method)
    -> WillRenameFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillRenameFilesRequest::id(std::variant<long, std::string> id)
    -> WillRenameFilesRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WillRenameFilesResponse::id(long id) -> WillRenameFilesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillRenameFilesResponse::id(std::string id) -> WillRenameFilesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillRenameFilesResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  lsp_runtime_error("WillRenameFilesResponse::result() - not implemented yet");
}

auto WillRenameFilesResponse::result(
    std::variant<WorkspaceEdit, std::nullptr_t> result)
    -> WillRenameFilesResponse& {
  lsp_runtime_error("WillRenameFilesResponse::result() - not implemented yet");
  return *this;
}

auto WillDeleteFilesRequest::method(std::string method)
    -> WillDeleteFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillDeleteFilesRequest::id(std::variant<long, std::string> id)
    -> WillDeleteFilesRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WillDeleteFilesResponse::id(long id) -> WillDeleteFilesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillDeleteFilesResponse::id(std::string id) -> WillDeleteFilesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillDeleteFilesResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  lsp_runtime_error("WillDeleteFilesResponse::result() - not implemented yet");
}

auto WillDeleteFilesResponse::result(
    std::variant<WorkspaceEdit, std::nullptr_t> result)
    -> WillDeleteFilesResponse& {
  lsp_runtime_error("WillDeleteFilesResponse::result() - not implemented yet");
  return *this;
}

auto MonikerRequest::method(std::string method) -> MonikerRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto MonikerRequest::id(std::variant<long, std::string> id) -> MonikerRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto MonikerRequest::params() const -> MonikerParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return MonikerParams(repr_->at("params"));
}

auto MonikerRequest::params(MonikerParams params) -> MonikerRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto MonikerResponse::id(long id) -> MonikerResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto MonikerResponse::id(std::string id) -> MonikerResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto MonikerResponse::result() const
    -> std::variant<Vector<Moniker>, std::nullptr_t> {
  lsp_runtime_error("MonikerResponse::result() - not implemented yet");
}

auto MonikerResponse::result(
    std::variant<Vector<Moniker>, std::nullptr_t> result) -> MonikerResponse& {
  lsp_runtime_error("MonikerResponse::result() - not implemented yet");
  return *this;
}

auto TypeHierarchyPrepareRequest::method(std::string method)
    -> TypeHierarchyPrepareRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchyPrepareRequest::id(std::variant<long, std::string> id)
    -> TypeHierarchyPrepareRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto TypeHierarchyPrepareResponse::id(long id)
    -> TypeHierarchyPrepareResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeHierarchyPrepareResponse::id(std::string id)
    -> TypeHierarchyPrepareResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TypeHierarchyPrepareResponse::result() const
    -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> {
  lsp_runtime_error(
      "TypeHierarchyPrepareResponse::result() - not implemented yet");
}

auto TypeHierarchyPrepareResponse::result(
    std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result)
    -> TypeHierarchyPrepareResponse& {
  lsp_runtime_error(
      "TypeHierarchyPrepareResponse::result() - not implemented yet");
  return *this;
}

auto TypeHierarchySupertypesRequest::method(std::string method)
    -> TypeHierarchySupertypesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchySupertypesRequest::id(std::variant<long, std::string> id)
    -> TypeHierarchySupertypesRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto TypeHierarchySupertypesResponse::id(long id)
    -> TypeHierarchySupertypesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeHierarchySupertypesResponse::id(std::string id)
    -> TypeHierarchySupertypesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TypeHierarchySupertypesResponse::result() const
    -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> {
  lsp_runtime_error(
      "TypeHierarchySupertypesResponse::result() - not implemented yet");
}

auto TypeHierarchySupertypesResponse::result(
    std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result)
    -> TypeHierarchySupertypesResponse& {
  lsp_runtime_error(
      "TypeHierarchySupertypesResponse::result() - not implemented yet");
  return *this;
}

auto TypeHierarchySubtypesRequest::method(std::string method)
    -> TypeHierarchySubtypesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchySubtypesRequest::id(std::variant<long, std::string> id)
    -> TypeHierarchySubtypesRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto TypeHierarchySubtypesResponse::id(long id)
    -> TypeHierarchySubtypesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeHierarchySubtypesResponse::id(std::string id)
    -> TypeHierarchySubtypesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TypeHierarchySubtypesResponse::result() const
    -> std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> {
  lsp_runtime_error(
      "TypeHierarchySubtypesResponse::result() - not implemented yet");
}

auto TypeHierarchySubtypesResponse::result(
    std::variant<Vector<TypeHierarchyItem>, std::nullptr_t> result)
    -> TypeHierarchySubtypesResponse& {
  lsp_runtime_error(
      "TypeHierarchySubtypesResponse::result() - not implemented yet");
  return *this;
}

auto InlineValueRequest::method(std::string method) -> InlineValueRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineValueRequest::id(std::variant<long, std::string> id)
    -> InlineValueRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto InlineValueResponse::id(long id) -> InlineValueResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineValueResponse::id(std::string id) -> InlineValueResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlineValueResponse::result() const
    -> std::variant<Vector<InlineValue>, std::nullptr_t> {
  lsp_runtime_error("InlineValueResponse::result() - not implemented yet");
}

auto InlineValueResponse::result(
    std::variant<Vector<InlineValue>, std::nullptr_t> result)
    -> InlineValueResponse& {
  lsp_runtime_error("InlineValueResponse::result() - not implemented yet");
  return *this;
}

auto InlineValueRefreshRequest::method(std::string method)
    -> InlineValueRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineValueRefreshRequest::id(std::variant<long, std::string> id)
    -> InlineValueRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto InlineValueRefreshResponse::id(long id) -> InlineValueRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineValueRefreshResponse::id(std::string id)
    -> InlineValueRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlineValueRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto InlineValueRefreshResponse::result(std::nullptr_t result)
    -> InlineValueRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto InlayHintRequest::method(std::string method) -> InlayHintRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintRequest::id(std::variant<long, std::string> id)
    -> InlayHintRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto InlayHintRequest::params() const -> InlayHintParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InlayHintParams(repr_->at("params"));
}

auto InlayHintRequest::params(InlayHintParams params) -> InlayHintRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InlayHintResponse::id(long id) -> InlayHintResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintResponse::id(std::string id) -> InlayHintResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlayHintResponse::result() const
    -> std::variant<Vector<InlayHint>, std::nullptr_t> {
  lsp_runtime_error("InlayHintResponse::result() - not implemented yet");
}

auto InlayHintResponse::result(
    std::variant<Vector<InlayHint>, std::nullptr_t> result)
    -> InlayHintResponse& {
  lsp_runtime_error("InlayHintResponse::result() - not implemented yet");
  return *this;
}

auto InlayHintResolveRequest::method(std::string method)
    -> InlayHintResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintResolveRequest::id(std::variant<long, std::string> id)
    -> InlayHintResolveRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto InlayHintResolveResponse::id(long id) -> InlayHintResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintResolveResponse::id(std::string id) -> InlayHintResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlayHintResolveResponse::result() const -> InlayHint {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return InlayHint(repr_->at("result"));  // reference
}

auto InlayHintResolveResponse::result(InlayHint result)
    -> InlayHintResolveResponse& {
  lsp_runtime_error("InlayHintResolveResponse::result() - not implemented yet");
  return *this;
}

auto InlayHintRefreshRequest::method(std::string method)
    -> InlayHintRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintRefreshRequest::id(std::variant<long, std::string> id)
    -> InlayHintRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto InlayHintRefreshResponse::id(long id) -> InlayHintRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintRefreshResponse::id(std::string id) -> InlayHintRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlayHintRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto InlayHintRefreshResponse::result(std::nullptr_t result)
    -> InlayHintRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto DocumentDiagnosticRequest::method(std::string method)
    -> DocumentDiagnosticRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentDiagnosticRequest::id(std::variant<long, std::string> id)
    -> DocumentDiagnosticRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentDiagnosticResponse::id(long id) -> DocumentDiagnosticResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentDiagnosticResponse::id(std::string id)
    -> DocumentDiagnosticResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentDiagnosticResponse::result() const -> DocumentDiagnosticReport {
  lsp_runtime_error(
      "DocumentDiagnosticResponse::result() - not implemented yet");
}

auto DocumentDiagnosticResponse::result(DocumentDiagnosticReport result)
    -> DocumentDiagnosticResponse& {
  lsp_runtime_error(
      "DocumentDiagnosticResponse::result() - not implemented yet");
  return *this;
}

auto WorkspaceDiagnosticRequest::method(std::string method)
    -> WorkspaceDiagnosticRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceDiagnosticRequest::id(std::variant<long, std::string> id)
    -> WorkspaceDiagnosticRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WorkspaceDiagnosticResponse::id(long id) -> WorkspaceDiagnosticResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceDiagnosticResponse::id(std::string id)
    -> WorkspaceDiagnosticResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceDiagnosticResponse::result() const -> WorkspaceDiagnosticReport {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return WorkspaceDiagnosticReport(repr_->at("result"));  // reference
}

auto WorkspaceDiagnosticResponse::result(WorkspaceDiagnosticReport result)
    -> WorkspaceDiagnosticResponse& {
  lsp_runtime_error(
      "WorkspaceDiagnosticResponse::result() - not implemented yet");
  return *this;
}

auto DiagnosticRefreshRequest::method(std::string method)
    -> DiagnosticRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DiagnosticRefreshRequest::id(std::variant<long, std::string> id)
    -> DiagnosticRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto DiagnosticRefreshResponse::id(long id) -> DiagnosticRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DiagnosticRefreshResponse::id(std::string id)
    -> DiagnosticRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DiagnosticRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto DiagnosticRefreshResponse::result(std::nullptr_t result)
    -> DiagnosticRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto InlineCompletionRequest::method(std::string method)
    -> InlineCompletionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineCompletionRequest::id(std::variant<long, std::string> id)
    -> InlineCompletionRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto InlineCompletionResponse::id(long id) -> InlineCompletionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineCompletionResponse::id(std::string id) -> InlineCompletionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlineCompletionResponse::result() const
    -> std::variant<InlineCompletionList, Vector<InlineCompletionItem>,
                    std::nullptr_t> {
  lsp_runtime_error("InlineCompletionResponse::result() - not implemented yet");
}

auto InlineCompletionResponse::result(
    std::variant<InlineCompletionList, Vector<InlineCompletionItem>,
                 std::nullptr_t>
        result) -> InlineCompletionResponse& {
  lsp_runtime_error("InlineCompletionResponse::result() - not implemented yet");
  return *this;
}

auto TextDocumentContentRequest::method(std::string method)
    -> TextDocumentContentRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TextDocumentContentRequest::id(std::variant<long, std::string> id)
    -> TextDocumentContentRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto TextDocumentContentResponse::id(long id) -> TextDocumentContentResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TextDocumentContentResponse::id(std::string id)
    -> TextDocumentContentResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TextDocumentContentResponse::result() const -> TextDocumentContentResult {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return TextDocumentContentResult(repr_->at("result"));  // reference
}

auto TextDocumentContentResponse::result(TextDocumentContentResult result)
    -> TextDocumentContentResponse& {
  lsp_runtime_error(
      "TextDocumentContentResponse::result() - not implemented yet");
  return *this;
}

auto TextDocumentContentRefreshRequest::method(std::string method)
    -> TextDocumentContentRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TextDocumentContentRefreshRequest::id(std::variant<long, std::string> id)
    -> TextDocumentContentRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto TextDocumentContentRefreshResponse::id(long id)
    -> TextDocumentContentRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TextDocumentContentRefreshResponse::id(std::string id)
    -> TextDocumentContentRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TextDocumentContentRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto TextDocumentContentRefreshResponse::result(std::nullptr_t result)
    -> TextDocumentContentRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto RegistrationRequest::method(std::string method) -> RegistrationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto RegistrationRequest::id(std::variant<long, std::string> id)
    -> RegistrationRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto RegistrationResponse::id(long id) -> RegistrationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto RegistrationResponse::id(std::string id) -> RegistrationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto RegistrationResponse::result() const -> std::nullptr_t { return nullptr; }

auto RegistrationResponse::result(std::nullptr_t result)
    -> RegistrationResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto UnregistrationRequest::method(std::string method)
    -> UnregistrationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto UnregistrationRequest::id(std::variant<long, std::string> id)
    -> UnregistrationRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto UnregistrationResponse::id(long id) -> UnregistrationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto UnregistrationResponse::id(std::string id) -> UnregistrationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto UnregistrationResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto UnregistrationResponse::result(std::nullptr_t result)
    -> UnregistrationResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto InitializeRequest::method(std::string method) -> InitializeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InitializeRequest::id(std::variant<long, std::string> id)
    -> InitializeRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto InitializeRequest::params() const -> InitializeParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return InitializeParams(repr_->at("params"));
}

auto InitializeRequest::params(InitializeParams params) -> InitializeRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto InitializeResponse::id(long id) -> InitializeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InitializeResponse::id(std::string id) -> InitializeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InitializeResponse::result() const -> InitializeResult {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return InitializeResult(repr_->at("result"));  // reference
}

auto InitializeResponse::result(InitializeResult result)
    -> InitializeResponse& {
  lsp_runtime_error("InitializeResponse::result() - not implemented yet");
  return *this;
}

auto ShutdownRequest::method(std::string method) -> ShutdownRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShutdownRequest::id(std::variant<long, std::string> id)
    -> ShutdownRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto ShutdownResponse::id(long id) -> ShutdownResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShutdownResponse::id(std::string id) -> ShutdownResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ShutdownResponse::result() const -> std::nullptr_t { return nullptr; }

auto ShutdownResponse::result(std::nullptr_t result) -> ShutdownResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto ShowMessageRequest::method(std::string method) -> ShowMessageRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowMessageRequest::id(std::variant<long, std::string> id)
    -> ShowMessageRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ShowMessageResponse::id(long id) -> ShowMessageResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowMessageResponse::id(std::string id) -> ShowMessageResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ShowMessageResponse::result() const
    -> std::variant<MessageActionItem, std::nullptr_t> {
  lsp_runtime_error("ShowMessageResponse::result() - not implemented yet");
}

auto ShowMessageResponse::result(
    std::variant<MessageActionItem, std::nullptr_t> result)
    -> ShowMessageResponse& {
  lsp_runtime_error("ShowMessageResponse::result() - not implemented yet");
  return *this;
}

auto WillSaveTextDocumentWaitUntilRequest::method(std::string method)
    -> WillSaveTextDocumentWaitUntilRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillSaveTextDocumentWaitUntilRequest::id(
    std::variant<long, std::string> id)
    -> WillSaveTextDocumentWaitUntilRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WillSaveTextDocumentWaitUntilResponse::id(long id)
    -> WillSaveTextDocumentWaitUntilResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillSaveTextDocumentWaitUntilResponse::id(std::string id)
    -> WillSaveTextDocumentWaitUntilResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillSaveTextDocumentWaitUntilResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  lsp_runtime_error(
      "WillSaveTextDocumentWaitUntilResponse::result() - not implemented yet");
}

auto WillSaveTextDocumentWaitUntilResponse::result(
    std::variant<Vector<TextEdit>, std::nullptr_t> result)
    -> WillSaveTextDocumentWaitUntilResponse& {
  lsp_runtime_error(
      "WillSaveTextDocumentWaitUntilResponse::result() - not implemented yet");
  return *this;
}

auto CompletionRequest::method(std::string method) -> CompletionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CompletionRequest::id(std::variant<long, std::string> id)
    -> CompletionRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto CompletionRequest::params() const -> CompletionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CompletionParams(repr_->at("params"));
}

auto CompletionRequest::params(CompletionParams params) -> CompletionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CompletionResponse::id(long id) -> CompletionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CompletionResponse::id(std::string id) -> CompletionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CompletionResponse::result() const
    -> std::variant<Vector<CompletionItem>, CompletionList, std::nullptr_t> {
  lsp_runtime_error("CompletionResponse::result() - not implemented yet");
}

auto CompletionResponse::result(
    std::variant<Vector<CompletionItem>, CompletionList, std::nullptr_t> result)
    -> CompletionResponse& {
  lsp_runtime_error("CompletionResponse::result() - not implemented yet");
  return *this;
}

auto CompletionResolveRequest::method(std::string method)
    -> CompletionResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CompletionResolveRequest::id(std::variant<long, std::string> id)
    -> CompletionResolveRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto CompletionResolveResponse::id(long id) -> CompletionResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CompletionResolveResponse::id(std::string id)
    -> CompletionResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CompletionResolveResponse::result() const -> CompletionItem {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return CompletionItem(repr_->at("result"));  // reference
}

auto CompletionResolveResponse::result(CompletionItem result)
    -> CompletionResolveResponse& {
  lsp_runtime_error(
      "CompletionResolveResponse::result() - not implemented yet");
  return *this;
}

auto HoverRequest::method(std::string method) -> HoverRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto HoverRequest::id(std::variant<long, std::string> id) -> HoverRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto HoverRequest::params() const -> HoverParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return HoverParams(repr_->at("params"));
}

auto HoverRequest::params(HoverParams params) -> HoverRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto HoverResponse::id(long id) -> HoverResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto HoverResponse::id(std::string id) -> HoverResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto HoverResponse::result() const -> std::variant<Hover, std::nullptr_t> {
  lsp_runtime_error("HoverResponse::result() - not implemented yet");
}

auto HoverResponse::result(std::variant<Hover, std::nullptr_t> result)
    -> HoverResponse& {
  lsp_runtime_error("HoverResponse::result() - not implemented yet");
  return *this;
}

auto SignatureHelpRequest::method(std::string method) -> SignatureHelpRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SignatureHelpRequest::id(std::variant<long, std::string> id)
    -> SignatureHelpRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto SignatureHelpResponse::id(long id) -> SignatureHelpResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SignatureHelpResponse::id(std::string id) -> SignatureHelpResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SignatureHelpResponse::result() const
    -> std::variant<SignatureHelp, std::nullptr_t> {
  lsp_runtime_error("SignatureHelpResponse::result() - not implemented yet");
}

auto SignatureHelpResponse::result(
    std::variant<SignatureHelp, std::nullptr_t> result)
    -> SignatureHelpResponse& {
  lsp_runtime_error("SignatureHelpResponse::result() - not implemented yet");
  return *this;
}

auto DefinitionRequest::method(std::string method) -> DefinitionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DefinitionRequest::id(std::variant<long, std::string> id)
    -> DefinitionRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto DefinitionRequest::params() const -> DefinitionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return DefinitionParams(repr_->at("params"));
}

auto DefinitionRequest::params(DefinitionParams params) -> DefinitionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto DefinitionResponse::id(long id) -> DefinitionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DefinitionResponse::id(std::string id) -> DefinitionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DefinitionResponse::result() const
    -> std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> {
  lsp_runtime_error("DefinitionResponse::result() - not implemented yet");
}

auto DefinitionResponse::result(
    std::variant<Definition, Vector<DefinitionLink>, std::nullptr_t> result)
    -> DefinitionResponse& {
  lsp_runtime_error("DefinitionResponse::result() - not implemented yet");
  return *this;
}

auto ReferencesRequest::method(std::string method) -> ReferencesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ReferencesRequest::id(std::variant<long, std::string> id)
    -> ReferencesRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto ReferencesRequest::params() const -> ReferenceParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return ReferenceParams(repr_->at("params"));
}

auto ReferencesRequest::params(ReferenceParams params) -> ReferencesRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto ReferencesResponse::id(long id) -> ReferencesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ReferencesResponse::id(std::string id) -> ReferencesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ReferencesResponse::result() const
    -> std::variant<Vector<Location>, std::nullptr_t> {
  lsp_runtime_error("ReferencesResponse::result() - not implemented yet");
}

auto ReferencesResponse::result(
    std::variant<Vector<Location>, std::nullptr_t> result)
    -> ReferencesResponse& {
  lsp_runtime_error("ReferencesResponse::result() - not implemented yet");
  return *this;
}

auto DocumentHighlightRequest::method(std::string method)
    -> DocumentHighlightRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentHighlightRequest::id(std::variant<long, std::string> id)
    -> DocumentHighlightRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentHighlightResponse::id(long id) -> DocumentHighlightResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentHighlightResponse::id(std::string id)
    -> DocumentHighlightResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentHighlightResponse::result() const
    -> std::variant<Vector<DocumentHighlight>, std::nullptr_t> {
  lsp_runtime_error(
      "DocumentHighlightResponse::result() - not implemented yet");
}

auto DocumentHighlightResponse::result(
    std::variant<Vector<DocumentHighlight>, std::nullptr_t> result)
    -> DocumentHighlightResponse& {
  lsp_runtime_error(
      "DocumentHighlightResponse::result() - not implemented yet");
  return *this;
}

auto DocumentSymbolRequest::method(std::string method)
    -> DocumentSymbolRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentSymbolRequest::id(std::variant<long, std::string> id)
    -> DocumentSymbolRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentSymbolResponse::id(long id) -> DocumentSymbolResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentSymbolResponse::id(std::string id) -> DocumentSymbolResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentSymbolResponse::result() const
    -> std::variant<Vector<SymbolInformation>, Vector<DocumentSymbol>,
                    std::nullptr_t> {
  lsp_runtime_error("DocumentSymbolResponse::result() - not implemented yet");
}

auto DocumentSymbolResponse::result(
    std::variant<Vector<SymbolInformation>, Vector<DocumentSymbol>,
                 std::nullptr_t>
        result) -> DocumentSymbolResponse& {
  lsp_runtime_error("DocumentSymbolResponse::result() - not implemented yet");
  return *this;
}

auto CodeActionRequest::method(std::string method) -> CodeActionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeActionRequest::id(std::variant<long, std::string> id)
    -> CodeActionRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto CodeActionRequest::params() const -> CodeActionParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CodeActionParams(repr_->at("params"));
}

auto CodeActionRequest::params(CodeActionParams params) -> CodeActionRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CodeActionResponse::id(long id) -> CodeActionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeActionResponse::id(std::string id) -> CodeActionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeActionResponse::result() const
    -> std::variant<Vector<std::variant<Command, CodeAction>>, std::nullptr_t> {
  lsp_runtime_error("CodeActionResponse::result() - not implemented yet");
}

auto CodeActionResponse::result(
    std::variant<Vector<std::variant<Command, CodeAction>>, std::nullptr_t>
        result) -> CodeActionResponse& {
  lsp_runtime_error("CodeActionResponse::result() - not implemented yet");
  return *this;
}

auto CodeActionResolveRequest::method(std::string method)
    -> CodeActionResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeActionResolveRequest::id(std::variant<long, std::string> id)
    -> CodeActionResolveRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto CodeActionResolveResponse::id(long id) -> CodeActionResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeActionResolveResponse::id(std::string id)
    -> CodeActionResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeActionResolveResponse::result() const -> CodeAction {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return CodeAction(repr_->at("result"));  // reference
}

auto CodeActionResolveResponse::result(CodeAction result)
    -> CodeActionResolveResponse& {
  lsp_runtime_error(
      "CodeActionResolveResponse::result() - not implemented yet");
  return *this;
}

auto WorkspaceSymbolRequest::method(std::string method)
    -> WorkspaceSymbolRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceSymbolRequest::id(std::variant<long, std::string> id)
    -> WorkspaceSymbolRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WorkspaceSymbolResponse::id(long id) -> WorkspaceSymbolResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceSymbolResponse::id(std::string id) -> WorkspaceSymbolResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceSymbolResponse::result() const
    -> std::variant<Vector<SymbolInformation>, Vector<WorkspaceSymbol>,
                    std::nullptr_t> {
  lsp_runtime_error("WorkspaceSymbolResponse::result() - not implemented yet");
}

auto WorkspaceSymbolResponse::result(
    std::variant<Vector<SymbolInformation>, Vector<WorkspaceSymbol>,
                 std::nullptr_t>
        result) -> WorkspaceSymbolResponse& {
  lsp_runtime_error("WorkspaceSymbolResponse::result() - not implemented yet");
  return *this;
}

auto WorkspaceSymbolResolveRequest::method(std::string method)
    -> WorkspaceSymbolResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceSymbolResolveRequest::id(std::variant<long, std::string> id)
    -> WorkspaceSymbolResolveRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto WorkspaceSymbolResolveResponse::id(long id)
    -> WorkspaceSymbolResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceSymbolResolveResponse::id(std::string id)
    -> WorkspaceSymbolResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceSymbolResolveResponse::result() const -> WorkspaceSymbol {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return WorkspaceSymbol(repr_->at("result"));  // reference
}

auto WorkspaceSymbolResolveResponse::result(WorkspaceSymbol result)
    -> WorkspaceSymbolResolveResponse& {
  lsp_runtime_error(
      "WorkspaceSymbolResolveResponse::result() - not implemented yet");
  return *this;
}

auto CodeLensRequest::method(std::string method) -> CodeLensRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensRequest::id(std::variant<long, std::string> id)
    -> CodeLensRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto CodeLensRequest::params() const -> CodeLensParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return CodeLensParams(repr_->at("params"));
}

auto CodeLensRequest::params(CodeLensParams params) -> CodeLensRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto CodeLensResponse::id(long id) -> CodeLensResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensResponse::id(std::string id) -> CodeLensResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeLensResponse::result() const
    -> std::variant<Vector<CodeLens>, std::nullptr_t> {
  lsp_runtime_error("CodeLensResponse::result() - not implemented yet");
}

auto CodeLensResponse::result(
    std::variant<Vector<CodeLens>, std::nullptr_t> result)
    -> CodeLensResponse& {
  lsp_runtime_error("CodeLensResponse::result() - not implemented yet");
  return *this;
}

auto CodeLensResolveRequest::method(std::string method)
    -> CodeLensResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensResolveRequest::id(std::variant<long, std::string> id)
    -> CodeLensResolveRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto CodeLensResolveResponse::id(long id) -> CodeLensResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensResolveResponse::id(std::string id) -> CodeLensResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeLensResolveResponse::result() const -> CodeLens {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return CodeLens(repr_->at("result"));  // reference
}

auto CodeLensResolveResponse::result(CodeLens result)
    -> CodeLensResolveResponse& {
  lsp_runtime_error("CodeLensResolveResponse::result() - not implemented yet");
  return *this;
}

auto CodeLensRefreshRequest::method(std::string method)
    -> CodeLensRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensRefreshRequest::id(std::variant<long, std::string> id)
    -> CodeLensRefreshRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto CodeLensRefreshResponse::id(long id) -> CodeLensRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensRefreshResponse::id(std::string id) -> CodeLensRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeLensRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto CodeLensRefreshResponse::result(std::nullptr_t result)
    -> CodeLensRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
  return *this;
}

auto DocumentLinkRequest::method(std::string method) -> DocumentLinkRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentLinkRequest::id(std::variant<long, std::string> id)
    -> DocumentLinkRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentLinkResponse::id(long id) -> DocumentLinkResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentLinkResponse::id(std::string id) -> DocumentLinkResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentLinkResponse::result() const
    -> std::variant<Vector<DocumentLink>, std::nullptr_t> {
  lsp_runtime_error("DocumentLinkResponse::result() - not implemented yet");
}

auto DocumentLinkResponse::result(
    std::variant<Vector<DocumentLink>, std::nullptr_t> result)
    -> DocumentLinkResponse& {
  lsp_runtime_error("DocumentLinkResponse::result() - not implemented yet");
  return *this;
}

auto DocumentLinkResolveRequest::method(std::string method)
    -> DocumentLinkResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentLinkResolveRequest::id(std::variant<long, std::string> id)
    -> DocumentLinkResolveRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentLinkResolveResponse::id(long id) -> DocumentLinkResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentLinkResolveResponse::id(std::string id)
    -> DocumentLinkResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentLinkResolveResponse::result() const -> DocumentLink {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return DocumentLink(repr_->at("result"));  // reference
}

auto DocumentLinkResolveResponse::result(DocumentLink result)
    -> DocumentLinkResolveResponse& {
  lsp_runtime_error(
      "DocumentLinkResolveResponse::result() - not implemented yet");
  return *this;
}

auto DocumentFormattingRequest::method(std::string method)
    -> DocumentFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentFormattingRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentFormattingResponse::id(long id) -> DocumentFormattingResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentFormattingResponse::id(std::string id)
    -> DocumentFormattingResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  lsp_runtime_error(
      "DocumentFormattingResponse::result() - not implemented yet");
}

auto DocumentFormattingResponse::result(
    std::variant<Vector<TextEdit>, std::nullptr_t> result)
    -> DocumentFormattingResponse& {
  lsp_runtime_error(
      "DocumentFormattingResponse::result() - not implemented yet");
  return *this;
}

auto DocumentRangeFormattingRequest::method(std::string method)
    -> DocumentRangeFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentRangeFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentRangeFormattingRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentRangeFormattingResponse::id(long id)
    -> DocumentRangeFormattingResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentRangeFormattingResponse::id(std::string id)
    -> DocumentRangeFormattingResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentRangeFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  lsp_runtime_error(
      "DocumentRangeFormattingResponse::result() - not implemented yet");
}

auto DocumentRangeFormattingResponse::result(
    std::variant<Vector<TextEdit>, std::nullptr_t> result)
    -> DocumentRangeFormattingResponse& {
  lsp_runtime_error(
      "DocumentRangeFormattingResponse::result() - not implemented yet");
  return *this;
}

auto DocumentRangesFormattingRequest::method(std::string method)
    -> DocumentRangesFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentRangesFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentRangesFormattingRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentRangesFormattingResponse::id(long id)
    -> DocumentRangesFormattingResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentRangesFormattingResponse::id(std::string id)
    -> DocumentRangesFormattingResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentRangesFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  lsp_runtime_error(
      "DocumentRangesFormattingResponse::result() - not implemented yet");
}

auto DocumentRangesFormattingResponse::result(
    std::variant<Vector<TextEdit>, std::nullptr_t> result)
    -> DocumentRangesFormattingResponse& {
  lsp_runtime_error(
      "DocumentRangesFormattingResponse::result() - not implemented yet");
  return *this;
}

auto DocumentOnTypeFormattingRequest::method(std::string method)
    -> DocumentOnTypeFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentOnTypeFormattingRequest::id(std::variant<long, std::string> id)
    -> DocumentOnTypeFormattingRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto DocumentOnTypeFormattingResponse::id(long id)
    -> DocumentOnTypeFormattingResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentOnTypeFormattingResponse::id(std::string id)
    -> DocumentOnTypeFormattingResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentOnTypeFormattingResponse::result() const
    -> std::variant<Vector<TextEdit>, std::nullptr_t> {
  lsp_runtime_error(
      "DocumentOnTypeFormattingResponse::result() - not implemented yet");
}

auto DocumentOnTypeFormattingResponse::result(
    std::variant<Vector<TextEdit>, std::nullptr_t> result)
    -> DocumentOnTypeFormattingResponse& {
  lsp_runtime_error(
      "DocumentOnTypeFormattingResponse::result() - not implemented yet");
  return *this;
}

auto RenameRequest::method(std::string method) -> RenameRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto RenameRequest::id(std::variant<long, std::string> id) -> RenameRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto RenameRequest::params() const -> RenameParams {
  if (!repr_->contains("params")) repr_->emplace("params", json::object());
  return RenameParams(repr_->at("params"));
}

auto RenameRequest::params(RenameParams params) -> RenameRequest& {
  (*repr_)["params"] = std::move(params);
  return *this;
}

auto RenameResponse::id(long id) -> RenameResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto RenameResponse::id(std::string id) -> RenameResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto RenameResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  lsp_runtime_error("RenameResponse::result() - not implemented yet");
}

auto RenameResponse::result(std::variant<WorkspaceEdit, std::nullptr_t> result)
    -> RenameResponse& {
  lsp_runtime_error("RenameResponse::result() - not implemented yet");
  return *this;
}

auto PrepareRenameRequest::method(std::string method) -> PrepareRenameRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto PrepareRenameRequest::id(std::variant<long, std::string> id)
    -> PrepareRenameRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto PrepareRenameResponse::id(long id) -> PrepareRenameResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto PrepareRenameResponse::id(std::string id) -> PrepareRenameResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto PrepareRenameResponse::result() const
    -> std::variant<PrepareRenameResult, std::nullptr_t> {
  lsp_runtime_error("PrepareRenameResponse::result() - not implemented yet");
}

auto PrepareRenameResponse::result(
    std::variant<PrepareRenameResult, std::nullptr_t> result)
    -> PrepareRenameResponse& {
  lsp_runtime_error("PrepareRenameResponse::result() - not implemented yet");
  return *this;
}

auto ExecuteCommandRequest::method(std::string method)
    -> ExecuteCommandRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ExecuteCommandRequest::id(std::variant<long, std::string> id)
    -> ExecuteCommandRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ExecuteCommandResponse::id(long id) -> ExecuteCommandResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ExecuteCommandResponse::id(std::string id) -> ExecuteCommandResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ExecuteCommandResponse::result() const
    -> std::variant<LSPAny, std::nullptr_t> {
  lsp_runtime_error("ExecuteCommandResponse::result() - not implemented yet");
}

auto ExecuteCommandResponse::result(std::variant<LSPAny, std::nullptr_t> result)
    -> ExecuteCommandResponse& {
  lsp_runtime_error("ExecuteCommandResponse::result() - not implemented yet");
  return *this;
}

auto ApplyWorkspaceEditRequest::method(std::string method)
    -> ApplyWorkspaceEditRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ApplyWorkspaceEditRequest::id(std::variant<long, std::string> id)
    -> ApplyWorkspaceEditRequest& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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

auto ApplyWorkspaceEditResponse::id(long id) -> ApplyWorkspaceEditResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ApplyWorkspaceEditResponse::id(std::string id)
    -> ApplyWorkspaceEditResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ApplyWorkspaceEditResponse::result() const -> ApplyWorkspaceEditResult {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return ApplyWorkspaceEditResult(repr_->at("result"));  // reference
}

auto ApplyWorkspaceEditResponse::result(ApplyWorkspaceEditResult result)
    -> ApplyWorkspaceEditResponse& {
  lsp_runtime_error(
      "ApplyWorkspaceEditResponse::result() - not implemented yet");
  return *this;
}

auto DidChangeWorkspaceFoldersNotification::method(std::string method)
    -> DidChangeWorkspaceFoldersNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeWorkspaceFoldersNotification::id(
    std::variant<long, std::string> id)
    -> DidChangeWorkspaceFoldersNotification& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
}

auto DidChangeConfigurationNotification::method(std::string method)
    -> DidChangeConfigurationNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeConfigurationNotification::id(std::variant<long, std::string> id)
    -> DidChangeConfigurationNotification& {
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
  if (std::holds_alternative<long>(id)) {
    (*repr_)["id"] = std::get<long>(id);
  } else {
    (*repr_)["id"] = std::get<std::string>(id);
  }
  return *this;
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
