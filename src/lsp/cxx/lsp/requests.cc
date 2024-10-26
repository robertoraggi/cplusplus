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

auto ImplementationRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ImplementationRequest::method(std::string method)
    -> ImplementationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ImplementationRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ImplementationRequest::id(long id) -> ImplementationRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ImplementationRequest::id(std::string id) -> ImplementationRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ImplementationResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ImplementationResponse::id(long id) -> ImplementationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ImplementationResponse::id(std::string id) -> ImplementationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TypeDefinitionRequest::method() const -> std::string {
  return repr_->at("method");
}

auto TypeDefinitionRequest::method(std::string method)
    -> TypeDefinitionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeDefinitionRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TypeDefinitionRequest::id(long id) -> TypeDefinitionRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeDefinitionRequest::id(std::string id) -> TypeDefinitionRequest& {
  (*repr_)["id"] = std::move(id);
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

auto TypeDefinitionResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TypeDefinitionResponse::id(long id) -> TypeDefinitionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeDefinitionResponse::id(std::string id) -> TypeDefinitionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceFoldersRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WorkspaceFoldersRequest::method(std::string method)
    -> WorkspaceFoldersRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceFoldersRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkspaceFoldersRequest::id(long id) -> WorkspaceFoldersRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceFoldersRequest::id(std::string id) -> WorkspaceFoldersRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceFoldersResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkspaceFoldersResponse::id(long id) -> WorkspaceFoldersResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceFoldersResponse::id(std::string id) -> WorkspaceFoldersResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ConfigurationRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ConfigurationRequest::method(std::string method) -> ConfigurationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ConfigurationRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ConfigurationRequest::id(long id) -> ConfigurationRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ConfigurationRequest::id(std::string id) -> ConfigurationRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ConfigurationResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ConfigurationResponse::id(long id) -> ConfigurationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ConfigurationResponse::id(std::string id) -> ConfigurationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentColorRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentColorRequest::method(std::string method) -> DocumentColorRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentColorRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentColorRequest::id(long id) -> DocumentColorRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentColorRequest::id(std::string id) -> DocumentColorRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentColorResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentColorResponse::id(long id) -> DocumentColorResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentColorResponse::id(std::string id) -> DocumentColorResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ColorPresentationRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ColorPresentationRequest::method(std::string method)
    -> ColorPresentationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ColorPresentationRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ColorPresentationRequest::id(long id) -> ColorPresentationRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ColorPresentationRequest::id(std::string id) -> ColorPresentationRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ColorPresentationResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto FoldingRangeRequest::method() const -> std::string {
  return repr_->at("method");
}

auto FoldingRangeRequest::method(std::string method) -> FoldingRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto FoldingRangeRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto FoldingRangeRequest::id(long id) -> FoldingRangeRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto FoldingRangeRequest::id(std::string id) -> FoldingRangeRequest& {
  (*repr_)["id"] = std::move(id);
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

auto FoldingRangeResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto FoldingRangeResponse::id(long id) -> FoldingRangeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto FoldingRangeResponse::id(std::string id) -> FoldingRangeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto FoldingRangeRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto FoldingRangeRefreshRequest::method(std::string method)
    -> FoldingRangeRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto FoldingRangeRefreshRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto FoldingRangeRefreshRequest::id(long id) -> FoldingRangeRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto FoldingRangeRefreshRequest::id(std::string id)
    -> FoldingRangeRefreshRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto FoldingRangeRefreshResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DeclarationRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DeclarationRequest::method(std::string method) -> DeclarationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DeclarationRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DeclarationRequest::id(long id) -> DeclarationRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DeclarationRequest::id(std::string id) -> DeclarationRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DeclarationResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DeclarationResponse::id(long id) -> DeclarationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DeclarationResponse::id(std::string id) -> DeclarationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SelectionRangeRequest::method() const -> std::string {
  return repr_->at("method");
}

auto SelectionRangeRequest::method(std::string method)
    -> SelectionRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SelectionRangeRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SelectionRangeRequest::id(long id) -> SelectionRangeRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto SelectionRangeRequest::id(std::string id) -> SelectionRangeRequest& {
  (*repr_)["id"] = std::move(id);
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

auto SelectionRangeResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SelectionRangeResponse::id(long id) -> SelectionRangeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SelectionRangeResponse::id(std::string id) -> SelectionRangeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkDoneProgressCreateRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WorkDoneProgressCreateRequest::method(std::string method)
    -> WorkDoneProgressCreateRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkDoneProgressCreateRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkDoneProgressCreateRequest::id(long id)
    -> WorkDoneProgressCreateRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkDoneProgressCreateRequest::id(std::string id)
    -> WorkDoneProgressCreateRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WorkDoneProgressCreateResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto CallHierarchyPrepareRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CallHierarchyPrepareRequest::method(std::string method)
    -> CallHierarchyPrepareRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyPrepareRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CallHierarchyPrepareRequest::id(long id) -> CallHierarchyPrepareRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CallHierarchyPrepareRequest::id(std::string id)
    -> CallHierarchyPrepareRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CallHierarchyPrepareResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto CallHierarchyIncomingCallsRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CallHierarchyIncomingCallsRequest::method(std::string method)
    -> CallHierarchyIncomingCallsRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyIncomingCallsRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CallHierarchyIncomingCallsRequest::id(long id)
    -> CallHierarchyIncomingCallsRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CallHierarchyIncomingCallsRequest::id(std::string id)
    -> CallHierarchyIncomingCallsRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CallHierarchyIncomingCallsResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto CallHierarchyOutgoingCallsRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CallHierarchyOutgoingCallsRequest::method(std::string method)
    -> CallHierarchyOutgoingCallsRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CallHierarchyOutgoingCallsRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CallHierarchyOutgoingCallsRequest::id(long id)
    -> CallHierarchyOutgoingCallsRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CallHierarchyOutgoingCallsRequest::id(std::string id)
    -> CallHierarchyOutgoingCallsRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CallHierarchyOutgoingCallsResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto SemanticTokensRequest::method() const -> std::string {
  return repr_->at("method");
}

auto SemanticTokensRequest::method(std::string method)
    -> SemanticTokensRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SemanticTokensRequest::id(long id) -> SemanticTokensRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensRequest::id(std::string id) -> SemanticTokensRequest& {
  (*repr_)["id"] = std::move(id);
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

auto SemanticTokensResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SemanticTokensResponse::id(long id) -> SemanticTokensResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensResponse::id(std::string id) -> SemanticTokensResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SemanticTokensDeltaRequest::method() const -> std::string {
  return repr_->at("method");
}

auto SemanticTokensDeltaRequest::method(std::string method)
    -> SemanticTokensDeltaRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensDeltaRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SemanticTokensDeltaRequest::id(long id) -> SemanticTokensDeltaRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensDeltaRequest::id(std::string id)
    -> SemanticTokensDeltaRequest& {
  (*repr_)["id"] = std::move(id);
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

auto SemanticTokensDeltaResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto SemanticTokensRangeRequest::method() const -> std::string {
  return repr_->at("method");
}

auto SemanticTokensRangeRequest::method(std::string method)
    -> SemanticTokensRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRangeRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SemanticTokensRangeRequest::id(long id) -> SemanticTokensRangeRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensRangeRequest::id(std::string id)
    -> SemanticTokensRangeRequest& {
  (*repr_)["id"] = std::move(id);
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

auto SemanticTokensRangeResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto SemanticTokensRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto SemanticTokensRefreshRequest::method(std::string method)
    -> SemanticTokensRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SemanticTokensRefreshRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SemanticTokensRefreshRequest::id(long id)
    -> SemanticTokensRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto SemanticTokensRefreshRequest::id(std::string id)
    -> SemanticTokensRefreshRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SemanticTokensRefreshResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto ShowDocumentRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ShowDocumentRequest::method(std::string method) -> ShowDocumentRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowDocumentRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShowDocumentRequest::id(long id) -> ShowDocumentRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowDocumentRequest::id(std::string id) -> ShowDocumentRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ShowDocumentResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShowDocumentResponse::id(long id) -> ShowDocumentResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowDocumentResponse::id(std::string id) -> ShowDocumentResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto LinkedEditingRangeRequest::method() const -> std::string {
  return repr_->at("method");
}

auto LinkedEditingRangeRequest::method(std::string method)
    -> LinkedEditingRangeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LinkedEditingRangeRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto LinkedEditingRangeRequest::id(long id) -> LinkedEditingRangeRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto LinkedEditingRangeRequest::id(std::string id)
    -> LinkedEditingRangeRequest& {
  (*repr_)["id"] = std::move(id);
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

auto LinkedEditingRangeResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto WillCreateFilesRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WillCreateFilesRequest::method(std::string method)
    -> WillCreateFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillCreateFilesRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillCreateFilesRequest::id(long id) -> WillCreateFilesRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillCreateFilesRequest::id(std::string id) -> WillCreateFilesRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WillCreateFilesResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillCreateFilesResponse::id(long id) -> WillCreateFilesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillCreateFilesResponse::id(std::string id) -> WillCreateFilesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillRenameFilesRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WillRenameFilesRequest::method(std::string method)
    -> WillRenameFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillRenameFilesRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillRenameFilesRequest::id(long id) -> WillRenameFilesRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillRenameFilesRequest::id(std::string id) -> WillRenameFilesRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WillRenameFilesResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillRenameFilesResponse::id(long id) -> WillRenameFilesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillRenameFilesResponse::id(std::string id) -> WillRenameFilesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillDeleteFilesRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WillDeleteFilesRequest::method(std::string method)
    -> WillDeleteFilesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillDeleteFilesRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillDeleteFilesRequest::id(long id) -> WillDeleteFilesRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillDeleteFilesRequest::id(std::string id) -> WillDeleteFilesRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WillDeleteFilesResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillDeleteFilesResponse::id(long id) -> WillDeleteFilesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillDeleteFilesResponse::id(std::string id) -> WillDeleteFilesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto MonikerRequest::method() const -> std::string {
  return repr_->at("method");
}

auto MonikerRequest::method(std::string method) -> MonikerRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto MonikerRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto MonikerRequest::id(long id) -> MonikerRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto MonikerRequest::id(std::string id) -> MonikerRequest& {
  (*repr_)["id"] = std::move(id);
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

auto MonikerResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto MonikerResponse::id(long id) -> MonikerResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto MonikerResponse::id(std::string id) -> MonikerResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TypeHierarchyPrepareRequest::method() const -> std::string {
  return repr_->at("method");
}

auto TypeHierarchyPrepareRequest::method(std::string method)
    -> TypeHierarchyPrepareRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchyPrepareRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TypeHierarchyPrepareRequest::id(long id) -> TypeHierarchyPrepareRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeHierarchyPrepareRequest::id(std::string id)
    -> TypeHierarchyPrepareRequest& {
  (*repr_)["id"] = std::move(id);
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

auto TypeHierarchyPrepareResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto TypeHierarchySupertypesRequest::method() const -> std::string {
  return repr_->at("method");
}

auto TypeHierarchySupertypesRequest::method(std::string method)
    -> TypeHierarchySupertypesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchySupertypesRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TypeHierarchySupertypesRequest::id(long id)
    -> TypeHierarchySupertypesRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeHierarchySupertypesRequest::id(std::string id)
    -> TypeHierarchySupertypesRequest& {
  (*repr_)["id"] = std::move(id);
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

auto TypeHierarchySupertypesResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto TypeHierarchySubtypesRequest::method() const -> std::string {
  return repr_->at("method");
}

auto TypeHierarchySubtypesRequest::method(std::string method)
    -> TypeHierarchySubtypesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TypeHierarchySubtypesRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TypeHierarchySubtypesRequest::id(long id)
    -> TypeHierarchySubtypesRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto TypeHierarchySubtypesRequest::id(std::string id)
    -> TypeHierarchySubtypesRequest& {
  (*repr_)["id"] = std::move(id);
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

auto TypeHierarchySubtypesResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto InlineValueRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InlineValueRequest::method(std::string method) -> InlineValueRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineValueRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlineValueRequest::id(long id) -> InlineValueRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineValueRequest::id(std::string id) -> InlineValueRequest& {
  (*repr_)["id"] = std::move(id);
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

auto InlineValueResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlineValueResponse::id(long id) -> InlineValueResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineValueResponse::id(std::string id) -> InlineValueResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlineValueRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InlineValueRefreshRequest::method(std::string method)
    -> InlineValueRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineValueRefreshRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlineValueRefreshRequest::id(long id) -> InlineValueRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineValueRefreshRequest::id(std::string id)
    -> InlineValueRefreshRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlineValueRefreshResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto InlayHintRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InlayHintRequest::method(std::string method) -> InlayHintRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlayHintRequest::id(long id) -> InlayHintRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintRequest::id(std::string id) -> InlayHintRequest& {
  (*repr_)["id"] = std::move(id);
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

auto InlayHintResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlayHintResponse::id(long id) -> InlayHintResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintResponse::id(std::string id) -> InlayHintResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlayHintResolveRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InlayHintResolveRequest::method(std::string method)
    -> InlayHintResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintResolveRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlayHintResolveRequest::id(long id) -> InlayHintResolveRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintResolveRequest::id(std::string id) -> InlayHintResolveRequest& {
  (*repr_)["id"] = std::move(id);
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

auto InlayHintResolveResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlayHintResolveResponse::id(long id) -> InlayHintResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintResolveResponse::id(std::string id) -> InlayHintResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlayHintRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InlayHintRefreshRequest::method(std::string method)
    -> InlayHintRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlayHintRefreshRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlayHintRefreshRequest::id(long id) -> InlayHintRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintRefreshRequest::id(std::string id) -> InlayHintRefreshRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InlayHintRefreshResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlayHintRefreshResponse::id(long id) -> InlayHintRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlayHintRefreshResponse::id(std::string id) -> InlayHintRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentDiagnosticRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentDiagnosticRequest::method(std::string method)
    -> DocumentDiagnosticRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentDiagnosticRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentDiagnosticRequest::id(long id) -> DocumentDiagnosticRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentDiagnosticRequest::id(std::string id)
    -> DocumentDiagnosticRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentDiagnosticResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto WorkspaceDiagnosticRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WorkspaceDiagnosticRequest::method(std::string method)
    -> WorkspaceDiagnosticRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceDiagnosticRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkspaceDiagnosticRequest::id(long id) -> WorkspaceDiagnosticRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceDiagnosticRequest::id(std::string id)
    -> WorkspaceDiagnosticRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WorkspaceDiagnosticResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DiagnosticRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DiagnosticRefreshRequest::method(std::string method)
    -> DiagnosticRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DiagnosticRefreshRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DiagnosticRefreshRequest::id(long id) -> DiagnosticRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DiagnosticRefreshRequest::id(std::string id) -> DiagnosticRefreshRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DiagnosticRefreshResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto InlineCompletionRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InlineCompletionRequest::method(std::string method)
    -> InlineCompletionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InlineCompletionRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlineCompletionRequest::id(long id) -> InlineCompletionRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineCompletionRequest::id(std::string id) -> InlineCompletionRequest& {
  (*repr_)["id"] = std::move(id);
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

auto InlineCompletionResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InlineCompletionResponse::id(long id) -> InlineCompletionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InlineCompletionResponse::id(std::string id) -> InlineCompletionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto TextDocumentContentRequest::method() const -> std::string {
  return repr_->at("method");
}

auto TextDocumentContentRequest::method(std::string method)
    -> TextDocumentContentRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TextDocumentContentRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TextDocumentContentRequest::id(long id) -> TextDocumentContentRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto TextDocumentContentRequest::id(std::string id)
    -> TextDocumentContentRequest& {
  (*repr_)["id"] = std::move(id);
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

auto TextDocumentContentResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto TextDocumentContentRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto TextDocumentContentRefreshRequest::method(std::string method)
    -> TextDocumentContentRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TextDocumentContentRefreshRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TextDocumentContentRefreshRequest::id(long id)
    -> TextDocumentContentRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto TextDocumentContentRefreshRequest::id(std::string id)
    -> TextDocumentContentRefreshRequest& {
  (*repr_)["id"] = std::move(id);
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

auto TextDocumentContentRefreshResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto RegistrationRequest::method() const -> std::string {
  return repr_->at("method");
}

auto RegistrationRequest::method(std::string method) -> RegistrationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto RegistrationRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto RegistrationRequest::id(long id) -> RegistrationRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto RegistrationRequest::id(std::string id) -> RegistrationRequest& {
  (*repr_)["id"] = std::move(id);
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

auto RegistrationResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto RegistrationResponse::id(long id) -> RegistrationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto RegistrationResponse::id(std::string id) -> RegistrationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto UnregistrationRequest::method() const -> std::string {
  return repr_->at("method");
}

auto UnregistrationRequest::method(std::string method)
    -> UnregistrationRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto UnregistrationRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto UnregistrationRequest::id(long id) -> UnregistrationRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto UnregistrationRequest::id(std::string id) -> UnregistrationRequest& {
  (*repr_)["id"] = std::move(id);
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

auto UnregistrationResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto UnregistrationResponse::id(long id) -> UnregistrationResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto UnregistrationResponse::id(std::string id) -> UnregistrationResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto InitializeRequest::method() const -> std::string {
  return repr_->at("method");
}

auto InitializeRequest::method(std::string method) -> InitializeRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InitializeRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InitializeRequest::id(long id) -> InitializeRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto InitializeRequest::id(std::string id) -> InitializeRequest& {
  (*repr_)["id"] = std::move(id);
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

auto InitializeResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InitializeResponse::id(long id) -> InitializeResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto InitializeResponse::id(std::string id) -> InitializeResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ShutdownRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ShutdownRequest::method(std::string method) -> ShutdownRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShutdownRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShutdownRequest::id(long id) -> ShutdownRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShutdownRequest::id(std::string id) -> ShutdownRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ShutdownResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShutdownResponse::id(long id) -> ShutdownResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShutdownResponse::id(std::string id) -> ShutdownResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ShowMessageRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ShowMessageRequest::method(std::string method) -> ShowMessageRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowMessageRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShowMessageRequest::id(long id) -> ShowMessageRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowMessageRequest::id(std::string id) -> ShowMessageRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ShowMessageResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShowMessageResponse::id(long id) -> ShowMessageResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowMessageResponse::id(std::string id) -> ShowMessageResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WillSaveTextDocumentWaitUntilRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WillSaveTextDocumentWaitUntilRequest::method(std::string method)
    -> WillSaveTextDocumentWaitUntilRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillSaveTextDocumentWaitUntilRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillSaveTextDocumentWaitUntilRequest::id(long id)
    -> WillSaveTextDocumentWaitUntilRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillSaveTextDocumentWaitUntilRequest::id(std::string id)
    -> WillSaveTextDocumentWaitUntilRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WillSaveTextDocumentWaitUntilResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto CompletionRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CompletionRequest::method(std::string method) -> CompletionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CompletionRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CompletionRequest::id(long id) -> CompletionRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CompletionRequest::id(std::string id) -> CompletionRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CompletionResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CompletionResponse::id(long id) -> CompletionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CompletionResponse::id(std::string id) -> CompletionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CompletionResolveRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CompletionResolveRequest::method(std::string method)
    -> CompletionResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CompletionResolveRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CompletionResolveRequest::id(long id) -> CompletionResolveRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CompletionResolveRequest::id(std::string id) -> CompletionResolveRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CompletionResolveResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto HoverRequest::method() const -> std::string { return repr_->at("method"); }

auto HoverRequest::method(std::string method) -> HoverRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto HoverRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto HoverRequest::id(long id) -> HoverRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto HoverRequest::id(std::string id) -> HoverRequest& {
  (*repr_)["id"] = std::move(id);
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

auto HoverResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto HoverResponse::id(long id) -> HoverResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto HoverResponse::id(std::string id) -> HoverResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto SignatureHelpRequest::method() const -> std::string {
  return repr_->at("method");
}

auto SignatureHelpRequest::method(std::string method) -> SignatureHelpRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SignatureHelpRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SignatureHelpRequest::id(long id) -> SignatureHelpRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto SignatureHelpRequest::id(std::string id) -> SignatureHelpRequest& {
  (*repr_)["id"] = std::move(id);
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

auto SignatureHelpResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SignatureHelpResponse::id(long id) -> SignatureHelpResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto SignatureHelpResponse::id(std::string id) -> SignatureHelpResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DefinitionRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DefinitionRequest::method(std::string method) -> DefinitionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DefinitionRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DefinitionRequest::id(long id) -> DefinitionRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DefinitionRequest::id(std::string id) -> DefinitionRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DefinitionResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DefinitionResponse::id(long id) -> DefinitionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DefinitionResponse::id(std::string id) -> DefinitionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ReferencesRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ReferencesRequest::method(std::string method) -> ReferencesRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ReferencesRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ReferencesRequest::id(long id) -> ReferencesRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ReferencesRequest::id(std::string id) -> ReferencesRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ReferencesResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ReferencesResponse::id(long id) -> ReferencesResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ReferencesResponse::id(std::string id) -> ReferencesResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentHighlightRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentHighlightRequest::method(std::string method)
    -> DocumentHighlightRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentHighlightRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentHighlightRequest::id(long id) -> DocumentHighlightRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentHighlightRequest::id(std::string id) -> DocumentHighlightRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentHighlightResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DocumentSymbolRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentSymbolRequest::method(std::string method)
    -> DocumentSymbolRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentSymbolRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentSymbolRequest::id(long id) -> DocumentSymbolRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentSymbolRequest::id(std::string id) -> DocumentSymbolRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentSymbolResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentSymbolResponse::id(long id) -> DocumentSymbolResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentSymbolResponse::id(std::string id) -> DocumentSymbolResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeActionRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CodeActionRequest::method(std::string method) -> CodeActionRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeActionRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeActionRequest::id(long id) -> CodeActionRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeActionRequest::id(std::string id) -> CodeActionRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CodeActionResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeActionResponse::id(long id) -> CodeActionResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeActionResponse::id(std::string id) -> CodeActionResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeActionResolveRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CodeActionResolveRequest::method(std::string method)
    -> CodeActionResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeActionResolveRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeActionResolveRequest::id(long id) -> CodeActionResolveRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeActionResolveRequest::id(std::string id) -> CodeActionResolveRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CodeActionResolveResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto WorkspaceSymbolRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WorkspaceSymbolRequest::method(std::string method)
    -> WorkspaceSymbolRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceSymbolRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkspaceSymbolRequest::id(long id) -> WorkspaceSymbolRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceSymbolRequest::id(std::string id) -> WorkspaceSymbolRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WorkspaceSymbolResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkspaceSymbolResponse::id(long id) -> WorkspaceSymbolResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceSymbolResponse::id(std::string id) -> WorkspaceSymbolResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto WorkspaceSymbolResolveRequest::method() const -> std::string {
  return repr_->at("method");
}

auto WorkspaceSymbolResolveRequest::method(std::string method)
    -> WorkspaceSymbolResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkspaceSymbolResolveRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkspaceSymbolResolveRequest::id(long id)
    -> WorkspaceSymbolResolveRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkspaceSymbolResolveRequest::id(std::string id)
    -> WorkspaceSymbolResolveRequest& {
  (*repr_)["id"] = std::move(id);
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

auto WorkspaceSymbolResolveResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto CodeLensRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CodeLensRequest::method(std::string method) -> CodeLensRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeLensRequest::id(long id) -> CodeLensRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensRequest::id(std::string id) -> CodeLensRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CodeLensResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeLensResponse::id(long id) -> CodeLensResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensResponse::id(std::string id) -> CodeLensResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeLensResolveRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CodeLensResolveRequest::method(std::string method)
    -> CodeLensResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensResolveRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeLensResolveRequest::id(long id) -> CodeLensResolveRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensResolveRequest::id(std::string id) -> CodeLensResolveRequest& {
  (*repr_)["id"] = std::move(id);
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

auto CodeLensResolveResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeLensResolveResponse::id(long id) -> CodeLensResolveResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensResolveResponse::id(std::string id) -> CodeLensResolveResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeLensRefreshRequest::method() const -> std::string {
  return repr_->at("method");
}

auto CodeLensRefreshRequest::method(std::string method)
    -> CodeLensRefreshRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CodeLensRefreshRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeLensRefreshRequest::id(long id) -> CodeLensRefreshRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensRefreshRequest::id(std::string id) -> CodeLensRefreshRequest& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto CodeLensRefreshResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CodeLensRefreshResponse::id(long id) -> CodeLensRefreshResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto CodeLensRefreshResponse::id(std::string id) -> CodeLensRefreshResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentLinkRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentLinkRequest::method(std::string method) -> DocumentLinkRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentLinkRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentLinkRequest::id(long id) -> DocumentLinkRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentLinkRequest::id(std::string id) -> DocumentLinkRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentLinkResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentLinkResponse::id(long id) -> DocumentLinkResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentLinkResponse::id(std::string id) -> DocumentLinkResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DocumentLinkResolveRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentLinkResolveRequest::method(std::string method)
    -> DocumentLinkResolveRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentLinkResolveRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentLinkResolveRequest::id(long id) -> DocumentLinkResolveRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentLinkResolveRequest::id(std::string id)
    -> DocumentLinkResolveRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentLinkResolveResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DocumentFormattingRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentFormattingRequest::method(std::string method)
    -> DocumentFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentFormattingRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentFormattingRequest::id(long id) -> DocumentFormattingRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentFormattingRequest::id(std::string id)
    -> DocumentFormattingRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentFormattingResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DocumentRangeFormattingRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentRangeFormattingRequest::method(std::string method)
    -> DocumentRangeFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentRangeFormattingRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentRangeFormattingRequest::id(long id)
    -> DocumentRangeFormattingRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentRangeFormattingRequest::id(std::string id)
    -> DocumentRangeFormattingRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentRangeFormattingResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DocumentRangesFormattingRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentRangesFormattingRequest::method(std::string method)
    -> DocumentRangesFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentRangesFormattingRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentRangesFormattingRequest::id(long id)
    -> DocumentRangesFormattingRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentRangesFormattingRequest::id(std::string id)
    -> DocumentRangesFormattingRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentRangesFormattingResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto DocumentOnTypeFormattingRequest::method() const -> std::string {
  return repr_->at("method");
}

auto DocumentOnTypeFormattingRequest::method(std::string method)
    -> DocumentOnTypeFormattingRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DocumentOnTypeFormattingRequest::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DocumentOnTypeFormattingRequest::id(long id)
    -> DocumentOnTypeFormattingRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto DocumentOnTypeFormattingRequest::id(std::string id)
    -> DocumentOnTypeFormattingRequest& {
  (*repr_)["id"] = std::move(id);
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

auto DocumentOnTypeFormattingResponse::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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

auto RenameRequest::method() const -> std::string {
  return repr_->at("method");
}

auto RenameRequest::method(std::string method) -> RenameRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto RenameRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto RenameRequest::id(long id) -> RenameRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto RenameRequest::id(std::string id) -> RenameRequest& {
  (*repr_)["id"] = std::move(id);
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

auto RenameResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto RenameResponse::id(long id) -> RenameResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto RenameResponse::id(std::string id) -> RenameResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto PrepareRenameRequest::method() const -> std::string {
  return repr_->at("method");
}

auto PrepareRenameRequest::method(std::string method) -> PrepareRenameRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto PrepareRenameRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto PrepareRenameRequest::id(long id) -> PrepareRenameRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto PrepareRenameRequest::id(std::string id) -> PrepareRenameRequest& {
  (*repr_)["id"] = std::move(id);
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

auto PrepareRenameResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto PrepareRenameResponse::id(long id) -> PrepareRenameResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto PrepareRenameResponse::id(std::string id) -> PrepareRenameResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ExecuteCommandRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ExecuteCommandRequest::method(std::string method)
    -> ExecuteCommandRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ExecuteCommandRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ExecuteCommandRequest::id(long id) -> ExecuteCommandRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ExecuteCommandRequest::id(std::string id) -> ExecuteCommandRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ExecuteCommandResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ExecuteCommandResponse::id(long id) -> ExecuteCommandResponse& {
  (*repr_)["id"] = id;
  return *this;
}

auto ExecuteCommandResponse::id(std::string id) -> ExecuteCommandResponse& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto ApplyWorkspaceEditRequest::method() const -> std::string {
  return repr_->at("method");
}

auto ApplyWorkspaceEditRequest::method(std::string method)
    -> ApplyWorkspaceEditRequest& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ApplyWorkspaceEditRequest::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ApplyWorkspaceEditRequest::id(long id) -> ApplyWorkspaceEditRequest& {
  (*repr_)["id"] = id;
  return *this;
}

auto ApplyWorkspaceEditRequest::id(std::string id)
    -> ApplyWorkspaceEditRequest& {
  (*repr_)["id"] = std::move(id);
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

auto ApplyWorkspaceEditResponse::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
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
}  // namespace cxx::lsp
