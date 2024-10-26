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

auto ConfigurationResponse::result() const -> Vector<LSPAny> {
  lsp_runtime_error("ConfigurationResponse::result() - not implemented yet");
}

auto ConfigurationResponse::result(Vector<LSPAny> result)
    -> ConfigurationResponse& {
  lsp_runtime_error("ConfigurationResponse::result() - not implemented yet");
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

auto DocumentColorResponse::result() const -> Vector<ColorInformation> {
  lsp_runtime_error("DocumentColorResponse::result() - not implemented yet");
}

auto DocumentColorResponse::result(Vector<ColorInformation> result)
    -> DocumentColorResponse& {
  lsp_runtime_error("DocumentColorResponse::result() - not implemented yet");
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

auto FoldingRangeRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto FoldingRangeRefreshResponse::result(std::nullptr_t result)
    -> FoldingRangeRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto WorkDoneProgressCreateResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto WorkDoneProgressCreateResponse::result(std::nullptr_t result)
    -> WorkDoneProgressCreateResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto SemanticTokensRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto SemanticTokensRefreshResponse::result(std::nullptr_t result)
    -> SemanticTokensRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto ShowDocumentResponse::result() const -> ShowDocumentResult {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return ShowDocumentResult(repr_->at("result"));  // reference
}

auto ShowDocumentResponse::result(ShowDocumentResult result)
    -> ShowDocumentResponse& {
  lsp_runtime_error("ShowDocumentResponse::result() - not implemented yet");
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

auto MonikerResponse::result() const
    -> std::variant<Vector<Moniker>, std::nullptr_t> {
  lsp_runtime_error("MonikerResponse::result() - not implemented yet");
}

auto MonikerResponse::result(
    std::variant<Vector<Moniker>, std::nullptr_t> result) -> MonikerResponse& {
  lsp_runtime_error("MonikerResponse::result() - not implemented yet");
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

auto InlineValueRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto InlineValueRefreshResponse::result(std::nullptr_t result)
    -> InlineValueRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto InlayHintResolveResponse::result() const -> InlayHint {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return InlayHint(repr_->at("result"));  // reference
}

auto InlayHintResolveResponse::result(InlayHint result)
    -> InlayHintResolveResponse& {
  lsp_runtime_error("InlayHintResolveResponse::result() - not implemented yet");
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

auto InlayHintRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto InlayHintRefreshResponse::result(std::nullptr_t result)
    -> InlayHintRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto DiagnosticRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto DiagnosticRefreshResponse::result(std::nullptr_t result)
    -> DiagnosticRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto TextDocumentContentRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto TextDocumentContentRefreshResponse::result(std::nullptr_t result)
    -> TextDocumentContentRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto RegistrationResponse::result() const -> std::nullptr_t { return nullptr; }

auto RegistrationResponse::result(std::nullptr_t result)
    -> RegistrationResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto UnregistrationResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto UnregistrationResponse::result(std::nullptr_t result)
    -> UnregistrationResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto InitializeResponse::result() const -> InitializeResult {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return InitializeResult(repr_->at("result"));  // reference
}

auto InitializeResponse::result(InitializeResult result)
    -> InitializeResponse& {
  lsp_runtime_error("InitializeResponse::result() - not implemented yet");
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

auto ShutdownResponse::result() const -> std::nullptr_t { return nullptr; }

auto ShutdownResponse::result(std::nullptr_t result) -> ShutdownResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto HoverResponse::result() const -> std::variant<Hover, std::nullptr_t> {
  lsp_runtime_error("HoverResponse::result() - not implemented yet");
}

auto HoverResponse::result(std::variant<Hover, std::nullptr_t> result)
    -> HoverResponse& {
  lsp_runtime_error("HoverResponse::result() - not implemented yet");
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

auto CodeLensResolveResponse::result() const -> CodeLens {
  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;
  return CodeLens(repr_->at("result"));  // reference
}

auto CodeLensResolveResponse::result(CodeLens result)
    -> CodeLensResolveResponse& {
  lsp_runtime_error("CodeLensResolveResponse::result() - not implemented yet");
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

auto CodeLensRefreshResponse::result() const -> std::nullptr_t {
  return nullptr;
}

auto CodeLensRefreshResponse::result(std::nullptr_t result)
    -> CodeLensRefreshResponse& {
  (*repr_)["result"] = std::move(result);  // base
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

auto RenameResponse::result() const
    -> std::variant<WorkspaceEdit, std::nullptr_t> {
  lsp_runtime_error("RenameResponse::result() - not implemented yet");
}

auto RenameResponse::result(std::variant<WorkspaceEdit, std::nullptr_t> result)
    -> RenameResponse& {
  lsp_runtime_error("RenameResponse::result() - not implemented yet");
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

auto ExecuteCommandResponse::result() const
    -> std::variant<LSPAny, std::nullptr_t> {
  lsp_runtime_error("ExecuteCommandResponse::result() - not implemented yet");
}

auto ExecuteCommandResponse::result(std::variant<LSPAny, std::nullptr_t> result)
    -> ExecuteCommandResponse& {
  lsp_runtime_error("ExecuteCommandResponse::result() - not implemented yet");
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

auto DidChangeWorkspaceFoldersNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidChangeWorkspaceFoldersNotification::method(std::string method)
    -> DidChangeWorkspaceFoldersNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeWorkspaceFoldersNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidChangeWorkspaceFoldersNotification::id(long id)
    -> DidChangeWorkspaceFoldersNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidChangeWorkspaceFoldersNotification::id(std::string id)
    -> DidChangeWorkspaceFoldersNotification& {
  (*repr_)["id"] = std::move(id);
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

auto WorkDoneProgressCancelNotification::method() const -> std::string {
  return repr_->at("method");
}

auto WorkDoneProgressCancelNotification::method(std::string method)
    -> WorkDoneProgressCancelNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WorkDoneProgressCancelNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WorkDoneProgressCancelNotification::id(long id)
    -> WorkDoneProgressCancelNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto WorkDoneProgressCancelNotification::id(std::string id)
    -> WorkDoneProgressCancelNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidCreateFilesNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidCreateFilesNotification::method(std::string method)
    -> DidCreateFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidCreateFilesNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidCreateFilesNotification::id(long id) -> DidCreateFilesNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidCreateFilesNotification::id(std::string id)
    -> DidCreateFilesNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidRenameFilesNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidRenameFilesNotification::method(std::string method)
    -> DidRenameFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidRenameFilesNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidRenameFilesNotification::id(long id) -> DidRenameFilesNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidRenameFilesNotification::id(std::string id)
    -> DidRenameFilesNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidDeleteFilesNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidDeleteFilesNotification::method(std::string method)
    -> DidDeleteFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidDeleteFilesNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidDeleteFilesNotification::id(long id) -> DidDeleteFilesNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidDeleteFilesNotification::id(std::string id)
    -> DidDeleteFilesNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidOpenNotebookDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidOpenNotebookDocumentNotification::method(std::string method)
    -> DidOpenNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidOpenNotebookDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidOpenNotebookDocumentNotification::id(long id)
    -> DidOpenNotebookDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidOpenNotebookDocumentNotification::id(std::string id)
    -> DidOpenNotebookDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidChangeNotebookDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidChangeNotebookDocumentNotification::method(std::string method)
    -> DidChangeNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeNotebookDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidChangeNotebookDocumentNotification::id(long id)
    -> DidChangeNotebookDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidChangeNotebookDocumentNotification::id(std::string id)
    -> DidChangeNotebookDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidSaveNotebookDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidSaveNotebookDocumentNotification::method(std::string method)
    -> DidSaveNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidSaveNotebookDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidSaveNotebookDocumentNotification::id(long id)
    -> DidSaveNotebookDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidSaveNotebookDocumentNotification::id(std::string id)
    -> DidSaveNotebookDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidCloseNotebookDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidCloseNotebookDocumentNotification::method(std::string method)
    -> DidCloseNotebookDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidCloseNotebookDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidCloseNotebookDocumentNotification::id(long id)
    -> DidCloseNotebookDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidCloseNotebookDocumentNotification::id(std::string id)
    -> DidCloseNotebookDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto InitializedNotification::method() const -> std::string {
  return repr_->at("method");
}

auto InitializedNotification::method(std::string method)
    -> InitializedNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto InitializedNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto InitializedNotification::id(long id) -> InitializedNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto InitializedNotification::id(std::string id) -> InitializedNotification& {
  (*repr_)["id"] = std::move(id);
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

auto ExitNotification::method() const -> std::string {
  return repr_->at("method");
}

auto ExitNotification::method(std::string method) -> ExitNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ExitNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ExitNotification::id(long id) -> ExitNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto ExitNotification::id(std::string id) -> ExitNotification& {
  (*repr_)["id"] = std::move(id);
  return *this;
}

auto DidChangeConfigurationNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidChangeConfigurationNotification::method(std::string method)
    -> DidChangeConfigurationNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeConfigurationNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidChangeConfigurationNotification::id(long id)
    -> DidChangeConfigurationNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidChangeConfigurationNotification::id(std::string id)
    -> DidChangeConfigurationNotification& {
  (*repr_)["id"] = std::move(id);
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

auto ShowMessageNotification::method() const -> std::string {
  return repr_->at("method");
}

auto ShowMessageNotification::method(std::string method)
    -> ShowMessageNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ShowMessageNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ShowMessageNotification::id(long id) -> ShowMessageNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto ShowMessageNotification::id(std::string id) -> ShowMessageNotification& {
  (*repr_)["id"] = std::move(id);
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

auto LogMessageNotification::method() const -> std::string {
  return repr_->at("method");
}

auto LogMessageNotification::method(std::string method)
    -> LogMessageNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LogMessageNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto LogMessageNotification::id(long id) -> LogMessageNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto LogMessageNotification::id(std::string id) -> LogMessageNotification& {
  (*repr_)["id"] = std::move(id);
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

auto TelemetryEventNotification::method() const -> std::string {
  return repr_->at("method");
}

auto TelemetryEventNotification::method(std::string method)
    -> TelemetryEventNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto TelemetryEventNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto TelemetryEventNotification::id(long id) -> TelemetryEventNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto TelemetryEventNotification::id(std::string id)
    -> TelemetryEventNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidOpenTextDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidOpenTextDocumentNotification::method(std::string method)
    -> DidOpenTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidOpenTextDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidOpenTextDocumentNotification::id(long id)
    -> DidOpenTextDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidOpenTextDocumentNotification::id(std::string id)
    -> DidOpenTextDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidChangeTextDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidChangeTextDocumentNotification::method(std::string method)
    -> DidChangeTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeTextDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidChangeTextDocumentNotification::id(long id)
    -> DidChangeTextDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidChangeTextDocumentNotification::id(std::string id)
    -> DidChangeTextDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidCloseTextDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidCloseTextDocumentNotification::method(std::string method)
    -> DidCloseTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidCloseTextDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidCloseTextDocumentNotification::id(long id)
    -> DidCloseTextDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidCloseTextDocumentNotification::id(std::string id)
    -> DidCloseTextDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidSaveTextDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidSaveTextDocumentNotification::method(std::string method)
    -> DidSaveTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidSaveTextDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidSaveTextDocumentNotification::id(long id)
    -> DidSaveTextDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidSaveTextDocumentNotification::id(std::string id)
    -> DidSaveTextDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto WillSaveTextDocumentNotification::method() const -> std::string {
  return repr_->at("method");
}

auto WillSaveTextDocumentNotification::method(std::string method)
    -> WillSaveTextDocumentNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto WillSaveTextDocumentNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto WillSaveTextDocumentNotification::id(long id)
    -> WillSaveTextDocumentNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto WillSaveTextDocumentNotification::id(std::string id)
    -> WillSaveTextDocumentNotification& {
  (*repr_)["id"] = std::move(id);
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

auto DidChangeWatchedFilesNotification::method() const -> std::string {
  return repr_->at("method");
}

auto DidChangeWatchedFilesNotification::method(std::string method)
    -> DidChangeWatchedFilesNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto DidChangeWatchedFilesNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto DidChangeWatchedFilesNotification::id(long id)
    -> DidChangeWatchedFilesNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto DidChangeWatchedFilesNotification::id(std::string id)
    -> DidChangeWatchedFilesNotification& {
  (*repr_)["id"] = std::move(id);
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

auto PublishDiagnosticsNotification::method() const -> std::string {
  return repr_->at("method");
}

auto PublishDiagnosticsNotification::method(std::string method)
    -> PublishDiagnosticsNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto PublishDiagnosticsNotification::id() const
    -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto PublishDiagnosticsNotification::id(long id)
    -> PublishDiagnosticsNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto PublishDiagnosticsNotification::id(std::string id)
    -> PublishDiagnosticsNotification& {
  (*repr_)["id"] = std::move(id);
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

auto SetTraceNotification::method() const -> std::string {
  return repr_->at("method");
}

auto SetTraceNotification::method(std::string method) -> SetTraceNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto SetTraceNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto SetTraceNotification::id(long id) -> SetTraceNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto SetTraceNotification::id(std::string id) -> SetTraceNotification& {
  (*repr_)["id"] = std::move(id);
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

auto LogTraceNotification::method() const -> std::string {
  return repr_->at("method");
}

auto LogTraceNotification::method(std::string method) -> LogTraceNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto LogTraceNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto LogTraceNotification::id(long id) -> LogTraceNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto LogTraceNotification::id(std::string id) -> LogTraceNotification& {
  (*repr_)["id"] = std::move(id);
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

auto CancelNotification::method() const -> std::string {
  return repr_->at("method");
}

auto CancelNotification::method(std::string method) -> CancelNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto CancelNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto CancelNotification::id(long id) -> CancelNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto CancelNotification::id(std::string id) -> CancelNotification& {
  (*repr_)["id"] = std::move(id);
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

auto ProgressNotification::method() const -> std::string {
  return repr_->at("method");
}

auto ProgressNotification::method(std::string method) -> ProgressNotification& {
  (*repr_)["method"] = std::move(method);
  return *this;
}

auto ProgressNotification::id() const -> std::variant<long, std::string> {
  const auto& id = repr_->at("id");
  if (id.is_string()) return id.get<std::string>();
  return id.get<long>();
}

auto ProgressNotification::id(long id) -> ProgressNotification& {
  (*repr_)["id"] = id;
  return *this;
}

auto ProgressNotification::id(std::string id) -> ProgressNotification& {
  (*repr_)["id"] = std::move(id);
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
