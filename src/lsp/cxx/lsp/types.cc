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

#include <cxx/lsp/types.h>

namespace cxx::lsp {

ImplementationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto ImplementationParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto ImplementationParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto ImplementationParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ImplementationParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ImplementationParams::textDocument(TextDocumentIdentifier textDocument)
    -> ImplementationParams& {
  return *this;
}

auto ImplementationParams::position(Position position)
    -> ImplementationParams& {
  return *this;
}

auto ImplementationParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> ImplementationParams& {
  return *this;
}

auto ImplementationParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> ImplementationParams& {
  return *this;
}

Location::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto Location::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Location::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto Location::uri(std::string uri) -> Location& { return *this; }

auto Location::range(Range range) -> Location& { return *this; }

ImplementationRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto ImplementationRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ImplementationRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ImplementationRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> ImplementationRegistrationOptions& {
  return *this;
}

auto ImplementationRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> ImplementationRegistrationOptions& {
  return *this;
}

auto ImplementationRegistrationOptions::id(std::optional<std::string> id)
    -> ImplementationRegistrationOptions& {
  return *this;
}

TypeDefinitionParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto TypeDefinitionParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto TypeDefinitionParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto TypeDefinitionParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionParams::textDocument(TextDocumentIdentifier textDocument)
    -> TypeDefinitionParams& {
  return *this;
}

auto TypeDefinitionParams::position(Position position)
    -> TypeDefinitionParams& {
  return *this;
}

auto TypeDefinitionParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> TypeDefinitionParams& {
  return *this;
}

auto TypeDefinitionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> TypeDefinitionParams& {
  return *this;
}

TypeDefinitionRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto TypeDefinitionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeDefinitionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TypeDefinitionRegistrationOptions& {
  return *this;
}

auto TypeDefinitionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> TypeDefinitionRegistrationOptions& {
  return *this;
}

auto TypeDefinitionRegistrationOptions::id(std::optional<std::string> id)
    -> TypeDefinitionRegistrationOptions& {
  return *this;
}

WorkspaceFolder::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("name")) return false;
  return true;
}

auto WorkspaceFolder::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFolder::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFolder::uri(std::string uri) -> WorkspaceFolder& { return *this; }

auto WorkspaceFolder::name(std::string name) -> WorkspaceFolder& {
  return *this;
}

DidChangeWorkspaceFoldersParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("event")) return false;
  return true;
}

auto DidChangeWorkspaceFoldersParams::event() const
    -> WorkspaceFoldersChangeEvent {
  const auto& value = repr_["event"];

  return WorkspaceFoldersChangeEvent(value);
}

auto DidChangeWorkspaceFoldersParams::event(WorkspaceFoldersChangeEvent event)
    -> DidChangeWorkspaceFoldersParams& {
  return *this;
}

ConfigurationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto ConfigurationParams::items() const -> Vector<ConfigurationItem> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<ConfigurationItem>(value);
}

auto ConfigurationParams::items(Vector<ConfigurationItem> items)
    -> ConfigurationParams& {
  return *this;
}

DocumentColorParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DocumentColorParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentColorParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentColorParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentColorParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentColorParams& {
  return *this;
}

auto DocumentColorParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentColorParams& {
  return *this;
}

auto DocumentColorParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DocumentColorParams& {
  return *this;
}

ColorInformation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("color")) return false;
  return true;
}

auto ColorInformation::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto ColorInformation::color() const -> Color {
  const auto& value = repr_["color"];

  return Color(value);
}

auto ColorInformation::range(Range range) -> ColorInformation& { return *this; }

auto ColorInformation::color(Color color) -> ColorInformation& { return *this; }

DocumentColorRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DocumentColorRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentColorRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentColorRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentColorRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentColorRegistrationOptions& {
  return *this;
}

auto DocumentColorRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentColorRegistrationOptions& {
  return *this;
}

auto DocumentColorRegistrationOptions::id(std::optional<std::string> id)
    -> DocumentColorRegistrationOptions& {
  return *this;
}

ColorPresentationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("color")) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto ColorPresentationParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto ColorPresentationParams::color() const -> Color {
  const auto& value = repr_["color"];

  return Color(value);
}

auto ColorPresentationParams::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto ColorPresentationParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ColorPresentationParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ColorPresentationParams::textDocument(TextDocumentIdentifier textDocument)
    -> ColorPresentationParams& {
  return *this;
}

auto ColorPresentationParams::color(Color color) -> ColorPresentationParams& {
  return *this;
}

auto ColorPresentationParams::range(Range range) -> ColorPresentationParams& {
  return *this;
}

auto ColorPresentationParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> ColorPresentationParams& {
  return *this;
}

auto ColorPresentationParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> ColorPresentationParams& {
  return *this;
}

ColorPresentation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("label")) return false;
  return true;
}

auto ColorPresentation::label() const -> std::string {
  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ColorPresentation::textEdit() const -> std::optional<TextEdit> {
  if (!repr_.contains("textEdit")) return std::nullopt;

  const auto& value = repr_["textEdit"];

  return TextEdit(value);
}

auto ColorPresentation::additionalTextEdits() const
    -> std::optional<Vector<TextEdit>> {
  if (!repr_.contains("additionalTextEdits")) return std::nullopt;

  const auto& value = repr_["additionalTextEdits"];

  assert(value.is_array());
  return Vector<TextEdit>(value);
}

auto ColorPresentation::label(std::string label) -> ColorPresentation& {
  return *this;
}

auto ColorPresentation::textEdit(std::optional<TextEdit> textEdit)
    -> ColorPresentation& {
  return *this;
}

auto ColorPresentation::additionalTextEdits(
    std::optional<Vector<TextEdit>> additionalTextEdits) -> ColorPresentation& {
  return *this;
}

WorkDoneProgressOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkDoneProgressOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkDoneProgressOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> WorkDoneProgressOptions& {
  return *this;
}

TextDocumentRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto TextDocumentRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TextDocumentRegistrationOptions& {
  return *this;
}

FoldingRangeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto FoldingRangeParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto FoldingRangeParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeParams::textDocument(TextDocumentIdentifier textDocument)
    -> FoldingRangeParams& {
  return *this;
}

auto FoldingRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> FoldingRangeParams& {
  return *this;
}

auto FoldingRangeParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> FoldingRangeParams& {
  return *this;
}

FoldingRange::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("startLine")) return false;
  if (!repr_.contains("endLine")) return false;
  return true;
}

auto FoldingRange::startLine() const -> long {
  const auto& value = repr_["startLine"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::startCharacter() const -> std::optional<long> {
  if (!repr_.contains("startCharacter")) return std::nullopt;

  const auto& value = repr_["startCharacter"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::endLine() const -> long {
  const auto& value = repr_["endLine"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::endCharacter() const -> std::optional<long> {
  if (!repr_.contains("endCharacter")) return std::nullopt;

  const auto& value = repr_["endCharacter"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::kind() const -> std::optional<FoldingRangeKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  lsp_runtime_error("FoldingRange::kind: not implement yet");
}

auto FoldingRange::collapsedText() const -> std::optional<std::string> {
  if (!repr_.contains("collapsedText")) return std::nullopt;

  const auto& value = repr_["collapsedText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FoldingRange::startLine(long startLine) -> FoldingRange& { return *this; }

auto FoldingRange::startCharacter(std::optional<long> startCharacter)
    -> FoldingRange& {
  return *this;
}

auto FoldingRange::endLine(long endLine) -> FoldingRange& { return *this; }

auto FoldingRange::endCharacter(std::optional<long> endCharacter)
    -> FoldingRange& {
  return *this;
}

auto FoldingRange::kind(std::optional<FoldingRangeKind> kind) -> FoldingRange& {
  return *this;
}

auto FoldingRange::collapsedText(std::optional<std::string> collapsedText)
    -> FoldingRange& {
  return *this;
}

FoldingRangeRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto FoldingRangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FoldingRangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> FoldingRangeRegistrationOptions& {
  return *this;
}

auto FoldingRangeRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> FoldingRangeRegistrationOptions& {
  return *this;
}

auto FoldingRangeRegistrationOptions::id(std::optional<std::string> id)
    -> FoldingRangeRegistrationOptions& {
  return *this;
}

DeclarationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto DeclarationParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DeclarationParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto DeclarationParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DeclarationParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DeclarationParams::textDocument(TextDocumentIdentifier textDocument)
    -> DeclarationParams& {
  return *this;
}

auto DeclarationParams::position(Position position) -> DeclarationParams& {
  return *this;
}

auto DeclarationParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DeclarationParams& {
  return *this;
}

auto DeclarationParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DeclarationParams& {
  return *this;
}

DeclarationRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DeclarationRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DeclarationRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeclarationRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DeclarationRegistrationOptions& {
  return *this;
}

auto DeclarationRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DeclarationRegistrationOptions& {
  return *this;
}

auto DeclarationRegistrationOptions::id(std::optional<std::string> id)
    -> DeclarationRegistrationOptions& {
  return *this;
}

SelectionRangeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("positions")) return false;
  return true;
}

auto SelectionRangeParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SelectionRangeParams::positions() const -> Vector<Position> {
  const auto& value = repr_["positions"];

  assert(value.is_array());
  return Vector<Position>(value);
}

auto SelectionRangeParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeParams::textDocument(TextDocumentIdentifier textDocument)
    -> SelectionRangeParams& {
  return *this;
}

auto SelectionRangeParams::positions(Vector<Position> positions)
    -> SelectionRangeParams& {
  return *this;
}

auto SelectionRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SelectionRangeParams& {
  return *this;
}

auto SelectionRangeParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> SelectionRangeParams& {
  return *this;
}

SelectionRange::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto SelectionRange::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto SelectionRange::parent() const -> std::optional<SelectionRange> {
  if (!repr_.contains("parent")) return std::nullopt;

  const auto& value = repr_["parent"];

  return SelectionRange(value);
}

auto SelectionRange::range(Range range) -> SelectionRange& { return *this; }

auto SelectionRange::parent(std::optional<SelectionRange> parent)
    -> SelectionRange& {
  return *this;
}

SelectionRangeRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto SelectionRangeRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SelectionRangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SelectionRangeRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> SelectionRangeRegistrationOptions& {
  return *this;
}

auto SelectionRangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> SelectionRangeRegistrationOptions& {
  return *this;
}

auto SelectionRangeRegistrationOptions::id(std::optional<std::string> id)
    -> SelectionRangeRegistrationOptions& {
  return *this;
}

WorkDoneProgressCreateParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("token")) return false;
  return true;
}

auto WorkDoneProgressCreateParams::token() const -> ProgressToken {
  const auto& value = repr_["token"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressCreateParams::token(ProgressToken token)
    -> WorkDoneProgressCreateParams& {
  return *this;
}

WorkDoneProgressCancelParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("token")) return false;
  return true;
}

auto WorkDoneProgressCancelParams::token() const -> ProgressToken {
  const auto& value = repr_["token"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressCancelParams::token(ProgressToken token)
    -> WorkDoneProgressCancelParams& {
  return *this;
}

CallHierarchyPrepareParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto CallHierarchyPrepareParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CallHierarchyPrepareParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto CallHierarchyPrepareParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyPrepareParams::textDocument(
    TextDocumentIdentifier textDocument) -> CallHierarchyPrepareParams& {
  return *this;
}

auto CallHierarchyPrepareParams::position(Position position)
    -> CallHierarchyPrepareParams& {
  return *this;
}

auto CallHierarchyPrepareParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> CallHierarchyPrepareParams& {
  return *this;
}

CallHierarchyItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("name")) return false;
  if (!repr_.contains("kind")) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("selectionRange")) return false;
  return true;
}

auto CallHierarchyItem::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyItem::kind() const -> SymbolKind {
  const auto& value = repr_["kind"];

  return SymbolKind(value);
}

auto CallHierarchyItem::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto CallHierarchyItem::detail() const -> std::optional<std::string> {
  if (!repr_.contains("detail")) return std::nullopt;

  const auto& value = repr_["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyItem::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyItem::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto CallHierarchyItem::selectionRange() const -> Range {
  const auto& value = repr_["selectionRange"];

  return Range(value);
}

auto CallHierarchyItem::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CallHierarchyItem::name(std::string name) -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::kind(SymbolKind kind) -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::tags(std::optional<Vector<SymbolTag>> tags)
    -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::detail(std::optional<std::string> detail)
    -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::uri(std::string uri) -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::range(Range range) -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::selectionRange(Range selectionRange)
    -> CallHierarchyItem& {
  return *this;
}

auto CallHierarchyItem::data(std::optional<LSPAny> data) -> CallHierarchyItem& {
  return *this;
}

CallHierarchyRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto CallHierarchyRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CallHierarchyRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CallHierarchyRegistrationOptions& {
  return *this;
}

auto CallHierarchyRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CallHierarchyRegistrationOptions& {
  return *this;
}

auto CallHierarchyRegistrationOptions::id(std::optional<std::string> id)
    -> CallHierarchyRegistrationOptions& {
  return *this;
}

CallHierarchyIncomingCallsParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("item")) return false;
  return true;
}

auto CallHierarchyIncomingCallsParams::item() const -> CallHierarchyItem {
  const auto& value = repr_["item"];

  return CallHierarchyItem(value);
}

auto CallHierarchyIncomingCallsParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyIncomingCallsParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyIncomingCallsParams::item(CallHierarchyItem item)
    -> CallHierarchyIncomingCallsParams& {
  return *this;
}

auto CallHierarchyIncomingCallsParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> CallHierarchyIncomingCallsParams& {
  return *this;
}

auto CallHierarchyIncomingCallsParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> CallHierarchyIncomingCallsParams& {
  return *this;
}

CallHierarchyIncomingCall::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("from")) return false;
  if (!repr_.contains("fromRanges")) return false;
  return true;
}

auto CallHierarchyIncomingCall::from() const -> CallHierarchyItem {
  const auto& value = repr_["from"];

  return CallHierarchyItem(value);
}

auto CallHierarchyIncomingCall::fromRanges() const -> Vector<Range> {
  const auto& value = repr_["fromRanges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto CallHierarchyIncomingCall::from(CallHierarchyItem from)
    -> CallHierarchyIncomingCall& {
  return *this;
}

auto CallHierarchyIncomingCall::fromRanges(Vector<Range> fromRanges)
    -> CallHierarchyIncomingCall& {
  return *this;
}

CallHierarchyOutgoingCallsParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("item")) return false;
  return true;
}

auto CallHierarchyOutgoingCallsParams::item() const -> CallHierarchyItem {
  const auto& value = repr_["item"];

  return CallHierarchyItem(value);
}

auto CallHierarchyOutgoingCallsParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyOutgoingCallsParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyOutgoingCallsParams::item(CallHierarchyItem item)
    -> CallHierarchyOutgoingCallsParams& {
  return *this;
}

auto CallHierarchyOutgoingCallsParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> CallHierarchyOutgoingCallsParams& {
  return *this;
}

auto CallHierarchyOutgoingCallsParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> CallHierarchyOutgoingCallsParams& {
  return *this;
}

CallHierarchyOutgoingCall::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("to")) return false;
  if (!repr_.contains("fromRanges")) return false;
  return true;
}

auto CallHierarchyOutgoingCall::to() const -> CallHierarchyItem {
  const auto& value = repr_["to"];

  return CallHierarchyItem(value);
}

auto CallHierarchyOutgoingCall::fromRanges() const -> Vector<Range> {
  const auto& value = repr_["fromRanges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto CallHierarchyOutgoingCall::to(CallHierarchyItem to)
    -> CallHierarchyOutgoingCall& {
  return *this;
}

auto CallHierarchyOutgoingCall::fromRanges(Vector<Range> fromRanges)
    -> CallHierarchyOutgoingCall& {
  return *this;
}

SemanticTokensParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto SemanticTokensParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SemanticTokensParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensParams::textDocument(TextDocumentIdentifier textDocument)
    -> SemanticTokensParams& {
  return *this;
}

auto SemanticTokensParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SemanticTokensParams& {
  return *this;
}

auto SemanticTokensParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> SemanticTokensParams& {
  return *this;
}

SemanticTokens::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("data")) return false;
  return true;
}

auto SemanticTokens::resultId() const -> std::optional<std::string> {
  if (!repr_.contains("resultId")) return std::nullopt;

  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokens::data() const -> Vector<long> {
  const auto& value = repr_["data"];

  assert(value.is_array());
  return Vector<long>(value);
}

auto SemanticTokens::resultId(std::optional<std::string> resultId)
    -> SemanticTokens& {
  return *this;
}

auto SemanticTokens::data(Vector<long> data) -> SemanticTokens& {
  return *this;
}

SemanticTokensPartialResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("data")) return false;
  return true;
}

auto SemanticTokensPartialResult::data() const -> Vector<long> {
  const auto& value = repr_["data"];

  assert(value.is_array());
  return Vector<long>(value);
}

auto SemanticTokensPartialResult::data(Vector<long> data)
    -> SemanticTokensPartialResult& {
  return *this;
}

SemanticTokensRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  if (!repr_.contains("legend")) return false;
  return true;
}

auto SemanticTokensRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRegistrationOptions::legend() const -> SemanticTokensLegend {
  const auto& value = repr_["legend"];

  return SemanticTokensLegend(value);
}

auto SemanticTokensRegistrationOptions::range() const
    -> std::optional<std::variant<std::monostate, bool, json>> {
  if (!repr_.contains("range")) return std::nullopt;

  const auto& value = repr_["range"];

  std::variant<std::monostate, bool, json> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRegistrationOptions::full() const -> std::optional<
    std::variant<std::monostate, bool, SemanticTokensFullDelta>> {
  if (!repr_.contains("full")) return std::nullopt;

  const auto& value = repr_["full"];

  std::variant<std::monostate, bool, SemanticTokensFullDelta> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokensRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> SemanticTokensRegistrationOptions& {
  return *this;
}

auto SemanticTokensRegistrationOptions::legend(SemanticTokensLegend legend)
    -> SemanticTokensRegistrationOptions& {
  return *this;
}

auto SemanticTokensRegistrationOptions::range(
    std::optional<std::variant<std::monostate, bool, json>> range)
    -> SemanticTokensRegistrationOptions& {
  return *this;
}

auto SemanticTokensRegistrationOptions::full(
    std::optional<std::variant<std::monostate, bool, SemanticTokensFullDelta>>
        full) -> SemanticTokensRegistrationOptions& {
  return *this;
}

auto SemanticTokensRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> SemanticTokensRegistrationOptions& {
  return *this;
}

auto SemanticTokensRegistrationOptions::id(std::optional<std::string> id)
    -> SemanticTokensRegistrationOptions& {
  return *this;
}

SemanticTokensDeltaParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("previousResultId")) return false;
  return true;
}

auto SemanticTokensDeltaParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SemanticTokensDeltaParams::previousResultId() const -> std::string {
  const auto& value = repr_["previousResultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokensDeltaParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensDeltaParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensDeltaParams::textDocument(
    TextDocumentIdentifier textDocument) -> SemanticTokensDeltaParams& {
  return *this;
}

auto SemanticTokensDeltaParams::previousResultId(std::string previousResultId)
    -> SemanticTokensDeltaParams& {
  return *this;
}

auto SemanticTokensDeltaParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SemanticTokensDeltaParams& {
  return *this;
}

auto SemanticTokensDeltaParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> SemanticTokensDeltaParams& {
  return *this;
}

SemanticTokensDelta::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("edits")) return false;
  return true;
}

auto SemanticTokensDelta::resultId() const -> std::optional<std::string> {
  if (!repr_.contains("resultId")) return std::nullopt;

  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokensDelta::edits() const -> Vector<SemanticTokensEdit> {
  const auto& value = repr_["edits"];

  assert(value.is_array());
  return Vector<SemanticTokensEdit>(value);
}

auto SemanticTokensDelta::resultId(std::optional<std::string> resultId)
    -> SemanticTokensDelta& {
  return *this;
}

auto SemanticTokensDelta::edits(Vector<SemanticTokensEdit> edits)
    -> SemanticTokensDelta& {
  return *this;
}

SemanticTokensDeltaPartialResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("edits")) return false;
  return true;
}

auto SemanticTokensDeltaPartialResult::edits() const
    -> Vector<SemanticTokensEdit> {
  const auto& value = repr_["edits"];

  assert(value.is_array());
  return Vector<SemanticTokensEdit>(value);
}

auto SemanticTokensDeltaPartialResult::edits(Vector<SemanticTokensEdit> edits)
    -> SemanticTokensDeltaPartialResult& {
  return *this;
}

SemanticTokensRangeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto SemanticTokensRangeParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SemanticTokensRangeParams::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto SemanticTokensRangeParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRangeParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRangeParams::textDocument(
    TextDocumentIdentifier textDocument) -> SemanticTokensRangeParams& {
  return *this;
}

auto SemanticTokensRangeParams::range(Range range)
    -> SemanticTokensRangeParams& {
  return *this;
}

auto SemanticTokensRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SemanticTokensRangeParams& {
  return *this;
}

auto SemanticTokensRangeParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> SemanticTokensRangeParams& {
  return *this;
}

ShowDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto ShowDocumentParams::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ShowDocumentParams::external() const -> std::optional<bool> {
  if (!repr_.contains("external")) return std::nullopt;

  const auto& value = repr_["external"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentParams::takeFocus() const -> std::optional<bool> {
  if (!repr_.contains("takeFocus")) return std::nullopt;

  const auto& value = repr_["takeFocus"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentParams::selection() const -> std::optional<Range> {
  if (!repr_.contains("selection")) return std::nullopt;

  const auto& value = repr_["selection"];

  return Range(value);
}

auto ShowDocumentParams::uri(std::string uri) -> ShowDocumentParams& {
  return *this;
}

auto ShowDocumentParams::external(std::optional<bool> external)
    -> ShowDocumentParams& {
  return *this;
}

auto ShowDocumentParams::takeFocus(std::optional<bool> takeFocus)
    -> ShowDocumentParams& {
  return *this;
}

auto ShowDocumentParams::selection(std::optional<Range> selection)
    -> ShowDocumentParams& {
  return *this;
}

ShowDocumentResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("success")) return false;
  return true;
}

auto ShowDocumentResult::success() const -> bool {
  const auto& value = repr_["success"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentResult::success(bool success) -> ShowDocumentResult& {
  return *this;
}

LinkedEditingRangeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto LinkedEditingRangeParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto LinkedEditingRangeParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto LinkedEditingRangeParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto LinkedEditingRangeParams::textDocument(TextDocumentIdentifier textDocument)
    -> LinkedEditingRangeParams& {
  return *this;
}

auto LinkedEditingRangeParams::position(Position position)
    -> LinkedEditingRangeParams& {
  return *this;
}

auto LinkedEditingRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> LinkedEditingRangeParams& {
  return *this;
}

LinkedEditingRanges::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("ranges")) return false;
  return true;
}

auto LinkedEditingRanges::ranges() const -> Vector<Range> {
  const auto& value = repr_["ranges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto LinkedEditingRanges::wordPattern() const -> std::optional<std::string> {
  if (!repr_.contains("wordPattern")) return std::nullopt;

  const auto& value = repr_["wordPattern"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LinkedEditingRanges::ranges(Vector<Range> ranges) -> LinkedEditingRanges& {
  return *this;
}

auto LinkedEditingRanges::wordPattern(std::optional<std::string> wordPattern)
    -> LinkedEditingRanges& {
  return *this;
}

LinkedEditingRangeRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto LinkedEditingRangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto LinkedEditingRangeRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto LinkedEditingRangeRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LinkedEditingRangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> LinkedEditingRangeRegistrationOptions& {
  return *this;
}

auto LinkedEditingRangeRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> LinkedEditingRangeRegistrationOptions& {
  return *this;
}

auto LinkedEditingRangeRegistrationOptions::id(std::optional<std::string> id)
    -> LinkedEditingRangeRegistrationOptions& {
  return *this;
}

CreateFilesParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("files")) return false;
  return true;
}

auto CreateFilesParams::files() const -> Vector<FileCreate> {
  const auto& value = repr_["files"];

  assert(value.is_array());
  return Vector<FileCreate>(value);
}

auto CreateFilesParams::files(Vector<FileCreate> files) -> CreateFilesParams& {
  return *this;
}

WorkspaceEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceEdit::changes() const
    -> std::optional<Map<std::string, Vector<TextEdit>>> {
  if (!repr_.contains("changes")) return std::nullopt;

  const auto& value = repr_["changes"];

  assert(value.is_object());
  return Map<std::string, Vector<TextEdit>>(value);
}

auto WorkspaceEdit::documentChanges() const
    -> std::optional<Vector<std::variant<std::monostate, TextDocumentEdit,
                                         CreateFile, RenameFile, DeleteFile>>> {
  if (!repr_.contains("documentChanges")) return std::nullopt;

  const auto& value = repr_["documentChanges"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, TextDocumentEdit, CreateFile,
                             RenameFile, DeleteFile>>(value);
}

auto WorkspaceEdit::changeAnnotations() const
    -> std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>> {
  if (!repr_.contains("changeAnnotations")) return std::nullopt;

  const auto& value = repr_["changeAnnotations"];

  assert(value.is_object());
  return Map<ChangeAnnotationIdentifier, ChangeAnnotation>(value);
}

auto WorkspaceEdit::changes(
    std::optional<Map<std::string, Vector<TextEdit>>> changes)
    -> WorkspaceEdit& {
  return *this;
}

auto WorkspaceEdit::documentChanges(
    std::optional<Vector<std::variant<std::monostate, TextDocumentEdit,
                                      CreateFile, RenameFile, DeleteFile>>>
        documentChanges) -> WorkspaceEdit& {
  return *this;
}

auto WorkspaceEdit::changeAnnotations(
    std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>>
        changeAnnotations) -> WorkspaceEdit& {
  return *this;
}

FileOperationRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("filters")) return false;
  return true;
}

auto FileOperationRegistrationOptions::filters() const
    -> Vector<FileOperationFilter> {
  const auto& value = repr_["filters"];

  assert(value.is_array());
  return Vector<FileOperationFilter>(value);
}

auto FileOperationRegistrationOptions::filters(
    Vector<FileOperationFilter> filters) -> FileOperationRegistrationOptions& {
  return *this;
}

RenameFilesParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("files")) return false;
  return true;
}

auto RenameFilesParams::files() const -> Vector<FileRename> {
  const auto& value = repr_["files"];

  assert(value.is_array());
  return Vector<FileRename>(value);
}

auto RenameFilesParams::files(Vector<FileRename> files) -> RenameFilesParams& {
  return *this;
}

DeleteFilesParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("files")) return false;
  return true;
}

auto DeleteFilesParams::files() const -> Vector<FileDelete> {
  const auto& value = repr_["files"];

  assert(value.is_array());
  return Vector<FileDelete>(value);
}

auto DeleteFilesParams::files(Vector<FileDelete> files) -> DeleteFilesParams& {
  return *this;
}

MonikerParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto MonikerParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto MonikerParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto MonikerParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerParams::partialResultToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerParams::textDocument(TextDocumentIdentifier textDocument)
    -> MonikerParams& {
  return *this;
}

auto MonikerParams::position(Position position) -> MonikerParams& {
  return *this;
}

auto MonikerParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> MonikerParams& {
  return *this;
}

auto MonikerParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> MonikerParams& {
  return *this;
}

Moniker::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("scheme")) return false;
  if (!repr_.contains("identifier")) return false;
  if (!repr_.contains("unique")) return false;
  return true;
}

auto Moniker::scheme() const -> std::string {
  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Moniker::identifier() const -> std::string {
  const auto& value = repr_["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Moniker::unique() const -> UniquenessLevel {
  const auto& value = repr_["unique"];

  lsp_runtime_error("Moniker::unique: not implement yet");
}

auto Moniker::kind() const -> std::optional<MonikerKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  lsp_runtime_error("Moniker::kind: not implement yet");
}

auto Moniker::scheme(std::string scheme) -> Moniker& { return *this; }

auto Moniker::identifier(std::string identifier) -> Moniker& { return *this; }

auto Moniker::unique(UniquenessLevel unique) -> Moniker& { return *this; }

auto Moniker::kind(std::optional<MonikerKind> kind) -> Moniker& {
  return *this;
}

MonikerRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto MonikerRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto MonikerRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> MonikerRegistrationOptions& {
  return *this;
}

auto MonikerRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> MonikerRegistrationOptions& {
  return *this;
}

TypeHierarchyPrepareParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto TypeHierarchyPrepareParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto TypeHierarchyPrepareParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto TypeHierarchyPrepareParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchyPrepareParams::textDocument(
    TextDocumentIdentifier textDocument) -> TypeHierarchyPrepareParams& {
  return *this;
}

auto TypeHierarchyPrepareParams::position(Position position)
    -> TypeHierarchyPrepareParams& {
  return *this;
}

auto TypeHierarchyPrepareParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> TypeHierarchyPrepareParams& {
  return *this;
}

TypeHierarchyItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("name")) return false;
  if (!repr_.contains("kind")) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("selectionRange")) return false;
  return true;
}

auto TypeHierarchyItem::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyItem::kind() const -> SymbolKind {
  const auto& value = repr_["kind"];

  return SymbolKind(value);
}

auto TypeHierarchyItem::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto TypeHierarchyItem::detail() const -> std::optional<std::string> {
  if (!repr_.contains("detail")) return std::nullopt;

  const auto& value = repr_["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyItem::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyItem::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto TypeHierarchyItem::selectionRange() const -> Range {
  const auto& value = repr_["selectionRange"];

  return Range(value);
}

auto TypeHierarchyItem::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto TypeHierarchyItem::name(std::string name) -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::kind(SymbolKind kind) -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::tags(std::optional<Vector<SymbolTag>> tags)
    -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::detail(std::optional<std::string> detail)
    -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::uri(std::string uri) -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::range(Range range) -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::selectionRange(Range selectionRange)
    -> TypeHierarchyItem& {
  return *this;
}

auto TypeHierarchyItem::data(std::optional<LSPAny> data) -> TypeHierarchyItem& {
  return *this;
}

TypeHierarchyRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto TypeHierarchyRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchyRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeHierarchyRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TypeHierarchyRegistrationOptions& {
  return *this;
}

auto TypeHierarchyRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> TypeHierarchyRegistrationOptions& {
  return *this;
}

auto TypeHierarchyRegistrationOptions::id(std::optional<std::string> id)
    -> TypeHierarchyRegistrationOptions& {
  return *this;
}

TypeHierarchySupertypesParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("item")) return false;
  return true;
}

auto TypeHierarchySupertypesParams::item() const -> TypeHierarchyItem {
  const auto& value = repr_["item"];

  return TypeHierarchyItem(value);
}

auto TypeHierarchySupertypesParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySupertypesParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySupertypesParams::item(TypeHierarchyItem item)
    -> TypeHierarchySupertypesParams& {
  return *this;
}

auto TypeHierarchySupertypesParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> TypeHierarchySupertypesParams& {
  return *this;
}

auto TypeHierarchySupertypesParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> TypeHierarchySupertypesParams& {
  return *this;
}

TypeHierarchySubtypesParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("item")) return false;
  return true;
}

auto TypeHierarchySubtypesParams::item() const -> TypeHierarchyItem {
  const auto& value = repr_["item"];

  return TypeHierarchyItem(value);
}

auto TypeHierarchySubtypesParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySubtypesParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySubtypesParams::item(TypeHierarchyItem item)
    -> TypeHierarchySubtypesParams& {
  return *this;
}

auto TypeHierarchySubtypesParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> TypeHierarchySubtypesParams& {
  return *this;
}

auto TypeHierarchySubtypesParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> TypeHierarchySubtypesParams& {
  return *this;
}

InlineValueParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("context")) return false;
  return true;
}

auto InlineValueParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto InlineValueParams::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto InlineValueParams::context() const -> InlineValueContext {
  const auto& value = repr_["context"];

  return InlineValueContext(value);
}

auto InlineValueParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InlineValueParams::textDocument(TextDocumentIdentifier textDocument)
    -> InlineValueParams& {
  return *this;
}

auto InlineValueParams::range(Range range) -> InlineValueParams& {
  return *this;
}

auto InlineValueParams::context(InlineValueContext context)
    -> InlineValueParams& {
  return *this;
}

auto InlineValueParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> InlineValueParams& {
  return *this;
}

InlineValueRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto InlineValueRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineValueRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> InlineValueRegistrationOptions& {
  return *this;
}

auto InlineValueRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> InlineValueRegistrationOptions& {
  return *this;
}

auto InlineValueRegistrationOptions::id(std::optional<std::string> id)
    -> InlineValueRegistrationOptions& {
  return *this;
}

InlayHintParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto InlayHintParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto InlayHintParams::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto InlayHintParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintParams::textDocument(TextDocumentIdentifier textDocument)
    -> InlayHintParams& {
  return *this;
}

auto InlayHintParams::range(Range range) -> InlayHintParams& { return *this; }

auto InlayHintParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> InlayHintParams& {
  return *this;
}

InlayHint::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("position")) return false;
  if (!repr_.contains("label")) return false;
  return true;
}

auto InlayHint::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto InlayHint::label() const
    -> std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>> {
  const auto& value = repr_["label"];

  std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHint::kind() const -> std::optional<InlayHintKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  return InlayHintKind(value);
}

auto InlayHint::textEdits() const -> std::optional<Vector<TextEdit>> {
  if (!repr_.contains("textEdits")) return std::nullopt;

  const auto& value = repr_["textEdits"];

  assert(value.is_array());
  return Vector<TextEdit>(value);
}

auto InlayHint::tooltip() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_.contains("tooltip")) return std::nullopt;

  const auto& value = repr_["tooltip"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHint::paddingLeft() const -> std::optional<bool> {
  if (!repr_.contains("paddingLeft")) return std::nullopt;

  const auto& value = repr_["paddingLeft"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHint::paddingRight() const -> std::optional<bool> {
  if (!repr_.contains("paddingRight")) return std::nullopt;

  const auto& value = repr_["paddingRight"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHint::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto InlayHint::position(Position position) -> InlayHint& { return *this; }

auto InlayHint::label(
    std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>> label)
    -> InlayHint& {
  return *this;
}

auto InlayHint::kind(std::optional<InlayHintKind> kind) -> InlayHint& {
  return *this;
}

auto InlayHint::textEdits(std::optional<Vector<TextEdit>> textEdits)
    -> InlayHint& {
  return *this;
}

auto InlayHint::tooltip(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        tooltip) -> InlayHint& {
  return *this;
}

auto InlayHint::paddingLeft(std::optional<bool> paddingLeft) -> InlayHint& {
  return *this;
}

auto InlayHint::paddingRight(std::optional<bool> paddingRight) -> InlayHint& {
  return *this;
}

auto InlayHint::data(std::optional<LSPAny> data) -> InlayHint& { return *this; }

InlayHintRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto InlayHintRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlayHintRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> InlayHintRegistrationOptions& {
  return *this;
}

auto InlayHintRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> InlayHintRegistrationOptions& {
  return *this;
}

auto InlayHintRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> InlayHintRegistrationOptions& {
  return *this;
}

auto InlayHintRegistrationOptions::id(std::optional<std::string> id)
    -> InlayHintRegistrationOptions& {
  return *this;
}

DocumentDiagnosticParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DocumentDiagnosticParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentDiagnosticParams::identifier() const
    -> std::optional<std::string> {
  if (!repr_.contains("identifier")) return std::nullopt;

  const auto& value = repr_["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentDiagnosticParams::previousResultId() const
    -> std::optional<std::string> {
  if (!repr_.contains("previousResultId")) return std::nullopt;

  const auto& value = repr_["previousResultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentDiagnosticParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentDiagnosticParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentDiagnosticParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentDiagnosticParams& {
  return *this;
}

auto DocumentDiagnosticParams::identifier(std::optional<std::string> identifier)
    -> DocumentDiagnosticParams& {
  return *this;
}

auto DocumentDiagnosticParams::previousResultId(
    std::optional<std::string> previousResultId) -> DocumentDiagnosticParams& {
  return *this;
}

auto DocumentDiagnosticParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentDiagnosticParams& {
  return *this;
}

auto DocumentDiagnosticParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> DocumentDiagnosticParams& {
  return *this;
}

DocumentDiagnosticReportPartialResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("relatedDocuments")) return false;
  return true;
}

auto DocumentDiagnosticReportPartialResult::relatedDocuments() const
    -> Map<std::string,
           std::variant<std::monostate, FullDocumentDiagnosticReport,
                        UnchangedDocumentDiagnosticReport>> {
  const auto& value = repr_["relatedDocuments"];

  assert(value.is_object());
  return Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>(value);
}

auto DocumentDiagnosticReportPartialResult::relatedDocuments(
    Map<std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                  UnchangedDocumentDiagnosticReport>>
        relatedDocuments) -> DocumentDiagnosticReportPartialResult& {
  return *this;
}

DiagnosticServerCancellationData::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("retriggerRequest")) return false;
  return true;
}

auto DiagnosticServerCancellationData::retriggerRequest() const -> bool {
  const auto& value = repr_["retriggerRequest"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticServerCancellationData::retriggerRequest(bool retriggerRequest)
    -> DiagnosticServerCancellationData& {
  return *this;
}

DiagnosticRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  if (!repr_.contains("interFileDependencies")) return false;
  if (!repr_.contains("workspaceDiagnostics")) return false;
  return true;
}

auto DiagnosticRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DiagnosticRegistrationOptions::identifier() const
    -> std::optional<std::string> {
  if (!repr_.contains("identifier")) return std::nullopt;

  const auto& value = repr_["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticRegistrationOptions::interFileDependencies() const -> bool {
  const auto& value = repr_["interFileDependencies"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticRegistrationOptions::workspaceDiagnostics() const -> bool {
  const auto& value = repr_["workspaceDiagnostics"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DiagnosticRegistrationOptions& {
  return *this;
}

auto DiagnosticRegistrationOptions::identifier(
    std::optional<std::string> identifier) -> DiagnosticRegistrationOptions& {
  return *this;
}

auto DiagnosticRegistrationOptions::interFileDependencies(
    bool interFileDependencies) -> DiagnosticRegistrationOptions& {
  return *this;
}

auto DiagnosticRegistrationOptions::workspaceDiagnostics(
    bool workspaceDiagnostics) -> DiagnosticRegistrationOptions& {
  return *this;
}

auto DiagnosticRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DiagnosticRegistrationOptions& {
  return *this;
}

auto DiagnosticRegistrationOptions::id(std::optional<std::string> id)
    -> DiagnosticRegistrationOptions& {
  return *this;
}

WorkspaceDiagnosticParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("previousResultIds")) return false;
  return true;
}

auto WorkspaceDiagnosticParams::identifier() const
    -> std::optional<std::string> {
  if (!repr_.contains("identifier")) return std::nullopt;

  const auto& value = repr_["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceDiagnosticParams::previousResultIds() const
    -> Vector<PreviousResultId> {
  const auto& value = repr_["previousResultIds"];

  assert(value.is_array());
  return Vector<PreviousResultId>(value);
}

auto WorkspaceDiagnosticParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceDiagnosticParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceDiagnosticParams::identifier(
    std::optional<std::string> identifier) -> WorkspaceDiagnosticParams& {
  return *this;
}

auto WorkspaceDiagnosticParams::previousResultIds(
    Vector<PreviousResultId> previousResultIds) -> WorkspaceDiagnosticParams& {
  return *this;
}

auto WorkspaceDiagnosticParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> WorkspaceDiagnosticParams& {
  return *this;
}

auto WorkspaceDiagnosticParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> WorkspaceDiagnosticParams& {
  return *this;
}

WorkspaceDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto WorkspaceDiagnosticReport::items() const
    -> Vector<WorkspaceDocumentDiagnosticReport> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<WorkspaceDocumentDiagnosticReport>(value);
}

auto WorkspaceDiagnosticReport::items(
    Vector<WorkspaceDocumentDiagnosticReport> items)
    -> WorkspaceDiagnosticReport& {
  return *this;
}

WorkspaceDiagnosticReportPartialResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto WorkspaceDiagnosticReportPartialResult::items() const
    -> Vector<WorkspaceDocumentDiagnosticReport> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<WorkspaceDocumentDiagnosticReport>(value);
}

auto WorkspaceDiagnosticReportPartialResult::items(
    Vector<WorkspaceDocumentDiagnosticReport> items)
    -> WorkspaceDiagnosticReportPartialResult& {
  return *this;
}

DidOpenNotebookDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookDocument")) return false;
  if (!repr_.contains("cellTextDocuments")) return false;
  return true;
}

auto DidOpenNotebookDocumentParams::notebookDocument() const
    -> NotebookDocument {
  const auto& value = repr_["notebookDocument"];

  return NotebookDocument(value);
}

auto DidOpenNotebookDocumentParams::cellTextDocuments() const
    -> Vector<TextDocumentItem> {
  const auto& value = repr_["cellTextDocuments"];

  assert(value.is_array());
  return Vector<TextDocumentItem>(value);
}

auto DidOpenNotebookDocumentParams::notebookDocument(
    NotebookDocument notebookDocument) -> DidOpenNotebookDocumentParams& {
  return *this;
}

auto DidOpenNotebookDocumentParams::cellTextDocuments(
    Vector<TextDocumentItem> cellTextDocuments)
    -> DidOpenNotebookDocumentParams& {
  return *this;
}

NotebookDocumentSyncRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookSelector")) return false;
  return true;
}

auto NotebookDocumentSyncRegistrationOptions::notebookSelector() const
    -> Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                           NotebookDocumentFilterWithCells>> {
  const auto& value = repr_["notebookSelector"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>(value);
}

auto NotebookDocumentSyncRegistrationOptions::save() const
    -> std::optional<bool> {
  if (!repr_.contains("save")) return std::nullopt;

  const auto& value = repr_["save"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentSyncRegistrationOptions::notebookSelector(
    Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                        NotebookDocumentFilterWithCells>>
        notebookSelector) -> NotebookDocumentSyncRegistrationOptions& {
  return *this;
}

auto NotebookDocumentSyncRegistrationOptions::save(std::optional<bool> save)
    -> NotebookDocumentSyncRegistrationOptions& {
  return *this;
}

auto NotebookDocumentSyncRegistrationOptions::id(std::optional<std::string> id)
    -> NotebookDocumentSyncRegistrationOptions& {
  return *this;
}

DidChangeNotebookDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookDocument")) return false;
  if (!repr_.contains("change")) return false;
  return true;
}

auto DidChangeNotebookDocumentParams::notebookDocument() const
    -> VersionedNotebookDocumentIdentifier {
  const auto& value = repr_["notebookDocument"];

  return VersionedNotebookDocumentIdentifier(value);
}

auto DidChangeNotebookDocumentParams::change() const
    -> NotebookDocumentChangeEvent {
  const auto& value = repr_["change"];

  return NotebookDocumentChangeEvent(value);
}

auto DidChangeNotebookDocumentParams::notebookDocument(
    VersionedNotebookDocumentIdentifier notebookDocument)
    -> DidChangeNotebookDocumentParams& {
  return *this;
}

auto DidChangeNotebookDocumentParams::change(NotebookDocumentChangeEvent change)
    -> DidChangeNotebookDocumentParams& {
  return *this;
}

DidSaveNotebookDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookDocument")) return false;
  return true;
}

auto DidSaveNotebookDocumentParams::notebookDocument() const
    -> NotebookDocumentIdentifier {
  const auto& value = repr_["notebookDocument"];

  return NotebookDocumentIdentifier(value);
}

auto DidSaveNotebookDocumentParams::notebookDocument(
    NotebookDocumentIdentifier notebookDocument)
    -> DidSaveNotebookDocumentParams& {
  return *this;
}

DidCloseNotebookDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookDocument")) return false;
  if (!repr_.contains("cellTextDocuments")) return false;
  return true;
}

auto DidCloseNotebookDocumentParams::notebookDocument() const
    -> NotebookDocumentIdentifier {
  const auto& value = repr_["notebookDocument"];

  return NotebookDocumentIdentifier(value);
}

auto DidCloseNotebookDocumentParams::cellTextDocuments() const
    -> Vector<TextDocumentIdentifier> {
  const auto& value = repr_["cellTextDocuments"];

  assert(value.is_array());
  return Vector<TextDocumentIdentifier>(value);
}

auto DidCloseNotebookDocumentParams::notebookDocument(
    NotebookDocumentIdentifier notebookDocument)
    -> DidCloseNotebookDocumentParams& {
  return *this;
}

auto DidCloseNotebookDocumentParams::cellTextDocuments(
    Vector<TextDocumentIdentifier> cellTextDocuments)
    -> DidCloseNotebookDocumentParams& {
  return *this;
}

InlineCompletionParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("context")) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto InlineCompletionParams::context() const -> InlineCompletionContext {
  const auto& value = repr_["context"];

  return InlineCompletionContext(value);
}

auto InlineCompletionParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto InlineCompletionParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto InlineCompletionParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InlineCompletionParams::context(InlineCompletionContext context)
    -> InlineCompletionParams& {
  return *this;
}

auto InlineCompletionParams::textDocument(TextDocumentIdentifier textDocument)
    -> InlineCompletionParams& {
  return *this;
}

auto InlineCompletionParams::position(Position position)
    -> InlineCompletionParams& {
  return *this;
}

auto InlineCompletionParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> InlineCompletionParams& {
  return *this;
}

InlineCompletionList::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto InlineCompletionList::items() const -> Vector<InlineCompletionItem> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<InlineCompletionItem>(value);
}

auto InlineCompletionList::items(Vector<InlineCompletionItem> items)
    -> InlineCompletionList& {
  return *this;
}

InlineCompletionItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("insertText")) return false;
  return true;
}

auto InlineCompletionItem::insertText() const
    -> std::variant<std::monostate, std::string, StringValue> {
  const auto& value = repr_["insertText"];

  std::variant<std::monostate, std::string, StringValue> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineCompletionItem::filterText() const -> std::optional<std::string> {
  if (!repr_.contains("filterText")) return std::nullopt;

  const auto& value = repr_["filterText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineCompletionItem::range() const -> std::optional<Range> {
  if (!repr_.contains("range")) return std::nullopt;

  const auto& value = repr_["range"];

  return Range(value);
}

auto InlineCompletionItem::command() const -> std::optional<Command> {
  if (!repr_.contains("command")) return std::nullopt;

  const auto& value = repr_["command"];

  return Command(value);
}

auto InlineCompletionItem::insertText(
    std::variant<std::monostate, std::string, StringValue> insertText)
    -> InlineCompletionItem& {
  return *this;
}

auto InlineCompletionItem::filterText(std::optional<std::string> filterText)
    -> InlineCompletionItem& {
  return *this;
}

auto InlineCompletionItem::range(std::optional<Range> range)
    -> InlineCompletionItem& {
  return *this;
}

auto InlineCompletionItem::command(std::optional<Command> command)
    -> InlineCompletionItem& {
  return *this;
}

InlineCompletionRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto InlineCompletionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineCompletionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineCompletionRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineCompletionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> InlineCompletionRegistrationOptions& {
  return *this;
}

auto InlineCompletionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> InlineCompletionRegistrationOptions& {
  return *this;
}

auto InlineCompletionRegistrationOptions::id(std::optional<std::string> id)
    -> InlineCompletionRegistrationOptions& {
  return *this;
}

TextDocumentContentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto TextDocumentContentParams::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentParams::uri(std::string uri)
    -> TextDocumentContentParams& {
  return *this;
}

TextDocumentContentResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("text")) return false;
  return true;
}

auto TextDocumentContentResult::text() const -> std::string {
  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentResult::text(std::string text)
    -> TextDocumentContentResult& {
  return *this;
}

TextDocumentContentRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("schemes")) return false;
  return true;
}

auto TextDocumentContentRegistrationOptions::schemes() const
    -> Vector<std::string> {
  const auto& value = repr_["schemes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto TextDocumentContentRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentRegistrationOptions::schemes(
    Vector<std::string> schemes) -> TextDocumentContentRegistrationOptions& {
  return *this;
}

auto TextDocumentContentRegistrationOptions::id(std::optional<std::string> id)
    -> TextDocumentContentRegistrationOptions& {
  return *this;
}

TextDocumentContentRefreshParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto TextDocumentContentRefreshParams::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentRefreshParams::uri(std::string uri)
    -> TextDocumentContentRefreshParams& {
  return *this;
}

RegistrationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("registrations")) return false;
  return true;
}

auto RegistrationParams::registrations() const -> Vector<Registration> {
  const auto& value = repr_["registrations"];

  assert(value.is_array());
  return Vector<Registration>(value);
}

auto RegistrationParams::registrations(Vector<Registration> registrations)
    -> RegistrationParams& {
  return *this;
}

UnregistrationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("unregisterations")) return false;
  return true;
}

auto UnregistrationParams::unregisterations() const -> Vector<Unregistration> {
  const auto& value = repr_["unregisterations"];

  assert(value.is_array());
  return Vector<Unregistration>(value);
}

auto UnregistrationParams::unregisterations(
    Vector<Unregistration> unregisterations) -> UnregistrationParams& {
  return *this;
}

InitializeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("processId")) return false;
  if (!repr_.contains("rootUri")) return false;
  if (!repr_.contains("capabilities")) return false;
  return true;
}

auto InitializeParams::processId() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  const auto& value = repr_["processId"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::clientInfo() const -> std::optional<ClientInfo> {
  if (!repr_.contains("clientInfo")) return std::nullopt;

  const auto& value = repr_["clientInfo"];

  return ClientInfo(value);
}

auto InitializeParams::locale() const -> std::optional<std::string> {
  if (!repr_.contains("locale")) return std::nullopt;

  const auto& value = repr_["locale"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InitializeParams::rootPath() const -> std::optional<
    std::variant<std::monostate, std::string, std::nullptr_t>> {
  if (!repr_.contains("rootPath")) return std::nullopt;

  const auto& value = repr_["rootPath"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::rootUri() const
    -> std::variant<std::monostate, std::string, std::nullptr_t> {
  const auto& value = repr_["rootUri"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::capabilities() const -> ClientCapabilities {
  const auto& value = repr_["capabilities"];

  return ClientCapabilities(value);
}

auto InitializeParams::initializationOptions() const -> std::optional<LSPAny> {
  if (!repr_.contains("initializationOptions")) return std::nullopt;

  const auto& value = repr_["initializationOptions"];

  assert(value.is_object());
  return LSPAny(value);
}

auto InitializeParams::trace() const -> std::optional<TraceValue> {
  if (!repr_.contains("trace")) return std::nullopt;

  const auto& value = repr_["trace"];

  lsp_runtime_error("InitializeParams::trace: not implement yet");
}

auto InitializeParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::workspaceFolders() const -> std::optional<
    std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>> {
  if (!repr_.contains("workspaceFolders")) return std::nullopt;

  const auto& value = repr_["workspaceFolders"];

  std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::processId(
    std::variant<std::monostate, int, std::nullptr_t> processId)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::clientInfo(std::optional<ClientInfo> clientInfo)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::locale(std::optional<std::string> locale)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::rootPath(
    std::optional<std::variant<std::monostate, std::string, std::nullptr_t>>
        rootPath) -> InitializeParams& {
  return *this;
}

auto InitializeParams::rootUri(
    std::variant<std::monostate, std::string, std::nullptr_t> rootUri)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::capabilities(ClientCapabilities capabilities)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::initializationOptions(
    std::optional<LSPAny> initializationOptions) -> InitializeParams& {
  return *this;
}

auto InitializeParams::trace(std::optional<TraceValue> trace)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> InitializeParams& {
  return *this;
}

auto InitializeParams::workspaceFolders(
    std::optional<
        std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>
        workspaceFolders) -> InitializeParams& {
  return *this;
}

InitializeResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("capabilities")) return false;
  return true;
}

auto InitializeResult::capabilities() const -> ServerCapabilities {
  const auto& value = repr_["capabilities"];

  return ServerCapabilities(value);
}

auto InitializeResult::serverInfo() const -> std::optional<ServerInfo> {
  if (!repr_.contains("serverInfo")) return std::nullopt;

  const auto& value = repr_["serverInfo"];

  return ServerInfo(value);
}

auto InitializeResult::capabilities(ServerCapabilities capabilities)
    -> InitializeResult& {
  return *this;
}

auto InitializeResult::serverInfo(std::optional<ServerInfo> serverInfo)
    -> InitializeResult& {
  return *this;
}

InitializeError::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("retry")) return false;
  return true;
}

auto InitializeError::retry() const -> bool {
  const auto& value = repr_["retry"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InitializeError::retry(bool retry) -> InitializeError& { return *this; }

InitializedParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

DidChangeConfigurationParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("settings")) return false;
  return true;
}

auto DidChangeConfigurationParams::settings() const -> LSPAny {
  const auto& value = repr_["settings"];

  assert(value.is_object());
  return LSPAny(value);
}

auto DidChangeConfigurationParams::settings(LSPAny settings)
    -> DidChangeConfigurationParams& {
  return *this;
}

DidChangeConfigurationRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DidChangeConfigurationRegistrationOptions::section() const
    -> std::optional<
        std::variant<std::monostate, std::string, Vector<std::string>>> {
  if (!repr_.contains("section")) return std::nullopt;

  const auto& value = repr_["section"];

  std::variant<std::monostate, std::string, Vector<std::string>> result;

  details::try_emplace(result, value);

  return result;
}

auto DidChangeConfigurationRegistrationOptions::section(
    std::optional<
        std::variant<std::monostate, std::string, Vector<std::string>>>
        section) -> DidChangeConfigurationRegistrationOptions& {
  return *this;
}

ShowMessageParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("type")) return false;
  if (!repr_.contains("message")) return false;
  return true;
}

auto ShowMessageParams::type() const -> MessageType {
  const auto& value = repr_["type"];

  return MessageType(value);
}

auto ShowMessageParams::message() const -> std::string {
  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ShowMessageParams::type(MessageType type) -> ShowMessageParams& {
  return *this;
}

auto ShowMessageParams::message(std::string message) -> ShowMessageParams& {
  return *this;
}

ShowMessageRequestParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("type")) return false;
  if (!repr_.contains("message")) return false;
  return true;
}

auto ShowMessageRequestParams::type() const -> MessageType {
  const auto& value = repr_["type"];

  return MessageType(value);
}

auto ShowMessageRequestParams::message() const -> std::string {
  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ShowMessageRequestParams::actions() const
    -> std::optional<Vector<MessageActionItem>> {
  if (!repr_.contains("actions")) return std::nullopt;

  const auto& value = repr_["actions"];

  assert(value.is_array());
  return Vector<MessageActionItem>(value);
}

auto ShowMessageRequestParams::type(MessageType type)
    -> ShowMessageRequestParams& {
  return *this;
}

auto ShowMessageRequestParams::message(std::string message)
    -> ShowMessageRequestParams& {
  return *this;
}

auto ShowMessageRequestParams::actions(
    std::optional<Vector<MessageActionItem>> actions)
    -> ShowMessageRequestParams& {
  return *this;
}

MessageActionItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("title")) return false;
  return true;
}

auto MessageActionItem::title() const -> std::string {
  const auto& value = repr_["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MessageActionItem::title(std::string title) -> MessageActionItem& {
  return *this;
}

LogMessageParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("type")) return false;
  if (!repr_.contains("message")) return false;
  return true;
}

auto LogMessageParams::type() const -> MessageType {
  const auto& value = repr_["type"];

  return MessageType(value);
}

auto LogMessageParams::message() const -> std::string {
  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LogMessageParams::type(MessageType type) -> LogMessageParams& {
  return *this;
}

auto LogMessageParams::message(std::string message) -> LogMessageParams& {
  return *this;
}

DidOpenTextDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DidOpenTextDocumentParams::textDocument() const -> TextDocumentItem {
  const auto& value = repr_["textDocument"];

  return TextDocumentItem(value);
}

auto DidOpenTextDocumentParams::textDocument(TextDocumentItem textDocument)
    -> DidOpenTextDocumentParams& {
  return *this;
}

DidChangeTextDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("contentChanges")) return false;
  return true;
}

auto DidChangeTextDocumentParams::textDocument() const
    -> VersionedTextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return VersionedTextDocumentIdentifier(value);
}

auto DidChangeTextDocumentParams::contentChanges() const
    -> Vector<TextDocumentContentChangeEvent> {
  const auto& value = repr_["contentChanges"];

  assert(value.is_array());
  return Vector<TextDocumentContentChangeEvent>(value);
}

auto DidChangeTextDocumentParams::textDocument(
    VersionedTextDocumentIdentifier textDocument)
    -> DidChangeTextDocumentParams& {
  return *this;
}

auto DidChangeTextDocumentParams::contentChanges(
    Vector<TextDocumentContentChangeEvent> contentChanges)
    -> DidChangeTextDocumentParams& {
  return *this;
}

TextDocumentChangeRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("syncKind")) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto TextDocumentChangeRegistrationOptions::syncKind() const
    -> TextDocumentSyncKind {
  const auto& value = repr_["syncKind"];

  return TextDocumentSyncKind(value);
}

auto TextDocumentChangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentChangeRegistrationOptions::syncKind(
    TextDocumentSyncKind syncKind) -> TextDocumentChangeRegistrationOptions& {
  return *this;
}

auto TextDocumentChangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TextDocumentChangeRegistrationOptions& {
  return *this;
}

DidCloseTextDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DidCloseTextDocumentParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DidCloseTextDocumentParams::textDocument(
    TextDocumentIdentifier textDocument) -> DidCloseTextDocumentParams& {
  return *this;
}

DidSaveTextDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DidSaveTextDocumentParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DidSaveTextDocumentParams::text() const -> std::optional<std::string> {
  if (!repr_.contains("text")) return std::nullopt;

  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DidSaveTextDocumentParams::textDocument(
    TextDocumentIdentifier textDocument) -> DidSaveTextDocumentParams& {
  return *this;
}

auto DidSaveTextDocumentParams::text(std::optional<std::string> text)
    -> DidSaveTextDocumentParams& {
  return *this;
}

TextDocumentSaveRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto TextDocumentSaveRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentSaveRegistrationOptions::includeText() const
    -> std::optional<bool> {
  if (!repr_.contains("includeText")) return std::nullopt;

  const auto& value = repr_["includeText"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSaveRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TextDocumentSaveRegistrationOptions& {
  return *this;
}

auto TextDocumentSaveRegistrationOptions::includeText(
    std::optional<bool> includeText) -> TextDocumentSaveRegistrationOptions& {
  return *this;
}

WillSaveTextDocumentParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("reason")) return false;
  return true;
}

auto WillSaveTextDocumentParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto WillSaveTextDocumentParams::reason() const -> TextDocumentSaveReason {
  const auto& value = repr_["reason"];

  return TextDocumentSaveReason(value);
}

auto WillSaveTextDocumentParams::textDocument(
    TextDocumentIdentifier textDocument) -> WillSaveTextDocumentParams& {
  return *this;
}

auto WillSaveTextDocumentParams::reason(TextDocumentSaveReason reason)
    -> WillSaveTextDocumentParams& {
  return *this;
}

TextEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("newText")) return false;
  return true;
}

auto TextEdit::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto TextEdit::newText() const -> std::string {
  const auto& value = repr_["newText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextEdit::range(Range range) -> TextEdit& { return *this; }

auto TextEdit::newText(std::string newText) -> TextEdit& { return *this; }

DidChangeWatchedFilesParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("changes")) return false;
  return true;
}

auto DidChangeWatchedFilesParams::changes() const -> Vector<FileEvent> {
  const auto& value = repr_["changes"];

  assert(value.is_array());
  return Vector<FileEvent>(value);
}

auto DidChangeWatchedFilesParams::changes(Vector<FileEvent> changes)
    -> DidChangeWatchedFilesParams& {
  return *this;
}

DidChangeWatchedFilesRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("watchers")) return false;
  return true;
}

auto DidChangeWatchedFilesRegistrationOptions::watchers() const
    -> Vector<FileSystemWatcher> {
  const auto& value = repr_["watchers"];

  assert(value.is_array());
  return Vector<FileSystemWatcher>(value);
}

auto DidChangeWatchedFilesRegistrationOptions::watchers(
    Vector<FileSystemWatcher> watchers)
    -> DidChangeWatchedFilesRegistrationOptions& {
  return *this;
}

PublishDiagnosticsParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("diagnostics")) return false;
  return true;
}

auto PublishDiagnosticsParams::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PublishDiagnosticsParams::version() const -> std::optional<int> {
  if (!repr_.contains("version")) return std::nullopt;

  const auto& value = repr_["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto PublishDiagnosticsParams::diagnostics() const -> Vector<Diagnostic> {
  const auto& value = repr_["diagnostics"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto PublishDiagnosticsParams::uri(std::string uri)
    -> PublishDiagnosticsParams& {
  return *this;
}

auto PublishDiagnosticsParams::version(std::optional<int> version)
    -> PublishDiagnosticsParams& {
  return *this;
}

auto PublishDiagnosticsParams::diagnostics(Vector<Diagnostic> diagnostics)
    -> PublishDiagnosticsParams& {
  return *this;
}

CompletionParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto CompletionParams::context() const -> std::optional<CompletionContext> {
  if (!repr_.contains("context")) return std::nullopt;

  const auto& value = repr_["context"];

  return CompletionContext(value);
}

auto CompletionParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CompletionParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto CompletionParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionParams::context(std::optional<CompletionContext> context)
    -> CompletionParams& {
  return *this;
}

auto CompletionParams::textDocument(TextDocumentIdentifier textDocument)
    -> CompletionParams& {
  return *this;
}

auto CompletionParams::position(Position position) -> CompletionParams& {
  return *this;
}

auto CompletionParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> CompletionParams& {
  return *this;
}

auto CompletionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> CompletionParams& {
  return *this;
}

CompletionItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("label")) return false;
  return true;
}

auto CompletionItem::label() const -> std::string {
  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::labelDetails() const
    -> std::optional<CompletionItemLabelDetails> {
  if (!repr_.contains("labelDetails")) return std::nullopt;

  const auto& value = repr_["labelDetails"];

  return CompletionItemLabelDetails(value);
}

auto CompletionItem::kind() const -> std::optional<CompletionItemKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  return CompletionItemKind(value);
}

auto CompletionItem::tags() const -> std::optional<Vector<CompletionItemTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<CompletionItemTag>(value);
}

auto CompletionItem::detail() const -> std::optional<std::string> {
  if (!repr_.contains("detail")) return std::nullopt;

  const auto& value = repr_["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::documentation() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_.contains("documentation")) return std::nullopt;

  const auto& value = repr_["documentation"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionItem::deprecated() const -> std::optional<bool> {
  if (!repr_.contains("deprecated")) return std::nullopt;

  const auto& value = repr_["deprecated"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionItem::preselect() const -> std::optional<bool> {
  if (!repr_.contains("preselect")) return std::nullopt;

  const auto& value = repr_["preselect"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionItem::sortText() const -> std::optional<std::string> {
  if (!repr_.contains("sortText")) return std::nullopt;

  const auto& value = repr_["sortText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::filterText() const -> std::optional<std::string> {
  if (!repr_.contains("filterText")) return std::nullopt;

  const auto& value = repr_["filterText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::insertText() const -> std::optional<std::string> {
  if (!repr_.contains("insertText")) return std::nullopt;

  const auto& value = repr_["insertText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::insertTextFormat() const
    -> std::optional<InsertTextFormat> {
  if (!repr_.contains("insertTextFormat")) return std::nullopt;

  const auto& value = repr_["insertTextFormat"];

  return InsertTextFormat(value);
}

auto CompletionItem::insertTextMode() const -> std::optional<InsertTextMode> {
  if (!repr_.contains("insertTextMode")) return std::nullopt;

  const auto& value = repr_["insertTextMode"];

  return InsertTextMode(value);
}

auto CompletionItem::textEdit() const -> std::optional<
    std::variant<std::monostate, TextEdit, InsertReplaceEdit>> {
  if (!repr_.contains("textEdit")) return std::nullopt;

  const auto& value = repr_["textEdit"];

  std::variant<std::monostate, TextEdit, InsertReplaceEdit> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionItem::textEditText() const -> std::optional<std::string> {
  if (!repr_.contains("textEditText")) return std::nullopt;

  const auto& value = repr_["textEditText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::additionalTextEdits() const
    -> std::optional<Vector<TextEdit>> {
  if (!repr_.contains("additionalTextEdits")) return std::nullopt;

  const auto& value = repr_["additionalTextEdits"];

  assert(value.is_array());
  return Vector<TextEdit>(value);
}

auto CompletionItem::commitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("commitCharacters")) return std::nullopt;

  const auto& value = repr_["commitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionItem::command() const -> std::optional<Command> {
  if (!repr_.contains("command")) return std::nullopt;

  const auto& value = repr_["command"];

  return Command(value);
}

auto CompletionItem::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CompletionItem::label(std::string label) -> CompletionItem& {
  return *this;
}

auto CompletionItem::labelDetails(
    std::optional<CompletionItemLabelDetails> labelDetails) -> CompletionItem& {
  return *this;
}

auto CompletionItem::kind(std::optional<CompletionItemKind> kind)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::tags(std::optional<Vector<CompletionItemTag>> tags)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::detail(std::optional<std::string> detail)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::documentation(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        documentation) -> CompletionItem& {
  return *this;
}

auto CompletionItem::deprecated(std::optional<bool> deprecated)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::preselect(std::optional<bool> preselect)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::sortText(std::optional<std::string> sortText)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::filterText(std::optional<std::string> filterText)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::insertText(std::optional<std::string> insertText)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::insertTextFormat(
    std::optional<InsertTextFormat> insertTextFormat) -> CompletionItem& {
  return *this;
}

auto CompletionItem::insertTextMode(
    std::optional<InsertTextMode> insertTextMode) -> CompletionItem& {
  return *this;
}

auto CompletionItem::textEdit(
    std::optional<std::variant<std::monostate, TextEdit, InsertReplaceEdit>>
        textEdit) -> CompletionItem& {
  return *this;
}

auto CompletionItem::textEditText(std::optional<std::string> textEditText)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::additionalTextEdits(
    std::optional<Vector<TextEdit>> additionalTextEdits) -> CompletionItem& {
  return *this;
}

auto CompletionItem::commitCharacters(
    std::optional<Vector<std::string>> commitCharacters) -> CompletionItem& {
  return *this;
}

auto CompletionItem::command(std::optional<Command> command)
    -> CompletionItem& {
  return *this;
}

auto CompletionItem::data(std::optional<LSPAny> data) -> CompletionItem& {
  return *this;
}

CompletionList::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("isIncomplete")) return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto CompletionList::isIncomplete() const -> bool {
  const auto& value = repr_["isIncomplete"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionList::itemDefaults() const
    -> std::optional<CompletionItemDefaults> {
  if (!repr_.contains("itemDefaults")) return std::nullopt;

  const auto& value = repr_["itemDefaults"];

  return CompletionItemDefaults(value);
}

auto CompletionList::applyKind() const
    -> std::optional<CompletionItemApplyKinds> {
  if (!repr_.contains("applyKind")) return std::nullopt;

  const auto& value = repr_["applyKind"];

  return CompletionItemApplyKinds(value);
}

auto CompletionList::items() const -> Vector<CompletionItem> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<CompletionItem>(value);
}

auto CompletionList::isIncomplete(bool isIncomplete) -> CompletionList& {
  return *this;
}

auto CompletionList::itemDefaults(
    std::optional<CompletionItemDefaults> itemDefaults) -> CompletionList& {
  return *this;
}

auto CompletionList::applyKind(
    std::optional<CompletionItemApplyKinds> applyKind) -> CompletionList& {
  return *this;
}

auto CompletionList::items(Vector<CompletionItem> items) -> CompletionList& {
  return *this;
}

CompletionRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto CompletionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionRegistrationOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("triggerCharacters")) return std::nullopt;

  const auto& value = repr_["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionRegistrationOptions::allCommitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("allCommitCharacters")) return std::nullopt;

  const auto& value = repr_["allCommitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionRegistrationOptions::completionItem() const
    -> std::optional<ServerCompletionItemOptions> {
  if (!repr_.contains("completionItem")) return std::nullopt;

  const auto& value = repr_["completionItem"];

  return ServerCompletionItemOptions(value);
}

auto CompletionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CompletionRegistrationOptions& {
  return *this;
}

auto CompletionRegistrationOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> CompletionRegistrationOptions& {
  return *this;
}

auto CompletionRegistrationOptions::allCommitCharacters(
    std::optional<Vector<std::string>> allCommitCharacters)
    -> CompletionRegistrationOptions& {
  return *this;
}

auto CompletionRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> CompletionRegistrationOptions& {
  return *this;
}

auto CompletionRegistrationOptions::completionItem(
    std::optional<ServerCompletionItemOptions> completionItem)
    -> CompletionRegistrationOptions& {
  return *this;
}

auto CompletionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CompletionRegistrationOptions& {
  return *this;
}

HoverParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto HoverParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto HoverParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto HoverParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto HoverParams::textDocument(TextDocumentIdentifier textDocument)
    -> HoverParams& {
  return *this;
}

auto HoverParams::position(Position position) -> HoverParams& { return *this; }

auto HoverParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> HoverParams& {
  return *this;
}

Hover::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("contents")) return false;
  return true;
}

auto Hover::contents() const
    -> std::variant<std::monostate, MarkupContent, MarkedString,
                    Vector<MarkedString>> {
  const auto& value = repr_["contents"];

  std::variant<std::monostate, MarkupContent, MarkedString,
               Vector<MarkedString>>
      result;

  details::try_emplace(result, value);

  return result;
}

auto Hover::range() const -> std::optional<Range> {
  if (!repr_.contains("range")) return std::nullopt;

  const auto& value = repr_["range"];

  return Range(value);
}

auto Hover::contents(std::variant<std::monostate, MarkupContent, MarkedString,
                                  Vector<MarkedString>>
                         contents) -> Hover& {
  return *this;
}

auto Hover::range(std::optional<Range> range) -> Hover& { return *this; }

HoverRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto HoverRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto HoverRegistrationOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto HoverRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> HoverRegistrationOptions& {
  return *this;
}

auto HoverRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> HoverRegistrationOptions& {
  return *this;
}

SignatureHelpParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto SignatureHelpParams::context() const
    -> std::optional<SignatureHelpContext> {
  if (!repr_.contains("context")) return std::nullopt;

  const auto& value = repr_["context"];

  return SignatureHelpContext(value);
}

auto SignatureHelpParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SignatureHelpParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto SignatureHelpParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelpParams::context(std::optional<SignatureHelpContext> context)
    -> SignatureHelpParams& {
  return *this;
}

auto SignatureHelpParams::textDocument(TextDocumentIdentifier textDocument)
    -> SignatureHelpParams& {
  return *this;
}

auto SignatureHelpParams::position(Position position) -> SignatureHelpParams& {
  return *this;
}

auto SignatureHelpParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SignatureHelpParams& {
  return *this;
}

SignatureHelp::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("signatures")) return false;
  return true;
}

auto SignatureHelp::signatures() const -> Vector<SignatureInformation> {
  const auto& value = repr_["signatures"];

  assert(value.is_array());
  return Vector<SignatureInformation>(value);
}

auto SignatureHelp::activeSignature() const -> std::optional<long> {
  if (!repr_.contains("activeSignature")) return std::nullopt;

  const auto& value = repr_["activeSignature"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto SignatureHelp::activeParameter() const
    -> std::optional<std::variant<std::monostate, long, std::nullptr_t>> {
  if (!repr_.contains("activeParameter")) return std::nullopt;

  const auto& value = repr_["activeParameter"];

  std::variant<std::monostate, long, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelp::signatures(Vector<SignatureInformation> signatures)
    -> SignatureHelp& {
  return *this;
}

auto SignatureHelp::activeSignature(std::optional<long> activeSignature)
    -> SignatureHelp& {
  return *this;
}

auto SignatureHelp::activeParameter(
    std::optional<std::variant<std::monostate, long, std::nullptr_t>>
        activeParameter) -> SignatureHelp& {
  return *this;
}

SignatureHelpRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto SignatureHelpRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelpRegistrationOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("triggerCharacters")) return std::nullopt;

  const auto& value = repr_["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpRegistrationOptions::retriggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("retriggerCharacters")) return std::nullopt;

  const auto& value = repr_["retriggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> SignatureHelpRegistrationOptions& {
  return *this;
}

auto SignatureHelpRegistrationOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> SignatureHelpRegistrationOptions& {
  return *this;
}

auto SignatureHelpRegistrationOptions::retriggerCharacters(
    std::optional<Vector<std::string>> retriggerCharacters)
    -> SignatureHelpRegistrationOptions& {
  return *this;
}

auto SignatureHelpRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SignatureHelpRegistrationOptions& {
  return *this;
}

DefinitionParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto DefinitionParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DefinitionParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto DefinitionParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionParams::textDocument(TextDocumentIdentifier textDocument)
    -> DefinitionParams& {
  return *this;
}

auto DefinitionParams::position(Position position) -> DefinitionParams& {
  return *this;
}

auto DefinitionParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> DefinitionParams& {
  return *this;
}

auto DefinitionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DefinitionParams& {
  return *this;
}

DefinitionRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DefinitionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DefinitionRegistrationOptions& {
  return *this;
}

auto DefinitionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DefinitionRegistrationOptions& {
  return *this;
}

ReferenceParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("context")) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto ReferenceParams::context() const -> ReferenceContext {
  const auto& value = repr_["context"];

  return ReferenceContext(value);
}

auto ReferenceParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto ReferenceParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto ReferenceParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ReferenceParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ReferenceParams::context(ReferenceContext context) -> ReferenceParams& {
  return *this;
}

auto ReferenceParams::textDocument(TextDocumentIdentifier textDocument)
    -> ReferenceParams& {
  return *this;
}

auto ReferenceParams::position(Position position) -> ReferenceParams& {
  return *this;
}

auto ReferenceParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> ReferenceParams& {
  return *this;
}

auto ReferenceParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> ReferenceParams& {
  return *this;
}

ReferenceRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto ReferenceRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ReferenceRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> ReferenceRegistrationOptions& {
  return *this;
}

auto ReferenceRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> ReferenceRegistrationOptions& {
  return *this;
}

DocumentHighlightParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto DocumentHighlightParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentHighlightParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto DocumentHighlightParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentHighlightParams& {
  return *this;
}

auto DocumentHighlightParams::position(Position position)
    -> DocumentHighlightParams& {
  return *this;
}

auto DocumentHighlightParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentHighlightParams& {
  return *this;
}

auto DocumentHighlightParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> DocumentHighlightParams& {
  return *this;
}

DocumentHighlight::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto DocumentHighlight::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto DocumentHighlight::kind() const -> std::optional<DocumentHighlightKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  return DocumentHighlightKind(value);
}

auto DocumentHighlight::range(Range range) -> DocumentHighlight& {
  return *this;
}

auto DocumentHighlight::kind(std::optional<DocumentHighlightKind> kind)
    -> DocumentHighlight& {
  return *this;
}

DocumentHighlightRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DocumentHighlightRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentHighlightRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentHighlightRegistrationOptions& {
  return *this;
}

auto DocumentHighlightRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentHighlightRegistrationOptions& {
  return *this;
}

DocumentSymbolParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DocumentSymbolParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentSymbolParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentSymbolParams& {
  return *this;
}

auto DocumentSymbolParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentSymbolParams& {
  return *this;
}

auto DocumentSymbolParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DocumentSymbolParams& {
  return *this;
}

SymbolInformation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("location")) return false;
  if (!repr_.contains("name")) return false;
  if (!repr_.contains("kind")) return false;
  return true;
}

auto SymbolInformation::deprecated() const -> std::optional<bool> {
  if (!repr_.contains("deprecated")) return std::nullopt;

  const auto& value = repr_["deprecated"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SymbolInformation::location() const -> Location {
  const auto& value = repr_["location"];

  return Location(value);
}

auto SymbolInformation::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SymbolInformation::kind() const -> SymbolKind {
  const auto& value = repr_["kind"];

  return SymbolKind(value);
}

auto SymbolInformation::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto SymbolInformation::containerName() const -> std::optional<std::string> {
  if (!repr_.contains("containerName")) return std::nullopt;

  const auto& value = repr_["containerName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SymbolInformation::deprecated(std::optional<bool> deprecated)
    -> SymbolInformation& {
  return *this;
}

auto SymbolInformation::location(Location location) -> SymbolInformation& {
  return *this;
}

auto SymbolInformation::name(std::string name) -> SymbolInformation& {
  return *this;
}

auto SymbolInformation::kind(SymbolKind kind) -> SymbolInformation& {
  return *this;
}

auto SymbolInformation::tags(std::optional<Vector<SymbolTag>> tags)
    -> SymbolInformation& {
  return *this;
}

auto SymbolInformation::containerName(std::optional<std::string> containerName)
    -> SymbolInformation& {
  return *this;
}

DocumentSymbol::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("name")) return false;
  if (!repr_.contains("kind")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("selectionRange")) return false;
  return true;
}

auto DocumentSymbol::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbol::detail() const -> std::optional<std::string> {
  if (!repr_.contains("detail")) return std::nullopt;

  const auto& value = repr_["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbol::kind() const -> SymbolKind {
  const auto& value = repr_["kind"];

  return SymbolKind(value);
}

auto DocumentSymbol::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto DocumentSymbol::deprecated() const -> std::optional<bool> {
  if (!repr_.contains("deprecated")) return std::nullopt;

  const auto& value = repr_["deprecated"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbol::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto DocumentSymbol::selectionRange() const -> Range {
  const auto& value = repr_["selectionRange"];

  return Range(value);
}

auto DocumentSymbol::children() const -> std::optional<Vector<DocumentSymbol>> {
  if (!repr_.contains("children")) return std::nullopt;

  const auto& value = repr_["children"];

  assert(value.is_array());
  return Vector<DocumentSymbol>(value);
}

auto DocumentSymbol::name(std::string name) -> DocumentSymbol& { return *this; }

auto DocumentSymbol::detail(std::optional<std::string> detail)
    -> DocumentSymbol& {
  return *this;
}

auto DocumentSymbol::kind(SymbolKind kind) -> DocumentSymbol& { return *this; }

auto DocumentSymbol::tags(std::optional<Vector<SymbolTag>> tags)
    -> DocumentSymbol& {
  return *this;
}

auto DocumentSymbol::deprecated(std::optional<bool> deprecated)
    -> DocumentSymbol& {
  return *this;
}

auto DocumentSymbol::range(Range range) -> DocumentSymbol& { return *this; }

auto DocumentSymbol::selectionRange(Range selectionRange) -> DocumentSymbol& {
  return *this;
}

auto DocumentSymbol::children(std::optional<Vector<DocumentSymbol>> children)
    -> DocumentSymbol& {
  return *this;
}

DocumentSymbolRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DocumentSymbolRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolRegistrationOptions::label() const
    -> std::optional<std::string> {
  if (!repr_.contains("label")) return std::nullopt;

  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbolRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentSymbolRegistrationOptions& {
  return *this;
}

auto DocumentSymbolRegistrationOptions::label(std::optional<std::string> label)
    -> DocumentSymbolRegistrationOptions& {
  return *this;
}

auto DocumentSymbolRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentSymbolRegistrationOptions& {
  return *this;
}

CodeActionParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("context")) return false;
  return true;
}

auto CodeActionParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CodeActionParams::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto CodeActionParams::context() const -> CodeActionContext {
  const auto& value = repr_["context"];

  return CodeActionContext(value);
}

auto CodeActionParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionParams::textDocument(TextDocumentIdentifier textDocument)
    -> CodeActionParams& {
  return *this;
}

auto CodeActionParams::range(Range range) -> CodeActionParams& { return *this; }

auto CodeActionParams::context(CodeActionContext context) -> CodeActionParams& {
  return *this;
}

auto CodeActionParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> CodeActionParams& {
  return *this;
}

auto CodeActionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> CodeActionParams& {
  return *this;
}

Command::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("title")) return false;
  if (!repr_.contains("command")) return false;
  return true;
}

auto Command::title() const -> std::string {
  const auto& value = repr_["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Command::tooltip() const -> std::optional<std::string> {
  if (!repr_.contains("tooltip")) return std::nullopt;

  const auto& value = repr_["tooltip"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Command::command() const -> std::string {
  const auto& value = repr_["command"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Command::arguments() const -> std::optional<Vector<LSPAny>> {
  if (!repr_.contains("arguments")) return std::nullopt;

  const auto& value = repr_["arguments"];

  assert(value.is_array());
  return Vector<LSPAny>(value);
}

auto Command::title(std::string title) -> Command& { return *this; }

auto Command::tooltip(std::optional<std::string> tooltip) -> Command& {
  return *this;
}

auto Command::command(std::string command) -> Command& { return *this; }

auto Command::arguments(std::optional<Vector<LSPAny>> arguments) -> Command& {
  return *this;
}

CodeAction::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("title")) return false;
  return true;
}

auto CodeAction::title() const -> std::string {
  const auto& value = repr_["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CodeAction::kind() const -> std::optional<CodeActionKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  lsp_runtime_error("CodeAction::kind: not implement yet");
}

auto CodeAction::diagnostics() const -> std::optional<Vector<Diagnostic>> {
  if (!repr_.contains("diagnostics")) return std::nullopt;

  const auto& value = repr_["diagnostics"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto CodeAction::isPreferred() const -> std::optional<bool> {
  if (!repr_.contains("isPreferred")) return std::nullopt;

  const auto& value = repr_["isPreferred"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeAction::disabled() const -> std::optional<CodeActionDisabled> {
  if (!repr_.contains("disabled")) return std::nullopt;

  const auto& value = repr_["disabled"];

  return CodeActionDisabled(value);
}

auto CodeAction::edit() const -> std::optional<WorkspaceEdit> {
  if (!repr_.contains("edit")) return std::nullopt;

  const auto& value = repr_["edit"];

  return WorkspaceEdit(value);
}

auto CodeAction::command() const -> std::optional<Command> {
  if (!repr_.contains("command")) return std::nullopt;

  const auto& value = repr_["command"];

  return Command(value);
}

auto CodeAction::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CodeAction::tags() const -> std::optional<Vector<CodeActionTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<CodeActionTag>(value);
}

auto CodeAction::title(std::string title) -> CodeAction& { return *this; }

auto CodeAction::kind(std::optional<CodeActionKind> kind) -> CodeAction& {
  return *this;
}

auto CodeAction::diagnostics(std::optional<Vector<Diagnostic>> diagnostics)
    -> CodeAction& {
  return *this;
}

auto CodeAction::isPreferred(std::optional<bool> isPreferred) -> CodeAction& {
  return *this;
}

auto CodeAction::disabled(std::optional<CodeActionDisabled> disabled)
    -> CodeAction& {
  return *this;
}

auto CodeAction::edit(std::optional<WorkspaceEdit> edit) -> CodeAction& {
  return *this;
}

auto CodeAction::command(std::optional<Command> command) -> CodeAction& {
  return *this;
}

auto CodeAction::data(std::optional<LSPAny> data) -> CodeAction& {
  return *this;
}

auto CodeAction::tags(std::optional<Vector<CodeActionTag>> tags)
    -> CodeAction& {
  return *this;
}

CodeActionRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto CodeActionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionRegistrationOptions::codeActionKinds() const
    -> std::optional<Vector<CodeActionKind>> {
  if (!repr_.contains("codeActionKinds")) return std::nullopt;

  const auto& value = repr_["codeActionKinds"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto CodeActionRegistrationOptions::documentation() const
    -> std::optional<Vector<CodeActionKindDocumentation>> {
  if (!repr_.contains("documentation")) return std::nullopt;

  const auto& value = repr_["documentation"];

  assert(value.is_array());
  return Vector<CodeActionKindDocumentation>(value);
}

auto CodeActionRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CodeActionRegistrationOptions& {
  return *this;
}

auto CodeActionRegistrationOptions::codeActionKinds(
    std::optional<Vector<CodeActionKind>> codeActionKinds)
    -> CodeActionRegistrationOptions& {
  return *this;
}

auto CodeActionRegistrationOptions::documentation(
    std::optional<Vector<CodeActionKindDocumentation>> documentation)
    -> CodeActionRegistrationOptions& {
  return *this;
}

auto CodeActionRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> CodeActionRegistrationOptions& {
  return *this;
}

auto CodeActionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CodeActionRegistrationOptions& {
  return *this;
}

WorkspaceSymbolParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("query")) return false;
  return true;
}

auto WorkspaceSymbolParams::query() const -> std::string {
  const auto& value = repr_["query"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceSymbolParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbolParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbolParams::query(std::string query) -> WorkspaceSymbolParams& {
  return *this;
}

auto WorkspaceSymbolParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> WorkspaceSymbolParams& {
  return *this;
}

auto WorkspaceSymbolParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> WorkspaceSymbolParams& {
  return *this;
}

WorkspaceSymbol::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("location")) return false;
  if (!repr_.contains("name")) return false;
  if (!repr_.contains("kind")) return false;
  return true;
}

auto WorkspaceSymbol::location() const
    -> std::variant<std::monostate, Location, LocationUriOnly> {
  const auto& value = repr_["location"];

  std::variant<std::monostate, Location, LocationUriOnly> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbol::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto WorkspaceSymbol::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceSymbol::kind() const -> SymbolKind {
  const auto& value = repr_["kind"];

  return SymbolKind(value);
}

auto WorkspaceSymbol::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto WorkspaceSymbol::containerName() const -> std::optional<std::string> {
  if (!repr_.contains("containerName")) return std::nullopt;

  const auto& value = repr_["containerName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceSymbol::location(
    std::variant<std::monostate, Location, LocationUriOnly> location)
    -> WorkspaceSymbol& {
  return *this;
}

auto WorkspaceSymbol::data(std::optional<LSPAny> data) -> WorkspaceSymbol& {
  return *this;
}

auto WorkspaceSymbol::name(std::string name) -> WorkspaceSymbol& {
  return *this;
}

auto WorkspaceSymbol::kind(SymbolKind kind) -> WorkspaceSymbol& {
  return *this;
}

auto WorkspaceSymbol::tags(std::optional<Vector<SymbolTag>> tags)
    -> WorkspaceSymbol& {
  return *this;
}

auto WorkspaceSymbol::containerName(std::optional<std::string> containerName)
    -> WorkspaceSymbol& {
  return *this;
}

WorkspaceSymbolRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceSymbolRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider)
    -> WorkspaceSymbolRegistrationOptions& {
  return *this;
}

auto WorkspaceSymbolRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> WorkspaceSymbolRegistrationOptions& {
  return *this;
}

CodeLensParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto CodeLensParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CodeLensParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensParams::textDocument(TextDocumentIdentifier textDocument)
    -> CodeLensParams& {
  return *this;
}

auto CodeLensParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> CodeLensParams& {
  return *this;
}

auto CodeLensParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> CodeLensParams& {
  return *this;
}

CodeLens::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto CodeLens::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto CodeLens::command() const -> std::optional<Command> {
  if (!repr_.contains("command")) return std::nullopt;

  const auto& value = repr_["command"];

  return Command(value);
}

auto CodeLens::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CodeLens::range(Range range) -> CodeLens& { return *this; }

auto CodeLens::command(std::optional<Command> command) -> CodeLens& {
  return *this;
}

auto CodeLens::data(std::optional<LSPAny> data) -> CodeLens& { return *this; }

CodeLensRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto CodeLensRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CodeLensRegistrationOptions& {
  return *this;
}

auto CodeLensRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> CodeLensRegistrationOptions& {
  return *this;
}

auto CodeLensRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CodeLensRegistrationOptions& {
  return *this;
}

DocumentLinkParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  return true;
}

auto DocumentLinkParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentLinkParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentLinkParams& {
  return *this;
}

auto DocumentLinkParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentLinkParams& {
  return *this;
}

auto DocumentLinkParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DocumentLinkParams& {
  return *this;
}

DocumentLink::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto DocumentLink::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto DocumentLink::target() const -> std::optional<std::string> {
  if (!repr_.contains("target")) return std::nullopt;

  const auto& value = repr_["target"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentLink::tooltip() const -> std::optional<std::string> {
  if (!repr_.contains("tooltip")) return std::nullopt;

  const auto& value = repr_["tooltip"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentLink::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto DocumentLink::range(Range range) -> DocumentLink& { return *this; }

auto DocumentLink::target(std::optional<std::string> target) -> DocumentLink& {
  return *this;
}

auto DocumentLink::tooltip(std::optional<std::string> tooltip)
    -> DocumentLink& {
  return *this;
}

auto DocumentLink::data(std::optional<LSPAny> data) -> DocumentLink& {
  return *this;
}

DocumentLinkRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DocumentLinkRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentLinkRegistrationOptions& {
  return *this;
}

auto DocumentLinkRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> DocumentLinkRegistrationOptions& {
  return *this;
}

auto DocumentLinkRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentLinkRegistrationOptions& {
  return *this;
}

DocumentFormattingParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("options")) return false;
  return true;
}

auto DocumentFormattingParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentFormattingParams::options() const -> FormattingOptions {
  const auto& value = repr_["options"];

  return FormattingOptions(value);
}

auto DocumentFormattingParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentFormattingParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentFormattingParams& {
  return *this;
}

auto DocumentFormattingParams::options(FormattingOptions options)
    -> DocumentFormattingParams& {
  return *this;
}

auto DocumentFormattingParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentFormattingParams& {
  return *this;
}

DocumentFormattingRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DocumentFormattingRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentFormattingRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentFormattingRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentFormattingRegistrationOptions& {
  return *this;
}

auto DocumentFormattingRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentFormattingRegistrationOptions& {
  return *this;
}

DocumentRangeFormattingParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("options")) return false;
  return true;
}

auto DocumentRangeFormattingParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentRangeFormattingParams::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto DocumentRangeFormattingParams::options() const -> FormattingOptions {
  const auto& value = repr_["options"];

  return FormattingOptions(value);
}

auto DocumentRangeFormattingParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangeFormattingParams::textDocument(
    TextDocumentIdentifier textDocument) -> DocumentRangeFormattingParams& {
  return *this;
}

auto DocumentRangeFormattingParams::range(Range range)
    -> DocumentRangeFormattingParams& {
  return *this;
}

auto DocumentRangeFormattingParams::options(FormattingOptions options)
    -> DocumentRangeFormattingParams& {
  return *this;
}

auto DocumentRangeFormattingParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> DocumentRangeFormattingParams& {
  return *this;
}

DocumentRangeFormattingRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto DocumentRangeFormattingRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangeFormattingRegistrationOptions::rangesSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("rangesSupport")) return std::nullopt;

  const auto& value = repr_["rangesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentRangeFormattingRegistrationOptions& {
  return *this;
}

auto DocumentRangeFormattingRegistrationOptions::rangesSupport(
    std::optional<bool> rangesSupport)
    -> DocumentRangeFormattingRegistrationOptions& {
  return *this;
}

auto DocumentRangeFormattingRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentRangeFormattingRegistrationOptions& {
  return *this;
}

DocumentRangesFormattingParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("ranges")) return false;
  if (!repr_.contains("options")) return false;
  return true;
}

auto DocumentRangesFormattingParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentRangesFormattingParams::ranges() const -> Vector<Range> {
  const auto& value = repr_["ranges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto DocumentRangesFormattingParams::options() const -> FormattingOptions {
  const auto& value = repr_["options"];

  return FormattingOptions(value);
}

auto DocumentRangesFormattingParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangesFormattingParams::textDocument(
    TextDocumentIdentifier textDocument) -> DocumentRangesFormattingParams& {
  return *this;
}

auto DocumentRangesFormattingParams::ranges(Vector<Range> ranges)
    -> DocumentRangesFormattingParams& {
  return *this;
}

auto DocumentRangesFormattingParams::options(FormattingOptions options)
    -> DocumentRangesFormattingParams& {
  return *this;
}

auto DocumentRangesFormattingParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> DocumentRangesFormattingParams& {
  return *this;
}

DocumentOnTypeFormattingParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  if (!repr_.contains("ch")) return false;
  if (!repr_.contains("options")) return false;
  return true;
}

auto DocumentOnTypeFormattingParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentOnTypeFormattingParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto DocumentOnTypeFormattingParams::ch() const -> std::string {
  const auto& value = repr_["ch"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentOnTypeFormattingParams::options() const -> FormattingOptions {
  const auto& value = repr_["options"];

  return FormattingOptions(value);
}

auto DocumentOnTypeFormattingParams::textDocument(
    TextDocumentIdentifier textDocument) -> DocumentOnTypeFormattingParams& {
  return *this;
}

auto DocumentOnTypeFormattingParams::position(Position position)
    -> DocumentOnTypeFormattingParams& {
  return *this;
}

auto DocumentOnTypeFormattingParams::ch(std::string ch)
    -> DocumentOnTypeFormattingParams& {
  return *this;
}

auto DocumentOnTypeFormattingParams::options(FormattingOptions options)
    -> DocumentOnTypeFormattingParams& {
  return *this;
}

DocumentOnTypeFormattingRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  if (!repr_.contains("firstTriggerCharacter")) return false;
  return true;
}

auto DocumentOnTypeFormattingRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentOnTypeFormattingRegistrationOptions::firstTriggerCharacter() const
    -> std::string {
  const auto& value = repr_["firstTriggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentOnTypeFormattingRegistrationOptions::moreTriggerCharacter() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("moreTriggerCharacter")) return std::nullopt;

  const auto& value = repr_["moreTriggerCharacter"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto DocumentOnTypeFormattingRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentOnTypeFormattingRegistrationOptions& {
  return *this;
}

auto DocumentOnTypeFormattingRegistrationOptions::firstTriggerCharacter(
    std::string firstTriggerCharacter)
    -> DocumentOnTypeFormattingRegistrationOptions& {
  return *this;
}

auto DocumentOnTypeFormattingRegistrationOptions::moreTriggerCharacter(
    std::optional<Vector<std::string>> moreTriggerCharacter)
    -> DocumentOnTypeFormattingRegistrationOptions& {
  return *this;
}

RenameParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  if (!repr_.contains("newName")) return false;
  return true;
}

auto RenameParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto RenameParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto RenameParams::newName() const -> std::string {
  const auto& value = repr_["newName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto RenameParams::textDocument(TextDocumentIdentifier textDocument)
    -> RenameParams& {
  return *this;
}

auto RenameParams::position(Position position) -> RenameParams& {
  return *this;
}

auto RenameParams::newName(std::string newName) -> RenameParams& {
  return *this;
}

auto RenameParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> RenameParams& {
  return *this;
}

RenameRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("documentSelector")) return false;
  return true;
}

auto RenameRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  const auto& value = repr_["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto RenameRegistrationOptions::prepareProvider() const -> std::optional<bool> {
  if (!repr_.contains("prepareProvider")) return std::nullopt;

  const auto& value = repr_["prepareProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> RenameRegistrationOptions& {
  return *this;
}

auto RenameRegistrationOptions::prepareProvider(
    std::optional<bool> prepareProvider) -> RenameRegistrationOptions& {
  return *this;
}

auto RenameRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> RenameRegistrationOptions& {
  return *this;
}

PrepareRenameParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto PrepareRenameParams::textDocument() const -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto PrepareRenameParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto PrepareRenameParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto PrepareRenameParams::textDocument(TextDocumentIdentifier textDocument)
    -> PrepareRenameParams& {
  return *this;
}

auto PrepareRenameParams::position(Position position) -> PrepareRenameParams& {
  return *this;
}

auto PrepareRenameParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> PrepareRenameParams& {
  return *this;
}

ExecuteCommandParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("command")) return false;
  return true;
}

auto ExecuteCommandParams::command() const -> std::string {
  const auto& value = repr_["command"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ExecuteCommandParams::arguments() const -> std::optional<Vector<LSPAny>> {
  if (!repr_.contains("arguments")) return std::nullopt;

  const auto& value = repr_["arguments"];

  assert(value.is_array());
  return Vector<LSPAny>(value);
}

auto ExecuteCommandParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ExecuteCommandParams::command(std::string command)
    -> ExecuteCommandParams& {
  return *this;
}

auto ExecuteCommandParams::arguments(std::optional<Vector<LSPAny>> arguments)
    -> ExecuteCommandParams& {
  return *this;
}

auto ExecuteCommandParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> ExecuteCommandParams& {
  return *this;
}

ExecuteCommandRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("commands")) return false;
  return true;
}

auto ExecuteCommandRegistrationOptions::commands() const
    -> Vector<std::string> {
  const auto& value = repr_["commands"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ExecuteCommandRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecuteCommandRegistrationOptions::commands(Vector<std::string> commands)
    -> ExecuteCommandRegistrationOptions& {
  return *this;
}

auto ExecuteCommandRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> ExecuteCommandRegistrationOptions& {
  return *this;
}

ApplyWorkspaceEditParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("edit")) return false;
  return true;
}

auto ApplyWorkspaceEditParams::label() const -> std::optional<std::string> {
  if (!repr_.contains("label")) return std::nullopt;

  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ApplyWorkspaceEditParams::edit() const -> WorkspaceEdit {
  const auto& value = repr_["edit"];

  return WorkspaceEdit(value);
}

auto ApplyWorkspaceEditParams::metadata() const
    -> std::optional<WorkspaceEditMetadata> {
  if (!repr_.contains("metadata")) return std::nullopt;

  const auto& value = repr_["metadata"];

  return WorkspaceEditMetadata(value);
}

auto ApplyWorkspaceEditParams::label(std::optional<std::string> label)
    -> ApplyWorkspaceEditParams& {
  return *this;
}

auto ApplyWorkspaceEditParams::edit(WorkspaceEdit edit)
    -> ApplyWorkspaceEditParams& {
  return *this;
}

auto ApplyWorkspaceEditParams::metadata(
    std::optional<WorkspaceEditMetadata> metadata)
    -> ApplyWorkspaceEditParams& {
  return *this;
}

ApplyWorkspaceEditResult::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("applied")) return false;
  return true;
}

auto ApplyWorkspaceEditResult::applied() const -> bool {
  const auto& value = repr_["applied"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ApplyWorkspaceEditResult::failureReason() const
    -> std::optional<std::string> {
  if (!repr_.contains("failureReason")) return std::nullopt;

  const auto& value = repr_["failureReason"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ApplyWorkspaceEditResult::failedChange() const -> std::optional<long> {
  if (!repr_.contains("failedChange")) return std::nullopt;

  const auto& value = repr_["failedChange"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto ApplyWorkspaceEditResult::applied(bool applied)
    -> ApplyWorkspaceEditResult& {
  return *this;
}

auto ApplyWorkspaceEditResult::failureReason(
    std::optional<std::string> failureReason) -> ApplyWorkspaceEditResult& {
  return *this;
}

auto ApplyWorkspaceEditResult::failedChange(std::optional<long> failedChange)
    -> ApplyWorkspaceEditResult& {
  return *this;
}

WorkDoneProgressBegin::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "begin") return false;
  if (!repr_.contains("title")) return false;
  return true;
}

auto WorkDoneProgressBegin::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressBegin::title() const -> std::string {
  const auto& value = repr_["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressBegin::cancellable() const -> std::optional<bool> {
  if (!repr_.contains("cancellable")) return std::nullopt;

  const auto& value = repr_["cancellable"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkDoneProgressBegin::message() const -> std::optional<std::string> {
  if (!repr_.contains("message")) return std::nullopt;

  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressBegin::percentage() const -> std::optional<long> {
  if (!repr_.contains("percentage")) return std::nullopt;

  const auto& value = repr_["percentage"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto WorkDoneProgressBegin::kind(std::string kind) -> WorkDoneProgressBegin& {
  return *this;
}

auto WorkDoneProgressBegin::title(std::string title) -> WorkDoneProgressBegin& {
  return *this;
}

auto WorkDoneProgressBegin::cancellable(std::optional<bool> cancellable)
    -> WorkDoneProgressBegin& {
  return *this;
}

auto WorkDoneProgressBegin::message(std::optional<std::string> message)
    -> WorkDoneProgressBegin& {
  return *this;
}

auto WorkDoneProgressBegin::percentage(std::optional<long> percentage)
    -> WorkDoneProgressBegin& {
  return *this;
}

WorkDoneProgressReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "report") return false;
  return true;
}

auto WorkDoneProgressReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressReport::cancellable() const -> std::optional<bool> {
  if (!repr_.contains("cancellable")) return std::nullopt;

  const auto& value = repr_["cancellable"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkDoneProgressReport::message() const -> std::optional<std::string> {
  if (!repr_.contains("message")) return std::nullopt;

  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressReport::percentage() const -> std::optional<long> {
  if (!repr_.contains("percentage")) return std::nullopt;

  const auto& value = repr_["percentage"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto WorkDoneProgressReport::kind(std::string kind) -> WorkDoneProgressReport& {
  return *this;
}

auto WorkDoneProgressReport::cancellable(std::optional<bool> cancellable)
    -> WorkDoneProgressReport& {
  return *this;
}

auto WorkDoneProgressReport::message(std::optional<std::string> message)
    -> WorkDoneProgressReport& {
  return *this;
}

auto WorkDoneProgressReport::percentage(std::optional<long> percentage)
    -> WorkDoneProgressReport& {
  return *this;
}

WorkDoneProgressEnd::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "end") return false;
  return true;
}

auto WorkDoneProgressEnd::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressEnd::message() const -> std::optional<std::string> {
  if (!repr_.contains("message")) return std::nullopt;

  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressEnd::kind(std::string kind) -> WorkDoneProgressEnd& {
  return *this;
}

auto WorkDoneProgressEnd::message(std::optional<std::string> message)
    -> WorkDoneProgressEnd& {
  return *this;
}

SetTraceParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto SetTraceParams::value() const -> TraceValue {
  const auto& value = repr_["value"];

  lsp_runtime_error("SetTraceParams::value: not implement yet");
}

auto SetTraceParams::value(TraceValue value) -> SetTraceParams& {
  return *this;
}

LogTraceParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("message")) return false;
  return true;
}

auto LogTraceParams::message() const -> std::string {
  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LogTraceParams::verbose() const -> std::optional<std::string> {
  if (!repr_.contains("verbose")) return std::nullopt;

  const auto& value = repr_["verbose"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LogTraceParams::message(std::string message) -> LogTraceParams& {
  return *this;
}

auto LogTraceParams::verbose(std::optional<std::string> verbose)
    -> LogTraceParams& {
  return *this;
}

CancelParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("id")) return false;
  return true;
}

auto CancelParams::id() const
    -> std::variant<std::monostate, int, std::string> {
  const auto& value = repr_["id"];

  std::variant<std::monostate, int, std::string> result;

  details::try_emplace(result, value);

  return result;
}

auto CancelParams::id(std::variant<std::monostate, int, std::string> id)
    -> CancelParams& {
  return *this;
}

ProgressParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("token")) return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto ProgressParams::token() const -> ProgressToken {
  const auto& value = repr_["token"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ProgressParams::value() const -> LSPAny {
  const auto& value = repr_["value"];

  assert(value.is_object());
  return LSPAny(value);
}

auto ProgressParams::token(ProgressToken token) -> ProgressParams& {
  return *this;
}

auto ProgressParams::value(LSPAny value) -> ProgressParams& { return *this; }

TextDocumentPositionParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("position")) return false;
  return true;
}

auto TextDocumentPositionParams::textDocument() const
    -> TextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return TextDocumentIdentifier(value);
}

auto TextDocumentPositionParams::position() const -> Position {
  const auto& value = repr_["position"];

  return Position(value);
}

auto TextDocumentPositionParams::textDocument(
    TextDocumentIdentifier textDocument) -> TextDocumentPositionParams& {
  return *this;
}

auto TextDocumentPositionParams::position(Position position)
    -> TextDocumentPositionParams& {
  return *this;
}

WorkDoneProgressParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkDoneProgressParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> WorkDoneProgressParams& {
  return *this;
}

PartialResultParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto PartialResultParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_.contains("partialResultToken")) return std::nullopt;

  const auto& value = repr_["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto PartialResultParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> PartialResultParams& {
  return *this;
}

LocationLink::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("targetUri")) return false;
  if (!repr_.contains("targetRange")) return false;
  if (!repr_.contains("targetSelectionRange")) return false;
  return true;
}

auto LocationLink::originSelectionRange() const -> std::optional<Range> {
  if (!repr_.contains("originSelectionRange")) return std::nullopt;

  const auto& value = repr_["originSelectionRange"];

  return Range(value);
}

auto LocationLink::targetUri() const -> std::string {
  const auto& value = repr_["targetUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LocationLink::targetRange() const -> Range {
  const auto& value = repr_["targetRange"];

  return Range(value);
}

auto LocationLink::targetSelectionRange() const -> Range {
  const auto& value = repr_["targetSelectionRange"];

  return Range(value);
}

auto LocationLink::originSelectionRange(
    std::optional<Range> originSelectionRange) -> LocationLink& {
  return *this;
}

auto LocationLink::targetUri(std::string targetUri) -> LocationLink& {
  return *this;
}

auto LocationLink::targetRange(Range targetRange) -> LocationLink& {
  return *this;
}

auto LocationLink::targetSelectionRange(Range targetSelectionRange)
    -> LocationLink& {
  return *this;
}

Range::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("start")) return false;
  if (!repr_.contains("end")) return false;
  return true;
}

auto Range::start() const -> Position {
  const auto& value = repr_["start"];

  return Position(value);
}

auto Range::end() const -> Position {
  const auto& value = repr_["end"];

  return Position(value);
}

auto Range::start(Position start) -> Range& { return *this; }

auto Range::end(Position end) -> Range& { return *this; }

ImplementationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ImplementationOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> ImplementationOptions& {
  return *this;
}

StaticRegistrationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto StaticRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_.contains("id")) return std::nullopt;

  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto StaticRegistrationOptions::id(std::optional<std::string> id)
    -> StaticRegistrationOptions& {
  return *this;
}

TypeDefinitionOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TypeDefinitionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> TypeDefinitionOptions& {
  return *this;
}

WorkspaceFoldersChangeEvent::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("added")) return false;
  if (!repr_.contains("removed")) return false;
  return true;
}

auto WorkspaceFoldersChangeEvent::added() const -> Vector<WorkspaceFolder> {
  const auto& value = repr_["added"];

  assert(value.is_array());
  return Vector<WorkspaceFolder>(value);
}

auto WorkspaceFoldersChangeEvent::removed() const -> Vector<WorkspaceFolder> {
  const auto& value = repr_["removed"];

  assert(value.is_array());
  return Vector<WorkspaceFolder>(value);
}

auto WorkspaceFoldersChangeEvent::added(Vector<WorkspaceFolder> added)
    -> WorkspaceFoldersChangeEvent& {
  return *this;
}

auto WorkspaceFoldersChangeEvent::removed(Vector<WorkspaceFolder> removed)
    -> WorkspaceFoldersChangeEvent& {
  return *this;
}

ConfigurationItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ConfigurationItem::scopeUri() const -> std::optional<std::string> {
  if (!repr_.contains("scopeUri")) return std::nullopt;

  const auto& value = repr_["scopeUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ConfigurationItem::section() const -> std::optional<std::string> {
  if (!repr_.contains("section")) return std::nullopt;

  const auto& value = repr_["section"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ConfigurationItem::scopeUri(std::optional<std::string> scopeUri)
    -> ConfigurationItem& {
  return *this;
}

auto ConfigurationItem::section(std::optional<std::string> section)
    -> ConfigurationItem& {
  return *this;
}

TextDocumentIdentifier::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto TextDocumentIdentifier::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentIdentifier::uri(std::string uri) -> TextDocumentIdentifier& {
  return *this;
}

Color::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("red")) return false;
  if (!repr_.contains("green")) return false;
  if (!repr_.contains("blue")) return false;
  if (!repr_.contains("alpha")) return false;
  return true;
}

auto Color::red() const -> double {
  const auto& value = repr_["red"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::green() const -> double {
  const auto& value = repr_["green"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::blue() const -> double {
  const auto& value = repr_["blue"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::alpha() const -> double {
  const auto& value = repr_["alpha"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::red(double red) -> Color& { return *this; }

auto Color::green(double green) -> Color& { return *this; }

auto Color::blue(double blue) -> Color& { return *this; }

auto Color::alpha(double alpha) -> Color& { return *this; }

DocumentColorOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentColorOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentColorOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentColorOptions& {
  return *this;
}

FoldingRangeOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto FoldingRangeOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> FoldingRangeOptions& {
  return *this;
}

DeclarationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DeclarationOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DeclarationOptions& {
  return *this;
}

Position::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("line")) return false;
  if (!repr_.contains("character")) return false;
  return true;
}

auto Position::line() const -> long {
  const auto& value = repr_["line"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto Position::character() const -> long {
  const auto& value = repr_["character"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto Position::line(long line) -> Position& { return *this; }

auto Position::character(long character) -> Position& { return *this; }

SelectionRangeOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SelectionRangeOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SelectionRangeOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SelectionRangeOptions& {
  return *this;
}

CallHierarchyOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CallHierarchyOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CallHierarchyOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CallHierarchyOptions& {
  return *this;
}

SemanticTokensOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("legend")) return false;
  return true;
}

auto SemanticTokensOptions::legend() const -> SemanticTokensLegend {
  const auto& value = repr_["legend"];

  return SemanticTokensLegend(value);
}

auto SemanticTokensOptions::range() const
    -> std::optional<std::variant<std::monostate, bool, json>> {
  if (!repr_.contains("range")) return std::nullopt;

  const auto& value = repr_["range"];

  std::variant<std::monostate, bool, json> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensOptions::full() const -> std::optional<
    std::variant<std::monostate, bool, SemanticTokensFullDelta>> {
  if (!repr_.contains("full")) return std::nullopt;

  const auto& value = repr_["full"];

  std::variant<std::monostate, bool, SemanticTokensFullDelta> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensOptions::legend(SemanticTokensLegend legend)
    -> SemanticTokensOptions& {
  return *this;
}

auto SemanticTokensOptions::range(
    std::optional<std::variant<std::monostate, bool, json>> range)
    -> SemanticTokensOptions& {
  return *this;
}

auto SemanticTokensOptions::full(
    std::optional<std::variant<std::monostate, bool, SemanticTokensFullDelta>>
        full) -> SemanticTokensOptions& {
  return *this;
}

auto SemanticTokensOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SemanticTokensOptions& {
  return *this;
}

SemanticTokensEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("start")) return false;
  if (!repr_.contains("deleteCount")) return false;
  return true;
}

auto SemanticTokensEdit::start() const -> long {
  const auto& value = repr_["start"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto SemanticTokensEdit::deleteCount() const -> long {
  const auto& value = repr_["deleteCount"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto SemanticTokensEdit::data() const -> std::optional<Vector<long>> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_array());
  return Vector<long>(value);
}

auto SemanticTokensEdit::start(long start) -> SemanticTokensEdit& {
  return *this;
}

auto SemanticTokensEdit::deleteCount(long deleteCount) -> SemanticTokensEdit& {
  return *this;
}

auto SemanticTokensEdit::data(std::optional<Vector<long>> data)
    -> SemanticTokensEdit& {
  return *this;
}

LinkedEditingRangeOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto LinkedEditingRangeOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto LinkedEditingRangeOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> LinkedEditingRangeOptions& {
  return *this;
}

FileCreate::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto FileCreate::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileCreate::uri(std::string uri) -> FileCreate& { return *this; }

TextDocumentEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("textDocument")) return false;
  if (!repr_.contains("edits")) return false;
  return true;
}

auto TextDocumentEdit::textDocument() const
    -> OptionalVersionedTextDocumentIdentifier {
  const auto& value = repr_["textDocument"];

  return OptionalVersionedTextDocumentIdentifier(value);
}

auto TextDocumentEdit::edits() const
    -> Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                           SnippetTextEdit>> {
  const auto& value = repr_["edits"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                             SnippetTextEdit>>(value);
}

auto TextDocumentEdit::textDocument(
    OptionalVersionedTextDocumentIdentifier textDocument) -> TextDocumentEdit& {
  return *this;
}

auto TextDocumentEdit::edits(
    Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                        SnippetTextEdit>>
        edits) -> TextDocumentEdit& {
  return *this;
}

CreateFile::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "create") return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto CreateFile::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CreateFile::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CreateFile::options() const -> std::optional<CreateFileOptions> {
  if (!repr_.contains("options")) return std::nullopt;

  const auto& value = repr_["options"];

  return CreateFileOptions(value);
}

auto CreateFile::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_.contains("annotationId")) return std::nullopt;

  const auto& value = repr_["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CreateFile::kind(std::string kind) -> CreateFile& { return *this; }

auto CreateFile::uri(std::string uri) -> CreateFile& { return *this; }

auto CreateFile::options(std::optional<CreateFileOptions> options)
    -> CreateFile& {
  return *this;
}

auto CreateFile::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId) -> CreateFile& {
  return *this;
}

RenameFile::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "rename") return false;
  if (!repr_.contains("oldUri")) return false;
  if (!repr_.contains("newUri")) return false;
  return true;
}

auto RenameFile::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::oldUri() const -> std::string {
  const auto& value = repr_["oldUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::newUri() const -> std::string {
  const auto& value = repr_["newUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::options() const -> std::optional<RenameFileOptions> {
  if (!repr_.contains("options")) return std::nullopt;

  const auto& value = repr_["options"];

  return RenameFileOptions(value);
}

auto RenameFile::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_.contains("annotationId")) return std::nullopt;

  const auto& value = repr_["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::kind(std::string kind) -> RenameFile& { return *this; }

auto RenameFile::oldUri(std::string oldUri) -> RenameFile& { return *this; }

auto RenameFile::newUri(std::string newUri) -> RenameFile& { return *this; }

auto RenameFile::options(std::optional<RenameFileOptions> options)
    -> RenameFile& {
  return *this;
}

auto RenameFile::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId) -> RenameFile& {
  return *this;
}

DeleteFile::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "delete") return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto DeleteFile::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeleteFile::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeleteFile::options() const -> std::optional<DeleteFileOptions> {
  if (!repr_.contains("options")) return std::nullopt;

  const auto& value = repr_["options"];

  return DeleteFileOptions(value);
}

auto DeleteFile::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_.contains("annotationId")) return std::nullopt;

  const auto& value = repr_["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeleteFile::kind(std::string kind) -> DeleteFile& { return *this; }

auto DeleteFile::uri(std::string uri) -> DeleteFile& { return *this; }

auto DeleteFile::options(std::optional<DeleteFileOptions> options)
    -> DeleteFile& {
  return *this;
}

auto DeleteFile::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId) -> DeleteFile& {
  return *this;
}

ChangeAnnotation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("label")) return false;
  return true;
}

auto ChangeAnnotation::label() const -> std::string {
  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ChangeAnnotation::needsConfirmation() const -> std::optional<bool> {
  if (!repr_.contains("needsConfirmation")) return std::nullopt;

  const auto& value = repr_["needsConfirmation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ChangeAnnotation::description() const -> std::optional<std::string> {
  if (!repr_.contains("description")) return std::nullopt;

  const auto& value = repr_["description"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ChangeAnnotation::label(std::string label) -> ChangeAnnotation& {
  return *this;
}

auto ChangeAnnotation::needsConfirmation(std::optional<bool> needsConfirmation)
    -> ChangeAnnotation& {
  return *this;
}

auto ChangeAnnotation::description(std::optional<std::string> description)
    -> ChangeAnnotation& {
  return *this;
}

FileOperationFilter::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("pattern")) return false;
  return true;
}

auto FileOperationFilter::scheme() const -> std::optional<std::string> {
  if (!repr_.contains("scheme")) return std::nullopt;

  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileOperationFilter::pattern() const -> FileOperationPattern {
  const auto& value = repr_["pattern"];

  return FileOperationPattern(value);
}

auto FileOperationFilter::scheme(std::optional<std::string> scheme)
    -> FileOperationFilter& {
  return *this;
}

auto FileOperationFilter::pattern(FileOperationPattern pattern)
    -> FileOperationFilter& {
  return *this;
}

FileRename::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("oldUri")) return false;
  if (!repr_.contains("newUri")) return false;
  return true;
}

auto FileRename::oldUri() const -> std::string {
  const auto& value = repr_["oldUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileRename::newUri() const -> std::string {
  const auto& value = repr_["newUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileRename::oldUri(std::string oldUri) -> FileRename& { return *this; }

auto FileRename::newUri(std::string newUri) -> FileRename& { return *this; }

FileDelete::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto FileDelete::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileDelete::uri(std::string uri) -> FileDelete& { return *this; }

MonikerOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto MonikerOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto MonikerOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> MonikerOptions& {
  return *this;
}

TypeHierarchyOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TypeHierarchyOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeHierarchyOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> TypeHierarchyOptions& {
  return *this;
}

InlineValueContext::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("frameId")) return false;
  if (!repr_.contains("stoppedLocation")) return false;
  return true;
}

auto InlineValueContext::frameId() const -> int {
  const auto& value = repr_["frameId"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto InlineValueContext::stoppedLocation() const -> Range {
  const auto& value = repr_["stoppedLocation"];

  return Range(value);
}

auto InlineValueContext::frameId(int frameId) -> InlineValueContext& {
  return *this;
}

auto InlineValueContext::stoppedLocation(Range stoppedLocation)
    -> InlineValueContext& {
  return *this;
}

InlineValueText::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("text")) return false;
  return true;
}

auto InlineValueText::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto InlineValueText::text() const -> std::string {
  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueText::range(Range range) -> InlineValueText& { return *this; }

auto InlineValueText::text(std::string text) -> InlineValueText& {
  return *this;
}

InlineValueVariableLookup::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("caseSensitiveLookup")) return false;
  return true;
}

auto InlineValueVariableLookup::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto InlineValueVariableLookup::variableName() const
    -> std::optional<std::string> {
  if (!repr_.contains("variableName")) return std::nullopt;

  const auto& value = repr_["variableName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueVariableLookup::caseSensitiveLookup() const -> bool {
  const auto& value = repr_["caseSensitiveLookup"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueVariableLookup::range(Range range)
    -> InlineValueVariableLookup& {
  return *this;
}

auto InlineValueVariableLookup::variableName(
    std::optional<std::string> variableName) -> InlineValueVariableLookup& {
  return *this;
}

auto InlineValueVariableLookup::caseSensitiveLookup(bool caseSensitiveLookup)
    -> InlineValueVariableLookup& {
  return *this;
}

InlineValueEvaluatableExpression::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  return true;
}

auto InlineValueEvaluatableExpression::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto InlineValueEvaluatableExpression::expression() const
    -> std::optional<std::string> {
  if (!repr_.contains("expression")) return std::nullopt;

  const auto& value = repr_["expression"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueEvaluatableExpression::range(Range range)
    -> InlineValueEvaluatableExpression& {
  return *this;
}

auto InlineValueEvaluatableExpression::expression(
    std::optional<std::string> expression)
    -> InlineValueEvaluatableExpression& {
  return *this;
}

InlineValueOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlineValueOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> InlineValueOptions& {
  return *this;
}

InlayHintLabelPart::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto InlayHintLabelPart::value() const -> std::string {
  const auto& value = repr_["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlayHintLabelPart::tooltip() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_.contains("tooltip")) return std::nullopt;

  const auto& value = repr_["tooltip"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintLabelPart::location() const -> std::optional<Location> {
  if (!repr_.contains("location")) return std::nullopt;

  const auto& value = repr_["location"];

  return Location(value);
}

auto InlayHintLabelPart::command() const -> std::optional<Command> {
  if (!repr_.contains("command")) return std::nullopt;

  const auto& value = repr_["command"];

  return Command(value);
}

auto InlayHintLabelPart::value(std::string value) -> InlayHintLabelPart& {
  return *this;
}

auto InlayHintLabelPart::tooltip(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        tooltip) -> InlayHintLabelPart& {
  return *this;
}

auto InlayHintLabelPart::location(std::optional<Location> location)
    -> InlayHintLabelPart& {
  return *this;
}

auto InlayHintLabelPart::command(std::optional<Command> command)
    -> InlayHintLabelPart& {
  return *this;
}

MarkupContent::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto MarkupContent::kind() const -> MarkupKind {
  const auto& value = repr_["kind"];

  lsp_runtime_error("MarkupContent::kind: not implement yet");
}

auto MarkupContent::value() const -> std::string {
  const auto& value = repr_["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkupContent::kind(MarkupKind kind) -> MarkupContent& { return *this; }

auto MarkupContent::value(std::string value) -> MarkupContent& { return *this; }

InlayHintOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlayHintOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> InlayHintOptions& {
  return *this;
}

auto InlayHintOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> InlayHintOptions& {
  return *this;
}

RelatedFullDocumentDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "full") return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto RelatedFullDocumentDiagnosticReport::relatedDocuments() const
    -> std::optional<Map<
        std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                  UnchangedDocumentDiagnosticReport>>> {
  if (!repr_.contains("relatedDocuments")) return std::nullopt;

  const auto& value = repr_["relatedDocuments"];

  assert(value.is_object());
  return Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>(value);
}

auto RelatedFullDocumentDiagnosticReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedFullDocumentDiagnosticReport::resultId() const
    -> std::optional<std::string> {
  if (!repr_.contains("resultId")) return std::nullopt;

  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedFullDocumentDiagnosticReport::items() const -> Vector<Diagnostic> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto RelatedFullDocumentDiagnosticReport::relatedDocuments(
    std::optional<Map<std::string,
                      std::variant<std::monostate, FullDocumentDiagnosticReport,
                                   UnchangedDocumentDiagnosticReport>>>
        relatedDocuments) -> RelatedFullDocumentDiagnosticReport& {
  return *this;
}

auto RelatedFullDocumentDiagnosticReport::kind(std::string kind)
    -> RelatedFullDocumentDiagnosticReport& {
  return *this;
}

auto RelatedFullDocumentDiagnosticReport::resultId(
    std::optional<std::string> resultId)
    -> RelatedFullDocumentDiagnosticReport& {
  return *this;
}

auto RelatedFullDocumentDiagnosticReport::items(Vector<Diagnostic> items)
    -> RelatedFullDocumentDiagnosticReport& {
  return *this;
}

RelatedUnchangedDocumentDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "unchanged") return false;
  if (!repr_.contains("resultId")) return false;
  return true;
}

auto RelatedUnchangedDocumentDiagnosticReport::relatedDocuments() const
    -> std::optional<Map<
        std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                  UnchangedDocumentDiagnosticReport>>> {
  if (!repr_.contains("relatedDocuments")) return std::nullopt;

  const auto& value = repr_["relatedDocuments"];

  assert(value.is_object());
  return Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>(value);
}

auto RelatedUnchangedDocumentDiagnosticReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedUnchangedDocumentDiagnosticReport::resultId() const -> std::string {
  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedUnchangedDocumentDiagnosticReport::relatedDocuments(
    std::optional<Map<std::string,
                      std::variant<std::monostate, FullDocumentDiagnosticReport,
                                   UnchangedDocumentDiagnosticReport>>>
        relatedDocuments) -> RelatedUnchangedDocumentDiagnosticReport& {
  return *this;
}

auto RelatedUnchangedDocumentDiagnosticReport::kind(std::string kind)
    -> RelatedUnchangedDocumentDiagnosticReport& {
  return *this;
}

auto RelatedUnchangedDocumentDiagnosticReport::resultId(std::string resultId)
    -> RelatedUnchangedDocumentDiagnosticReport& {
  return *this;
}

FullDocumentDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "full") return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto FullDocumentDiagnosticReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FullDocumentDiagnosticReport::resultId() const
    -> std::optional<std::string> {
  if (!repr_.contains("resultId")) return std::nullopt;

  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FullDocumentDiagnosticReport::items() const -> Vector<Diagnostic> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto FullDocumentDiagnosticReport::kind(std::string kind)
    -> FullDocumentDiagnosticReport& {
  return *this;
}

auto FullDocumentDiagnosticReport::resultId(std::optional<std::string> resultId)
    -> FullDocumentDiagnosticReport& {
  return *this;
}

auto FullDocumentDiagnosticReport::items(Vector<Diagnostic> items)
    -> FullDocumentDiagnosticReport& {
  return *this;
}

UnchangedDocumentDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "unchanged") return false;
  if (!repr_.contains("resultId")) return false;
  return true;
}

auto UnchangedDocumentDiagnosticReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto UnchangedDocumentDiagnosticReport::resultId() const -> std::string {
  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto UnchangedDocumentDiagnosticReport::kind(std::string kind)
    -> UnchangedDocumentDiagnosticReport& {
  return *this;
}

auto UnchangedDocumentDiagnosticReport::resultId(std::string resultId)
    -> UnchangedDocumentDiagnosticReport& {
  return *this;
}

DiagnosticOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("interFileDependencies")) return false;
  if (!repr_.contains("workspaceDiagnostics")) return false;
  return true;
}

auto DiagnosticOptions::identifier() const -> std::optional<std::string> {
  if (!repr_.contains("identifier")) return std::nullopt;

  const auto& value = repr_["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticOptions::interFileDependencies() const -> bool {
  const auto& value = repr_["interFileDependencies"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticOptions::workspaceDiagnostics() const -> bool {
  const auto& value = repr_["workspaceDiagnostics"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticOptions::identifier(std::optional<std::string> identifier)
    -> DiagnosticOptions& {
  return *this;
}

auto DiagnosticOptions::interFileDependencies(bool interFileDependencies)
    -> DiagnosticOptions& {
  return *this;
}

auto DiagnosticOptions::workspaceDiagnostics(bool workspaceDiagnostics)
    -> DiagnosticOptions& {
  return *this;
}

auto DiagnosticOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DiagnosticOptions& {
  return *this;
}

PreviousResultId::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto PreviousResultId::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PreviousResultId::value() const -> std::string {
  const auto& value = repr_["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PreviousResultId::uri(std::string uri) -> PreviousResultId& {
  return *this;
}

auto PreviousResultId::value(std::string value) -> PreviousResultId& {
  return *this;
}

NotebookDocument::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("notebookType")) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("cells")) return false;
  return true;
}

auto NotebookDocument::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocument::notebookType() const -> std::string {
  const auto& value = repr_["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocument::version() const -> int {
  const auto& value = repr_["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto NotebookDocument::metadata() const -> std::optional<LSPObject> {
  if (!repr_.contains("metadata")) return std::nullopt;

  const auto& value = repr_["metadata"];

  assert(value.is_object());
  return LSPObject(value);
}

auto NotebookDocument::cells() const -> Vector<NotebookCell> {
  const auto& value = repr_["cells"];

  assert(value.is_array());
  return Vector<NotebookCell>(value);
}

auto NotebookDocument::uri(std::string uri) -> NotebookDocument& {
  return *this;
}

auto NotebookDocument::notebookType(std::string notebookType)
    -> NotebookDocument& {
  return *this;
}

auto NotebookDocument::version(int version) -> NotebookDocument& {
  return *this;
}

auto NotebookDocument::metadata(std::optional<LSPObject> metadata)
    -> NotebookDocument& {
  return *this;
}

auto NotebookDocument::cells(Vector<NotebookCell> cells) -> NotebookDocument& {
  return *this;
}

TextDocumentItem::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("languageId")) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("text")) return false;
  return true;
}

auto TextDocumentItem::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentItem::languageId() const -> LanguageKind {
  const auto& value = repr_["languageId"];

  lsp_runtime_error("TextDocumentItem::languageId: not implement yet");
}

auto TextDocumentItem::version() const -> int {
  const auto& value = repr_["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto TextDocumentItem::text() const -> std::string {
  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentItem::uri(std::string uri) -> TextDocumentItem& {
  return *this;
}

auto TextDocumentItem::languageId(LanguageKind languageId)
    -> TextDocumentItem& {
  return *this;
}

auto TextDocumentItem::version(int version) -> TextDocumentItem& {
  return *this;
}

auto TextDocumentItem::text(std::string text) -> TextDocumentItem& {
  return *this;
}

NotebookDocumentSyncOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookSelector")) return false;
  return true;
}

auto NotebookDocumentSyncOptions::notebookSelector() const
    -> Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                           NotebookDocumentFilterWithCells>> {
  const auto& value = repr_["notebookSelector"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>(value);
}

auto NotebookDocumentSyncOptions::save() const -> std::optional<bool> {
  if (!repr_.contains("save")) return std::nullopt;

  const auto& value = repr_["save"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncOptions::notebookSelector(
    Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                        NotebookDocumentFilterWithCells>>
        notebookSelector) -> NotebookDocumentSyncOptions& {
  return *this;
}

auto NotebookDocumentSyncOptions::save(std::optional<bool> save)
    -> NotebookDocumentSyncOptions& {
  return *this;
}

VersionedNotebookDocumentIdentifier::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto VersionedNotebookDocumentIdentifier::version() const -> int {
  const auto& value = repr_["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto VersionedNotebookDocumentIdentifier::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto VersionedNotebookDocumentIdentifier::version(int version)
    -> VersionedNotebookDocumentIdentifier& {
  return *this;
}

auto VersionedNotebookDocumentIdentifier::uri(std::string uri)
    -> VersionedNotebookDocumentIdentifier& {
  return *this;
}

NotebookDocumentChangeEvent::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto NotebookDocumentChangeEvent::metadata() const -> std::optional<LSPObject> {
  if (!repr_.contains("metadata")) return std::nullopt;

  const auto& value = repr_["metadata"];

  assert(value.is_object());
  return LSPObject(value);
}

auto NotebookDocumentChangeEvent::cells() const
    -> std::optional<NotebookDocumentCellChanges> {
  if (!repr_.contains("cells")) return std::nullopt;

  const auto& value = repr_["cells"];

  return NotebookDocumentCellChanges(value);
}

auto NotebookDocumentChangeEvent::metadata(std::optional<LSPObject> metadata)
    -> NotebookDocumentChangeEvent& {
  return *this;
}

auto NotebookDocumentChangeEvent::cells(
    std::optional<NotebookDocumentCellChanges> cells)
    -> NotebookDocumentChangeEvent& {
  return *this;
}

NotebookDocumentIdentifier::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto NotebookDocumentIdentifier::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentIdentifier::uri(std::string uri)
    -> NotebookDocumentIdentifier& {
  return *this;
}

InlineCompletionContext::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("triggerKind")) return false;
  return true;
}

auto InlineCompletionContext::triggerKind() const
    -> InlineCompletionTriggerKind {
  const auto& value = repr_["triggerKind"];

  return InlineCompletionTriggerKind(value);
}

auto InlineCompletionContext::selectedCompletionInfo() const
    -> std::optional<SelectedCompletionInfo> {
  if (!repr_.contains("selectedCompletionInfo")) return std::nullopt;

  const auto& value = repr_["selectedCompletionInfo"];

  return SelectedCompletionInfo(value);
}

auto InlineCompletionContext::triggerKind(
    InlineCompletionTriggerKind triggerKind) -> InlineCompletionContext& {
  return *this;
}

auto InlineCompletionContext::selectedCompletionInfo(
    std::optional<SelectedCompletionInfo> selectedCompletionInfo)
    -> InlineCompletionContext& {
  return *this;
}

StringValue::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "snippet") return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto StringValue::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto StringValue::value() const -> std::string {
  const auto& value = repr_["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto StringValue::kind(std::string kind) -> StringValue& { return *this; }

auto StringValue::value(std::string value) -> StringValue& { return *this; }

InlineCompletionOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlineCompletionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineCompletionOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> InlineCompletionOptions& {
  return *this;
}

TextDocumentContentOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("schemes")) return false;
  return true;
}

auto TextDocumentContentOptions::schemes() const -> Vector<std::string> {
  const auto& value = repr_["schemes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto TextDocumentContentOptions::schemes(Vector<std::string> schemes)
    -> TextDocumentContentOptions& {
  return *this;
}

Registration::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("id")) return false;
  if (!repr_.contains("method")) return false;
  return true;
}

auto Registration::id() const -> std::string {
  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Registration::method() const -> std::string {
  const auto& value = repr_["method"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Registration::registerOptions() const -> std::optional<LSPAny> {
  if (!repr_.contains("registerOptions")) return std::nullopt;

  const auto& value = repr_["registerOptions"];

  assert(value.is_object());
  return LSPAny(value);
}

auto Registration::id(std::string id) -> Registration& { return *this; }

auto Registration::method(std::string method) -> Registration& { return *this; }

auto Registration::registerOptions(std::optional<LSPAny> registerOptions)
    -> Registration& {
  return *this;
}

Unregistration::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("id")) return false;
  if (!repr_.contains("method")) return false;
  return true;
}

auto Unregistration::id() const -> std::string {
  const auto& value = repr_["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Unregistration::method() const -> std::string {
  const auto& value = repr_["method"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Unregistration::id(std::string id) -> Unregistration& { return *this; }

auto Unregistration::method(std::string method) -> Unregistration& {
  return *this;
}

_InitializeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("processId")) return false;
  if (!repr_.contains("rootUri")) return false;
  if (!repr_.contains("capabilities")) return false;
  return true;
}

auto _InitializeParams::processId() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  const auto& value = repr_["processId"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::clientInfo() const -> std::optional<ClientInfo> {
  if (!repr_.contains("clientInfo")) return std::nullopt;

  const auto& value = repr_["clientInfo"];

  return ClientInfo(value);
}

auto _InitializeParams::locale() const -> std::optional<std::string> {
  if (!repr_.contains("locale")) return std::nullopt;

  const auto& value = repr_["locale"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto _InitializeParams::rootPath() const -> std::optional<
    std::variant<std::monostate, std::string, std::nullptr_t>> {
  if (!repr_.contains("rootPath")) return std::nullopt;

  const auto& value = repr_["rootPath"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::rootUri() const
    -> std::variant<std::monostate, std::string, std::nullptr_t> {
  const auto& value = repr_["rootUri"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::capabilities() const -> ClientCapabilities {
  const auto& value = repr_["capabilities"];

  return ClientCapabilities(value);
}

auto _InitializeParams::initializationOptions() const -> std::optional<LSPAny> {
  if (!repr_.contains("initializationOptions")) return std::nullopt;

  const auto& value = repr_["initializationOptions"];

  assert(value.is_object());
  return LSPAny(value);
}

auto _InitializeParams::trace() const -> std::optional<TraceValue> {
  if (!repr_.contains("trace")) return std::nullopt;

  const auto& value = repr_["trace"];

  lsp_runtime_error("_InitializeParams::trace: not implement yet");
}

auto _InitializeParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_.contains("workDoneToken")) return std::nullopt;

  const auto& value = repr_["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::processId(
    std::variant<std::monostate, int, std::nullptr_t> processId)
    -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::clientInfo(std::optional<ClientInfo> clientInfo)
    -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::locale(std::optional<std::string> locale)
    -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::rootPath(
    std::optional<std::variant<std::monostate, std::string, std::nullptr_t>>
        rootPath) -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::rootUri(
    std::variant<std::monostate, std::string, std::nullptr_t> rootUri)
    -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::capabilities(ClientCapabilities capabilities)
    -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::initializationOptions(
    std::optional<LSPAny> initializationOptions) -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::trace(std::optional<TraceValue> trace)
    -> _InitializeParams& {
  return *this;
}

auto _InitializeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> _InitializeParams& {
  return *this;
}

WorkspaceFoldersInitializeParams::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceFoldersInitializeParams::workspaceFolders() const
    -> std::optional<
        std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>> {
  if (!repr_.contains("workspaceFolders")) return std::nullopt;

  const auto& value = repr_["workspaceFolders"];

  std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFoldersInitializeParams::workspaceFolders(
    std::optional<
        std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>
        workspaceFolders) -> WorkspaceFoldersInitializeParams& {
  return *this;
}

ServerCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ServerCapabilities::positionEncoding() const
    -> std::optional<PositionEncodingKind> {
  if (!repr_.contains("positionEncoding")) return std::nullopt;

  const auto& value = repr_["positionEncoding"];

  lsp_runtime_error("ServerCapabilities::positionEncoding: not implement yet");
}

auto ServerCapabilities::textDocumentSync() const
    -> std::optional<std::variant<std::monostate, TextDocumentSyncOptions,
                                  TextDocumentSyncKind>> {
  if (!repr_.contains("textDocumentSync")) return std::nullopt;

  const auto& value = repr_["textDocumentSync"];

  std::variant<std::monostate, TextDocumentSyncOptions, TextDocumentSyncKind>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::notebookDocumentSync() const
    -> std::optional<std::variant<std::monostate, NotebookDocumentSyncOptions,
                                  NotebookDocumentSyncRegistrationOptions>> {
  if (!repr_.contains("notebookDocumentSync")) return std::nullopt;

  const auto& value = repr_["notebookDocumentSync"];

  std::variant<std::monostate, NotebookDocumentSyncOptions,
               NotebookDocumentSyncRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::completionProvider() const
    -> std::optional<CompletionOptions> {
  if (!repr_.contains("completionProvider")) return std::nullopt;

  const auto& value = repr_["completionProvider"];

  return CompletionOptions(value);
}

auto ServerCapabilities::hoverProvider() const
    -> std::optional<std::variant<std::monostate, bool, HoverOptions>> {
  if (!repr_.contains("hoverProvider")) return std::nullopt;

  const auto& value = repr_["hoverProvider"];

  std::variant<std::monostate, bool, HoverOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::signatureHelpProvider() const
    -> std::optional<SignatureHelpOptions> {
  if (!repr_.contains("signatureHelpProvider")) return std::nullopt;

  const auto& value = repr_["signatureHelpProvider"];

  return SignatureHelpOptions(value);
}

auto ServerCapabilities::declarationProvider() const
    -> std::optional<std::variant<std::monostate, bool, DeclarationOptions,
                                  DeclarationRegistrationOptions>> {
  if (!repr_.contains("declarationProvider")) return std::nullopt;

  const auto& value = repr_["declarationProvider"];

  std::variant<std::monostate, bool, DeclarationOptions,
               DeclarationRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::definitionProvider() const
    -> std::optional<std::variant<std::monostate, bool, DefinitionOptions>> {
  if (!repr_.contains("definitionProvider")) return std::nullopt;

  const auto& value = repr_["definitionProvider"];

  std::variant<std::monostate, bool, DefinitionOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::typeDefinitionProvider() const
    -> std::optional<std::variant<std::monostate, bool, TypeDefinitionOptions,
                                  TypeDefinitionRegistrationOptions>> {
  if (!repr_.contains("typeDefinitionProvider")) return std::nullopt;

  const auto& value = repr_["typeDefinitionProvider"];

  std::variant<std::monostate, bool, TypeDefinitionOptions,
               TypeDefinitionRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::implementationProvider() const
    -> std::optional<std::variant<std::monostate, bool, ImplementationOptions,
                                  ImplementationRegistrationOptions>> {
  if (!repr_.contains("implementationProvider")) return std::nullopt;

  const auto& value = repr_["implementationProvider"];

  std::variant<std::monostate, bool, ImplementationOptions,
               ImplementationRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::referencesProvider() const
    -> std::optional<std::variant<std::monostate, bool, ReferenceOptions>> {
  if (!repr_.contains("referencesProvider")) return std::nullopt;

  const auto& value = repr_["referencesProvider"];

  std::variant<std::monostate, bool, ReferenceOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentHighlightProvider() const -> std::optional<
    std::variant<std::monostate, bool, DocumentHighlightOptions>> {
  if (!repr_.contains("documentHighlightProvider")) return std::nullopt;

  const auto& value = repr_["documentHighlightProvider"];

  std::variant<std::monostate, bool, DocumentHighlightOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentSymbolProvider() const -> std::optional<
    std::variant<std::monostate, bool, DocumentSymbolOptions>> {
  if (!repr_.contains("documentSymbolProvider")) return std::nullopt;

  const auto& value = repr_["documentSymbolProvider"];

  std::variant<std::monostate, bool, DocumentSymbolOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::codeActionProvider() const
    -> std::optional<std::variant<std::monostate, bool, CodeActionOptions>> {
  if (!repr_.contains("codeActionProvider")) return std::nullopt;

  const auto& value = repr_["codeActionProvider"];

  std::variant<std::monostate, bool, CodeActionOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::codeLensProvider() const
    -> std::optional<CodeLensOptions> {
  if (!repr_.contains("codeLensProvider")) return std::nullopt;

  const auto& value = repr_["codeLensProvider"];

  return CodeLensOptions(value);
}

auto ServerCapabilities::documentLinkProvider() const
    -> std::optional<DocumentLinkOptions> {
  if (!repr_.contains("documentLinkProvider")) return std::nullopt;

  const auto& value = repr_["documentLinkProvider"];

  return DocumentLinkOptions(value);
}

auto ServerCapabilities::colorProvider() const
    -> std::optional<std::variant<std::monostate, bool, DocumentColorOptions,
                                  DocumentColorRegistrationOptions>> {
  if (!repr_.contains("colorProvider")) return std::nullopt;

  const auto& value = repr_["colorProvider"];

  std::variant<std::monostate, bool, DocumentColorOptions,
               DocumentColorRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::workspaceSymbolProvider() const -> std::optional<
    std::variant<std::monostate, bool, WorkspaceSymbolOptions>> {
  if (!repr_.contains("workspaceSymbolProvider")) return std::nullopt;

  const auto& value = repr_["workspaceSymbolProvider"];

  std::variant<std::monostate, bool, WorkspaceSymbolOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentFormattingProvider() const -> std::optional<
    std::variant<std::monostate, bool, DocumentFormattingOptions>> {
  if (!repr_.contains("documentFormattingProvider")) return std::nullopt;

  const auto& value = repr_["documentFormattingProvider"];

  std::variant<std::monostate, bool, DocumentFormattingOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentRangeFormattingProvider() const
    -> std::optional<
        std::variant<std::monostate, bool, DocumentRangeFormattingOptions>> {
  if (!repr_.contains("documentRangeFormattingProvider")) return std::nullopt;

  const auto& value = repr_["documentRangeFormattingProvider"];

  std::variant<std::monostate, bool, DocumentRangeFormattingOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentOnTypeFormattingProvider() const
    -> std::optional<DocumentOnTypeFormattingOptions> {
  if (!repr_.contains("documentOnTypeFormattingProvider")) return std::nullopt;

  const auto& value = repr_["documentOnTypeFormattingProvider"];

  return DocumentOnTypeFormattingOptions(value);
}

auto ServerCapabilities::renameProvider() const
    -> std::optional<std::variant<std::monostate, bool, RenameOptions>> {
  if (!repr_.contains("renameProvider")) return std::nullopt;

  const auto& value = repr_["renameProvider"];

  std::variant<std::monostate, bool, RenameOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::foldingRangeProvider() const
    -> std::optional<std::variant<std::monostate, bool, FoldingRangeOptions,
                                  FoldingRangeRegistrationOptions>> {
  if (!repr_.contains("foldingRangeProvider")) return std::nullopt;

  const auto& value = repr_["foldingRangeProvider"];

  std::variant<std::monostate, bool, FoldingRangeOptions,
               FoldingRangeRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::selectionRangeProvider() const
    -> std::optional<std::variant<std::monostate, bool, SelectionRangeOptions,
                                  SelectionRangeRegistrationOptions>> {
  if (!repr_.contains("selectionRangeProvider")) return std::nullopt;

  const auto& value = repr_["selectionRangeProvider"];

  std::variant<std::monostate, bool, SelectionRangeOptions,
               SelectionRangeRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::executeCommandProvider() const
    -> std::optional<ExecuteCommandOptions> {
  if (!repr_.contains("executeCommandProvider")) return std::nullopt;

  const auto& value = repr_["executeCommandProvider"];

  return ExecuteCommandOptions(value);
}

auto ServerCapabilities::callHierarchyProvider() const
    -> std::optional<std::variant<std::monostate, bool, CallHierarchyOptions,
                                  CallHierarchyRegistrationOptions>> {
  if (!repr_.contains("callHierarchyProvider")) return std::nullopt;

  const auto& value = repr_["callHierarchyProvider"];

  std::variant<std::monostate, bool, CallHierarchyOptions,
               CallHierarchyRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::linkedEditingRangeProvider() const -> std::optional<
    std::variant<std::monostate, bool, LinkedEditingRangeOptions,
                 LinkedEditingRangeRegistrationOptions>> {
  if (!repr_.contains("linkedEditingRangeProvider")) return std::nullopt;

  const auto& value = repr_["linkedEditingRangeProvider"];

  std::variant<std::monostate, bool, LinkedEditingRangeOptions,
               LinkedEditingRangeRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::semanticTokensProvider() const
    -> std::optional<std::variant<std::monostate, SemanticTokensOptions,
                                  SemanticTokensRegistrationOptions>> {
  if (!repr_.contains("semanticTokensProvider")) return std::nullopt;

  const auto& value = repr_["semanticTokensProvider"];

  std::variant<std::monostate, SemanticTokensOptions,
               SemanticTokensRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::monikerProvider() const
    -> std::optional<std::variant<std::monostate, bool, MonikerOptions,
                                  MonikerRegistrationOptions>> {
  if (!repr_.contains("monikerProvider")) return std::nullopt;

  const auto& value = repr_["monikerProvider"];

  std::variant<std::monostate, bool, MonikerOptions, MonikerRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::typeHierarchyProvider() const
    -> std::optional<std::variant<std::monostate, bool, TypeHierarchyOptions,
                                  TypeHierarchyRegistrationOptions>> {
  if (!repr_.contains("typeHierarchyProvider")) return std::nullopt;

  const auto& value = repr_["typeHierarchyProvider"];

  std::variant<std::monostate, bool, TypeHierarchyOptions,
               TypeHierarchyRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::inlineValueProvider() const
    -> std::optional<std::variant<std::monostate, bool, InlineValueOptions,
                                  InlineValueRegistrationOptions>> {
  if (!repr_.contains("inlineValueProvider")) return std::nullopt;

  const auto& value = repr_["inlineValueProvider"];

  std::variant<std::monostate, bool, InlineValueOptions,
               InlineValueRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::inlayHintProvider() const
    -> std::optional<std::variant<std::monostate, bool, InlayHintOptions,
                                  InlayHintRegistrationOptions>> {
  if (!repr_.contains("inlayHintProvider")) return std::nullopt;

  const auto& value = repr_["inlayHintProvider"];

  std::variant<std::monostate, bool, InlayHintOptions,
               InlayHintRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::diagnosticProvider() const
    -> std::optional<std::variant<std::monostate, DiagnosticOptions,
                                  DiagnosticRegistrationOptions>> {
  if (!repr_.contains("diagnosticProvider")) return std::nullopt;

  const auto& value = repr_["diagnosticProvider"];

  std::variant<std::monostate, DiagnosticOptions, DiagnosticRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::inlineCompletionProvider() const -> std::optional<
    std::variant<std::monostate, bool, InlineCompletionOptions>> {
  if (!repr_.contains("inlineCompletionProvider")) return std::nullopt;

  const auto& value = repr_["inlineCompletionProvider"];

  std::variant<std::monostate, bool, InlineCompletionOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::workspace() const -> std::optional<WorkspaceOptions> {
  if (!repr_.contains("workspace")) return std::nullopt;

  const auto& value = repr_["workspace"];

  return WorkspaceOptions(value);
}

auto ServerCapabilities::experimental() const -> std::optional<LSPAny> {
  if (!repr_.contains("experimental")) return std::nullopt;

  const auto& value = repr_["experimental"];

  assert(value.is_object());
  return LSPAny(value);
}

auto ServerCapabilities::positionEncoding(
    std::optional<PositionEncodingKind> positionEncoding)
    -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::textDocumentSync(
    std::optional<std::variant<std::monostate, TextDocumentSyncOptions,
                               TextDocumentSyncKind>>
        textDocumentSync) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::notebookDocumentSync(
    std::optional<std::variant<std::monostate, NotebookDocumentSyncOptions,
                               NotebookDocumentSyncRegistrationOptions>>
        notebookDocumentSync) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::completionProvider(
    std::optional<CompletionOptions> completionProvider)
    -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::hoverProvider(
    std::optional<std::variant<std::monostate, bool, HoverOptions>>
        hoverProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::signatureHelpProvider(
    std::optional<SignatureHelpOptions> signatureHelpProvider)
    -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::declarationProvider(
    std::optional<std::variant<std::monostate, bool, DeclarationOptions,
                               DeclarationRegistrationOptions>>
        declarationProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::definitionProvider(
    std::optional<std::variant<std::monostate, bool, DefinitionOptions>>
        definitionProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::typeDefinitionProvider(
    std::optional<std::variant<std::monostate, bool, TypeDefinitionOptions,
                               TypeDefinitionRegistrationOptions>>
        typeDefinitionProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::implementationProvider(
    std::optional<std::variant<std::monostate, bool, ImplementationOptions,
                               ImplementationRegistrationOptions>>
        implementationProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::referencesProvider(
    std::optional<std::variant<std::monostate, bool, ReferenceOptions>>
        referencesProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::documentHighlightProvider(
    std::optional<std::variant<std::monostate, bool, DocumentHighlightOptions>>
        documentHighlightProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::documentSymbolProvider(
    std::optional<std::variant<std::monostate, bool, DocumentSymbolOptions>>
        documentSymbolProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::codeActionProvider(
    std::optional<std::variant<std::monostate, bool, CodeActionOptions>>
        codeActionProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::codeLensProvider(
    std::optional<CodeLensOptions> codeLensProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::documentLinkProvider(
    std::optional<DocumentLinkOptions> documentLinkProvider)
    -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::colorProvider(
    std::optional<std::variant<std::monostate, bool, DocumentColorOptions,
                               DocumentColorRegistrationOptions>>
        colorProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::workspaceSymbolProvider(
    std::optional<std::variant<std::monostate, bool, WorkspaceSymbolOptions>>
        workspaceSymbolProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::documentFormattingProvider(
    std::optional<std::variant<std::monostate, bool, DocumentFormattingOptions>>
        documentFormattingProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::documentRangeFormattingProvider(
    std::optional<
        std::variant<std::monostate, bool, DocumentRangeFormattingOptions>>
        documentRangeFormattingProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::documentOnTypeFormattingProvider(
    std::optional<DocumentOnTypeFormattingOptions>
        documentOnTypeFormattingProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::renameProvider(
    std::optional<std::variant<std::monostate, bool, RenameOptions>>
        renameProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::foldingRangeProvider(
    std::optional<std::variant<std::monostate, bool, FoldingRangeOptions,
                               FoldingRangeRegistrationOptions>>
        foldingRangeProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::selectionRangeProvider(
    std::optional<std::variant<std::monostate, bool, SelectionRangeOptions,
                               SelectionRangeRegistrationOptions>>
        selectionRangeProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::executeCommandProvider(
    std::optional<ExecuteCommandOptions> executeCommandProvider)
    -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::callHierarchyProvider(
    std::optional<std::variant<std::monostate, bool, CallHierarchyOptions,
                               CallHierarchyRegistrationOptions>>
        callHierarchyProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::linkedEditingRangeProvider(
    std::optional<std::variant<std::monostate, bool, LinkedEditingRangeOptions,
                               LinkedEditingRangeRegistrationOptions>>
        linkedEditingRangeProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::semanticTokensProvider(
    std::optional<std::variant<std::monostate, SemanticTokensOptions,
                               SemanticTokensRegistrationOptions>>
        semanticTokensProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::monikerProvider(
    std::optional<std::variant<std::monostate, bool, MonikerOptions,
                               MonikerRegistrationOptions>>
        monikerProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::typeHierarchyProvider(
    std::optional<std::variant<std::monostate, bool, TypeHierarchyOptions,
                               TypeHierarchyRegistrationOptions>>
        typeHierarchyProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::inlineValueProvider(
    std::optional<std::variant<std::monostate, bool, InlineValueOptions,
                               InlineValueRegistrationOptions>>
        inlineValueProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::inlayHintProvider(
    std::optional<std::variant<std::monostate, bool, InlayHintOptions,
                               InlayHintRegistrationOptions>>
        inlayHintProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::diagnosticProvider(
    std::optional<std::variant<std::monostate, DiagnosticOptions,
                               DiagnosticRegistrationOptions>>
        diagnosticProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::inlineCompletionProvider(
    std::optional<std::variant<std::monostate, bool, InlineCompletionOptions>>
        inlineCompletionProvider) -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::workspace(std::optional<WorkspaceOptions> workspace)
    -> ServerCapabilities& {
  return *this;
}

auto ServerCapabilities::experimental(std::optional<LSPAny> experimental)
    -> ServerCapabilities& {
  return *this;
}

ServerInfo::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("name")) return false;
  return true;
}

auto ServerInfo::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ServerInfo::version() const -> std::optional<std::string> {
  if (!repr_.contains("version")) return std::nullopt;

  const auto& value = repr_["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ServerInfo::name(std::string name) -> ServerInfo& { return *this; }

auto ServerInfo::version(std::optional<std::string> version) -> ServerInfo& {
  return *this;
}

VersionedTextDocumentIdentifier::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto VersionedTextDocumentIdentifier::version() const -> int {
  const auto& value = repr_["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto VersionedTextDocumentIdentifier::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto VersionedTextDocumentIdentifier::version(int version)
    -> VersionedTextDocumentIdentifier& {
  return *this;
}

auto VersionedTextDocumentIdentifier::uri(std::string uri)
    -> VersionedTextDocumentIdentifier& {
  return *this;
}

SaveOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SaveOptions::includeText() const -> std::optional<bool> {
  if (!repr_.contains("includeText")) return std::nullopt;

  const auto& value = repr_["includeText"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SaveOptions::includeText(std::optional<bool> includeText) -> SaveOptions& {
  return *this;
}

FileEvent::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("type")) return false;
  return true;
}

auto FileEvent::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileEvent::type() const -> FileChangeType {
  const auto& value = repr_["type"];

  return FileChangeType(value);
}

auto FileEvent::uri(std::string uri) -> FileEvent& { return *this; }

auto FileEvent::type(FileChangeType type) -> FileEvent& { return *this; }

FileSystemWatcher::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("globPattern")) return false;
  return true;
}

auto FileSystemWatcher::globPattern() const -> GlobPattern {
  const auto& value = repr_["globPattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto FileSystemWatcher::kind() const -> std::optional<WatchKind> {
  if (!repr_.contains("kind")) return std::nullopt;

  const auto& value = repr_["kind"];

  return WatchKind(value);
}

auto FileSystemWatcher::globPattern(GlobPattern globPattern)
    -> FileSystemWatcher& {
  return *this;
}

auto FileSystemWatcher::kind(std::optional<WatchKind> kind)
    -> FileSystemWatcher& {
  return *this;
}

Diagnostic::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("message")) return false;
  return true;
}

auto Diagnostic::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto Diagnostic::severity() const -> std::optional<DiagnosticSeverity> {
  if (!repr_.contains("severity")) return std::nullopt;

  const auto& value = repr_["severity"];

  return DiagnosticSeverity(value);
}

auto Diagnostic::code() const
    -> std::optional<std::variant<std::monostate, int, std::string>> {
  if (!repr_.contains("code")) return std::nullopt;

  const auto& value = repr_["code"];

  std::variant<std::monostate, int, std::string> result;

  details::try_emplace(result, value);

  return result;
}

auto Diagnostic::codeDescription() const -> std::optional<CodeDescription> {
  if (!repr_.contains("codeDescription")) return std::nullopt;

  const auto& value = repr_["codeDescription"];

  return CodeDescription(value);
}

auto Diagnostic::source() const -> std::optional<std::string> {
  if (!repr_.contains("source")) return std::nullopt;

  const auto& value = repr_["source"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Diagnostic::message() const -> std::string {
  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Diagnostic::tags() const -> std::optional<Vector<DiagnosticTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<DiagnosticTag>(value);
}

auto Diagnostic::relatedInformation() const
    -> std::optional<Vector<DiagnosticRelatedInformation>> {
  if (!repr_.contains("relatedInformation")) return std::nullopt;

  const auto& value = repr_["relatedInformation"];

  assert(value.is_array());
  return Vector<DiagnosticRelatedInformation>(value);
}

auto Diagnostic::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto Diagnostic::range(Range range) -> Diagnostic& { return *this; }

auto Diagnostic::severity(std::optional<DiagnosticSeverity> severity)
    -> Diagnostic& {
  return *this;
}

auto Diagnostic::code(
    std::optional<std::variant<std::monostate, int, std::string>> code)
    -> Diagnostic& {
  return *this;
}

auto Diagnostic::codeDescription(std::optional<CodeDescription> codeDescription)
    -> Diagnostic& {
  return *this;
}

auto Diagnostic::source(std::optional<std::string> source) -> Diagnostic& {
  return *this;
}

auto Diagnostic::message(std::string message) -> Diagnostic& { return *this; }

auto Diagnostic::tags(std::optional<Vector<DiagnosticTag>> tags)
    -> Diagnostic& {
  return *this;
}

auto Diagnostic::relatedInformation(
    std::optional<Vector<DiagnosticRelatedInformation>> relatedInformation)
    -> Diagnostic& {
  return *this;
}

auto Diagnostic::data(std::optional<LSPAny> data) -> Diagnostic& {
  return *this;
}

CompletionContext::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("triggerKind")) return false;
  return true;
}

auto CompletionContext::triggerKind() const -> CompletionTriggerKind {
  const auto& value = repr_["triggerKind"];

  return CompletionTriggerKind(value);
}

auto CompletionContext::triggerCharacter() const -> std::optional<std::string> {
  if (!repr_.contains("triggerCharacter")) return std::nullopt;

  const auto& value = repr_["triggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionContext::triggerKind(CompletionTriggerKind triggerKind)
    -> CompletionContext& {
  return *this;
}

auto CompletionContext::triggerCharacter(
    std::optional<std::string> triggerCharacter) -> CompletionContext& {
  return *this;
}

CompletionItemLabelDetails::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CompletionItemLabelDetails::detail() const -> std::optional<std::string> {
  if (!repr_.contains("detail")) return std::nullopt;

  const auto& value = repr_["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItemLabelDetails::description() const
    -> std::optional<std::string> {
  if (!repr_.contains("description")) return std::nullopt;

  const auto& value = repr_["description"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItemLabelDetails::detail(std::optional<std::string> detail)
    -> CompletionItemLabelDetails& {
  return *this;
}

auto CompletionItemLabelDetails::description(
    std::optional<std::string> description) -> CompletionItemLabelDetails& {
  return *this;
}

InsertReplaceEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("newText")) return false;
  if (!repr_.contains("insert")) return false;
  if (!repr_.contains("replace")) return false;
  return true;
}

auto InsertReplaceEdit::newText() const -> std::string {
  const auto& value = repr_["newText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InsertReplaceEdit::insert() const -> Range {
  const auto& value = repr_["insert"];

  return Range(value);
}

auto InsertReplaceEdit::replace() const -> Range {
  const auto& value = repr_["replace"];

  return Range(value);
}

auto InsertReplaceEdit::newText(std::string newText) -> InsertReplaceEdit& {
  return *this;
}

auto InsertReplaceEdit::insert(Range insert) -> InsertReplaceEdit& {
  return *this;
}

auto InsertReplaceEdit::replace(Range replace) -> InsertReplaceEdit& {
  return *this;
}

CompletionItemDefaults::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CompletionItemDefaults::commitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("commitCharacters")) return std::nullopt;

  const auto& value = repr_["commitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionItemDefaults::editRange() const -> std::optional<
    std::variant<std::monostate, Range, EditRangeWithInsertReplace>> {
  if (!repr_.contains("editRange")) return std::nullopt;

  const auto& value = repr_["editRange"];

  std::variant<std::monostate, Range, EditRangeWithInsertReplace> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionItemDefaults::insertTextFormat() const
    -> std::optional<InsertTextFormat> {
  if (!repr_.contains("insertTextFormat")) return std::nullopt;

  const auto& value = repr_["insertTextFormat"];

  return InsertTextFormat(value);
}

auto CompletionItemDefaults::insertTextMode() const
    -> std::optional<InsertTextMode> {
  if (!repr_.contains("insertTextMode")) return std::nullopt;

  const auto& value = repr_["insertTextMode"];

  return InsertTextMode(value);
}

auto CompletionItemDefaults::data() const -> std::optional<LSPAny> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CompletionItemDefaults::commitCharacters(
    std::optional<Vector<std::string>> commitCharacters)
    -> CompletionItemDefaults& {
  return *this;
}

auto CompletionItemDefaults::editRange(
    std::optional<
        std::variant<std::monostate, Range, EditRangeWithInsertReplace>>
        editRange) -> CompletionItemDefaults& {
  return *this;
}

auto CompletionItemDefaults::insertTextFormat(
    std::optional<InsertTextFormat> insertTextFormat)
    -> CompletionItemDefaults& {
  return *this;
}

auto CompletionItemDefaults::insertTextMode(
    std::optional<InsertTextMode> insertTextMode) -> CompletionItemDefaults& {
  return *this;
}

auto CompletionItemDefaults::data(std::optional<LSPAny> data)
    -> CompletionItemDefaults& {
  return *this;
}

CompletionItemApplyKinds::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CompletionItemApplyKinds::commitCharacters() const
    -> std::optional<ApplyKind> {
  if (!repr_.contains("commitCharacters")) return std::nullopt;

  const auto& value = repr_["commitCharacters"];

  return ApplyKind(value);
}

auto CompletionItemApplyKinds::data() const -> std::optional<ApplyKind> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  return ApplyKind(value);
}

auto CompletionItemApplyKinds::commitCharacters(
    std::optional<ApplyKind> commitCharacters) -> CompletionItemApplyKinds& {
  return *this;
}

auto CompletionItemApplyKinds::data(std::optional<ApplyKind> data)
    -> CompletionItemApplyKinds& {
  return *this;
}

CompletionOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CompletionOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("triggerCharacters")) return std::nullopt;

  const auto& value = repr_["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionOptions::allCommitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("allCommitCharacters")) return std::nullopt;

  const auto& value = repr_["allCommitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionOptions::completionItem() const
    -> std::optional<ServerCompletionItemOptions> {
  if (!repr_.contains("completionItem")) return std::nullopt;

  const auto& value = repr_["completionItem"];

  return ServerCompletionItemOptions(value);
}

auto CompletionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> CompletionOptions& {
  return *this;
}

auto CompletionOptions::allCommitCharacters(
    std::optional<Vector<std::string>> allCommitCharacters)
    -> CompletionOptions& {
  return *this;
}

auto CompletionOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> CompletionOptions& {
  return *this;
}

auto CompletionOptions::completionItem(
    std::optional<ServerCompletionItemOptions> completionItem)
    -> CompletionOptions& {
  return *this;
}

auto CompletionOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> CompletionOptions& {
  return *this;
}

HoverOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto HoverOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto HoverOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> HoverOptions& {
  return *this;
}

SignatureHelpContext::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("triggerKind")) return false;
  if (!repr_.contains("isRetrigger")) return false;
  return true;
}

auto SignatureHelpContext::triggerKind() const -> SignatureHelpTriggerKind {
  const auto& value = repr_["triggerKind"];

  return SignatureHelpTriggerKind(value);
}

auto SignatureHelpContext::triggerCharacter() const
    -> std::optional<std::string> {
  if (!repr_.contains("triggerCharacter")) return std::nullopt;

  const auto& value = repr_["triggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SignatureHelpContext::isRetrigger() const -> bool {
  const auto& value = repr_["isRetrigger"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpContext::activeSignatureHelp() const
    -> std::optional<SignatureHelp> {
  if (!repr_.contains("activeSignatureHelp")) return std::nullopt;

  const auto& value = repr_["activeSignatureHelp"];

  return SignatureHelp(value);
}

auto SignatureHelpContext::triggerKind(SignatureHelpTriggerKind triggerKind)
    -> SignatureHelpContext& {
  return *this;
}

auto SignatureHelpContext::triggerCharacter(
    std::optional<std::string> triggerCharacter) -> SignatureHelpContext& {
  return *this;
}

auto SignatureHelpContext::isRetrigger(bool isRetrigger)
    -> SignatureHelpContext& {
  return *this;
}

auto SignatureHelpContext::activeSignatureHelp(
    std::optional<SignatureHelp> activeSignatureHelp) -> SignatureHelpContext& {
  return *this;
}

SignatureInformation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("label")) return false;
  return true;
}

auto SignatureInformation::label() const -> std::string {
  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SignatureInformation::documentation() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_.contains("documentation")) return std::nullopt;

  const auto& value = repr_["documentation"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureInformation::parameters() const
    -> std::optional<Vector<ParameterInformation>> {
  if (!repr_.contains("parameters")) return std::nullopt;

  const auto& value = repr_["parameters"];

  assert(value.is_array());
  return Vector<ParameterInformation>(value);
}

auto SignatureInformation::activeParameter() const
    -> std::optional<std::variant<std::monostate, long, std::nullptr_t>> {
  if (!repr_.contains("activeParameter")) return std::nullopt;

  const auto& value = repr_["activeParameter"];

  std::variant<std::monostate, long, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureInformation::label(std::string label) -> SignatureInformation& {
  return *this;
}

auto SignatureInformation::documentation(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        documentation) -> SignatureInformation& {
  return *this;
}

auto SignatureInformation::parameters(
    std::optional<Vector<ParameterInformation>> parameters)
    -> SignatureInformation& {
  return *this;
}

auto SignatureInformation::activeParameter(
    std::optional<std::variant<std::monostate, long, std::nullptr_t>>
        activeParameter) -> SignatureInformation& {
  return *this;
}

SignatureHelpOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SignatureHelpOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("triggerCharacters")) return std::nullopt;

  const auto& value = repr_["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpOptions::retriggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("retriggerCharacters")) return std::nullopt;

  const auto& value = repr_["retriggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> SignatureHelpOptions& {
  return *this;
}

auto SignatureHelpOptions::retriggerCharacters(
    std::optional<Vector<std::string>> retriggerCharacters)
    -> SignatureHelpOptions& {
  return *this;
}

auto SignatureHelpOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SignatureHelpOptions& {
  return *this;
}

DefinitionOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DefinitionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DefinitionOptions& {
  return *this;
}

ReferenceContext::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("includeDeclaration")) return false;
  return true;
}

auto ReferenceContext::includeDeclaration() const -> bool {
  const auto& value = repr_["includeDeclaration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceContext::includeDeclaration(bool includeDeclaration)
    -> ReferenceContext& {
  return *this;
}

ReferenceOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ReferenceOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> ReferenceOptions& {
  return *this;
}

DocumentHighlightOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentHighlightOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentHighlightOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentHighlightOptions& {
  return *this;
}

BaseSymbolInformation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("name")) return false;
  if (!repr_.contains("kind")) return false;
  return true;
}

auto BaseSymbolInformation::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto BaseSymbolInformation::kind() const -> SymbolKind {
  const auto& value = repr_["kind"];

  return SymbolKind(value);
}

auto BaseSymbolInformation::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_.contains("tags")) return std::nullopt;

  const auto& value = repr_["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto BaseSymbolInformation::containerName() const
    -> std::optional<std::string> {
  if (!repr_.contains("containerName")) return std::nullopt;

  const auto& value = repr_["containerName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto BaseSymbolInformation::name(std::string name) -> BaseSymbolInformation& {
  return *this;
}

auto BaseSymbolInformation::kind(SymbolKind kind) -> BaseSymbolInformation& {
  return *this;
}

auto BaseSymbolInformation::tags(std::optional<Vector<SymbolTag>> tags)
    -> BaseSymbolInformation& {
  return *this;
}

auto BaseSymbolInformation::containerName(
    std::optional<std::string> containerName) -> BaseSymbolInformation& {
  return *this;
}

DocumentSymbolOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentSymbolOptions::label() const -> std::optional<std::string> {
  if (!repr_.contains("label")) return std::nullopt;

  const auto& value = repr_["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbolOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolOptions::label(std::optional<std::string> label)
    -> DocumentSymbolOptions& {
  return *this;
}

auto DocumentSymbolOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentSymbolOptions& {
  return *this;
}

CodeActionContext::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("diagnostics")) return false;
  return true;
}

auto CodeActionContext::diagnostics() const -> Vector<Diagnostic> {
  const auto& value = repr_["diagnostics"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto CodeActionContext::only() const -> std::optional<Vector<CodeActionKind>> {
  if (!repr_.contains("only")) return std::nullopt;

  const auto& value = repr_["only"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto CodeActionContext::triggerKind() const
    -> std::optional<CodeActionTriggerKind> {
  if (!repr_.contains("triggerKind")) return std::nullopt;

  const auto& value = repr_["triggerKind"];

  return CodeActionTriggerKind(value);
}

auto CodeActionContext::diagnostics(Vector<Diagnostic> diagnostics)
    -> CodeActionContext& {
  return *this;
}

auto CodeActionContext::only(std::optional<Vector<CodeActionKind>> only)
    -> CodeActionContext& {
  return *this;
}

auto CodeActionContext::triggerKind(
    std::optional<CodeActionTriggerKind> triggerKind) -> CodeActionContext& {
  return *this;
}

CodeActionDisabled::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("reason")) return false;
  return true;
}

auto CodeActionDisabled::reason() const -> std::string {
  const auto& value = repr_["reason"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CodeActionDisabled::reason(std::string reason) -> CodeActionDisabled& {
  return *this;
}

CodeActionOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CodeActionOptions::codeActionKinds() const
    -> std::optional<Vector<CodeActionKind>> {
  if (!repr_.contains("codeActionKinds")) return std::nullopt;

  const auto& value = repr_["codeActionKinds"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto CodeActionOptions::documentation() const
    -> std::optional<Vector<CodeActionKindDocumentation>> {
  if (!repr_.contains("documentation")) return std::nullopt;

  const auto& value = repr_["documentation"];

  assert(value.is_array());
  return Vector<CodeActionKindDocumentation>(value);
}

auto CodeActionOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionOptions::codeActionKinds(
    std::optional<Vector<CodeActionKind>> codeActionKinds)
    -> CodeActionOptions& {
  return *this;
}

auto CodeActionOptions::documentation(
    std::optional<Vector<CodeActionKindDocumentation>> documentation)
    -> CodeActionOptions& {
  return *this;
}

auto CodeActionOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> CodeActionOptions& {
  return *this;
}

auto CodeActionOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> CodeActionOptions& {
  return *this;
}

LocationUriOnly::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto LocationUriOnly::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LocationUriOnly::uri(std::string uri) -> LocationUriOnly& { return *this; }

WorkspaceSymbolOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceSymbolOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> WorkspaceSymbolOptions& {
  return *this;
}

auto WorkspaceSymbolOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> WorkspaceSymbolOptions& {
  return *this;
}

CodeLensOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CodeLensOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> CodeLensOptions& {
  return *this;
}

auto CodeLensOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> CodeLensOptions& {
  return *this;
}

DocumentLinkOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentLinkOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_.contains("resolveProvider")) return std::nullopt;

  const auto& value = repr_["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> DocumentLinkOptions& {
  return *this;
}

auto DocumentLinkOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DocumentLinkOptions& {
  return *this;
}

FormattingOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("tabSize")) return false;
  if (!repr_.contains("insertSpaces")) return false;
  return true;
}

auto FormattingOptions::tabSize() const -> long {
  const auto& value = repr_["tabSize"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FormattingOptions::insertSpaces() const -> bool {
  const auto& value = repr_["insertSpaces"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::trimTrailingWhitespace() const -> std::optional<bool> {
  if (!repr_.contains("trimTrailingWhitespace")) return std::nullopt;

  const auto& value = repr_["trimTrailingWhitespace"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::insertFinalNewline() const -> std::optional<bool> {
  if (!repr_.contains("insertFinalNewline")) return std::nullopt;

  const auto& value = repr_["insertFinalNewline"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::trimFinalNewlines() const -> std::optional<bool> {
  if (!repr_.contains("trimFinalNewlines")) return std::nullopt;

  const auto& value = repr_["trimFinalNewlines"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::tabSize(long tabSize) -> FormattingOptions& {
  return *this;
}

auto FormattingOptions::insertSpaces(bool insertSpaces) -> FormattingOptions& {
  return *this;
}

auto FormattingOptions::trimTrailingWhitespace(
    std::optional<bool> trimTrailingWhitespace) -> FormattingOptions& {
  return *this;
}

auto FormattingOptions::insertFinalNewline(
    std::optional<bool> insertFinalNewline) -> FormattingOptions& {
  return *this;
}

auto FormattingOptions::trimFinalNewlines(std::optional<bool> trimFinalNewlines)
    -> FormattingOptions& {
  return *this;
}

DocumentFormattingOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentFormattingOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentFormattingOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentFormattingOptions& {
  return *this;
}

DocumentRangeFormattingOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentRangeFormattingOptions::rangesSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("rangesSupport")) return std::nullopt;

  const auto& value = repr_["rangesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingOptions::rangesSupport(
    std::optional<bool> rangesSupport) -> DocumentRangeFormattingOptions& {
  return *this;
}

auto DocumentRangeFormattingOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentRangeFormattingOptions& {
  return *this;
}

DocumentOnTypeFormattingOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("firstTriggerCharacter")) return false;
  return true;
}

auto DocumentOnTypeFormattingOptions::firstTriggerCharacter() const
    -> std::string {
  const auto& value = repr_["firstTriggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentOnTypeFormattingOptions::moreTriggerCharacter() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("moreTriggerCharacter")) return std::nullopt;

  const auto& value = repr_["moreTriggerCharacter"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto DocumentOnTypeFormattingOptions::firstTriggerCharacter(
    std::string firstTriggerCharacter) -> DocumentOnTypeFormattingOptions& {
  return *this;
}

auto DocumentOnTypeFormattingOptions::moreTriggerCharacter(
    std::optional<Vector<std::string>> moreTriggerCharacter)
    -> DocumentOnTypeFormattingOptions& {
  return *this;
}

RenameOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto RenameOptions::prepareProvider() const -> std::optional<bool> {
  if (!repr_.contains("prepareProvider")) return std::nullopt;

  const auto& value = repr_["prepareProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameOptions::prepareProvider(std::optional<bool> prepareProvider)
    -> RenameOptions& {
  return *this;
}

auto RenameOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> RenameOptions& {
  return *this;
}

PrepareRenamePlaceholder::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("placeholder")) return false;
  return true;
}

auto PrepareRenamePlaceholder::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto PrepareRenamePlaceholder::placeholder() const -> std::string {
  const auto& value = repr_["placeholder"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PrepareRenamePlaceholder::range(Range range) -> PrepareRenamePlaceholder& {
  return *this;
}

auto PrepareRenamePlaceholder::placeholder(std::string placeholder)
    -> PrepareRenamePlaceholder& {
  return *this;
}

PrepareRenameDefaultBehavior::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("defaultBehavior")) return false;
  return true;
}

auto PrepareRenameDefaultBehavior::defaultBehavior() const -> bool {
  const auto& value = repr_["defaultBehavior"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PrepareRenameDefaultBehavior::defaultBehavior(bool defaultBehavior)
    -> PrepareRenameDefaultBehavior& {
  return *this;
}

ExecuteCommandOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("commands")) return false;
  return true;
}

auto ExecuteCommandOptions::commands() const -> Vector<std::string> {
  const auto& value = repr_["commands"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ExecuteCommandOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecuteCommandOptions::commands(Vector<std::string> commands)
    -> ExecuteCommandOptions& {
  return *this;
}

auto ExecuteCommandOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> ExecuteCommandOptions& {
  return *this;
}

WorkspaceEditMetadata::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceEditMetadata::isRefactoring() const -> std::optional<bool> {
  if (!repr_.contains("isRefactoring")) return std::nullopt;

  const auto& value = repr_["isRefactoring"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditMetadata::isRefactoring(std::optional<bool> isRefactoring)
    -> WorkspaceEditMetadata& {
  return *this;
}

SemanticTokensLegend::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("tokenTypes")) return false;
  if (!repr_.contains("tokenModifiers")) return false;
  return true;
}

auto SemanticTokensLegend::tokenTypes() const -> Vector<std::string> {
  const auto& value = repr_["tokenTypes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensLegend::tokenModifiers() const -> Vector<std::string> {
  const auto& value = repr_["tokenModifiers"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensLegend::tokenTypes(Vector<std::string> tokenTypes)
    -> SemanticTokensLegend& {
  return *this;
}

auto SemanticTokensLegend::tokenModifiers(Vector<std::string> tokenModifiers)
    -> SemanticTokensLegend& {
  return *this;
}

SemanticTokensFullDelta::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SemanticTokensFullDelta::delta() const -> std::optional<bool> {
  if (!repr_.contains("delta")) return std::nullopt;

  const auto& value = repr_["delta"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensFullDelta::delta(std::optional<bool> delta)
    -> SemanticTokensFullDelta& {
  return *this;
}

OptionalVersionedTextDocumentIdentifier::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("uri")) return false;
  return true;
}

auto OptionalVersionedTextDocumentIdentifier::version() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  const auto& value = repr_["version"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto OptionalVersionedTextDocumentIdentifier::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto OptionalVersionedTextDocumentIdentifier::version(
    std::variant<std::monostate, int, std::nullptr_t> version)
    -> OptionalVersionedTextDocumentIdentifier& {
  return *this;
}

auto OptionalVersionedTextDocumentIdentifier::uri(std::string uri)
    -> OptionalVersionedTextDocumentIdentifier& {
  return *this;
}

AnnotatedTextEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("annotationId")) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("newText")) return false;
  return true;
}

auto AnnotatedTextEdit::annotationId() const -> ChangeAnnotationIdentifier {
  const auto& value = repr_["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto AnnotatedTextEdit::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto AnnotatedTextEdit::newText() const -> std::string {
  const auto& value = repr_["newText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto AnnotatedTextEdit::annotationId(ChangeAnnotationIdentifier annotationId)
    -> AnnotatedTextEdit& {
  return *this;
}

auto AnnotatedTextEdit::range(Range range) -> AnnotatedTextEdit& {
  return *this;
}

auto AnnotatedTextEdit::newText(std::string newText) -> AnnotatedTextEdit& {
  return *this;
}

SnippetTextEdit::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("snippet")) return false;
  return true;
}

auto SnippetTextEdit::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto SnippetTextEdit::snippet() const -> StringValue {
  const auto& value = repr_["snippet"];

  return StringValue(value);
}

auto SnippetTextEdit::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_.contains("annotationId")) return std::nullopt;

  const auto& value = repr_["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SnippetTextEdit::range(Range range) -> SnippetTextEdit& { return *this; }

auto SnippetTextEdit::snippet(StringValue snippet) -> SnippetTextEdit& {
  return *this;
}

auto SnippetTextEdit::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId)
    -> SnippetTextEdit& {
  return *this;
}

ResourceOperation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  return true;
}

auto ResourceOperation::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ResourceOperation::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_.contains("annotationId")) return std::nullopt;

  const auto& value = repr_["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ResourceOperation::kind(std::string kind) -> ResourceOperation& {
  return *this;
}

auto ResourceOperation::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId)
    -> ResourceOperation& {
  return *this;
}

CreateFileOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CreateFileOptions::overwrite() const -> std::optional<bool> {
  if (!repr_.contains("overwrite")) return std::nullopt;

  const auto& value = repr_["overwrite"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CreateFileOptions::ignoreIfExists() const -> std::optional<bool> {
  if (!repr_.contains("ignoreIfExists")) return std::nullopt;

  const auto& value = repr_["ignoreIfExists"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CreateFileOptions::overwrite(std::optional<bool> overwrite)
    -> CreateFileOptions& {
  return *this;
}

auto CreateFileOptions::ignoreIfExists(std::optional<bool> ignoreIfExists)
    -> CreateFileOptions& {
  return *this;
}

RenameFileOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto RenameFileOptions::overwrite() const -> std::optional<bool> {
  if (!repr_.contains("overwrite")) return std::nullopt;

  const auto& value = repr_["overwrite"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameFileOptions::ignoreIfExists() const -> std::optional<bool> {
  if (!repr_.contains("ignoreIfExists")) return std::nullopt;

  const auto& value = repr_["ignoreIfExists"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameFileOptions::overwrite(std::optional<bool> overwrite)
    -> RenameFileOptions& {
  return *this;
}

auto RenameFileOptions::ignoreIfExists(std::optional<bool> ignoreIfExists)
    -> RenameFileOptions& {
  return *this;
}

DeleteFileOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DeleteFileOptions::recursive() const -> std::optional<bool> {
  if (!repr_.contains("recursive")) return std::nullopt;

  const auto& value = repr_["recursive"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeleteFileOptions::ignoreIfNotExists() const -> std::optional<bool> {
  if (!repr_.contains("ignoreIfNotExists")) return std::nullopt;

  const auto& value = repr_["ignoreIfNotExists"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeleteFileOptions::recursive(std::optional<bool> recursive)
    -> DeleteFileOptions& {
  return *this;
}

auto DeleteFileOptions::ignoreIfNotExists(std::optional<bool> ignoreIfNotExists)
    -> DeleteFileOptions& {
  return *this;
}

FileOperationPattern::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("glob")) return false;
  return true;
}

auto FileOperationPattern::glob() const -> std::string {
  const auto& value = repr_["glob"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileOperationPattern::matches() const
    -> std::optional<FileOperationPatternKind> {
  if (!repr_.contains("matches")) return std::nullopt;

  const auto& value = repr_["matches"];

  lsp_runtime_error("FileOperationPattern::matches: not implement yet");
}

auto FileOperationPattern::options() const
    -> std::optional<FileOperationPatternOptions> {
  if (!repr_.contains("options")) return std::nullopt;

  const auto& value = repr_["options"];

  return FileOperationPatternOptions(value);
}

auto FileOperationPattern::glob(std::string glob) -> FileOperationPattern& {
  return *this;
}

auto FileOperationPattern::matches(
    std::optional<FileOperationPatternKind> matches) -> FileOperationPattern& {
  return *this;
}

auto FileOperationPattern::options(
    std::optional<FileOperationPatternOptions> options)
    -> FileOperationPattern& {
  return *this;
}

WorkspaceFullDocumentDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "full") return false;
  if (!repr_.contains("items")) return false;
  return true;
}

auto WorkspaceFullDocumentDiagnosticReport::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFullDocumentDiagnosticReport::version() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  const auto& value = repr_["version"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFullDocumentDiagnosticReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFullDocumentDiagnosticReport::resultId() const
    -> std::optional<std::string> {
  if (!repr_.contains("resultId")) return std::nullopt;

  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFullDocumentDiagnosticReport::items() const
    -> Vector<Diagnostic> {
  const auto& value = repr_["items"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto WorkspaceFullDocumentDiagnosticReport::uri(std::string uri)
    -> WorkspaceFullDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::version(
    std::variant<std::monostate, int, std::nullptr_t> version)
    -> WorkspaceFullDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::kind(std::string kind)
    -> WorkspaceFullDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::resultId(
    std::optional<std::string> resultId)
    -> WorkspaceFullDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::items(Vector<Diagnostic> items)
    -> WorkspaceFullDocumentDiagnosticReport& {
  return *this;
}

WorkspaceUnchangedDocumentDiagnosticReport::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("uri")) return false;
  if (!repr_.contains("version")) return false;
  if (!repr_.contains("kind")) return false;
  if (repr_["kind"] != "unchanged") return false;
  if (!repr_.contains("resultId")) return false;
  return true;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::uri() const -> std::string {
  const auto& value = repr_["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceUnchangedDocumentDiagnosticReport::version() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  const auto& value = repr_["version"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::kind() const -> std::string {
  const auto& value = repr_["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceUnchangedDocumentDiagnosticReport::resultId() const
    -> std::string {
  const auto& value = repr_["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceUnchangedDocumentDiagnosticReport::uri(std::string uri)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::version(
    std::variant<std::monostate, int, std::nullptr_t> version)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::kind(std::string kind)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  return *this;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::resultId(std::string resultId)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  return *this;
}

NotebookCell::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (!repr_.contains("document")) return false;
  return true;
}

auto NotebookCell::kind() const -> NotebookCellKind {
  const auto& value = repr_["kind"];

  return NotebookCellKind(value);
}

auto NotebookCell::document() const -> std::string {
  const auto& value = repr_["document"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookCell::metadata() const -> std::optional<LSPObject> {
  if (!repr_.contains("metadata")) return std::nullopt;

  const auto& value = repr_["metadata"];

  assert(value.is_object());
  return LSPObject(value);
}

auto NotebookCell::executionSummary() const -> std::optional<ExecutionSummary> {
  if (!repr_.contains("executionSummary")) return std::nullopt;

  const auto& value = repr_["executionSummary"];

  return ExecutionSummary(value);
}

auto NotebookCell::kind(NotebookCellKind kind) -> NotebookCell& {
  return *this;
}

auto NotebookCell::document(std::string document) -> NotebookCell& {
  return *this;
}

auto NotebookCell::metadata(std::optional<LSPObject> metadata)
    -> NotebookCell& {
  return *this;
}

auto NotebookCell::executionSummary(
    std::optional<ExecutionSummary> executionSummary) -> NotebookCell& {
  return *this;
}

NotebookDocumentFilterWithNotebook::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebook")) return false;
  return true;
}

auto NotebookDocumentFilterWithNotebook::notebook() const
    -> std::variant<std::monostate, std::string, NotebookDocumentFilter> {
  const auto& value = repr_["notebook"];

  std::variant<std::monostate, std::string, NotebookDocumentFilter> result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterWithNotebook::cells() const
    -> std::optional<Vector<NotebookCellLanguage>> {
  if (!repr_.contains("cells")) return std::nullopt;

  const auto& value = repr_["cells"];

  assert(value.is_array());
  return Vector<NotebookCellLanguage>(value);
}

auto NotebookDocumentFilterWithNotebook::notebook(
    std::variant<std::monostate, std::string, NotebookDocumentFilter> notebook)
    -> NotebookDocumentFilterWithNotebook& {
  return *this;
}

auto NotebookDocumentFilterWithNotebook::cells(
    std::optional<Vector<NotebookCellLanguage>> cells)
    -> NotebookDocumentFilterWithNotebook& {
  return *this;
}

NotebookDocumentFilterWithCells::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("cells")) return false;
  return true;
}

auto NotebookDocumentFilterWithCells::notebook() const -> std::optional<
    std::variant<std::monostate, std::string, NotebookDocumentFilter>> {
  if (!repr_.contains("notebook")) return std::nullopt;

  const auto& value = repr_["notebook"];

  std::variant<std::monostate, std::string, NotebookDocumentFilter> result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterWithCells::cells() const
    -> Vector<NotebookCellLanguage> {
  const auto& value = repr_["cells"];

  assert(value.is_array());
  return Vector<NotebookCellLanguage>(value);
}

auto NotebookDocumentFilterWithCells::notebook(
    std::optional<
        std::variant<std::monostate, std::string, NotebookDocumentFilter>>
        notebook) -> NotebookDocumentFilterWithCells& {
  return *this;
}

auto NotebookDocumentFilterWithCells::cells(Vector<NotebookCellLanguage> cells)
    -> NotebookDocumentFilterWithCells& {
  return *this;
}

NotebookDocumentCellChanges::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto NotebookDocumentCellChanges::structure() const
    -> std::optional<NotebookDocumentCellChangeStructure> {
  if (!repr_.contains("structure")) return std::nullopt;

  const auto& value = repr_["structure"];

  return NotebookDocumentCellChangeStructure(value);
}

auto NotebookDocumentCellChanges::data() const
    -> std::optional<Vector<NotebookCell>> {
  if (!repr_.contains("data")) return std::nullopt;

  const auto& value = repr_["data"];

  assert(value.is_array());
  return Vector<NotebookCell>(value);
}

auto NotebookDocumentCellChanges::textContent() const
    -> std::optional<Vector<NotebookDocumentCellContentChanges>> {
  if (!repr_.contains("textContent")) return std::nullopt;

  const auto& value = repr_["textContent"];

  assert(value.is_array());
  return Vector<NotebookDocumentCellContentChanges>(value);
}

auto NotebookDocumentCellChanges::structure(
    std::optional<NotebookDocumentCellChangeStructure> structure)
    -> NotebookDocumentCellChanges& {
  return *this;
}

auto NotebookDocumentCellChanges::data(std::optional<Vector<NotebookCell>> data)
    -> NotebookDocumentCellChanges& {
  return *this;
}

auto NotebookDocumentCellChanges::textContent(
    std::optional<Vector<NotebookDocumentCellContentChanges>> textContent)
    -> NotebookDocumentCellChanges& {
  return *this;
}

SelectedCompletionInfo::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("text")) return false;
  return true;
}

auto SelectedCompletionInfo::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto SelectedCompletionInfo::text() const -> std::string {
  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SelectedCompletionInfo::range(Range range) -> SelectedCompletionInfo& {
  return *this;
}

auto SelectedCompletionInfo::text(std::string text) -> SelectedCompletionInfo& {
  return *this;
}

ClientInfo::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("name")) return false;
  return true;
}

auto ClientInfo::name() const -> std::string {
  const auto& value = repr_["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ClientInfo::version() const -> std::optional<std::string> {
  if (!repr_.contains("version")) return std::nullopt;

  const auto& value = repr_["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ClientInfo::name(std::string name) -> ClientInfo& { return *this; }

auto ClientInfo::version(std::optional<std::string> version) -> ClientInfo& {
  return *this;
}

ClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientCapabilities::workspace() const
    -> std::optional<WorkspaceClientCapabilities> {
  if (!repr_.contains("workspace")) return std::nullopt;

  const auto& value = repr_["workspace"];

  return WorkspaceClientCapabilities(value);
}

auto ClientCapabilities::textDocument() const
    -> std::optional<TextDocumentClientCapabilities> {
  if (!repr_.contains("textDocument")) return std::nullopt;

  const auto& value = repr_["textDocument"];

  return TextDocumentClientCapabilities(value);
}

auto ClientCapabilities::notebookDocument() const
    -> std::optional<NotebookDocumentClientCapabilities> {
  if (!repr_.contains("notebookDocument")) return std::nullopt;

  const auto& value = repr_["notebookDocument"];

  return NotebookDocumentClientCapabilities(value);
}

auto ClientCapabilities::window() const
    -> std::optional<WindowClientCapabilities> {
  if (!repr_.contains("window")) return std::nullopt;

  const auto& value = repr_["window"];

  return WindowClientCapabilities(value);
}

auto ClientCapabilities::general() const
    -> std::optional<GeneralClientCapabilities> {
  if (!repr_.contains("general")) return std::nullopt;

  const auto& value = repr_["general"];

  return GeneralClientCapabilities(value);
}

auto ClientCapabilities::experimental() const -> std::optional<LSPAny> {
  if (!repr_.contains("experimental")) return std::nullopt;

  const auto& value = repr_["experimental"];

  assert(value.is_object());
  return LSPAny(value);
}

auto ClientCapabilities::workspace(
    std::optional<WorkspaceClientCapabilities> workspace)
    -> ClientCapabilities& {
  return *this;
}

auto ClientCapabilities::textDocument(
    std::optional<TextDocumentClientCapabilities> textDocument)
    -> ClientCapabilities& {
  return *this;
}

auto ClientCapabilities::notebookDocument(
    std::optional<NotebookDocumentClientCapabilities> notebookDocument)
    -> ClientCapabilities& {
  return *this;
}

auto ClientCapabilities::window(std::optional<WindowClientCapabilities> window)
    -> ClientCapabilities& {
  return *this;
}

auto ClientCapabilities::general(
    std::optional<GeneralClientCapabilities> general) -> ClientCapabilities& {
  return *this;
}

auto ClientCapabilities::experimental(std::optional<LSPAny> experimental)
    -> ClientCapabilities& {
  return *this;
}

TextDocumentSyncOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TextDocumentSyncOptions::openClose() const -> std::optional<bool> {
  if (!repr_.contains("openClose")) return std::nullopt;

  const auto& value = repr_["openClose"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncOptions::change() const
    -> std::optional<TextDocumentSyncKind> {
  if (!repr_.contains("change")) return std::nullopt;

  const auto& value = repr_["change"];

  return TextDocumentSyncKind(value);
}

auto TextDocumentSyncOptions::willSave() const -> std::optional<bool> {
  if (!repr_.contains("willSave")) return std::nullopt;

  const auto& value = repr_["willSave"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncOptions::willSaveWaitUntil() const -> std::optional<bool> {
  if (!repr_.contains("willSaveWaitUntil")) return std::nullopt;

  const auto& value = repr_["willSaveWaitUntil"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncOptions::save() const
    -> std::optional<std::variant<std::monostate, bool, SaveOptions>> {
  if (!repr_.contains("save")) return std::nullopt;

  const auto& value = repr_["save"];

  std::variant<std::monostate, bool, SaveOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentSyncOptions::openClose(std::optional<bool> openClose)
    -> TextDocumentSyncOptions& {
  return *this;
}

auto TextDocumentSyncOptions::change(std::optional<TextDocumentSyncKind> change)
    -> TextDocumentSyncOptions& {
  return *this;
}

auto TextDocumentSyncOptions::willSave(std::optional<bool> willSave)
    -> TextDocumentSyncOptions& {
  return *this;
}

auto TextDocumentSyncOptions::willSaveWaitUntil(
    std::optional<bool> willSaveWaitUntil) -> TextDocumentSyncOptions& {
  return *this;
}

auto TextDocumentSyncOptions::save(
    std::optional<std::variant<std::monostate, bool, SaveOptions>> save)
    -> TextDocumentSyncOptions& {
  return *this;
}

WorkspaceOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceOptions::workspaceFolders() const
    -> std::optional<WorkspaceFoldersServerCapabilities> {
  if (!repr_.contains("workspaceFolders")) return std::nullopt;

  const auto& value = repr_["workspaceFolders"];

  return WorkspaceFoldersServerCapabilities(value);
}

auto WorkspaceOptions::fileOperations() const
    -> std::optional<FileOperationOptions> {
  if (!repr_.contains("fileOperations")) return std::nullopt;

  const auto& value = repr_["fileOperations"];

  return FileOperationOptions(value);
}

auto WorkspaceOptions::textDocumentContent() const
    -> std::optional<std::variant<std::monostate, TextDocumentContentOptions,
                                  TextDocumentContentRegistrationOptions>> {
  if (!repr_.contains("textDocumentContent")) return std::nullopt;

  const auto& value = repr_["textDocumentContent"];

  std::variant<std::monostate, TextDocumentContentOptions,
               TextDocumentContentRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceOptions::workspaceFolders(
    std::optional<WorkspaceFoldersServerCapabilities> workspaceFolders)
    -> WorkspaceOptions& {
  return *this;
}

auto WorkspaceOptions::fileOperations(
    std::optional<FileOperationOptions> fileOperations) -> WorkspaceOptions& {
  return *this;
}

auto WorkspaceOptions::textDocumentContent(
    std::optional<std::variant<std::monostate, TextDocumentContentOptions,
                               TextDocumentContentRegistrationOptions>>
        textDocumentContent) -> WorkspaceOptions& {
  return *this;
}

TextDocumentContentChangePartial::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("range")) return false;
  if (!repr_.contains("text")) return false;
  return true;
}

auto TextDocumentContentChangePartial::range() const -> Range {
  const auto& value = repr_["range"];

  return Range(value);
}

auto TextDocumentContentChangePartial::rangeLength() const
    -> std::optional<long> {
  if (!repr_.contains("rangeLength")) return std::nullopt;

  const auto& value = repr_["rangeLength"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto TextDocumentContentChangePartial::text() const -> std::string {
  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentChangePartial::range(Range range)
    -> TextDocumentContentChangePartial& {
  return *this;
}

auto TextDocumentContentChangePartial::rangeLength(
    std::optional<long> rangeLength) -> TextDocumentContentChangePartial& {
  return *this;
}

auto TextDocumentContentChangePartial::text(std::string text)
    -> TextDocumentContentChangePartial& {
  return *this;
}

TextDocumentContentChangeWholeDocument::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("text")) return false;
  return true;
}

auto TextDocumentContentChangeWholeDocument::text() const -> std::string {
  const auto& value = repr_["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentChangeWholeDocument::text(std::string text)
    -> TextDocumentContentChangeWholeDocument& {
  return *this;
}

CodeDescription::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("href")) return false;
  return true;
}

auto CodeDescription::href() const -> std::string {
  const auto& value = repr_["href"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CodeDescription::href(std::string href) -> CodeDescription& {
  return *this;
}

DiagnosticRelatedInformation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("location")) return false;
  if (!repr_.contains("message")) return false;
  return true;
}

auto DiagnosticRelatedInformation::location() const -> Location {
  const auto& value = repr_["location"];

  return Location(value);
}

auto DiagnosticRelatedInformation::message() const -> std::string {
  const auto& value = repr_["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticRelatedInformation::location(Location location)
    -> DiagnosticRelatedInformation& {
  return *this;
}

auto DiagnosticRelatedInformation::message(std::string message)
    -> DiagnosticRelatedInformation& {
  return *this;
}

EditRangeWithInsertReplace::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("insert")) return false;
  if (!repr_.contains("replace")) return false;
  return true;
}

auto EditRangeWithInsertReplace::insert() const -> Range {
  const auto& value = repr_["insert"];

  return Range(value);
}

auto EditRangeWithInsertReplace::replace() const -> Range {
  const auto& value = repr_["replace"];

  return Range(value);
}

auto EditRangeWithInsertReplace::insert(Range insert)
    -> EditRangeWithInsertReplace& {
  return *this;
}

auto EditRangeWithInsertReplace::replace(Range replace)
    -> EditRangeWithInsertReplace& {
  return *this;
}

ServerCompletionItemOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ServerCompletionItemOptions::labelDetailsSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("labelDetailsSupport")) return std::nullopt;

  const auto& value = repr_["labelDetailsSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ServerCompletionItemOptions::labelDetailsSupport(
    std::optional<bool> labelDetailsSupport) -> ServerCompletionItemOptions& {
  return *this;
}

MarkedStringWithLanguage::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("language")) return false;
  if (!repr_.contains("value")) return false;
  return true;
}

auto MarkedStringWithLanguage::language() const -> std::string {
  const auto& value = repr_["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkedStringWithLanguage::value() const -> std::string {
  const auto& value = repr_["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkedStringWithLanguage::language(std::string language)
    -> MarkedStringWithLanguage& {
  return *this;
}

auto MarkedStringWithLanguage::value(std::string value)
    -> MarkedStringWithLanguage& {
  return *this;
}

ParameterInformation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("label")) return false;
  return true;
}

auto ParameterInformation::label() const
    -> std::variant<std::monostate, std::string, std::tuple<long, long>> {
  const auto& value = repr_["label"];

  std::variant<std::monostate, std::string, std::tuple<long, long>> result;

  details::try_emplace(result, value);

  return result;
}

auto ParameterInformation::documentation() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_.contains("documentation")) return std::nullopt;

  const auto& value = repr_["documentation"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto ParameterInformation::label(
    std::variant<std::monostate, std::string, std::tuple<long, long>> label)
    -> ParameterInformation& {
  return *this;
}

auto ParameterInformation::documentation(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        documentation) -> ParameterInformation& {
  return *this;
}

CodeActionKindDocumentation::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("kind")) return false;
  if (!repr_.contains("command")) return false;
  return true;
}

auto CodeActionKindDocumentation::kind() const -> CodeActionKind {
  const auto& value = repr_["kind"];

  lsp_runtime_error("CodeActionKindDocumentation::kind: not implement yet");
}

auto CodeActionKindDocumentation::command() const -> Command {
  const auto& value = repr_["command"];

  return Command(value);
}

auto CodeActionKindDocumentation::kind(CodeActionKind kind)
    -> CodeActionKindDocumentation& {
  return *this;
}

auto CodeActionKindDocumentation::command(Command command)
    -> CodeActionKindDocumentation& {
  return *this;
}

NotebookCellTextDocumentFilter::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebook")) return false;
  return true;
}

auto NotebookCellTextDocumentFilter::notebook() const
    -> std::variant<std::monostate, std::string, NotebookDocumentFilter> {
  const auto& value = repr_["notebook"];

  std::variant<std::monostate, std::string, NotebookDocumentFilter> result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookCellTextDocumentFilter::language() const
    -> std::optional<std::string> {
  if (!repr_.contains("language")) return std::nullopt;

  const auto& value = repr_["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookCellTextDocumentFilter::notebook(
    std::variant<std::monostate, std::string, NotebookDocumentFilter> notebook)
    -> NotebookCellTextDocumentFilter& {
  return *this;
}

auto NotebookCellTextDocumentFilter::language(
    std::optional<std::string> language) -> NotebookCellTextDocumentFilter& {
  return *this;
}

FileOperationPatternOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto FileOperationPatternOptions::ignoreCase() const -> std::optional<bool> {
  if (!repr_.contains("ignoreCase")) return std::nullopt;

  const auto& value = repr_["ignoreCase"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationPatternOptions::ignoreCase(std::optional<bool> ignoreCase)
    -> FileOperationPatternOptions& {
  return *this;
}

ExecutionSummary::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("executionOrder")) return false;
  return true;
}

auto ExecutionSummary::executionOrder() const -> long {
  const auto& value = repr_["executionOrder"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto ExecutionSummary::success() const -> std::optional<bool> {
  if (!repr_.contains("success")) return std::nullopt;

  const auto& value = repr_["success"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecutionSummary::executionOrder(long executionOrder)
    -> ExecutionSummary& {
  return *this;
}

auto ExecutionSummary::success(std::optional<bool> success)
    -> ExecutionSummary& {
  return *this;
}

NotebookCellLanguage::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("language")) return false;
  return true;
}

auto NotebookCellLanguage::language() const -> std::string {
  const auto& value = repr_["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookCellLanguage::language(std::string language)
    -> NotebookCellLanguage& {
  return *this;
}

NotebookDocumentCellChangeStructure::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("array")) return false;
  return true;
}

auto NotebookDocumentCellChangeStructure::array() const
    -> NotebookCellArrayChange {
  const auto& value = repr_["array"];

  return NotebookCellArrayChange(value);
}

auto NotebookDocumentCellChangeStructure::didOpen() const
    -> std::optional<Vector<TextDocumentItem>> {
  if (!repr_.contains("didOpen")) return std::nullopt;

  const auto& value = repr_["didOpen"];

  assert(value.is_array());
  return Vector<TextDocumentItem>(value);
}

auto NotebookDocumentCellChangeStructure::didClose() const
    -> std::optional<Vector<TextDocumentIdentifier>> {
  if (!repr_.contains("didClose")) return std::nullopt;

  const auto& value = repr_["didClose"];

  assert(value.is_array());
  return Vector<TextDocumentIdentifier>(value);
}

auto NotebookDocumentCellChangeStructure::array(NotebookCellArrayChange array)
    -> NotebookDocumentCellChangeStructure& {
  return *this;
}

auto NotebookDocumentCellChangeStructure::didOpen(
    std::optional<Vector<TextDocumentItem>> didOpen)
    -> NotebookDocumentCellChangeStructure& {
  return *this;
}

auto NotebookDocumentCellChangeStructure::didClose(
    std::optional<Vector<TextDocumentIdentifier>> didClose)
    -> NotebookDocumentCellChangeStructure& {
  return *this;
}

NotebookDocumentCellContentChanges::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("document")) return false;
  if (!repr_.contains("changes")) return false;
  return true;
}

auto NotebookDocumentCellContentChanges::document() const
    -> VersionedTextDocumentIdentifier {
  const auto& value = repr_["document"];

  return VersionedTextDocumentIdentifier(value);
}

auto NotebookDocumentCellContentChanges::changes() const
    -> Vector<TextDocumentContentChangeEvent> {
  const auto& value = repr_["changes"];

  assert(value.is_array());
  return Vector<TextDocumentContentChangeEvent>(value);
}

auto NotebookDocumentCellContentChanges::document(
    VersionedTextDocumentIdentifier document)
    -> NotebookDocumentCellContentChanges& {
  return *this;
}

auto NotebookDocumentCellContentChanges::changes(
    Vector<TextDocumentContentChangeEvent> changes)
    -> NotebookDocumentCellContentChanges& {
  return *this;
}

WorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceClientCapabilities::applyEdit() const -> std::optional<bool> {
  if (!repr_.contains("applyEdit")) return std::nullopt;

  const auto& value = repr_["applyEdit"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceClientCapabilities::workspaceEdit() const
    -> std::optional<WorkspaceEditClientCapabilities> {
  if (!repr_.contains("workspaceEdit")) return std::nullopt;

  const auto& value = repr_["workspaceEdit"];

  return WorkspaceEditClientCapabilities(value);
}

auto WorkspaceClientCapabilities::didChangeConfiguration() const
    -> std::optional<DidChangeConfigurationClientCapabilities> {
  if (!repr_.contains("didChangeConfiguration")) return std::nullopt;

  const auto& value = repr_["didChangeConfiguration"];

  return DidChangeConfigurationClientCapabilities(value);
}

auto WorkspaceClientCapabilities::didChangeWatchedFiles() const
    -> std::optional<DidChangeWatchedFilesClientCapabilities> {
  if (!repr_.contains("didChangeWatchedFiles")) return std::nullopt;

  const auto& value = repr_["didChangeWatchedFiles"];

  return DidChangeWatchedFilesClientCapabilities(value);
}

auto WorkspaceClientCapabilities::symbol() const
    -> std::optional<WorkspaceSymbolClientCapabilities> {
  if (!repr_.contains("symbol")) return std::nullopt;

  const auto& value = repr_["symbol"];

  return WorkspaceSymbolClientCapabilities(value);
}

auto WorkspaceClientCapabilities::executeCommand() const
    -> std::optional<ExecuteCommandClientCapabilities> {
  if (!repr_.contains("executeCommand")) return std::nullopt;

  const auto& value = repr_["executeCommand"];

  return ExecuteCommandClientCapabilities(value);
}

auto WorkspaceClientCapabilities::workspaceFolders() const
    -> std::optional<bool> {
  if (!repr_.contains("workspaceFolders")) return std::nullopt;

  const auto& value = repr_["workspaceFolders"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceClientCapabilities::configuration() const -> std::optional<bool> {
  if (!repr_.contains("configuration")) return std::nullopt;

  const auto& value = repr_["configuration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceClientCapabilities::semanticTokens() const
    -> std::optional<SemanticTokensWorkspaceClientCapabilities> {
  if (!repr_.contains("semanticTokens")) return std::nullopt;

  const auto& value = repr_["semanticTokens"];

  return SemanticTokensWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::codeLens() const
    -> std::optional<CodeLensWorkspaceClientCapabilities> {
  if (!repr_.contains("codeLens")) return std::nullopt;

  const auto& value = repr_["codeLens"];

  return CodeLensWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::fileOperations() const
    -> std::optional<FileOperationClientCapabilities> {
  if (!repr_.contains("fileOperations")) return std::nullopt;

  const auto& value = repr_["fileOperations"];

  return FileOperationClientCapabilities(value);
}

auto WorkspaceClientCapabilities::inlineValue() const
    -> std::optional<InlineValueWorkspaceClientCapabilities> {
  if (!repr_.contains("inlineValue")) return std::nullopt;

  const auto& value = repr_["inlineValue"];

  return InlineValueWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::inlayHint() const
    -> std::optional<InlayHintWorkspaceClientCapabilities> {
  if (!repr_.contains("inlayHint")) return std::nullopt;

  const auto& value = repr_["inlayHint"];

  return InlayHintWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::diagnostics() const
    -> std::optional<DiagnosticWorkspaceClientCapabilities> {
  if (!repr_.contains("diagnostics")) return std::nullopt;

  const auto& value = repr_["diagnostics"];

  return DiagnosticWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::foldingRange() const
    -> std::optional<FoldingRangeWorkspaceClientCapabilities> {
  if (!repr_.contains("foldingRange")) return std::nullopt;

  const auto& value = repr_["foldingRange"];

  return FoldingRangeWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::textDocumentContent() const
    -> std::optional<TextDocumentContentClientCapabilities> {
  if (!repr_.contains("textDocumentContent")) return std::nullopt;

  const auto& value = repr_["textDocumentContent"];

  return TextDocumentContentClientCapabilities(value);
}

auto WorkspaceClientCapabilities::applyEdit(std::optional<bool> applyEdit)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::workspaceEdit(
    std::optional<WorkspaceEditClientCapabilities> workspaceEdit)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::didChangeConfiguration(
    std::optional<DidChangeConfigurationClientCapabilities>
        didChangeConfiguration) -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::didChangeWatchedFiles(
    std::optional<DidChangeWatchedFilesClientCapabilities>
        didChangeWatchedFiles) -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::symbol(
    std::optional<WorkspaceSymbolClientCapabilities> symbol)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::executeCommand(
    std::optional<ExecuteCommandClientCapabilities> executeCommand)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::workspaceFolders(
    std::optional<bool> workspaceFolders) -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::configuration(
    std::optional<bool> configuration) -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::semanticTokens(
    std::optional<SemanticTokensWorkspaceClientCapabilities> semanticTokens)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::codeLens(
    std::optional<CodeLensWorkspaceClientCapabilities> codeLens)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::fileOperations(
    std::optional<FileOperationClientCapabilities> fileOperations)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::inlineValue(
    std::optional<InlineValueWorkspaceClientCapabilities> inlineValue)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::inlayHint(
    std::optional<InlayHintWorkspaceClientCapabilities> inlayHint)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::diagnostics(
    std::optional<DiagnosticWorkspaceClientCapabilities> diagnostics)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::foldingRange(
    std::optional<FoldingRangeWorkspaceClientCapabilities> foldingRange)
    -> WorkspaceClientCapabilities& {
  return *this;
}

auto WorkspaceClientCapabilities::textDocumentContent(
    std::optional<TextDocumentContentClientCapabilities> textDocumentContent)
    -> WorkspaceClientCapabilities& {
  return *this;
}

TextDocumentClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TextDocumentClientCapabilities::synchronization() const
    -> std::optional<TextDocumentSyncClientCapabilities> {
  if (!repr_.contains("synchronization")) return std::nullopt;

  const auto& value = repr_["synchronization"];

  return TextDocumentSyncClientCapabilities(value);
}

auto TextDocumentClientCapabilities::filters() const
    -> std::optional<TextDocumentFilterClientCapabilities> {
  if (!repr_.contains("filters")) return std::nullopt;

  const auto& value = repr_["filters"];

  return TextDocumentFilterClientCapabilities(value);
}

auto TextDocumentClientCapabilities::completion() const
    -> std::optional<CompletionClientCapabilities> {
  if (!repr_.contains("completion")) return std::nullopt;

  const auto& value = repr_["completion"];

  return CompletionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::hover() const
    -> std::optional<HoverClientCapabilities> {
  if (!repr_.contains("hover")) return std::nullopt;

  const auto& value = repr_["hover"];

  return HoverClientCapabilities(value);
}

auto TextDocumentClientCapabilities::signatureHelp() const
    -> std::optional<SignatureHelpClientCapabilities> {
  if (!repr_.contains("signatureHelp")) return std::nullopt;

  const auto& value = repr_["signatureHelp"];

  return SignatureHelpClientCapabilities(value);
}

auto TextDocumentClientCapabilities::declaration() const
    -> std::optional<DeclarationClientCapabilities> {
  if (!repr_.contains("declaration")) return std::nullopt;

  const auto& value = repr_["declaration"];

  return DeclarationClientCapabilities(value);
}

auto TextDocumentClientCapabilities::definition() const
    -> std::optional<DefinitionClientCapabilities> {
  if (!repr_.contains("definition")) return std::nullopt;

  const auto& value = repr_["definition"];

  return DefinitionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::typeDefinition() const
    -> std::optional<TypeDefinitionClientCapabilities> {
  if (!repr_.contains("typeDefinition")) return std::nullopt;

  const auto& value = repr_["typeDefinition"];

  return TypeDefinitionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::implementation() const
    -> std::optional<ImplementationClientCapabilities> {
  if (!repr_.contains("implementation")) return std::nullopt;

  const auto& value = repr_["implementation"];

  return ImplementationClientCapabilities(value);
}

auto TextDocumentClientCapabilities::references() const
    -> std::optional<ReferenceClientCapabilities> {
  if (!repr_.contains("references")) return std::nullopt;

  const auto& value = repr_["references"];

  return ReferenceClientCapabilities(value);
}

auto TextDocumentClientCapabilities::documentHighlight() const
    -> std::optional<DocumentHighlightClientCapabilities> {
  if (!repr_.contains("documentHighlight")) return std::nullopt;

  const auto& value = repr_["documentHighlight"];

  return DocumentHighlightClientCapabilities(value);
}

auto TextDocumentClientCapabilities::documentSymbol() const
    -> std::optional<DocumentSymbolClientCapabilities> {
  if (!repr_.contains("documentSymbol")) return std::nullopt;

  const auto& value = repr_["documentSymbol"];

  return DocumentSymbolClientCapabilities(value);
}

auto TextDocumentClientCapabilities::codeAction() const
    -> std::optional<CodeActionClientCapabilities> {
  if (!repr_.contains("codeAction")) return std::nullopt;

  const auto& value = repr_["codeAction"];

  return CodeActionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::codeLens() const
    -> std::optional<CodeLensClientCapabilities> {
  if (!repr_.contains("codeLens")) return std::nullopt;

  const auto& value = repr_["codeLens"];

  return CodeLensClientCapabilities(value);
}

auto TextDocumentClientCapabilities::documentLink() const
    -> std::optional<DocumentLinkClientCapabilities> {
  if (!repr_.contains("documentLink")) return std::nullopt;

  const auto& value = repr_["documentLink"];

  return DocumentLinkClientCapabilities(value);
}

auto TextDocumentClientCapabilities::colorProvider() const
    -> std::optional<DocumentColorClientCapabilities> {
  if (!repr_.contains("colorProvider")) return std::nullopt;

  const auto& value = repr_["colorProvider"];

  return DocumentColorClientCapabilities(value);
}

auto TextDocumentClientCapabilities::formatting() const
    -> std::optional<DocumentFormattingClientCapabilities> {
  if (!repr_.contains("formatting")) return std::nullopt;

  const auto& value = repr_["formatting"];

  return DocumentFormattingClientCapabilities(value);
}

auto TextDocumentClientCapabilities::rangeFormatting() const
    -> std::optional<DocumentRangeFormattingClientCapabilities> {
  if (!repr_.contains("rangeFormatting")) return std::nullopt;

  const auto& value = repr_["rangeFormatting"];

  return DocumentRangeFormattingClientCapabilities(value);
}

auto TextDocumentClientCapabilities::onTypeFormatting() const
    -> std::optional<DocumentOnTypeFormattingClientCapabilities> {
  if (!repr_.contains("onTypeFormatting")) return std::nullopt;

  const auto& value = repr_["onTypeFormatting"];

  return DocumentOnTypeFormattingClientCapabilities(value);
}

auto TextDocumentClientCapabilities::rename() const
    -> std::optional<RenameClientCapabilities> {
  if (!repr_.contains("rename")) return std::nullopt;

  const auto& value = repr_["rename"];

  return RenameClientCapabilities(value);
}

auto TextDocumentClientCapabilities::foldingRange() const
    -> std::optional<FoldingRangeClientCapabilities> {
  if (!repr_.contains("foldingRange")) return std::nullopt;

  const auto& value = repr_["foldingRange"];

  return FoldingRangeClientCapabilities(value);
}

auto TextDocumentClientCapabilities::selectionRange() const
    -> std::optional<SelectionRangeClientCapabilities> {
  if (!repr_.contains("selectionRange")) return std::nullopt;

  const auto& value = repr_["selectionRange"];

  return SelectionRangeClientCapabilities(value);
}

auto TextDocumentClientCapabilities::publishDiagnostics() const
    -> std::optional<PublishDiagnosticsClientCapabilities> {
  if (!repr_.contains("publishDiagnostics")) return std::nullopt;

  const auto& value = repr_["publishDiagnostics"];

  return PublishDiagnosticsClientCapabilities(value);
}

auto TextDocumentClientCapabilities::callHierarchy() const
    -> std::optional<CallHierarchyClientCapabilities> {
  if (!repr_.contains("callHierarchy")) return std::nullopt;

  const auto& value = repr_["callHierarchy"];

  return CallHierarchyClientCapabilities(value);
}

auto TextDocumentClientCapabilities::semanticTokens() const
    -> std::optional<SemanticTokensClientCapabilities> {
  if (!repr_.contains("semanticTokens")) return std::nullopt;

  const auto& value = repr_["semanticTokens"];

  return SemanticTokensClientCapabilities(value);
}

auto TextDocumentClientCapabilities::linkedEditingRange() const
    -> std::optional<LinkedEditingRangeClientCapabilities> {
  if (!repr_.contains("linkedEditingRange")) return std::nullopt;

  const auto& value = repr_["linkedEditingRange"];

  return LinkedEditingRangeClientCapabilities(value);
}

auto TextDocumentClientCapabilities::moniker() const
    -> std::optional<MonikerClientCapabilities> {
  if (!repr_.contains("moniker")) return std::nullopt;

  const auto& value = repr_["moniker"];

  return MonikerClientCapabilities(value);
}

auto TextDocumentClientCapabilities::typeHierarchy() const
    -> std::optional<TypeHierarchyClientCapabilities> {
  if (!repr_.contains("typeHierarchy")) return std::nullopt;

  const auto& value = repr_["typeHierarchy"];

  return TypeHierarchyClientCapabilities(value);
}

auto TextDocumentClientCapabilities::inlineValue() const
    -> std::optional<InlineValueClientCapabilities> {
  if (!repr_.contains("inlineValue")) return std::nullopt;

  const auto& value = repr_["inlineValue"];

  return InlineValueClientCapabilities(value);
}

auto TextDocumentClientCapabilities::inlayHint() const
    -> std::optional<InlayHintClientCapabilities> {
  if (!repr_.contains("inlayHint")) return std::nullopt;

  const auto& value = repr_["inlayHint"];

  return InlayHintClientCapabilities(value);
}

auto TextDocumentClientCapabilities::diagnostic() const
    -> std::optional<DiagnosticClientCapabilities> {
  if (!repr_.contains("diagnostic")) return std::nullopt;

  const auto& value = repr_["diagnostic"];

  return DiagnosticClientCapabilities(value);
}

auto TextDocumentClientCapabilities::inlineCompletion() const
    -> std::optional<InlineCompletionClientCapabilities> {
  if (!repr_.contains("inlineCompletion")) return std::nullopt;

  const auto& value = repr_["inlineCompletion"];

  return InlineCompletionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::synchronization(
    std::optional<TextDocumentSyncClientCapabilities> synchronization)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::filters(
    std::optional<TextDocumentFilterClientCapabilities> filters)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::completion(
    std::optional<CompletionClientCapabilities> completion)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::hover(
    std::optional<HoverClientCapabilities> hover)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::signatureHelp(
    std::optional<SignatureHelpClientCapabilities> signatureHelp)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::declaration(
    std::optional<DeclarationClientCapabilities> declaration)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::definition(
    std::optional<DefinitionClientCapabilities> definition)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::typeDefinition(
    std::optional<TypeDefinitionClientCapabilities> typeDefinition)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::implementation(
    std::optional<ImplementationClientCapabilities> implementation)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::references(
    std::optional<ReferenceClientCapabilities> references)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::documentHighlight(
    std::optional<DocumentHighlightClientCapabilities> documentHighlight)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::documentSymbol(
    std::optional<DocumentSymbolClientCapabilities> documentSymbol)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::codeAction(
    std::optional<CodeActionClientCapabilities> codeAction)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::codeLens(
    std::optional<CodeLensClientCapabilities> codeLens)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::documentLink(
    std::optional<DocumentLinkClientCapabilities> documentLink)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::colorProvider(
    std::optional<DocumentColorClientCapabilities> colorProvider)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::formatting(
    std::optional<DocumentFormattingClientCapabilities> formatting)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::rangeFormatting(
    std::optional<DocumentRangeFormattingClientCapabilities> rangeFormatting)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::onTypeFormatting(
    std::optional<DocumentOnTypeFormattingClientCapabilities> onTypeFormatting)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::rename(
    std::optional<RenameClientCapabilities> rename)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::foldingRange(
    std::optional<FoldingRangeClientCapabilities> foldingRange)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::selectionRange(
    std::optional<SelectionRangeClientCapabilities> selectionRange)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::publishDiagnostics(
    std::optional<PublishDiagnosticsClientCapabilities> publishDiagnostics)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::callHierarchy(
    std::optional<CallHierarchyClientCapabilities> callHierarchy)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::semanticTokens(
    std::optional<SemanticTokensClientCapabilities> semanticTokens)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::linkedEditingRange(
    std::optional<LinkedEditingRangeClientCapabilities> linkedEditingRange)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::moniker(
    std::optional<MonikerClientCapabilities> moniker)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::typeHierarchy(
    std::optional<TypeHierarchyClientCapabilities> typeHierarchy)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::inlineValue(
    std::optional<InlineValueClientCapabilities> inlineValue)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::inlayHint(
    std::optional<InlayHintClientCapabilities> inlayHint)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::diagnostic(
    std::optional<DiagnosticClientCapabilities> diagnostic)
    -> TextDocumentClientCapabilities& {
  return *this;
}

auto TextDocumentClientCapabilities::inlineCompletion(
    std::optional<InlineCompletionClientCapabilities> inlineCompletion)
    -> TextDocumentClientCapabilities& {
  return *this;
}

NotebookDocumentClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("synchronization")) return false;
  return true;
}

auto NotebookDocumentClientCapabilities::synchronization() const
    -> NotebookDocumentSyncClientCapabilities {
  const auto& value = repr_["synchronization"];

  return NotebookDocumentSyncClientCapabilities(value);
}

auto NotebookDocumentClientCapabilities::synchronization(
    NotebookDocumentSyncClientCapabilities synchronization)
    -> NotebookDocumentClientCapabilities& {
  return *this;
}

WindowClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WindowClientCapabilities::workDoneProgress() const -> std::optional<bool> {
  if (!repr_.contains("workDoneProgress")) return std::nullopt;

  const auto& value = repr_["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WindowClientCapabilities::showMessage() const
    -> std::optional<ShowMessageRequestClientCapabilities> {
  if (!repr_.contains("showMessage")) return std::nullopt;

  const auto& value = repr_["showMessage"];

  return ShowMessageRequestClientCapabilities(value);
}

auto WindowClientCapabilities::showDocument() const
    -> std::optional<ShowDocumentClientCapabilities> {
  if (!repr_.contains("showDocument")) return std::nullopt;

  const auto& value = repr_["showDocument"];

  return ShowDocumentClientCapabilities(value);
}

auto WindowClientCapabilities::workDoneProgress(
    std::optional<bool> workDoneProgress) -> WindowClientCapabilities& {
  return *this;
}

auto WindowClientCapabilities::showMessage(
    std::optional<ShowMessageRequestClientCapabilities> showMessage)
    -> WindowClientCapabilities& {
  return *this;
}

auto WindowClientCapabilities::showDocument(
    std::optional<ShowDocumentClientCapabilities> showDocument)
    -> WindowClientCapabilities& {
  return *this;
}

GeneralClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto GeneralClientCapabilities::staleRequestSupport() const
    -> std::optional<StaleRequestSupportOptions> {
  if (!repr_.contains("staleRequestSupport")) return std::nullopt;

  const auto& value = repr_["staleRequestSupport"];

  return StaleRequestSupportOptions(value);
}

auto GeneralClientCapabilities::regularExpressions() const
    -> std::optional<RegularExpressionsClientCapabilities> {
  if (!repr_.contains("regularExpressions")) return std::nullopt;

  const auto& value = repr_["regularExpressions"];

  return RegularExpressionsClientCapabilities(value);
}

auto GeneralClientCapabilities::markdown() const
    -> std::optional<MarkdownClientCapabilities> {
  if (!repr_.contains("markdown")) return std::nullopt;

  const auto& value = repr_["markdown"];

  return MarkdownClientCapabilities(value);
}

auto GeneralClientCapabilities::positionEncodings() const
    -> std::optional<Vector<PositionEncodingKind>> {
  if (!repr_.contains("positionEncodings")) return std::nullopt;

  const auto& value = repr_["positionEncodings"];

  assert(value.is_array());
  return Vector<PositionEncodingKind>(value);
}

auto GeneralClientCapabilities::staleRequestSupport(
    std::optional<StaleRequestSupportOptions> staleRequestSupport)
    -> GeneralClientCapabilities& {
  return *this;
}

auto GeneralClientCapabilities::regularExpressions(
    std::optional<RegularExpressionsClientCapabilities> regularExpressions)
    -> GeneralClientCapabilities& {
  return *this;
}

auto GeneralClientCapabilities::markdown(
    std::optional<MarkdownClientCapabilities> markdown)
    -> GeneralClientCapabilities& {
  return *this;
}

auto GeneralClientCapabilities::positionEncodings(
    std::optional<Vector<PositionEncodingKind>> positionEncodings)
    -> GeneralClientCapabilities& {
  return *this;
}

WorkspaceFoldersServerCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceFoldersServerCapabilities::supported() const
    -> std::optional<bool> {
  if (!repr_.contains("supported")) return std::nullopt;

  const auto& value = repr_["supported"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceFoldersServerCapabilities::changeNotifications() const
    -> std::optional<std::variant<std::monostate, std::string, bool>> {
  if (!repr_.contains("changeNotifications")) return std::nullopt;

  const auto& value = repr_["changeNotifications"];

  std::variant<std::monostate, std::string, bool> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFoldersServerCapabilities::supported(
    std::optional<bool> supported) -> WorkspaceFoldersServerCapabilities& {
  return *this;
}

auto WorkspaceFoldersServerCapabilities::changeNotifications(
    std::optional<std::variant<std::monostate, std::string, bool>>
        changeNotifications) -> WorkspaceFoldersServerCapabilities& {
  return *this;
}

FileOperationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto FileOperationOptions::didCreate() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_.contains("didCreate")) return std::nullopt;

  const auto& value = repr_["didCreate"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::willCreate() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_.contains("willCreate")) return std::nullopt;

  const auto& value = repr_["willCreate"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::didRename() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_.contains("didRename")) return std::nullopt;

  const auto& value = repr_["didRename"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::willRename() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_.contains("willRename")) return std::nullopt;

  const auto& value = repr_["willRename"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::didDelete() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_.contains("didDelete")) return std::nullopt;

  const auto& value = repr_["didDelete"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::willDelete() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_.contains("willDelete")) return std::nullopt;

  const auto& value = repr_["willDelete"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::didCreate(
    std::optional<FileOperationRegistrationOptions> didCreate)
    -> FileOperationOptions& {
  return *this;
}

auto FileOperationOptions::willCreate(
    std::optional<FileOperationRegistrationOptions> willCreate)
    -> FileOperationOptions& {
  return *this;
}

auto FileOperationOptions::didRename(
    std::optional<FileOperationRegistrationOptions> didRename)
    -> FileOperationOptions& {
  return *this;
}

auto FileOperationOptions::willRename(
    std::optional<FileOperationRegistrationOptions> willRename)
    -> FileOperationOptions& {
  return *this;
}

auto FileOperationOptions::didDelete(
    std::optional<FileOperationRegistrationOptions> didDelete)
    -> FileOperationOptions& {
  return *this;
}

auto FileOperationOptions::willDelete(
    std::optional<FileOperationRegistrationOptions> willDelete)
    -> FileOperationOptions& {
  return *this;
}

RelativePattern::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("baseUri")) return false;
  if (!repr_.contains("pattern")) return false;
  return true;
}

auto RelativePattern::baseUri() const
    -> std::variant<std::monostate, WorkspaceFolder, std::string> {
  const auto& value = repr_["baseUri"];

  std::variant<std::monostate, WorkspaceFolder, std::string> result;

  details::try_emplace(result, value);

  return result;
}

auto RelativePattern::pattern() const -> Pattern {
  const auto& value = repr_["pattern"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelativePattern::baseUri(
    std::variant<std::monostate, WorkspaceFolder, std::string> baseUri)
    -> RelativePattern& {
  return *this;
}

auto RelativePattern::pattern(Pattern pattern) -> RelativePattern& {
  return *this;
}

TextDocumentFilterLanguage::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("language")) return false;
  return true;
}

auto TextDocumentFilterLanguage::language() const -> std::string {
  const auto& value = repr_["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterLanguage::scheme() const -> std::optional<std::string> {
  if (!repr_.contains("scheme")) return std::nullopt;

  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterLanguage::pattern() const -> std::optional<GlobPattern> {
  if (!repr_.contains("pattern")) return std::nullopt;

  const auto& value = repr_["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentFilterLanguage::language(std::string language)
    -> TextDocumentFilterLanguage& {
  return *this;
}

auto TextDocumentFilterLanguage::scheme(std::optional<std::string> scheme)
    -> TextDocumentFilterLanguage& {
  return *this;
}

auto TextDocumentFilterLanguage::pattern(std::optional<GlobPattern> pattern)
    -> TextDocumentFilterLanguage& {
  return *this;
}

TextDocumentFilterScheme::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("scheme")) return false;
  return true;
}

auto TextDocumentFilterScheme::language() const -> std::optional<std::string> {
  if (!repr_.contains("language")) return std::nullopt;

  const auto& value = repr_["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterScheme::scheme() const -> std::string {
  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterScheme::pattern() const -> std::optional<GlobPattern> {
  if (!repr_.contains("pattern")) return std::nullopt;

  const auto& value = repr_["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentFilterScheme::language(std::optional<std::string> language)
    -> TextDocumentFilterScheme& {
  return *this;
}

auto TextDocumentFilterScheme::scheme(std::string scheme)
    -> TextDocumentFilterScheme& {
  return *this;
}

auto TextDocumentFilterScheme::pattern(std::optional<GlobPattern> pattern)
    -> TextDocumentFilterScheme& {
  return *this;
}

TextDocumentFilterPattern::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("pattern")) return false;
  return true;
}

auto TextDocumentFilterPattern::language() const -> std::optional<std::string> {
  if (!repr_.contains("language")) return std::nullopt;

  const auto& value = repr_["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterPattern::scheme() const -> std::optional<std::string> {
  if (!repr_.contains("scheme")) return std::nullopt;

  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterPattern::pattern() const -> GlobPattern {
  const auto& value = repr_["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentFilterPattern::language(std::optional<std::string> language)
    -> TextDocumentFilterPattern& {
  return *this;
}

auto TextDocumentFilterPattern::scheme(std::optional<std::string> scheme)
    -> TextDocumentFilterPattern& {
  return *this;
}

auto TextDocumentFilterPattern::pattern(GlobPattern pattern)
    -> TextDocumentFilterPattern& {
  return *this;
}

NotebookDocumentFilterNotebookType::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("notebookType")) return false;
  return true;
}

auto NotebookDocumentFilterNotebookType::notebookType() const -> std::string {
  const auto& value = repr_["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterNotebookType::scheme() const
    -> std::optional<std::string> {
  if (!repr_.contains("scheme")) return std::nullopt;

  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterNotebookType::pattern() const
    -> std::optional<GlobPattern> {
  if (!repr_.contains("pattern")) return std::nullopt;

  const auto& value = repr_["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterNotebookType::notebookType(std::string notebookType)
    -> NotebookDocumentFilterNotebookType& {
  return *this;
}

auto NotebookDocumentFilterNotebookType::scheme(
    std::optional<std::string> scheme) -> NotebookDocumentFilterNotebookType& {
  return *this;
}

auto NotebookDocumentFilterNotebookType::pattern(
    std::optional<GlobPattern> pattern) -> NotebookDocumentFilterNotebookType& {
  return *this;
}

NotebookDocumentFilterScheme::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("scheme")) return false;
  return true;
}

auto NotebookDocumentFilterScheme::notebookType() const
    -> std::optional<std::string> {
  if (!repr_.contains("notebookType")) return std::nullopt;

  const auto& value = repr_["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterScheme::scheme() const -> std::string {
  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterScheme::pattern() const
    -> std::optional<GlobPattern> {
  if (!repr_.contains("pattern")) return std::nullopt;

  const auto& value = repr_["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterScheme::notebookType(
    std::optional<std::string> notebookType) -> NotebookDocumentFilterScheme& {
  return *this;
}

auto NotebookDocumentFilterScheme::scheme(std::string scheme)
    -> NotebookDocumentFilterScheme& {
  return *this;
}

auto NotebookDocumentFilterScheme::pattern(std::optional<GlobPattern> pattern)
    -> NotebookDocumentFilterScheme& {
  return *this;
}

NotebookDocumentFilterPattern::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("pattern")) return false;
  return true;
}

auto NotebookDocumentFilterPattern::notebookType() const
    -> std::optional<std::string> {
  if (!repr_.contains("notebookType")) return std::nullopt;

  const auto& value = repr_["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterPattern::scheme() const
    -> std::optional<std::string> {
  if (!repr_.contains("scheme")) return std::nullopt;

  const auto& value = repr_["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterPattern::pattern() const -> GlobPattern {
  const auto& value = repr_["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterPattern::notebookType(
    std::optional<std::string> notebookType) -> NotebookDocumentFilterPattern& {
  return *this;
}

auto NotebookDocumentFilterPattern::scheme(std::optional<std::string> scheme)
    -> NotebookDocumentFilterPattern& {
  return *this;
}

auto NotebookDocumentFilterPattern::pattern(GlobPattern pattern)
    -> NotebookDocumentFilterPattern& {
  return *this;
}

NotebookCellArrayChange::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("start")) return false;
  if (!repr_.contains("deleteCount")) return false;
  return true;
}

auto NotebookCellArrayChange::start() const -> long {
  const auto& value = repr_["start"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto NotebookCellArrayChange::deleteCount() const -> long {
  const auto& value = repr_["deleteCount"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto NotebookCellArrayChange::cells() const
    -> std::optional<Vector<NotebookCell>> {
  if (!repr_.contains("cells")) return std::nullopt;

  const auto& value = repr_["cells"];

  assert(value.is_array());
  return Vector<NotebookCell>(value);
}

auto NotebookCellArrayChange::start(long start) -> NotebookCellArrayChange& {
  return *this;
}

auto NotebookCellArrayChange::deleteCount(long deleteCount)
    -> NotebookCellArrayChange& {
  return *this;
}

auto NotebookCellArrayChange::cells(std::optional<Vector<NotebookCell>> cells)
    -> NotebookCellArrayChange& {
  return *this;
}

WorkspaceEditClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceEditClientCapabilities::documentChanges() const
    -> std::optional<bool> {
  if (!repr_.contains("documentChanges")) return std::nullopt;

  const auto& value = repr_["documentChanges"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::resourceOperations() const
    -> std::optional<Vector<ResourceOperationKind>> {
  if (!repr_.contains("resourceOperations")) return std::nullopt;

  const auto& value = repr_["resourceOperations"];

  assert(value.is_array());
  return Vector<ResourceOperationKind>(value);
}

auto WorkspaceEditClientCapabilities::failureHandling() const
    -> std::optional<FailureHandlingKind> {
  if (!repr_.contains("failureHandling")) return std::nullopt;

  const auto& value = repr_["failureHandling"];

  lsp_runtime_error(
      "WorkspaceEditClientCapabilities::failureHandling: not implement yet");
}

auto WorkspaceEditClientCapabilities::normalizesLineEndings() const
    -> std::optional<bool> {
  if (!repr_.contains("normalizesLineEndings")) return std::nullopt;

  const auto& value = repr_["normalizesLineEndings"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::changeAnnotationSupport() const
    -> std::optional<ChangeAnnotationsSupportOptions> {
  if (!repr_.contains("changeAnnotationSupport")) return std::nullopt;

  const auto& value = repr_["changeAnnotationSupport"];

  return ChangeAnnotationsSupportOptions(value);
}

auto WorkspaceEditClientCapabilities::metadataSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("metadataSupport")) return std::nullopt;

  const auto& value = repr_["metadataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::snippetEditSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("snippetEditSupport")) return std::nullopt;

  const auto& value = repr_["snippetEditSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::documentChanges(
    std::optional<bool> documentChanges) -> WorkspaceEditClientCapabilities& {
  return *this;
}

auto WorkspaceEditClientCapabilities::resourceOperations(
    std::optional<Vector<ResourceOperationKind>> resourceOperations)
    -> WorkspaceEditClientCapabilities& {
  return *this;
}

auto WorkspaceEditClientCapabilities::failureHandling(
    std::optional<FailureHandlingKind> failureHandling)
    -> WorkspaceEditClientCapabilities& {
  return *this;
}

auto WorkspaceEditClientCapabilities::normalizesLineEndings(
    std::optional<bool> normalizesLineEndings)
    -> WorkspaceEditClientCapabilities& {
  return *this;
}

auto WorkspaceEditClientCapabilities::changeAnnotationSupport(
    std::optional<ChangeAnnotationsSupportOptions> changeAnnotationSupport)
    -> WorkspaceEditClientCapabilities& {
  return *this;
}

auto WorkspaceEditClientCapabilities::metadataSupport(
    std::optional<bool> metadataSupport) -> WorkspaceEditClientCapabilities& {
  return *this;
}

auto WorkspaceEditClientCapabilities::snippetEditSupport(
    std::optional<bool> snippetEditSupport)
    -> WorkspaceEditClientCapabilities& {
  return *this;
}

DidChangeConfigurationClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DidChangeConfigurationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DidChangeConfigurationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DidChangeConfigurationClientCapabilities& {
  return *this;
}

DidChangeWatchedFilesClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DidChangeWatchedFilesClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DidChangeWatchedFilesClientCapabilities::relativePatternSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("relativePatternSupport")) return std::nullopt;

  const auto& value = repr_["relativePatternSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DidChangeWatchedFilesClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DidChangeWatchedFilesClientCapabilities& {
  return *this;
}

auto DidChangeWatchedFilesClientCapabilities::relativePatternSupport(
    std::optional<bool> relativePatternSupport)
    -> DidChangeWatchedFilesClientCapabilities& {
  return *this;
}

WorkspaceSymbolClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto WorkspaceSymbolClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolClientCapabilities::symbolKind() const
    -> std::optional<ClientSymbolKindOptions> {
  if (!repr_.contains("symbolKind")) return std::nullopt;

  const auto& value = repr_["symbolKind"];

  return ClientSymbolKindOptions(value);
}

auto WorkspaceSymbolClientCapabilities::tagSupport() const
    -> std::optional<ClientSymbolTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return ClientSymbolTagOptions(value);
}

auto WorkspaceSymbolClientCapabilities::resolveSupport() const
    -> std::optional<ClientSymbolResolveOptions> {
  if (!repr_.contains("resolveSupport")) return std::nullopt;

  const auto& value = repr_["resolveSupport"];

  return ClientSymbolResolveOptions(value);
}

auto WorkspaceSymbolClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> WorkspaceSymbolClientCapabilities& {
  return *this;
}

auto WorkspaceSymbolClientCapabilities::symbolKind(
    std::optional<ClientSymbolKindOptions> symbolKind)
    -> WorkspaceSymbolClientCapabilities& {
  return *this;
}

auto WorkspaceSymbolClientCapabilities::tagSupport(
    std::optional<ClientSymbolTagOptions> tagSupport)
    -> WorkspaceSymbolClientCapabilities& {
  return *this;
}

auto WorkspaceSymbolClientCapabilities::resolveSupport(
    std::optional<ClientSymbolResolveOptions> resolveSupport)
    -> WorkspaceSymbolClientCapabilities& {
  return *this;
}

ExecuteCommandClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ExecuteCommandClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecuteCommandClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> ExecuteCommandClientCapabilities& {
  return *this;
}

SemanticTokensWorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SemanticTokensWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("refreshSupport")) return std::nullopt;

  const auto& value = repr_["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> SemanticTokensWorkspaceClientCapabilities& {
  return *this;
}

CodeLensWorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CodeLensWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("refreshSupport")) return std::nullopt;

  const auto& value = repr_["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> CodeLensWorkspaceClientCapabilities& {
  return *this;
}

FileOperationClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto FileOperationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::didCreate() const -> std::optional<bool> {
  if (!repr_.contains("didCreate")) return std::nullopt;

  const auto& value = repr_["didCreate"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::willCreate() const
    -> std::optional<bool> {
  if (!repr_.contains("willCreate")) return std::nullopt;

  const auto& value = repr_["willCreate"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::didRename() const -> std::optional<bool> {
  if (!repr_.contains("didRename")) return std::nullopt;

  const auto& value = repr_["didRename"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::willRename() const
    -> std::optional<bool> {
  if (!repr_.contains("willRename")) return std::nullopt;

  const auto& value = repr_["willRename"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::didDelete() const -> std::optional<bool> {
  if (!repr_.contains("didDelete")) return std::nullopt;

  const auto& value = repr_["didDelete"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::willDelete() const
    -> std::optional<bool> {
  if (!repr_.contains("willDelete")) return std::nullopt;

  const auto& value = repr_["willDelete"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> FileOperationClientCapabilities& {
  return *this;
}

auto FileOperationClientCapabilities::didCreate(std::optional<bool> didCreate)
    -> FileOperationClientCapabilities& {
  return *this;
}

auto FileOperationClientCapabilities::willCreate(std::optional<bool> willCreate)
    -> FileOperationClientCapabilities& {
  return *this;
}

auto FileOperationClientCapabilities::didRename(std::optional<bool> didRename)
    -> FileOperationClientCapabilities& {
  return *this;
}

auto FileOperationClientCapabilities::willRename(std::optional<bool> willRename)
    -> FileOperationClientCapabilities& {
  return *this;
}

auto FileOperationClientCapabilities::didDelete(std::optional<bool> didDelete)
    -> FileOperationClientCapabilities& {
  return *this;
}

auto FileOperationClientCapabilities::willDelete(std::optional<bool> willDelete)
    -> FileOperationClientCapabilities& {
  return *this;
}

InlineValueWorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlineValueWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("refreshSupport")) return std::nullopt;

  const auto& value = repr_["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> InlineValueWorkspaceClientCapabilities& {
  return *this;
}

InlayHintWorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlayHintWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("refreshSupport")) return std::nullopt;

  const auto& value = repr_["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> InlayHintWorkspaceClientCapabilities& {
  return *this;
}

DiagnosticWorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DiagnosticWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("refreshSupport")) return std::nullopt;

  const auto& value = repr_["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> DiagnosticWorkspaceClientCapabilities& {
  return *this;
}

FoldingRangeWorkspaceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto FoldingRangeWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("refreshSupport")) return std::nullopt;

  const auto& value = repr_["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> FoldingRangeWorkspaceClientCapabilities& {
  return *this;
}

TextDocumentContentClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TextDocumentContentClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentContentClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TextDocumentContentClientCapabilities& {
  return *this;
}

TextDocumentSyncClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TextDocumentSyncClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::willSave() const
    -> std::optional<bool> {
  if (!repr_.contains("willSave")) return std::nullopt;

  const auto& value = repr_["willSave"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::willSaveWaitUntil() const
    -> std::optional<bool> {
  if (!repr_.contains("willSaveWaitUntil")) return std::nullopt;

  const auto& value = repr_["willSaveWaitUntil"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::didSave() const
    -> std::optional<bool> {
  if (!repr_.contains("didSave")) return std::nullopt;

  const auto& value = repr_["didSave"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TextDocumentSyncClientCapabilities& {
  return *this;
}

auto TextDocumentSyncClientCapabilities::willSave(std::optional<bool> willSave)
    -> TextDocumentSyncClientCapabilities& {
  return *this;
}

auto TextDocumentSyncClientCapabilities::willSaveWaitUntil(
    std::optional<bool> willSaveWaitUntil)
    -> TextDocumentSyncClientCapabilities& {
  return *this;
}

auto TextDocumentSyncClientCapabilities::didSave(std::optional<bool> didSave)
    -> TextDocumentSyncClientCapabilities& {
  return *this;
}

TextDocumentFilterClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TextDocumentFilterClientCapabilities::relativePatternSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("relativePatternSupport")) return std::nullopt;

  const auto& value = repr_["relativePatternSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentFilterClientCapabilities::relativePatternSupport(
    std::optional<bool> relativePatternSupport)
    -> TextDocumentFilterClientCapabilities& {
  return *this;
}

CompletionClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CompletionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionClientCapabilities::completionItem() const
    -> std::optional<ClientCompletionItemOptions> {
  if (!repr_.contains("completionItem")) return std::nullopt;

  const auto& value = repr_["completionItem"];

  return ClientCompletionItemOptions(value);
}

auto CompletionClientCapabilities::completionItemKind() const
    -> std::optional<ClientCompletionItemOptionsKind> {
  if (!repr_.contains("completionItemKind")) return std::nullopt;

  const auto& value = repr_["completionItemKind"];

  return ClientCompletionItemOptionsKind(value);
}

auto CompletionClientCapabilities::insertTextMode() const
    -> std::optional<InsertTextMode> {
  if (!repr_.contains("insertTextMode")) return std::nullopt;

  const auto& value = repr_["insertTextMode"];

  return InsertTextMode(value);
}

auto CompletionClientCapabilities::contextSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("contextSupport")) return std::nullopt;

  const auto& value = repr_["contextSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionClientCapabilities::completionList() const
    -> std::optional<CompletionListCapabilities> {
  if (!repr_.contains("completionList")) return std::nullopt;

  const auto& value = repr_["completionList"];

  return CompletionListCapabilities(value);
}

auto CompletionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> CompletionClientCapabilities& {
  return *this;
}

auto CompletionClientCapabilities::completionItem(
    std::optional<ClientCompletionItemOptions> completionItem)
    -> CompletionClientCapabilities& {
  return *this;
}

auto CompletionClientCapabilities::completionItemKind(
    std::optional<ClientCompletionItemOptionsKind> completionItemKind)
    -> CompletionClientCapabilities& {
  return *this;
}

auto CompletionClientCapabilities::insertTextMode(
    std::optional<InsertTextMode> insertTextMode)
    -> CompletionClientCapabilities& {
  return *this;
}

auto CompletionClientCapabilities::contextSupport(
    std::optional<bool> contextSupport) -> CompletionClientCapabilities& {
  return *this;
}

auto CompletionClientCapabilities::completionList(
    std::optional<CompletionListCapabilities> completionList)
    -> CompletionClientCapabilities& {
  return *this;
}

HoverClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto HoverClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto HoverClientCapabilities::contentFormat() const
    -> std::optional<Vector<MarkupKind>> {
  if (!repr_.contains("contentFormat")) return std::nullopt;

  const auto& value = repr_["contentFormat"];

  assert(value.is_array());
  return Vector<MarkupKind>(value);
}

auto HoverClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> HoverClientCapabilities& {
  return *this;
}

auto HoverClientCapabilities::contentFormat(
    std::optional<Vector<MarkupKind>> contentFormat)
    -> HoverClientCapabilities& {
  return *this;
}

SignatureHelpClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SignatureHelpClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpClientCapabilities::signatureInformation() const
    -> std::optional<ClientSignatureInformationOptions> {
  if (!repr_.contains("signatureInformation")) return std::nullopt;

  const auto& value = repr_["signatureInformation"];

  return ClientSignatureInformationOptions(value);
}

auto SignatureHelpClientCapabilities::contextSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("contextSupport")) return std::nullopt;

  const auto& value = repr_["contextSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> SignatureHelpClientCapabilities& {
  return *this;
}

auto SignatureHelpClientCapabilities::signatureInformation(
    std::optional<ClientSignatureInformationOptions> signatureInformation)
    -> SignatureHelpClientCapabilities& {
  return *this;
}

auto SignatureHelpClientCapabilities::contextSupport(
    std::optional<bool> contextSupport) -> SignatureHelpClientCapabilities& {
  return *this;
}

DeclarationClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DeclarationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationClientCapabilities::linkSupport() const -> std::optional<bool> {
  if (!repr_.contains("linkSupport")) return std::nullopt;

  const auto& value = repr_["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> DeclarationClientCapabilities& {
  return *this;
}

auto DeclarationClientCapabilities::linkSupport(std::optional<bool> linkSupport)
    -> DeclarationClientCapabilities& {
  return *this;
}

DefinitionClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DefinitionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionClientCapabilities::linkSupport() const -> std::optional<bool> {
  if (!repr_.contains("linkSupport")) return std::nullopt;

  const auto& value = repr_["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> DefinitionClientCapabilities& {
  return *this;
}

auto DefinitionClientCapabilities::linkSupport(std::optional<bool> linkSupport)
    -> DefinitionClientCapabilities& {
  return *this;
}

TypeDefinitionClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TypeDefinitionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionClientCapabilities::linkSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("linkSupport")) return std::nullopt;

  const auto& value = repr_["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TypeDefinitionClientCapabilities& {
  return *this;
}

auto TypeDefinitionClientCapabilities::linkSupport(
    std::optional<bool> linkSupport) -> TypeDefinitionClientCapabilities& {
  return *this;
}

ImplementationClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ImplementationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationClientCapabilities::linkSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("linkSupport")) return std::nullopt;

  const auto& value = repr_["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> ImplementationClientCapabilities& {
  return *this;
}

auto ImplementationClientCapabilities::linkSupport(
    std::optional<bool> linkSupport) -> ImplementationClientCapabilities& {
  return *this;
}

ReferenceClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ReferenceClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> ReferenceClientCapabilities& {
  return *this;
}

DocumentHighlightClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentHighlightClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentHighlightClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentHighlightClientCapabilities& {
  return *this;
}

DocumentSymbolClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentSymbolClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolClientCapabilities::symbolKind() const
    -> std::optional<ClientSymbolKindOptions> {
  if (!repr_.contains("symbolKind")) return std::nullopt;

  const auto& value = repr_["symbolKind"];

  return ClientSymbolKindOptions(value);
}

auto DocumentSymbolClientCapabilities::hierarchicalDocumentSymbolSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("hierarchicalDocumentSymbolSupport")) return std::nullopt;

  const auto& value = repr_["hierarchicalDocumentSymbolSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolClientCapabilities::tagSupport() const
    -> std::optional<ClientSymbolTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return ClientSymbolTagOptions(value);
}

auto DocumentSymbolClientCapabilities::labelSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("labelSupport")) return std::nullopt;

  const auto& value = repr_["labelSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentSymbolClientCapabilities& {
  return *this;
}

auto DocumentSymbolClientCapabilities::symbolKind(
    std::optional<ClientSymbolKindOptions> symbolKind)
    -> DocumentSymbolClientCapabilities& {
  return *this;
}

auto DocumentSymbolClientCapabilities::hierarchicalDocumentSymbolSupport(
    std::optional<bool> hierarchicalDocumentSymbolSupport)
    -> DocumentSymbolClientCapabilities& {
  return *this;
}

auto DocumentSymbolClientCapabilities::tagSupport(
    std::optional<ClientSymbolTagOptions> tagSupport)
    -> DocumentSymbolClientCapabilities& {
  return *this;
}

auto DocumentSymbolClientCapabilities::labelSupport(
    std::optional<bool> labelSupport) -> DocumentSymbolClientCapabilities& {
  return *this;
}

CodeActionClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CodeActionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::codeActionLiteralSupport() const
    -> std::optional<ClientCodeActionLiteralOptions> {
  if (!repr_.contains("codeActionLiteralSupport")) return std::nullopt;

  const auto& value = repr_["codeActionLiteralSupport"];

  return ClientCodeActionLiteralOptions(value);
}

auto CodeActionClientCapabilities::isPreferredSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("isPreferredSupport")) return std::nullopt;

  const auto& value = repr_["isPreferredSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::disabledSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("disabledSupport")) return std::nullopt;

  const auto& value = repr_["disabledSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::dataSupport() const -> std::optional<bool> {
  if (!repr_.contains("dataSupport")) return std::nullopt;

  const auto& value = repr_["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::resolveSupport() const
    -> std::optional<ClientCodeActionResolveOptions> {
  if (!repr_.contains("resolveSupport")) return std::nullopt;

  const auto& value = repr_["resolveSupport"];

  return ClientCodeActionResolveOptions(value);
}

auto CodeActionClientCapabilities::honorsChangeAnnotations() const
    -> std::optional<bool> {
  if (!repr_.contains("honorsChangeAnnotations")) return std::nullopt;

  const auto& value = repr_["honorsChangeAnnotations"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::documentationSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("documentationSupport")) return std::nullopt;

  const auto& value = repr_["documentationSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::tagSupport() const
    -> std::optional<CodeActionTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return CodeActionTagOptions(value);
}

auto CodeActionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::codeActionLiteralSupport(
    std::optional<ClientCodeActionLiteralOptions> codeActionLiteralSupport)
    -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::isPreferredSupport(
    std::optional<bool> isPreferredSupport) -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::disabledSupport(
    std::optional<bool> disabledSupport) -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::dataSupport(std::optional<bool> dataSupport)
    -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::resolveSupport(
    std::optional<ClientCodeActionResolveOptions> resolveSupport)
    -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::honorsChangeAnnotations(
    std::optional<bool> honorsChangeAnnotations)
    -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::documentationSupport(
    std::optional<bool> documentationSupport) -> CodeActionClientCapabilities& {
  return *this;
}

auto CodeActionClientCapabilities::tagSupport(
    std::optional<CodeActionTagOptions> tagSupport)
    -> CodeActionClientCapabilities& {
  return *this;
}

CodeLensClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CodeLensClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensClientCapabilities::resolveSupport() const
    -> std::optional<ClientCodeLensResolveOptions> {
  if (!repr_.contains("resolveSupport")) return std::nullopt;

  const auto& value = repr_["resolveSupport"];

  return ClientCodeLensResolveOptions(value);
}

auto CodeLensClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> CodeLensClientCapabilities& {
  return *this;
}

auto CodeLensClientCapabilities::resolveSupport(
    std::optional<ClientCodeLensResolveOptions> resolveSupport)
    -> CodeLensClientCapabilities& {
  return *this;
}

DocumentLinkClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentLinkClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkClientCapabilities::tooltipSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("tooltipSupport")) return std::nullopt;

  const auto& value = repr_["tooltipSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentLinkClientCapabilities& {
  return *this;
}

auto DocumentLinkClientCapabilities::tooltipSupport(
    std::optional<bool> tooltipSupport) -> DocumentLinkClientCapabilities& {
  return *this;
}

DocumentColorClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentColorClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentColorClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentColorClientCapabilities& {
  return *this;
}

DocumentFormattingClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentFormattingClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentFormattingClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentFormattingClientCapabilities& {
  return *this;
}

DocumentRangeFormattingClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentRangeFormattingClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingClientCapabilities::rangesSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("rangesSupport")) return std::nullopt;

  const auto& value = repr_["rangesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentRangeFormattingClientCapabilities& {
  return *this;
}

auto DocumentRangeFormattingClientCapabilities::rangesSupport(
    std::optional<bool> rangesSupport)
    -> DocumentRangeFormattingClientCapabilities& {
  return *this;
}

DocumentOnTypeFormattingClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DocumentOnTypeFormattingClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentOnTypeFormattingClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentOnTypeFormattingClientCapabilities& {
  return *this;
}

RenameClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto RenameClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameClientCapabilities::prepareSupport() const -> std::optional<bool> {
  if (!repr_.contains("prepareSupport")) return std::nullopt;

  const auto& value = repr_["prepareSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameClientCapabilities::prepareSupportDefaultBehavior() const
    -> std::optional<PrepareSupportDefaultBehavior> {
  if (!repr_.contains("prepareSupportDefaultBehavior")) return std::nullopt;

  const auto& value = repr_["prepareSupportDefaultBehavior"];

  return PrepareSupportDefaultBehavior(value);
}

auto RenameClientCapabilities::honorsChangeAnnotations() const
    -> std::optional<bool> {
  if (!repr_.contains("honorsChangeAnnotations")) return std::nullopt;

  const auto& value = repr_["honorsChangeAnnotations"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> RenameClientCapabilities& {
  return *this;
}

auto RenameClientCapabilities::prepareSupport(
    std::optional<bool> prepareSupport) -> RenameClientCapabilities& {
  return *this;
}

auto RenameClientCapabilities::prepareSupportDefaultBehavior(
    std::optional<PrepareSupportDefaultBehavior> prepareSupportDefaultBehavior)
    -> RenameClientCapabilities& {
  return *this;
}

auto RenameClientCapabilities::honorsChangeAnnotations(
    std::optional<bool> honorsChangeAnnotations) -> RenameClientCapabilities& {
  return *this;
}

FoldingRangeClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto FoldingRangeClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeClientCapabilities::rangeLimit() const -> std::optional<long> {
  if (!repr_.contains("rangeLimit")) return std::nullopt;

  const auto& value = repr_["rangeLimit"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRangeClientCapabilities::lineFoldingOnly() const
    -> std::optional<bool> {
  if (!repr_.contains("lineFoldingOnly")) return std::nullopt;

  const auto& value = repr_["lineFoldingOnly"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeClientCapabilities::foldingRangeKind() const
    -> std::optional<ClientFoldingRangeKindOptions> {
  if (!repr_.contains("foldingRangeKind")) return std::nullopt;

  const auto& value = repr_["foldingRangeKind"];

  return ClientFoldingRangeKindOptions(value);
}

auto FoldingRangeClientCapabilities::foldingRange() const
    -> std::optional<ClientFoldingRangeOptions> {
  if (!repr_.contains("foldingRange")) return std::nullopt;

  const auto& value = repr_["foldingRange"];

  return ClientFoldingRangeOptions(value);
}

auto FoldingRangeClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> FoldingRangeClientCapabilities& {
  return *this;
}

auto FoldingRangeClientCapabilities::rangeLimit(std::optional<long> rangeLimit)
    -> FoldingRangeClientCapabilities& {
  return *this;
}

auto FoldingRangeClientCapabilities::lineFoldingOnly(
    std::optional<bool> lineFoldingOnly) -> FoldingRangeClientCapabilities& {
  return *this;
}

auto FoldingRangeClientCapabilities::foldingRangeKind(
    std::optional<ClientFoldingRangeKindOptions> foldingRangeKind)
    -> FoldingRangeClientCapabilities& {
  return *this;
}

auto FoldingRangeClientCapabilities::foldingRange(
    std::optional<ClientFoldingRangeOptions> foldingRange)
    -> FoldingRangeClientCapabilities& {
  return *this;
}

SelectionRangeClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto SelectionRangeClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SelectionRangeClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> SelectionRangeClientCapabilities& {
  return *this;
}

PublishDiagnosticsClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto PublishDiagnosticsClientCapabilities::versionSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("versionSupport")) return std::nullopt;

  const auto& value = repr_["versionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::relatedInformation() const
    -> std::optional<bool> {
  if (!repr_.contains("relatedInformation")) return std::nullopt;

  const auto& value = repr_["relatedInformation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::tagSupport() const
    -> std::optional<ClientDiagnosticsTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return ClientDiagnosticsTagOptions(value);
}

auto PublishDiagnosticsClientCapabilities::codeDescriptionSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("codeDescriptionSupport")) return std::nullopt;

  const auto& value = repr_["codeDescriptionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::dataSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("dataSupport")) return std::nullopt;

  const auto& value = repr_["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::versionSupport(
    std::optional<bool> versionSupport)
    -> PublishDiagnosticsClientCapabilities& {
  return *this;
}

auto PublishDiagnosticsClientCapabilities::relatedInformation(
    std::optional<bool> relatedInformation)
    -> PublishDiagnosticsClientCapabilities& {
  return *this;
}

auto PublishDiagnosticsClientCapabilities::tagSupport(
    std::optional<ClientDiagnosticsTagOptions> tagSupport)
    -> PublishDiagnosticsClientCapabilities& {
  return *this;
}

auto PublishDiagnosticsClientCapabilities::codeDescriptionSupport(
    std::optional<bool> codeDescriptionSupport)
    -> PublishDiagnosticsClientCapabilities& {
  return *this;
}

auto PublishDiagnosticsClientCapabilities::dataSupport(
    std::optional<bool> dataSupport) -> PublishDiagnosticsClientCapabilities& {
  return *this;
}

CallHierarchyClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CallHierarchyClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CallHierarchyClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> CallHierarchyClientCapabilities& {
  return *this;
}

SemanticTokensClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("requests")) return false;
  if (!repr_.contains("tokenTypes")) return false;
  if (!repr_.contains("tokenModifiers")) return false;
  if (!repr_.contains("formats")) return false;
  return true;
}

auto SemanticTokensClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::requests() const
    -> ClientSemanticTokensRequestOptions {
  const auto& value = repr_["requests"];

  return ClientSemanticTokensRequestOptions(value);
}

auto SemanticTokensClientCapabilities::tokenTypes() const
    -> Vector<std::string> {
  const auto& value = repr_["tokenTypes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensClientCapabilities::tokenModifiers() const
    -> Vector<std::string> {
  const auto& value = repr_["tokenModifiers"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensClientCapabilities::formats() const -> Vector<TokenFormat> {
  const auto& value = repr_["formats"];

  assert(value.is_array());
  return Vector<TokenFormat>(value);
}

auto SemanticTokensClientCapabilities::overlappingTokenSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("overlappingTokenSupport")) return std::nullopt;

  const auto& value = repr_["overlappingTokenSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::multilineTokenSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("multilineTokenSupport")) return std::nullopt;

  const auto& value = repr_["multilineTokenSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::serverCancelSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("serverCancelSupport")) return std::nullopt;

  const auto& value = repr_["serverCancelSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::augmentsSyntaxTokens() const
    -> std::optional<bool> {
  if (!repr_.contains("augmentsSyntaxTokens")) return std::nullopt;

  const auto& value = repr_["augmentsSyntaxTokens"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::requests(
    ClientSemanticTokensRequestOptions requests)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::tokenTypes(
    Vector<std::string> tokenTypes) -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::tokenModifiers(
    Vector<std::string> tokenModifiers) -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::formats(Vector<TokenFormat> formats)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::overlappingTokenSupport(
    std::optional<bool> overlappingTokenSupport)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::multilineTokenSupport(
    std::optional<bool> multilineTokenSupport)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::serverCancelSupport(
    std::optional<bool> serverCancelSupport)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

auto SemanticTokensClientCapabilities::augmentsSyntaxTokens(
    std::optional<bool> augmentsSyntaxTokens)
    -> SemanticTokensClientCapabilities& {
  return *this;
}

LinkedEditingRangeClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto LinkedEditingRangeClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto LinkedEditingRangeClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> LinkedEditingRangeClientCapabilities& {
  return *this;
}

MonikerClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto MonikerClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto MonikerClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> MonikerClientCapabilities& {
  return *this;
}

TypeHierarchyClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto TypeHierarchyClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeHierarchyClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TypeHierarchyClientCapabilities& {
  return *this;
}

InlineValueClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlineValueClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> InlineValueClientCapabilities& {
  return *this;
}

InlayHintClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlayHintClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintClientCapabilities::resolveSupport() const
    -> std::optional<ClientInlayHintResolveOptions> {
  if (!repr_.contains("resolveSupport")) return std::nullopt;

  const auto& value = repr_["resolveSupport"];

  return ClientInlayHintResolveOptions(value);
}

auto InlayHintClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> InlayHintClientCapabilities& {
  return *this;
}

auto InlayHintClientCapabilities::resolveSupport(
    std::optional<ClientInlayHintResolveOptions> resolveSupport)
    -> InlayHintClientCapabilities& {
  return *this;
}

DiagnosticClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DiagnosticClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::relatedDocumentSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("relatedDocumentSupport")) return std::nullopt;

  const auto& value = repr_["relatedDocumentSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::relatedInformation() const
    -> std::optional<bool> {
  if (!repr_.contains("relatedInformation")) return std::nullopt;

  const auto& value = repr_["relatedInformation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::tagSupport() const
    -> std::optional<ClientDiagnosticsTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return ClientDiagnosticsTagOptions(value);
}

auto DiagnosticClientCapabilities::codeDescriptionSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("codeDescriptionSupport")) return std::nullopt;

  const auto& value = repr_["codeDescriptionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::dataSupport() const -> std::optional<bool> {
  if (!repr_.contains("dataSupport")) return std::nullopt;

  const auto& value = repr_["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> DiagnosticClientCapabilities& {
  return *this;
}

auto DiagnosticClientCapabilities::relatedDocumentSupport(
    std::optional<bool> relatedDocumentSupport)
    -> DiagnosticClientCapabilities& {
  return *this;
}

auto DiagnosticClientCapabilities::relatedInformation(
    std::optional<bool> relatedInformation) -> DiagnosticClientCapabilities& {
  return *this;
}

auto DiagnosticClientCapabilities::tagSupport(
    std::optional<ClientDiagnosticsTagOptions> tagSupport)
    -> DiagnosticClientCapabilities& {
  return *this;
}

auto DiagnosticClientCapabilities::codeDescriptionSupport(
    std::optional<bool> codeDescriptionSupport)
    -> DiagnosticClientCapabilities& {
  return *this;
}

auto DiagnosticClientCapabilities::dataSupport(std::optional<bool> dataSupport)
    -> DiagnosticClientCapabilities& {
  return *this;
}

InlineCompletionClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto InlineCompletionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineCompletionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> InlineCompletionClientCapabilities& {
  return *this;
}

NotebookDocumentSyncClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto NotebookDocumentSyncClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_.contains("dynamicRegistration")) return std::nullopt;

  const auto& value = repr_["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncClientCapabilities::executionSummarySupport() const
    -> std::optional<bool> {
  if (!repr_.contains("executionSummarySupport")) return std::nullopt;

  const auto& value = repr_["executionSummarySupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> NotebookDocumentSyncClientCapabilities& {
  return *this;
}

auto NotebookDocumentSyncClientCapabilities::executionSummarySupport(
    std::optional<bool> executionSummarySupport)
    -> NotebookDocumentSyncClientCapabilities& {
  return *this;
}

ShowMessageRequestClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ShowMessageRequestClientCapabilities::messageActionItem() const
    -> std::optional<ClientShowMessageActionItemOptions> {
  if (!repr_.contains("messageActionItem")) return std::nullopt;

  const auto& value = repr_["messageActionItem"];

  return ClientShowMessageActionItemOptions(value);
}

auto ShowMessageRequestClientCapabilities::messageActionItem(
    std::optional<ClientShowMessageActionItemOptions> messageActionItem)
    -> ShowMessageRequestClientCapabilities& {
  return *this;
}

ShowDocumentClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("support")) return false;
  return true;
}

auto ShowDocumentClientCapabilities::support() const -> bool {
  const auto& value = repr_["support"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentClientCapabilities::support(bool support)
    -> ShowDocumentClientCapabilities& {
  return *this;
}

StaleRequestSupportOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("cancel")) return false;
  if (!repr_.contains("retryOnContentModified")) return false;
  return true;
}

auto StaleRequestSupportOptions::cancel() const -> bool {
  const auto& value = repr_["cancel"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto StaleRequestSupportOptions::retryOnContentModified() const
    -> Vector<std::string> {
  const auto& value = repr_["retryOnContentModified"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto StaleRequestSupportOptions::cancel(bool cancel)
    -> StaleRequestSupportOptions& {
  return *this;
}

auto StaleRequestSupportOptions::retryOnContentModified(
    Vector<std::string> retryOnContentModified) -> StaleRequestSupportOptions& {
  return *this;
}

RegularExpressionsClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("engine")) return false;
  return true;
}

auto RegularExpressionsClientCapabilities::engine() const
    -> RegularExpressionEngineKind {
  const auto& value = repr_["engine"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RegularExpressionsClientCapabilities::version() const
    -> std::optional<std::string> {
  if (!repr_.contains("version")) return std::nullopt;

  const auto& value = repr_["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RegularExpressionsClientCapabilities::engine(
    RegularExpressionEngineKind engine)
    -> RegularExpressionsClientCapabilities& {
  return *this;
}

auto RegularExpressionsClientCapabilities::version(
    std::optional<std::string> version)
    -> RegularExpressionsClientCapabilities& {
  return *this;
}

MarkdownClientCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("parser")) return false;
  return true;
}

auto MarkdownClientCapabilities::parser() const -> std::string {
  const auto& value = repr_["parser"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkdownClientCapabilities::version() const -> std::optional<std::string> {
  if (!repr_.contains("version")) return std::nullopt;

  const auto& value = repr_["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkdownClientCapabilities::allowedTags() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("allowedTags")) return std::nullopt;

  const auto& value = repr_["allowedTags"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto MarkdownClientCapabilities::parser(std::string parser)
    -> MarkdownClientCapabilities& {
  return *this;
}

auto MarkdownClientCapabilities::version(std::optional<std::string> version)
    -> MarkdownClientCapabilities& {
  return *this;
}

auto MarkdownClientCapabilities::allowedTags(
    std::optional<Vector<std::string>> allowedTags)
    -> MarkdownClientCapabilities& {
  return *this;
}

ChangeAnnotationsSupportOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ChangeAnnotationsSupportOptions::groupsOnLabel() const
    -> std::optional<bool> {
  if (!repr_.contains("groupsOnLabel")) return std::nullopt;

  const auto& value = repr_["groupsOnLabel"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ChangeAnnotationsSupportOptions::groupsOnLabel(
    std::optional<bool> groupsOnLabel) -> ChangeAnnotationsSupportOptions& {
  return *this;
}

ClientSymbolKindOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientSymbolKindOptions::valueSet() const
    -> std::optional<Vector<SymbolKind>> {
  if (!repr_.contains("valueSet")) return std::nullopt;

  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<SymbolKind>(value);
}

auto ClientSymbolKindOptions::valueSet(
    std::optional<Vector<SymbolKind>> valueSet) -> ClientSymbolKindOptions& {
  return *this;
}

ClientSymbolTagOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("valueSet")) return false;
  return true;
}

auto ClientSymbolTagOptions::valueSet() const -> Vector<SymbolTag> {
  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto ClientSymbolTagOptions::valueSet(Vector<SymbolTag> valueSet)
    -> ClientSymbolTagOptions& {
  return *this;
}

ClientSymbolResolveOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("properties")) return false;
  return true;
}

auto ClientSymbolResolveOptions::properties() const -> Vector<std::string> {
  const auto& value = repr_["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientSymbolResolveOptions::properties(Vector<std::string> properties)
    -> ClientSymbolResolveOptions& {
  return *this;
}

ClientCompletionItemOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientCompletionItemOptions::snippetSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("snippetSupport")) return std::nullopt;

  const auto& value = repr_["snippetSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::commitCharactersSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("commitCharactersSupport")) return std::nullopt;

  const auto& value = repr_["commitCharactersSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::documentationFormat() const
    -> std::optional<Vector<MarkupKind>> {
  if (!repr_.contains("documentationFormat")) return std::nullopt;

  const auto& value = repr_["documentationFormat"];

  assert(value.is_array());
  return Vector<MarkupKind>(value);
}

auto ClientCompletionItemOptions::deprecatedSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("deprecatedSupport")) return std::nullopt;

  const auto& value = repr_["deprecatedSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::preselectSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("preselectSupport")) return std::nullopt;

  const auto& value = repr_["preselectSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::tagSupport() const
    -> std::optional<CompletionItemTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return CompletionItemTagOptions(value);
}

auto ClientCompletionItemOptions::insertReplaceSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("insertReplaceSupport")) return std::nullopt;

  const auto& value = repr_["insertReplaceSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::resolveSupport() const
    -> std::optional<ClientCompletionItemResolveOptions> {
  if (!repr_.contains("resolveSupport")) return std::nullopt;

  const auto& value = repr_["resolveSupport"];

  return ClientCompletionItemResolveOptions(value);
}

auto ClientCompletionItemOptions::insertTextModeSupport() const
    -> std::optional<ClientCompletionItemInsertTextModeOptions> {
  if (!repr_.contains("insertTextModeSupport")) return std::nullopt;

  const auto& value = repr_["insertTextModeSupport"];

  return ClientCompletionItemInsertTextModeOptions(value);
}

auto ClientCompletionItemOptions::labelDetailsSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("labelDetailsSupport")) return std::nullopt;

  const auto& value = repr_["labelDetailsSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::snippetSupport(
    std::optional<bool> snippetSupport) -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::commitCharactersSupport(
    std::optional<bool> commitCharactersSupport)
    -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::documentationFormat(
    std::optional<Vector<MarkupKind>> documentationFormat)
    -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::deprecatedSupport(
    std::optional<bool> deprecatedSupport) -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::preselectSupport(
    std::optional<bool> preselectSupport) -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::tagSupport(
    std::optional<CompletionItemTagOptions> tagSupport)
    -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::insertReplaceSupport(
    std::optional<bool> insertReplaceSupport) -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::resolveSupport(
    std::optional<ClientCompletionItemResolveOptions> resolveSupport)
    -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::insertTextModeSupport(
    std::optional<ClientCompletionItemInsertTextModeOptions>
        insertTextModeSupport) -> ClientCompletionItemOptions& {
  return *this;
}

auto ClientCompletionItemOptions::labelDetailsSupport(
    std::optional<bool> labelDetailsSupport) -> ClientCompletionItemOptions& {
  return *this;
}

ClientCompletionItemOptionsKind::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientCompletionItemOptionsKind::valueSet() const
    -> std::optional<Vector<CompletionItemKind>> {
  if (!repr_.contains("valueSet")) return std::nullopt;

  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<CompletionItemKind>(value);
}

auto ClientCompletionItemOptionsKind::valueSet(
    std::optional<Vector<CompletionItemKind>> valueSet)
    -> ClientCompletionItemOptionsKind& {
  return *this;
}

CompletionListCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto CompletionListCapabilities::itemDefaults() const
    -> std::optional<Vector<std::string>> {
  if (!repr_.contains("itemDefaults")) return std::nullopt;

  const auto& value = repr_["itemDefaults"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionListCapabilities::applyKindSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("applyKindSupport")) return std::nullopt;

  const auto& value = repr_["applyKindSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionListCapabilities::itemDefaults(
    std::optional<Vector<std::string>> itemDefaults)
    -> CompletionListCapabilities& {
  return *this;
}

auto CompletionListCapabilities::applyKindSupport(
    std::optional<bool> applyKindSupport) -> CompletionListCapabilities& {
  return *this;
}

ClientSignatureInformationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientSignatureInformationOptions::documentationFormat() const
    -> std::optional<Vector<MarkupKind>> {
  if (!repr_.contains("documentationFormat")) return std::nullopt;

  const auto& value = repr_["documentationFormat"];

  assert(value.is_array());
  return Vector<MarkupKind>(value);
}

auto ClientSignatureInformationOptions::parameterInformation() const
    -> std::optional<ClientSignatureParameterInformationOptions> {
  if (!repr_.contains("parameterInformation")) return std::nullopt;

  const auto& value = repr_["parameterInformation"];

  return ClientSignatureParameterInformationOptions(value);
}

auto ClientSignatureInformationOptions::activeParameterSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("activeParameterSupport")) return std::nullopt;

  const auto& value = repr_["activeParameterSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSignatureInformationOptions::noActiveParameterSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("noActiveParameterSupport")) return std::nullopt;

  const auto& value = repr_["noActiveParameterSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSignatureInformationOptions::documentationFormat(
    std::optional<Vector<MarkupKind>> documentationFormat)
    -> ClientSignatureInformationOptions& {
  return *this;
}

auto ClientSignatureInformationOptions::parameterInformation(
    std::optional<ClientSignatureParameterInformationOptions>
        parameterInformation) -> ClientSignatureInformationOptions& {
  return *this;
}

auto ClientSignatureInformationOptions::activeParameterSupport(
    std::optional<bool> activeParameterSupport)
    -> ClientSignatureInformationOptions& {
  return *this;
}

auto ClientSignatureInformationOptions::noActiveParameterSupport(
    std::optional<bool> noActiveParameterSupport)
    -> ClientSignatureInformationOptions& {
  return *this;
}

ClientCodeActionLiteralOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("codeActionKind")) return false;
  return true;
}

auto ClientCodeActionLiteralOptions::codeActionKind() const
    -> ClientCodeActionKindOptions {
  const auto& value = repr_["codeActionKind"];

  return ClientCodeActionKindOptions(value);
}

auto ClientCodeActionLiteralOptions::codeActionKind(
    ClientCodeActionKindOptions codeActionKind)
    -> ClientCodeActionLiteralOptions& {
  return *this;
}

ClientCodeActionResolveOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("properties")) return false;
  return true;
}

auto ClientCodeActionResolveOptions::properties() const -> Vector<std::string> {
  const auto& value = repr_["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientCodeActionResolveOptions::properties(Vector<std::string> properties)
    -> ClientCodeActionResolveOptions& {
  return *this;
}

CodeActionTagOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("valueSet")) return false;
  return true;
}

auto CodeActionTagOptions::valueSet() const -> Vector<CodeActionTag> {
  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<CodeActionTag>(value);
}

auto CodeActionTagOptions::valueSet(Vector<CodeActionTag> valueSet)
    -> CodeActionTagOptions& {
  return *this;
}

ClientCodeLensResolveOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("properties")) return false;
  return true;
}

auto ClientCodeLensResolveOptions::properties() const -> Vector<std::string> {
  const auto& value = repr_["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientCodeLensResolveOptions::properties(Vector<std::string> properties)
    -> ClientCodeLensResolveOptions& {
  return *this;
}

ClientFoldingRangeKindOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientFoldingRangeKindOptions::valueSet() const
    -> std::optional<Vector<FoldingRangeKind>> {
  if (!repr_.contains("valueSet")) return std::nullopt;

  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<FoldingRangeKind>(value);
}

auto ClientFoldingRangeKindOptions::valueSet(
    std::optional<Vector<FoldingRangeKind>> valueSet)
    -> ClientFoldingRangeKindOptions& {
  return *this;
}

ClientFoldingRangeOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientFoldingRangeOptions::collapsedText() const -> std::optional<bool> {
  if (!repr_.contains("collapsedText")) return std::nullopt;

  const auto& value = repr_["collapsedText"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientFoldingRangeOptions::collapsedText(std::optional<bool> collapsedText)
    -> ClientFoldingRangeOptions& {
  return *this;
}

DiagnosticsCapabilities::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto DiagnosticsCapabilities::relatedInformation() const
    -> std::optional<bool> {
  if (!repr_.contains("relatedInformation")) return std::nullopt;

  const auto& value = repr_["relatedInformation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticsCapabilities::tagSupport() const
    -> std::optional<ClientDiagnosticsTagOptions> {
  if (!repr_.contains("tagSupport")) return std::nullopt;

  const auto& value = repr_["tagSupport"];

  return ClientDiagnosticsTagOptions(value);
}

auto DiagnosticsCapabilities::codeDescriptionSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("codeDescriptionSupport")) return std::nullopt;

  const auto& value = repr_["codeDescriptionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticsCapabilities::dataSupport() const -> std::optional<bool> {
  if (!repr_.contains("dataSupport")) return std::nullopt;

  const auto& value = repr_["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticsCapabilities::relatedInformation(
    std::optional<bool> relatedInformation) -> DiagnosticsCapabilities& {
  return *this;
}

auto DiagnosticsCapabilities::tagSupport(
    std::optional<ClientDiagnosticsTagOptions> tagSupport)
    -> DiagnosticsCapabilities& {
  return *this;
}

auto DiagnosticsCapabilities::codeDescriptionSupport(
    std::optional<bool> codeDescriptionSupport) -> DiagnosticsCapabilities& {
  return *this;
}

auto DiagnosticsCapabilities::dataSupport(std::optional<bool> dataSupport)
    -> DiagnosticsCapabilities& {
  return *this;
}

ClientSemanticTokensRequestOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientSemanticTokensRequestOptions::range() const
    -> std::optional<std::variant<std::monostate, bool, json>> {
  if (!repr_.contains("range")) return std::nullopt;

  const auto& value = repr_["range"];

  std::variant<std::monostate, bool, json> result;

  details::try_emplace(result, value);

  return result;
}

auto ClientSemanticTokensRequestOptions::full() const -> std::optional<
    std::variant<std::monostate, bool, ClientSemanticTokensRequestFullDelta>> {
  if (!repr_.contains("full")) return std::nullopt;

  const auto& value = repr_["full"];

  std::variant<std::monostate, bool, ClientSemanticTokensRequestFullDelta>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ClientSemanticTokensRequestOptions::range(
    std::optional<std::variant<std::monostate, bool, json>> range)
    -> ClientSemanticTokensRequestOptions& {
  return *this;
}

auto ClientSemanticTokensRequestOptions::full(
    std::optional<std::variant<std::monostate, bool,
                               ClientSemanticTokensRequestFullDelta>>
        full) -> ClientSemanticTokensRequestOptions& {
  return *this;
}

ClientInlayHintResolveOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("properties")) return false;
  return true;
}

auto ClientInlayHintResolveOptions::properties() const -> Vector<std::string> {
  const auto& value = repr_["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientInlayHintResolveOptions::properties(Vector<std::string> properties)
    -> ClientInlayHintResolveOptions& {
  return *this;
}

ClientShowMessageActionItemOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientShowMessageActionItemOptions::additionalPropertiesSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("additionalPropertiesSupport")) return std::nullopt;

  const auto& value = repr_["additionalPropertiesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientShowMessageActionItemOptions::additionalPropertiesSupport(
    std::optional<bool> additionalPropertiesSupport)
    -> ClientShowMessageActionItemOptions& {
  return *this;
}

CompletionItemTagOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("valueSet")) return false;
  return true;
}

auto CompletionItemTagOptions::valueSet() const -> Vector<CompletionItemTag> {
  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<CompletionItemTag>(value);
}

auto CompletionItemTagOptions::valueSet(Vector<CompletionItemTag> valueSet)
    -> CompletionItemTagOptions& {
  return *this;
}

ClientCompletionItemResolveOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("properties")) return false;
  return true;
}

auto ClientCompletionItemResolveOptions::properties() const
    -> Vector<std::string> {
  const auto& value = repr_["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientCompletionItemResolveOptions::properties(
    Vector<std::string> properties) -> ClientCompletionItemResolveOptions& {
  return *this;
}

ClientCompletionItemInsertTextModeOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("valueSet")) return false;
  return true;
}

auto ClientCompletionItemInsertTextModeOptions::valueSet() const
    -> Vector<InsertTextMode> {
  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<InsertTextMode>(value);
}

auto ClientCompletionItemInsertTextModeOptions::valueSet(
    Vector<InsertTextMode> valueSet)
    -> ClientCompletionItemInsertTextModeOptions& {
  return *this;
}

ClientSignatureParameterInformationOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientSignatureParameterInformationOptions::labelOffsetSupport() const
    -> std::optional<bool> {
  if (!repr_.contains("labelOffsetSupport")) return std::nullopt;

  const auto& value = repr_["labelOffsetSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSignatureParameterInformationOptions::labelOffsetSupport(
    std::optional<bool> labelOffsetSupport)
    -> ClientSignatureParameterInformationOptions& {
  return *this;
}

ClientCodeActionKindOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("valueSet")) return false;
  return true;
}

auto ClientCodeActionKindOptions::valueSet() const -> Vector<CodeActionKind> {
  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto ClientCodeActionKindOptions::valueSet(Vector<CodeActionKind> valueSet)
    -> ClientCodeActionKindOptions& {
  return *this;
}

ClientDiagnosticsTagOptions::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  if (!repr_.contains("valueSet")) return false;
  return true;
}

auto ClientDiagnosticsTagOptions::valueSet() const -> Vector<DiagnosticTag> {
  const auto& value = repr_["valueSet"];

  assert(value.is_array());
  return Vector<DiagnosticTag>(value);
}

auto ClientDiagnosticsTagOptions::valueSet(Vector<DiagnosticTag> valueSet)
    -> ClientDiagnosticsTagOptions& {
  return *this;
}

ClientSemanticTokensRequestFullDelta::operator bool() const {
  if (!repr_.is_object() || repr_.is_null()) return false;
  return true;
}

auto ClientSemanticTokensRequestFullDelta::delta() const
    -> std::optional<bool> {
  if (!repr_.contains("delta")) return std::nullopt;

  const auto& value = repr_["delta"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSemanticTokensRequestFullDelta::delta(std::optional<bool> delta)
    -> ClientSemanticTokensRequestFullDelta& {
  return *this;
}
}  // namespace cxx::lsp
