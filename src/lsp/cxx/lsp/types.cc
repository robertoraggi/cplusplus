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
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto ImplementationParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto ImplementationParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto ImplementationParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ImplementationParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ImplementationParams::textDocument(TextDocumentIdentifier textDocument)
    -> ImplementationParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto ImplementationParams::position(Position position)
    -> ImplementationParams& {
  repr_->emplace("position", position);
  return *this;
}

auto ImplementationParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> ImplementationParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("ImplementationParams::workDoneToken: not implement yet");
  return *this;
}

auto ImplementationParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> ImplementationParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "ImplementationParams::partialResultToken: not implement yet");
  return *this;
}

Location::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto Location::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Location::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto Location::uri(std::string uri) -> Location& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto Location::range(Range range) -> Location& {
  repr_->emplace("range", range);
  return *this;
}

ImplementationRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto ImplementationRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ImplementationRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ImplementationRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> ImplementationRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "ImplementationRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto ImplementationRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> ImplementationRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto ImplementationRegistrationOptions::id(std::optional<std::string> id)
    -> ImplementationRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

TypeDefinitionParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto TypeDefinitionParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto TypeDefinitionParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto TypeDefinitionParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionParams::textDocument(TextDocumentIdentifier textDocument)
    -> TypeDefinitionParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto TypeDefinitionParams::position(Position position)
    -> TypeDefinitionParams& {
  repr_->emplace("position", position);
  return *this;
}

auto TypeDefinitionParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> TypeDefinitionParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("TypeDefinitionParams::workDoneToken: not implement yet");
  return *this;
}

auto TypeDefinitionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> TypeDefinitionParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "TypeDefinitionParams::partialResultToken: not implement yet");
  return *this;
}

TypeDefinitionRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto TypeDefinitionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeDefinitionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeDefinitionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TypeDefinitionRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "TypeDefinitionRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto TypeDefinitionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> TypeDefinitionRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto TypeDefinitionRegistrationOptions::id(std::optional<std::string> id)
    -> TypeDefinitionRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

WorkspaceFolder::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("name")) return false;
  return true;
}

auto WorkspaceFolder::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFolder::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFolder::uri(std::string uri) -> WorkspaceFolder& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto WorkspaceFolder::name(std::string name) -> WorkspaceFolder& {
  repr_->emplace("name", std::move(name));
  return *this;
}

DidChangeWorkspaceFoldersParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("event")) return false;
  return true;
}

auto DidChangeWorkspaceFoldersParams::event() const
    -> WorkspaceFoldersChangeEvent {
  auto& value = (*repr_)["event"];

  return WorkspaceFoldersChangeEvent(value);
}

auto DidChangeWorkspaceFoldersParams::event(WorkspaceFoldersChangeEvent event)
    -> DidChangeWorkspaceFoldersParams& {
  repr_->emplace("event", event);
  return *this;
}

ConfigurationParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto ConfigurationParams::items() const -> Vector<ConfigurationItem> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<ConfigurationItem>(value);
}

auto ConfigurationParams::items(Vector<ConfigurationItem> items)
    -> ConfigurationParams& {
  lsp_runtime_error("ConfigurationParams::items: not implement yet");
  return *this;
}

DocumentColorParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DocumentColorParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentColorParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentColorParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentColorParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentColorParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentColorParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentColorParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("DocumentColorParams::workDoneToken: not implement yet");
  return *this;
}

auto DocumentColorParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DocumentColorParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentColorParams::partialResultToken: not implement yet");
  return *this;
}

ColorInformation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("color")) return false;
  return true;
}

auto ColorInformation::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto ColorInformation::color() const -> Color {
  auto& value = (*repr_)["color"];

  return Color(value);
}

auto ColorInformation::range(Range range) -> ColorInformation& {
  repr_->emplace("range", range);
  return *this;
}

auto ColorInformation::color(Color color) -> ColorInformation& {
  repr_->emplace("color", color);
  return *this;
}

DocumentColorRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DocumentColorRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentColorRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentColorRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentColorRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentColorRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentColorRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentColorRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentColorRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto DocumentColorRegistrationOptions::id(std::optional<std::string> id)
    -> DocumentColorRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

ColorPresentationParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("color")) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto ColorPresentationParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto ColorPresentationParams::color() const -> Color {
  auto& value = (*repr_)["color"];

  return Color(value);
}

auto ColorPresentationParams::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto ColorPresentationParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ColorPresentationParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ColorPresentationParams::textDocument(TextDocumentIdentifier textDocument)
    -> ColorPresentationParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto ColorPresentationParams::color(Color color) -> ColorPresentationParams& {
  repr_->emplace("color", color);
  return *this;
}

auto ColorPresentationParams::range(Range range) -> ColorPresentationParams& {
  repr_->emplace("range", range);
  return *this;
}

auto ColorPresentationParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> ColorPresentationParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "ColorPresentationParams::workDoneToken: not implement yet");
  return *this;
}

auto ColorPresentationParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> ColorPresentationParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "ColorPresentationParams::partialResultToken: not implement yet");
  return *this;
}

ColorPresentation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("label")) return false;
  return true;
}

auto ColorPresentation::label() const -> std::string {
  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ColorPresentation::textEdit() const -> std::optional<TextEdit> {
  if (!repr_->contains("textEdit")) return std::nullopt;

  auto& value = (*repr_)["textEdit"];

  return TextEdit(value);
}

auto ColorPresentation::additionalTextEdits() const
    -> std::optional<Vector<TextEdit>> {
  if (!repr_->contains("additionalTextEdits")) return std::nullopt;

  auto& value = (*repr_)["additionalTextEdits"];

  assert(value.is_array());
  return Vector<TextEdit>(value);
}

auto ColorPresentation::label(std::string label) -> ColorPresentation& {
  repr_->emplace("label", std::move(label));
  return *this;
}

auto ColorPresentation::textEdit(std::optional<TextEdit> textEdit)
    -> ColorPresentation& {
  if (!textEdit.has_value()) {
    repr_->erase("textEdit");
    return *this;
  }
  repr_->emplace("textEdit", textEdit.value());
  return *this;
}

auto ColorPresentation::additionalTextEdits(
    std::optional<Vector<TextEdit>> additionalTextEdits) -> ColorPresentation& {
  if (!additionalTextEdits.has_value()) {
    repr_->erase("additionalTextEdits");
    return *this;
  }
  lsp_runtime_error(
      "ColorPresentation::additionalTextEdits: not implement yet");
  return *this;
}

WorkDoneProgressOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkDoneProgressOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkDoneProgressOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> WorkDoneProgressOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

TextDocumentRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto TextDocumentRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TextDocumentRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "TextDocumentRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

FoldingRangeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto FoldingRangeParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto FoldingRangeParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeParams::textDocument(TextDocumentIdentifier textDocument)
    -> FoldingRangeParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto FoldingRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> FoldingRangeParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("FoldingRangeParams::workDoneToken: not implement yet");
  return *this;
}

auto FoldingRangeParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> FoldingRangeParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "FoldingRangeParams::partialResultToken: not implement yet");
  return *this;
}

FoldingRange::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("startLine")) return false;
  if (!repr_->contains("endLine")) return false;
  return true;
}

auto FoldingRange::startLine() const -> long {
  auto& value = (*repr_)["startLine"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::startCharacter() const -> std::optional<long> {
  if (!repr_->contains("startCharacter")) return std::nullopt;

  auto& value = (*repr_)["startCharacter"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::endLine() const -> long {
  auto& value = (*repr_)["endLine"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::endCharacter() const -> std::optional<long> {
  if (!repr_->contains("endCharacter")) return std::nullopt;

  auto& value = (*repr_)["endCharacter"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRange::kind() const -> std::optional<FoldingRangeKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  lsp_runtime_error("FoldingRange::kind: not implement yet");
}

auto FoldingRange::collapsedText() const -> std::optional<std::string> {
  if (!repr_->contains("collapsedText")) return std::nullopt;

  auto& value = (*repr_)["collapsedText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FoldingRange::startLine(long startLine) -> FoldingRange& {
  repr_->emplace("startLine", std::move(startLine));
  return *this;
}

auto FoldingRange::startCharacter(std::optional<long> startCharacter)
    -> FoldingRange& {
  if (!startCharacter.has_value()) {
    repr_->erase("startCharacter");
    return *this;
  }
  repr_->emplace("startCharacter", std::move(startCharacter.value()));
  return *this;
}

auto FoldingRange::endLine(long endLine) -> FoldingRange& {
  repr_->emplace("endLine", std::move(endLine));
  return *this;
}

auto FoldingRange::endCharacter(std::optional<long> endCharacter)
    -> FoldingRange& {
  if (!endCharacter.has_value()) {
    repr_->erase("endCharacter");
    return *this;
  }
  repr_->emplace("endCharacter", std::move(endCharacter.value()));
  return *this;
}

auto FoldingRange::kind(std::optional<FoldingRangeKind> kind) -> FoldingRange& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  lsp_runtime_error("FoldingRange::kind: not implement yet");
  return *this;
}

auto FoldingRange::collapsedText(std::optional<std::string> collapsedText)
    -> FoldingRange& {
  if (!collapsedText.has_value()) {
    repr_->erase("collapsedText");
    return *this;
  }
  repr_->emplace("collapsedText", std::move(collapsedText.value()));
  return *this;
}

FoldingRangeRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto FoldingRangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto FoldingRangeRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FoldingRangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> FoldingRangeRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "FoldingRangeRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto FoldingRangeRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> FoldingRangeRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto FoldingRangeRegistrationOptions::id(std::optional<std::string> id)
    -> FoldingRangeRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

DeclarationParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto DeclarationParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DeclarationParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto DeclarationParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DeclarationParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DeclarationParams::textDocument(TextDocumentIdentifier textDocument)
    -> DeclarationParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DeclarationParams::position(Position position) -> DeclarationParams& {
  repr_->emplace("position", position);
  return *this;
}

auto DeclarationParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DeclarationParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("DeclarationParams::workDoneToken: not implement yet");
  return *this;
}

auto DeclarationParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DeclarationParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("DeclarationParams::partialResultToken: not implement yet");
  return *this;
}

DeclarationRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DeclarationRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DeclarationRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeclarationRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DeclarationRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto DeclarationRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DeclarationRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DeclarationRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DeclarationRegistrationOptions::id(std::optional<std::string> id)
    -> DeclarationRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

SelectionRangeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("positions")) return false;
  return true;
}

auto SelectionRangeParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SelectionRangeParams::positions() const -> Vector<Position> {
  auto& value = (*repr_)["positions"];

  assert(value.is_array());
  return Vector<Position>(value);
}

auto SelectionRangeParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeParams::textDocument(TextDocumentIdentifier textDocument)
    -> SelectionRangeParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto SelectionRangeParams::positions(Vector<Position> positions)
    -> SelectionRangeParams& {
  lsp_runtime_error("SelectionRangeParams::positions: not implement yet");
  return *this;
}

auto SelectionRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SelectionRangeParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("SelectionRangeParams::workDoneToken: not implement yet");
  return *this;
}

auto SelectionRangeParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> SelectionRangeParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "SelectionRangeParams::partialResultToken: not implement yet");
  return *this;
}

SelectionRange::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto SelectionRange::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto SelectionRange::parent() const -> std::optional<SelectionRange> {
  if (!repr_->contains("parent")) return std::nullopt;

  auto& value = (*repr_)["parent"];

  return SelectionRange(value);
}

auto SelectionRange::range(Range range) -> SelectionRange& {
  repr_->emplace("range", range);
  return *this;
}

auto SelectionRange::parent(std::optional<SelectionRange> parent)
    -> SelectionRange& {
  if (!parent.has_value()) {
    repr_->erase("parent");
    return *this;
  }
  repr_->emplace("parent", parent.value());
  return *this;
}

SelectionRangeRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto SelectionRangeRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SelectionRangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SelectionRangeRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SelectionRangeRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> SelectionRangeRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto SelectionRangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> SelectionRangeRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "SelectionRangeRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto SelectionRangeRegistrationOptions::id(std::optional<std::string> id)
    -> SelectionRangeRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

WorkDoneProgressCreateParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("token")) return false;
  return true;
}

auto WorkDoneProgressCreateParams::token() const -> ProgressToken {
  auto& value = (*repr_)["token"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressCreateParams::token(ProgressToken token)
    -> WorkDoneProgressCreateParams& {
  lsp_runtime_error("WorkDoneProgressCreateParams::token: not implement yet");
  return *this;
}

WorkDoneProgressCancelParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("token")) return false;
  return true;
}

auto WorkDoneProgressCancelParams::token() const -> ProgressToken {
  auto& value = (*repr_)["token"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressCancelParams::token(ProgressToken token)
    -> WorkDoneProgressCancelParams& {
  lsp_runtime_error("WorkDoneProgressCancelParams::token: not implement yet");
  return *this;
}

CallHierarchyPrepareParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto CallHierarchyPrepareParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CallHierarchyPrepareParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto CallHierarchyPrepareParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyPrepareParams::textDocument(
    TextDocumentIdentifier textDocument) -> CallHierarchyPrepareParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto CallHierarchyPrepareParams::position(Position position)
    -> CallHierarchyPrepareParams& {
  repr_->emplace("position", position);
  return *this;
}

auto CallHierarchyPrepareParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> CallHierarchyPrepareParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "CallHierarchyPrepareParams::workDoneToken: not implement yet");
  return *this;
}

CallHierarchyItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("name")) return false;
  if (!repr_->contains("kind")) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("selectionRange")) return false;
  return true;
}

auto CallHierarchyItem::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyItem::kind() const -> SymbolKind {
  auto& value = (*repr_)["kind"];

  return SymbolKind(value);
}

auto CallHierarchyItem::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto CallHierarchyItem::detail() const -> std::optional<std::string> {
  if (!repr_->contains("detail")) return std::nullopt;

  auto& value = (*repr_)["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyItem::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyItem::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto CallHierarchyItem::selectionRange() const -> Range {
  auto& value = (*repr_)["selectionRange"];

  return Range(value);
}

auto CallHierarchyItem::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CallHierarchyItem::name(std::string name) -> CallHierarchyItem& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto CallHierarchyItem::kind(SymbolKind kind) -> CallHierarchyItem& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto CallHierarchyItem::tags(std::optional<Vector<SymbolTag>> tags)
    -> CallHierarchyItem& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("CallHierarchyItem::tags: not implement yet");
  return *this;
}

auto CallHierarchyItem::detail(std::optional<std::string> detail)
    -> CallHierarchyItem& {
  if (!detail.has_value()) {
    repr_->erase("detail");
    return *this;
  }
  repr_->emplace("detail", std::move(detail.value()));
  return *this;
}

auto CallHierarchyItem::uri(std::string uri) -> CallHierarchyItem& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto CallHierarchyItem::range(Range range) -> CallHierarchyItem& {
  repr_->emplace("range", range);
  return *this;
}

auto CallHierarchyItem::selectionRange(Range selectionRange)
    -> CallHierarchyItem& {
  repr_->emplace("selectionRange", selectionRange);
  return *this;
}

auto CallHierarchyItem::data(std::optional<LSPAny> data) -> CallHierarchyItem& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("CallHierarchyItem::data: not implement yet");
  return *this;
}

CallHierarchyRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto CallHierarchyRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CallHierarchyRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CallHierarchyRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CallHierarchyRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "CallHierarchyRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto CallHierarchyRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CallHierarchyRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto CallHierarchyRegistrationOptions::id(std::optional<std::string> id)
    -> CallHierarchyRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

CallHierarchyIncomingCallsParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("item")) return false;
  return true;
}

auto CallHierarchyIncomingCallsParams::item() const -> CallHierarchyItem {
  auto& value = (*repr_)["item"];

  return CallHierarchyItem(value);
}

auto CallHierarchyIncomingCallsParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyIncomingCallsParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyIncomingCallsParams::item(CallHierarchyItem item)
    -> CallHierarchyIncomingCallsParams& {
  repr_->emplace("item", item);
  return *this;
}

auto CallHierarchyIncomingCallsParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> CallHierarchyIncomingCallsParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "CallHierarchyIncomingCallsParams::workDoneToken: not implement yet");
  return *this;
}

auto CallHierarchyIncomingCallsParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> CallHierarchyIncomingCallsParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "CallHierarchyIncomingCallsParams::partialResultToken: not implement "
      "yet");
  return *this;
}

CallHierarchyIncomingCall::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("from")) return false;
  if (!repr_->contains("fromRanges")) return false;
  return true;
}

auto CallHierarchyIncomingCall::from() const -> CallHierarchyItem {
  auto& value = (*repr_)["from"];

  return CallHierarchyItem(value);
}

auto CallHierarchyIncomingCall::fromRanges() const -> Vector<Range> {
  auto& value = (*repr_)["fromRanges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto CallHierarchyIncomingCall::from(CallHierarchyItem from)
    -> CallHierarchyIncomingCall& {
  repr_->emplace("from", from);
  return *this;
}

auto CallHierarchyIncomingCall::fromRanges(Vector<Range> fromRanges)
    -> CallHierarchyIncomingCall& {
  lsp_runtime_error("CallHierarchyIncomingCall::fromRanges: not implement yet");
  return *this;
}

CallHierarchyOutgoingCallsParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("item")) return false;
  return true;
}

auto CallHierarchyOutgoingCallsParams::item() const -> CallHierarchyItem {
  auto& value = (*repr_)["item"];

  return CallHierarchyItem(value);
}

auto CallHierarchyOutgoingCallsParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyOutgoingCallsParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CallHierarchyOutgoingCallsParams::item(CallHierarchyItem item)
    -> CallHierarchyOutgoingCallsParams& {
  repr_->emplace("item", item);
  return *this;
}

auto CallHierarchyOutgoingCallsParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> CallHierarchyOutgoingCallsParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "CallHierarchyOutgoingCallsParams::workDoneToken: not implement yet");
  return *this;
}

auto CallHierarchyOutgoingCallsParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> CallHierarchyOutgoingCallsParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "CallHierarchyOutgoingCallsParams::partialResultToken: not implement "
      "yet");
  return *this;
}

CallHierarchyOutgoingCall::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("to")) return false;
  if (!repr_->contains("fromRanges")) return false;
  return true;
}

auto CallHierarchyOutgoingCall::to() const -> CallHierarchyItem {
  auto& value = (*repr_)["to"];

  return CallHierarchyItem(value);
}

auto CallHierarchyOutgoingCall::fromRanges() const -> Vector<Range> {
  auto& value = (*repr_)["fromRanges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto CallHierarchyOutgoingCall::to(CallHierarchyItem to)
    -> CallHierarchyOutgoingCall& {
  repr_->emplace("to", to);
  return *this;
}

auto CallHierarchyOutgoingCall::fromRanges(Vector<Range> fromRanges)
    -> CallHierarchyOutgoingCall& {
  lsp_runtime_error("CallHierarchyOutgoingCall::fromRanges: not implement yet");
  return *this;
}

SemanticTokensParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto SemanticTokensParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SemanticTokensParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensParams::textDocument(TextDocumentIdentifier textDocument)
    -> SemanticTokensParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto SemanticTokensParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SemanticTokensParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("SemanticTokensParams::workDoneToken: not implement yet");
  return *this;
}

auto SemanticTokensParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> SemanticTokensParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "SemanticTokensParams::partialResultToken: not implement yet");
  return *this;
}

SemanticTokens::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("data")) return false;
  return true;
}

auto SemanticTokens::resultId() const -> std::optional<std::string> {
  if (!repr_->contains("resultId")) return std::nullopt;

  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokens::data() const -> Vector<long> {
  auto& value = (*repr_)["data"];

  assert(value.is_array());
  return Vector<long>(value);
}

auto SemanticTokens::resultId(std::optional<std::string> resultId)
    -> SemanticTokens& {
  if (!resultId.has_value()) {
    repr_->erase("resultId");
    return *this;
  }
  repr_->emplace("resultId", std::move(resultId.value()));
  return *this;
}

auto SemanticTokens::data(Vector<long> data) -> SemanticTokens& {
  lsp_runtime_error("SemanticTokens::data: not implement yet");
  return *this;
}

SemanticTokensPartialResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("data")) return false;
  return true;
}

auto SemanticTokensPartialResult::data() const -> Vector<long> {
  auto& value = (*repr_)["data"];

  assert(value.is_array());
  return Vector<long>(value);
}

auto SemanticTokensPartialResult::data(Vector<long> data)
    -> SemanticTokensPartialResult& {
  lsp_runtime_error("SemanticTokensPartialResult::data: not implement yet");
  return *this;
}

SemanticTokensRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  if (!repr_->contains("legend")) return false;
  return true;
}

auto SemanticTokensRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRegistrationOptions::legend() const -> SemanticTokensLegend {
  auto& value = (*repr_)["legend"];

  return SemanticTokensLegend(value);
}

auto SemanticTokensRegistrationOptions::range() const
    -> std::optional<std::variant<std::monostate, bool, json>> {
  if (!repr_->contains("range")) return std::nullopt;

  auto& value = (*repr_)["range"];

  std::variant<std::monostate, bool, json> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRegistrationOptions::full() const -> std::optional<
    std::variant<std::monostate, bool, SemanticTokensFullDelta>> {
  if (!repr_->contains("full")) return std::nullopt;

  auto& value = (*repr_)["full"];

  std::variant<std::monostate, bool, SemanticTokensFullDelta> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokensRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> SemanticTokensRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "SemanticTokensRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto SemanticTokensRegistrationOptions::legend(SemanticTokensLegend legend)
    -> SemanticTokensRegistrationOptions& {
  repr_->emplace("legend", legend);
  return *this;
}

auto SemanticTokensRegistrationOptions::range(
    std::optional<std::variant<std::monostate, bool, json>> range)
    -> SemanticTokensRegistrationOptions& {
  if (!range.has_value()) {
    repr_->erase("range");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool range) { repr_->emplace("range", std::move(range)); }

    void operator()(json range) {
      lsp_runtime_error(
          "SemanticTokensRegistrationOptions::range: not implement yet");
    }
  } v{repr_};

  std::visit(v, range.value());

  return *this;
}

auto SemanticTokensRegistrationOptions::full(
    std::optional<std::variant<std::monostate, bool, SemanticTokensFullDelta>>
        full) -> SemanticTokensRegistrationOptions& {
  if (!full.has_value()) {
    repr_->erase("full");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool full) { repr_->emplace("full", std::move(full)); }

    void operator()(SemanticTokensFullDelta full) {
      repr_->emplace("full", full);
    }
  } v{repr_};

  std::visit(v, full.value());

  return *this;
}

auto SemanticTokensRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> SemanticTokensRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto SemanticTokensRegistrationOptions::id(std::optional<std::string> id)
    -> SemanticTokensRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

SemanticTokensDeltaParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("previousResultId")) return false;
  return true;
}

auto SemanticTokensDeltaParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SemanticTokensDeltaParams::previousResultId() const -> std::string {
  auto& value = (*repr_)["previousResultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokensDeltaParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensDeltaParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensDeltaParams::textDocument(
    TextDocumentIdentifier textDocument) -> SemanticTokensDeltaParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto SemanticTokensDeltaParams::previousResultId(std::string previousResultId)
    -> SemanticTokensDeltaParams& {
  repr_->emplace("previousResultId", std::move(previousResultId));
  return *this;
}

auto SemanticTokensDeltaParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SemanticTokensDeltaParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "SemanticTokensDeltaParams::workDoneToken: not implement yet");
  return *this;
}

auto SemanticTokensDeltaParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> SemanticTokensDeltaParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "SemanticTokensDeltaParams::partialResultToken: not implement yet");
  return *this;
}

SemanticTokensDelta::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("edits")) return false;
  return true;
}

auto SemanticTokensDelta::resultId() const -> std::optional<std::string> {
  if (!repr_->contains("resultId")) return std::nullopt;

  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SemanticTokensDelta::edits() const -> Vector<SemanticTokensEdit> {
  auto& value = (*repr_)["edits"];

  assert(value.is_array());
  return Vector<SemanticTokensEdit>(value);
}

auto SemanticTokensDelta::resultId(std::optional<std::string> resultId)
    -> SemanticTokensDelta& {
  if (!resultId.has_value()) {
    repr_->erase("resultId");
    return *this;
  }
  repr_->emplace("resultId", std::move(resultId.value()));
  return *this;
}

auto SemanticTokensDelta::edits(Vector<SemanticTokensEdit> edits)
    -> SemanticTokensDelta& {
  lsp_runtime_error("SemanticTokensDelta::edits: not implement yet");
  return *this;
}

SemanticTokensDeltaPartialResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("edits")) return false;
  return true;
}

auto SemanticTokensDeltaPartialResult::edits() const
    -> Vector<SemanticTokensEdit> {
  auto& value = (*repr_)["edits"];

  assert(value.is_array());
  return Vector<SemanticTokensEdit>(value);
}

auto SemanticTokensDeltaPartialResult::edits(Vector<SemanticTokensEdit> edits)
    -> SemanticTokensDeltaPartialResult& {
  lsp_runtime_error(
      "SemanticTokensDeltaPartialResult::edits: not implement yet");
  return *this;
}

SemanticTokensRangeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto SemanticTokensRangeParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SemanticTokensRangeParams::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto SemanticTokensRangeParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRangeParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensRangeParams::textDocument(
    TextDocumentIdentifier textDocument) -> SemanticTokensRangeParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto SemanticTokensRangeParams::range(Range range)
    -> SemanticTokensRangeParams& {
  repr_->emplace("range", range);
  return *this;
}

auto SemanticTokensRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SemanticTokensRangeParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "SemanticTokensRangeParams::workDoneToken: not implement yet");
  return *this;
}

auto SemanticTokensRangeParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> SemanticTokensRangeParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "SemanticTokensRangeParams::partialResultToken: not implement yet");
  return *this;
}

ShowDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto ShowDocumentParams::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ShowDocumentParams::external() const -> std::optional<bool> {
  if (!repr_->contains("external")) return std::nullopt;

  auto& value = (*repr_)["external"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentParams::takeFocus() const -> std::optional<bool> {
  if (!repr_->contains("takeFocus")) return std::nullopt;

  auto& value = (*repr_)["takeFocus"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentParams::selection() const -> std::optional<Range> {
  if (!repr_->contains("selection")) return std::nullopt;

  auto& value = (*repr_)["selection"];

  return Range(value);
}

auto ShowDocumentParams::uri(std::string uri) -> ShowDocumentParams& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto ShowDocumentParams::external(std::optional<bool> external)
    -> ShowDocumentParams& {
  if (!external.has_value()) {
    repr_->erase("external");
    return *this;
  }
  repr_->emplace("external", std::move(external.value()));
  return *this;
}

auto ShowDocumentParams::takeFocus(std::optional<bool> takeFocus)
    -> ShowDocumentParams& {
  if (!takeFocus.has_value()) {
    repr_->erase("takeFocus");
    return *this;
  }
  repr_->emplace("takeFocus", std::move(takeFocus.value()));
  return *this;
}

auto ShowDocumentParams::selection(std::optional<Range> selection)
    -> ShowDocumentParams& {
  if (!selection.has_value()) {
    repr_->erase("selection");
    return *this;
  }
  repr_->emplace("selection", selection.value());
  return *this;
}

ShowDocumentResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("success")) return false;
  return true;
}

auto ShowDocumentResult::success() const -> bool {
  auto& value = (*repr_)["success"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentResult::success(bool success) -> ShowDocumentResult& {
  repr_->emplace("success", std::move(success));
  return *this;
}

LinkedEditingRangeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto LinkedEditingRangeParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto LinkedEditingRangeParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto LinkedEditingRangeParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto LinkedEditingRangeParams::textDocument(TextDocumentIdentifier textDocument)
    -> LinkedEditingRangeParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto LinkedEditingRangeParams::position(Position position)
    -> LinkedEditingRangeParams& {
  repr_->emplace("position", position);
  return *this;
}

auto LinkedEditingRangeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> LinkedEditingRangeParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "LinkedEditingRangeParams::workDoneToken: not implement yet");
  return *this;
}

LinkedEditingRanges::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("ranges")) return false;
  return true;
}

auto LinkedEditingRanges::ranges() const -> Vector<Range> {
  auto& value = (*repr_)["ranges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto LinkedEditingRanges::wordPattern() const -> std::optional<std::string> {
  if (!repr_->contains("wordPattern")) return std::nullopt;

  auto& value = (*repr_)["wordPattern"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LinkedEditingRanges::ranges(Vector<Range> ranges) -> LinkedEditingRanges& {
  lsp_runtime_error("LinkedEditingRanges::ranges: not implement yet");
  return *this;
}

auto LinkedEditingRanges::wordPattern(std::optional<std::string> wordPattern)
    -> LinkedEditingRanges& {
  if (!wordPattern.has_value()) {
    repr_->erase("wordPattern");
    return *this;
  }
  repr_->emplace("wordPattern", std::move(wordPattern.value()));
  return *this;
}

LinkedEditingRangeRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto LinkedEditingRangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto LinkedEditingRangeRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto LinkedEditingRangeRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LinkedEditingRangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> LinkedEditingRangeRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "LinkedEditingRangeRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto LinkedEditingRangeRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> LinkedEditingRangeRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto LinkedEditingRangeRegistrationOptions::id(std::optional<std::string> id)
    -> LinkedEditingRangeRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

CreateFilesParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("files")) return false;
  return true;
}

auto CreateFilesParams::files() const -> Vector<FileCreate> {
  auto& value = (*repr_)["files"];

  assert(value.is_array());
  return Vector<FileCreate>(value);
}

auto CreateFilesParams::files(Vector<FileCreate> files) -> CreateFilesParams& {
  lsp_runtime_error("CreateFilesParams::files: not implement yet");
  return *this;
}

WorkspaceEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceEdit::changes() const
    -> std::optional<Map<std::string, Vector<TextEdit>>> {
  if (!repr_->contains("changes")) return std::nullopt;

  auto& value = (*repr_)["changes"];

  assert(value.is_object());
  return Map<std::string, Vector<TextEdit>>(value);
}

auto WorkspaceEdit::documentChanges() const
    -> std::optional<Vector<std::variant<std::monostate, TextDocumentEdit,
                                         CreateFile, RenameFile, DeleteFile>>> {
  if (!repr_->contains("documentChanges")) return std::nullopt;

  auto& value = (*repr_)["documentChanges"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, TextDocumentEdit, CreateFile,
                             RenameFile, DeleteFile>>(value);
}

auto WorkspaceEdit::changeAnnotations() const
    -> std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>> {
  if (!repr_->contains("changeAnnotations")) return std::nullopt;

  auto& value = (*repr_)["changeAnnotations"];

  assert(value.is_object());
  return Map<ChangeAnnotationIdentifier, ChangeAnnotation>(value);
}

auto WorkspaceEdit::changes(
    std::optional<Map<std::string, Vector<TextEdit>>> changes)
    -> WorkspaceEdit& {
  if (!changes.has_value()) {
    repr_->erase("changes");
    return *this;
  }
  lsp_runtime_error("WorkspaceEdit::changes: not implement yet");
  return *this;
}

auto WorkspaceEdit::documentChanges(
    std::optional<Vector<std::variant<std::monostate, TextDocumentEdit,
                                      CreateFile, RenameFile, DeleteFile>>>
        documentChanges) -> WorkspaceEdit& {
  if (!documentChanges.has_value()) {
    repr_->erase("documentChanges");
    return *this;
  }
  lsp_runtime_error("WorkspaceEdit::documentChanges: not implement yet");
  return *this;
}

auto WorkspaceEdit::changeAnnotations(
    std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>>
        changeAnnotations) -> WorkspaceEdit& {
  if (!changeAnnotations.has_value()) {
    repr_->erase("changeAnnotations");
    return *this;
  }
  lsp_runtime_error("WorkspaceEdit::changeAnnotations: not implement yet");
  return *this;
}

FileOperationRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("filters")) return false;
  return true;
}

auto FileOperationRegistrationOptions::filters() const
    -> Vector<FileOperationFilter> {
  auto& value = (*repr_)["filters"];

  assert(value.is_array());
  return Vector<FileOperationFilter>(value);
}

auto FileOperationRegistrationOptions::filters(
    Vector<FileOperationFilter> filters) -> FileOperationRegistrationOptions& {
  lsp_runtime_error(
      "FileOperationRegistrationOptions::filters: not implement yet");
  return *this;
}

RenameFilesParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("files")) return false;
  return true;
}

auto RenameFilesParams::files() const -> Vector<FileRename> {
  auto& value = (*repr_)["files"];

  assert(value.is_array());
  return Vector<FileRename>(value);
}

auto RenameFilesParams::files(Vector<FileRename> files) -> RenameFilesParams& {
  lsp_runtime_error("RenameFilesParams::files: not implement yet");
  return *this;
}

DeleteFilesParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("files")) return false;
  return true;
}

auto DeleteFilesParams::files() const -> Vector<FileDelete> {
  auto& value = (*repr_)["files"];

  assert(value.is_array());
  return Vector<FileDelete>(value);
}

auto DeleteFilesParams::files(Vector<FileDelete> files) -> DeleteFilesParams& {
  lsp_runtime_error("DeleteFilesParams::files: not implement yet");
  return *this;
}

MonikerParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto MonikerParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto MonikerParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto MonikerParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerParams::partialResultToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerParams::textDocument(TextDocumentIdentifier textDocument)
    -> MonikerParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto MonikerParams::position(Position position) -> MonikerParams& {
  repr_->emplace("position", position);
  return *this;
}

auto MonikerParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> MonikerParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("MonikerParams::workDoneToken: not implement yet");
  return *this;
}

auto MonikerParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> MonikerParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("MonikerParams::partialResultToken: not implement yet");
  return *this;
}

Moniker::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("scheme")) return false;
  if (!repr_->contains("identifier")) return false;
  if (!repr_->contains("unique")) return false;
  return true;
}

auto Moniker::scheme() const -> std::string {
  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Moniker::identifier() const -> std::string {
  auto& value = (*repr_)["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Moniker::unique() const -> UniquenessLevel {
  auto& value = (*repr_)["unique"];

  lsp_runtime_error("Moniker::unique: not implement yet");
}

auto Moniker::kind() const -> std::optional<MonikerKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  lsp_runtime_error("Moniker::kind: not implement yet");
}

auto Moniker::scheme(std::string scheme) -> Moniker& {
  repr_->emplace("scheme", std::move(scheme));
  return *this;
}

auto Moniker::identifier(std::string identifier) -> Moniker& {
  repr_->emplace("identifier", std::move(identifier));
  return *this;
}

auto Moniker::unique(UniquenessLevel unique) -> Moniker& {
  lsp_runtime_error("Moniker::unique: not implement yet");
  return *this;
}

auto Moniker::kind(std::optional<MonikerKind> kind) -> Moniker& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  lsp_runtime_error("Moniker::kind: not implement yet");
  return *this;
}

MonikerRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto MonikerRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto MonikerRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto MonikerRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> MonikerRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "MonikerRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto MonikerRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> MonikerRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

TypeHierarchyPrepareParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto TypeHierarchyPrepareParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto TypeHierarchyPrepareParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto TypeHierarchyPrepareParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchyPrepareParams::textDocument(
    TextDocumentIdentifier textDocument) -> TypeHierarchyPrepareParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto TypeHierarchyPrepareParams::position(Position position)
    -> TypeHierarchyPrepareParams& {
  repr_->emplace("position", position);
  return *this;
}

auto TypeHierarchyPrepareParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> TypeHierarchyPrepareParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "TypeHierarchyPrepareParams::workDoneToken: not implement yet");
  return *this;
}

TypeHierarchyItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("name")) return false;
  if (!repr_->contains("kind")) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("selectionRange")) return false;
  return true;
}

auto TypeHierarchyItem::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyItem::kind() const -> SymbolKind {
  auto& value = (*repr_)["kind"];

  return SymbolKind(value);
}

auto TypeHierarchyItem::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto TypeHierarchyItem::detail() const -> std::optional<std::string> {
  if (!repr_->contains("detail")) return std::nullopt;

  auto& value = (*repr_)["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyItem::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyItem::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto TypeHierarchyItem::selectionRange() const -> Range {
  auto& value = (*repr_)["selectionRange"];

  return Range(value);
}

auto TypeHierarchyItem::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto TypeHierarchyItem::name(std::string name) -> TypeHierarchyItem& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto TypeHierarchyItem::kind(SymbolKind kind) -> TypeHierarchyItem& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto TypeHierarchyItem::tags(std::optional<Vector<SymbolTag>> tags)
    -> TypeHierarchyItem& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("TypeHierarchyItem::tags: not implement yet");
  return *this;
}

auto TypeHierarchyItem::detail(std::optional<std::string> detail)
    -> TypeHierarchyItem& {
  if (!detail.has_value()) {
    repr_->erase("detail");
    return *this;
  }
  repr_->emplace("detail", std::move(detail.value()));
  return *this;
}

auto TypeHierarchyItem::uri(std::string uri) -> TypeHierarchyItem& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto TypeHierarchyItem::range(Range range) -> TypeHierarchyItem& {
  repr_->emplace("range", range);
  return *this;
}

auto TypeHierarchyItem::selectionRange(Range selectionRange)
    -> TypeHierarchyItem& {
  repr_->emplace("selectionRange", selectionRange);
  return *this;
}

auto TypeHierarchyItem::data(std::optional<LSPAny> data) -> TypeHierarchyItem& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("TypeHierarchyItem::data: not implement yet");
  return *this;
}

TypeHierarchyRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto TypeHierarchyRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchyRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeHierarchyRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TypeHierarchyRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TypeHierarchyRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "TypeHierarchyRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto TypeHierarchyRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> TypeHierarchyRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto TypeHierarchyRegistrationOptions::id(std::optional<std::string> id)
    -> TypeHierarchyRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

TypeHierarchySupertypesParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("item")) return false;
  return true;
}

auto TypeHierarchySupertypesParams::item() const -> TypeHierarchyItem {
  auto& value = (*repr_)["item"];

  return TypeHierarchyItem(value);
}

auto TypeHierarchySupertypesParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySupertypesParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySupertypesParams::item(TypeHierarchyItem item)
    -> TypeHierarchySupertypesParams& {
  repr_->emplace("item", item);
  return *this;
}

auto TypeHierarchySupertypesParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> TypeHierarchySupertypesParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "TypeHierarchySupertypesParams::workDoneToken: not implement yet");
  return *this;
}

auto TypeHierarchySupertypesParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> TypeHierarchySupertypesParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "TypeHierarchySupertypesParams::partialResultToken: not implement yet");
  return *this;
}

TypeHierarchySubtypesParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("item")) return false;
  return true;
}

auto TypeHierarchySubtypesParams::item() const -> TypeHierarchyItem {
  auto& value = (*repr_)["item"];

  return TypeHierarchyItem(value);
}

auto TypeHierarchySubtypesParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySubtypesParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto TypeHierarchySubtypesParams::item(TypeHierarchyItem item)
    -> TypeHierarchySubtypesParams& {
  repr_->emplace("item", item);
  return *this;
}

auto TypeHierarchySubtypesParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> TypeHierarchySubtypesParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "TypeHierarchySubtypesParams::workDoneToken: not implement yet");
  return *this;
}

auto TypeHierarchySubtypesParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> TypeHierarchySubtypesParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "TypeHierarchySubtypesParams::partialResultToken: not implement yet");
  return *this;
}

InlineValueParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("context")) return false;
  return true;
}

auto InlineValueParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto InlineValueParams::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto InlineValueParams::context() const -> InlineValueContext {
  auto& value = (*repr_)["context"];

  return InlineValueContext(value);
}

auto InlineValueParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InlineValueParams::textDocument(TextDocumentIdentifier textDocument)
    -> InlineValueParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto InlineValueParams::range(Range range) -> InlineValueParams& {
  repr_->emplace("range", range);
  return *this;
}

auto InlineValueParams::context(InlineValueContext context)
    -> InlineValueParams& {
  repr_->emplace("context", context);
  return *this;
}

auto InlineValueParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> InlineValueParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("InlineValueParams::workDoneToken: not implement yet");
  return *this;
}

InlineValueRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto InlineValueRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineValueRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> InlineValueRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto InlineValueRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> InlineValueRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "InlineValueRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto InlineValueRegistrationOptions::id(std::optional<std::string> id)
    -> InlineValueRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

InlayHintParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto InlayHintParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto InlayHintParams::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto InlayHintParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintParams::textDocument(TextDocumentIdentifier textDocument)
    -> InlayHintParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto InlayHintParams::range(Range range) -> InlayHintParams& {
  repr_->emplace("range", range);
  return *this;
}

auto InlayHintParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> InlayHintParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("InlayHintParams::workDoneToken: not implement yet");
  return *this;
}

InlayHint::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("position")) return false;
  if (!repr_->contains("label")) return false;
  return true;
}

auto InlayHint::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto InlayHint::label() const
    -> std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>> {
  auto& value = (*repr_)["label"];

  std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHint::kind() const -> std::optional<InlayHintKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  return InlayHintKind(value);
}

auto InlayHint::textEdits() const -> std::optional<Vector<TextEdit>> {
  if (!repr_->contains("textEdits")) return std::nullopt;

  auto& value = (*repr_)["textEdits"];

  assert(value.is_array());
  return Vector<TextEdit>(value);
}

auto InlayHint::tooltip() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_->contains("tooltip")) return std::nullopt;

  auto& value = (*repr_)["tooltip"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHint::paddingLeft() const -> std::optional<bool> {
  if (!repr_->contains("paddingLeft")) return std::nullopt;

  auto& value = (*repr_)["paddingLeft"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHint::paddingRight() const -> std::optional<bool> {
  if (!repr_->contains("paddingRight")) return std::nullopt;

  auto& value = (*repr_)["paddingRight"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHint::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto InlayHint::position(Position position) -> InlayHint& {
  repr_->emplace("position", position);
  return *this;
}

auto InlayHint::label(
    std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>> label)
    -> InlayHint& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string label) {
      repr_->emplace("label", std::move(label));
    }

    void operator()(Vector<InlayHintLabelPart> label) {
      lsp_runtime_error("InlayHint::label: not implement yet");
    }
  } v{repr_};

  std::visit(v, label);

  return *this;
}

auto InlayHint::kind(std::optional<InlayHintKind> kind) -> InlayHint& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  repr_->emplace("kind", static_cast<long>(kind.value()));
  return *this;
}

auto InlayHint::textEdits(std::optional<Vector<TextEdit>> textEdits)
    -> InlayHint& {
  if (!textEdits.has_value()) {
    repr_->erase("textEdits");
    return *this;
  }
  lsp_runtime_error("InlayHint::textEdits: not implement yet");
  return *this;
}

auto InlayHint::tooltip(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        tooltip) -> InlayHint& {
  if (!tooltip.has_value()) {
    repr_->erase("tooltip");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string tooltip) {
      repr_->emplace("tooltip", std::move(tooltip));
    }

    void operator()(MarkupContent tooltip) {
      repr_->emplace("tooltip", tooltip);
    }
  } v{repr_};

  std::visit(v, tooltip.value());

  return *this;
}

auto InlayHint::paddingLeft(std::optional<bool> paddingLeft) -> InlayHint& {
  if (!paddingLeft.has_value()) {
    repr_->erase("paddingLeft");
    return *this;
  }
  repr_->emplace("paddingLeft", std::move(paddingLeft.value()));
  return *this;
}

auto InlayHint::paddingRight(std::optional<bool> paddingRight) -> InlayHint& {
  if (!paddingRight.has_value()) {
    repr_->erase("paddingRight");
    return *this;
  }
  repr_->emplace("paddingRight", std::move(paddingRight.value()));
  return *this;
}

auto InlayHint::data(std::optional<LSPAny> data) -> InlayHint& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("InlayHint::data: not implement yet");
  return *this;
}

InlayHintRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto InlayHintRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlayHintRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> InlayHintRegistrationOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto InlayHintRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> InlayHintRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto InlayHintRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> InlayHintRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "InlayHintRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto InlayHintRegistrationOptions::id(std::optional<std::string> id)
    -> InlayHintRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

DocumentDiagnosticParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DocumentDiagnosticParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentDiagnosticParams::identifier() const
    -> std::optional<std::string> {
  if (!repr_->contains("identifier")) return std::nullopt;

  auto& value = (*repr_)["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentDiagnosticParams::previousResultId() const
    -> std::optional<std::string> {
  if (!repr_->contains("previousResultId")) return std::nullopt;

  auto& value = (*repr_)["previousResultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentDiagnosticParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentDiagnosticParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentDiagnosticParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentDiagnosticParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentDiagnosticParams::identifier(std::optional<std::string> identifier)
    -> DocumentDiagnosticParams& {
  if (!identifier.has_value()) {
    repr_->erase("identifier");
    return *this;
  }
  repr_->emplace("identifier", std::move(identifier.value()));
  return *this;
}

auto DocumentDiagnosticParams::previousResultId(
    std::optional<std::string> previousResultId) -> DocumentDiagnosticParams& {
  if (!previousResultId.has_value()) {
    repr_->erase("previousResultId");
    return *this;
  }
  repr_->emplace("previousResultId", std::move(previousResultId.value()));
  return *this;
}

auto DocumentDiagnosticParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentDiagnosticParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentDiagnosticParams::workDoneToken: not implement yet");
  return *this;
}

auto DocumentDiagnosticParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> DocumentDiagnosticParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentDiagnosticParams::partialResultToken: not implement yet");
  return *this;
}

DocumentDiagnosticReportPartialResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("relatedDocuments")) return false;
  return true;
}

auto DocumentDiagnosticReportPartialResult::relatedDocuments() const
    -> Map<std::string,
           std::variant<std::monostate, FullDocumentDiagnosticReport,
                        UnchangedDocumentDiagnosticReport>> {
  auto& value = (*repr_)["relatedDocuments"];

  assert(value.is_object());
  return Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>(value);
}

auto DocumentDiagnosticReportPartialResult::relatedDocuments(
    Map<std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                  UnchangedDocumentDiagnosticReport>>
        relatedDocuments) -> DocumentDiagnosticReportPartialResult& {
  lsp_runtime_error(
      "DocumentDiagnosticReportPartialResult::relatedDocuments: not implement "
      "yet");
  return *this;
}

DiagnosticServerCancellationData::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("retriggerRequest")) return false;
  return true;
}

auto DiagnosticServerCancellationData::retriggerRequest() const -> bool {
  auto& value = (*repr_)["retriggerRequest"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticServerCancellationData::retriggerRequest(bool retriggerRequest)
    -> DiagnosticServerCancellationData& {
  repr_->emplace("retriggerRequest", std::move(retriggerRequest));
  return *this;
}

DiagnosticRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  if (!repr_->contains("interFileDependencies")) return false;
  if (!repr_->contains("workspaceDiagnostics")) return false;
  return true;
}

auto DiagnosticRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DiagnosticRegistrationOptions::identifier() const
    -> std::optional<std::string> {
  if (!repr_->contains("identifier")) return std::nullopt;

  auto& value = (*repr_)["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticRegistrationOptions::interFileDependencies() const -> bool {
  auto& value = (*repr_)["interFileDependencies"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticRegistrationOptions::workspaceDiagnostics() const -> bool {
  auto& value = (*repr_)["workspaceDiagnostics"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DiagnosticRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DiagnosticRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DiagnosticRegistrationOptions::identifier(
    std::optional<std::string> identifier) -> DiagnosticRegistrationOptions& {
  if (!identifier.has_value()) {
    repr_->erase("identifier");
    return *this;
  }
  repr_->emplace("identifier", std::move(identifier.value()));
  return *this;
}

auto DiagnosticRegistrationOptions::interFileDependencies(
    bool interFileDependencies) -> DiagnosticRegistrationOptions& {
  repr_->emplace("interFileDependencies", std::move(interFileDependencies));
  return *this;
}

auto DiagnosticRegistrationOptions::workspaceDiagnostics(
    bool workspaceDiagnostics) -> DiagnosticRegistrationOptions& {
  repr_->emplace("workspaceDiagnostics", std::move(workspaceDiagnostics));
  return *this;
}

auto DiagnosticRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DiagnosticRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto DiagnosticRegistrationOptions::id(std::optional<std::string> id)
    -> DiagnosticRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

WorkspaceDiagnosticParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("previousResultIds")) return false;
  return true;
}

auto WorkspaceDiagnosticParams::identifier() const
    -> std::optional<std::string> {
  if (!repr_->contains("identifier")) return std::nullopt;

  auto& value = (*repr_)["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceDiagnosticParams::previousResultIds() const
    -> Vector<PreviousResultId> {
  auto& value = (*repr_)["previousResultIds"];

  assert(value.is_array());
  return Vector<PreviousResultId>(value);
}

auto WorkspaceDiagnosticParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceDiagnosticParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceDiagnosticParams::identifier(
    std::optional<std::string> identifier) -> WorkspaceDiagnosticParams& {
  if (!identifier.has_value()) {
    repr_->erase("identifier");
    return *this;
  }
  repr_->emplace("identifier", std::move(identifier.value()));
  return *this;
}

auto WorkspaceDiagnosticParams::previousResultIds(
    Vector<PreviousResultId> previousResultIds) -> WorkspaceDiagnosticParams& {
  lsp_runtime_error(
      "WorkspaceDiagnosticParams::previousResultIds: not implement yet");
  return *this;
}

auto WorkspaceDiagnosticParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> WorkspaceDiagnosticParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "WorkspaceDiagnosticParams::workDoneToken: not implement yet");
  return *this;
}

auto WorkspaceDiagnosticParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> WorkspaceDiagnosticParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "WorkspaceDiagnosticParams::partialResultToken: not implement yet");
  return *this;
}

WorkspaceDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto WorkspaceDiagnosticReport::items() const
    -> Vector<WorkspaceDocumentDiagnosticReport> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<WorkspaceDocumentDiagnosticReport>(value);
}

auto WorkspaceDiagnosticReport::items(
    Vector<WorkspaceDocumentDiagnosticReport> items)
    -> WorkspaceDiagnosticReport& {
  lsp_runtime_error("WorkspaceDiagnosticReport::items: not implement yet");
  return *this;
}

WorkspaceDiagnosticReportPartialResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto WorkspaceDiagnosticReportPartialResult::items() const
    -> Vector<WorkspaceDocumentDiagnosticReport> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<WorkspaceDocumentDiagnosticReport>(value);
}

auto WorkspaceDiagnosticReportPartialResult::items(
    Vector<WorkspaceDocumentDiagnosticReport> items)
    -> WorkspaceDiagnosticReportPartialResult& {
  lsp_runtime_error(
      "WorkspaceDiagnosticReportPartialResult::items: not implement yet");
  return *this;
}

DidOpenNotebookDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookDocument")) return false;
  if (!repr_->contains("cellTextDocuments")) return false;
  return true;
}

auto DidOpenNotebookDocumentParams::notebookDocument() const
    -> NotebookDocument {
  auto& value = (*repr_)["notebookDocument"];

  return NotebookDocument(value);
}

auto DidOpenNotebookDocumentParams::cellTextDocuments() const
    -> Vector<TextDocumentItem> {
  auto& value = (*repr_)["cellTextDocuments"];

  assert(value.is_array());
  return Vector<TextDocumentItem>(value);
}

auto DidOpenNotebookDocumentParams::notebookDocument(
    NotebookDocument notebookDocument) -> DidOpenNotebookDocumentParams& {
  repr_->emplace("notebookDocument", notebookDocument);
  return *this;
}

auto DidOpenNotebookDocumentParams::cellTextDocuments(
    Vector<TextDocumentItem> cellTextDocuments)
    -> DidOpenNotebookDocumentParams& {
  lsp_runtime_error(
      "DidOpenNotebookDocumentParams::cellTextDocuments: not implement yet");
  return *this;
}

NotebookDocumentSyncRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookSelector")) return false;
  return true;
}

auto NotebookDocumentSyncRegistrationOptions::notebookSelector() const
    -> Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                           NotebookDocumentFilterWithCells>> {
  auto& value = (*repr_)["notebookSelector"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>(value);
}

auto NotebookDocumentSyncRegistrationOptions::save() const
    -> std::optional<bool> {
  if (!repr_->contains("save")) return std::nullopt;

  auto& value = (*repr_)["save"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentSyncRegistrationOptions::notebookSelector(
    Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                        NotebookDocumentFilterWithCells>>
        notebookSelector) -> NotebookDocumentSyncRegistrationOptions& {
  lsp_runtime_error(
      "NotebookDocumentSyncRegistrationOptions::notebookSelector: not "
      "implement yet");
  return *this;
}

auto NotebookDocumentSyncRegistrationOptions::save(std::optional<bool> save)
    -> NotebookDocumentSyncRegistrationOptions& {
  if (!save.has_value()) {
    repr_->erase("save");
    return *this;
  }
  repr_->emplace("save", std::move(save.value()));
  return *this;
}

auto NotebookDocumentSyncRegistrationOptions::id(std::optional<std::string> id)
    -> NotebookDocumentSyncRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

DidChangeNotebookDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookDocument")) return false;
  if (!repr_->contains("change")) return false;
  return true;
}

auto DidChangeNotebookDocumentParams::notebookDocument() const
    -> VersionedNotebookDocumentIdentifier {
  auto& value = (*repr_)["notebookDocument"];

  return VersionedNotebookDocumentIdentifier(value);
}

auto DidChangeNotebookDocumentParams::change() const
    -> NotebookDocumentChangeEvent {
  auto& value = (*repr_)["change"];

  return NotebookDocumentChangeEvent(value);
}

auto DidChangeNotebookDocumentParams::notebookDocument(
    VersionedNotebookDocumentIdentifier notebookDocument)
    -> DidChangeNotebookDocumentParams& {
  repr_->emplace("notebookDocument", notebookDocument);
  return *this;
}

auto DidChangeNotebookDocumentParams::change(NotebookDocumentChangeEvent change)
    -> DidChangeNotebookDocumentParams& {
  repr_->emplace("change", change);
  return *this;
}

DidSaveNotebookDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookDocument")) return false;
  return true;
}

auto DidSaveNotebookDocumentParams::notebookDocument() const
    -> NotebookDocumentIdentifier {
  auto& value = (*repr_)["notebookDocument"];

  return NotebookDocumentIdentifier(value);
}

auto DidSaveNotebookDocumentParams::notebookDocument(
    NotebookDocumentIdentifier notebookDocument)
    -> DidSaveNotebookDocumentParams& {
  repr_->emplace("notebookDocument", notebookDocument);
  return *this;
}

DidCloseNotebookDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookDocument")) return false;
  if (!repr_->contains("cellTextDocuments")) return false;
  return true;
}

auto DidCloseNotebookDocumentParams::notebookDocument() const
    -> NotebookDocumentIdentifier {
  auto& value = (*repr_)["notebookDocument"];

  return NotebookDocumentIdentifier(value);
}

auto DidCloseNotebookDocumentParams::cellTextDocuments() const
    -> Vector<TextDocumentIdentifier> {
  auto& value = (*repr_)["cellTextDocuments"];

  assert(value.is_array());
  return Vector<TextDocumentIdentifier>(value);
}

auto DidCloseNotebookDocumentParams::notebookDocument(
    NotebookDocumentIdentifier notebookDocument)
    -> DidCloseNotebookDocumentParams& {
  repr_->emplace("notebookDocument", notebookDocument);
  return *this;
}

auto DidCloseNotebookDocumentParams::cellTextDocuments(
    Vector<TextDocumentIdentifier> cellTextDocuments)
    -> DidCloseNotebookDocumentParams& {
  lsp_runtime_error(
      "DidCloseNotebookDocumentParams::cellTextDocuments: not implement yet");
  return *this;
}

InlineCompletionParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("context")) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto InlineCompletionParams::context() const -> InlineCompletionContext {
  auto& value = (*repr_)["context"];

  return InlineCompletionContext(value);
}

auto InlineCompletionParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto InlineCompletionParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto InlineCompletionParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InlineCompletionParams::context(InlineCompletionContext context)
    -> InlineCompletionParams& {
  repr_->emplace("context", context);
  return *this;
}

auto InlineCompletionParams::textDocument(TextDocumentIdentifier textDocument)
    -> InlineCompletionParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto InlineCompletionParams::position(Position position)
    -> InlineCompletionParams& {
  repr_->emplace("position", position);
  return *this;
}

auto InlineCompletionParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> InlineCompletionParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("InlineCompletionParams::workDoneToken: not implement yet");
  return *this;
}

InlineCompletionList::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto InlineCompletionList::items() const -> Vector<InlineCompletionItem> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<InlineCompletionItem>(value);
}

auto InlineCompletionList::items(Vector<InlineCompletionItem> items)
    -> InlineCompletionList& {
  lsp_runtime_error("InlineCompletionList::items: not implement yet");
  return *this;
}

InlineCompletionItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("insertText")) return false;
  return true;
}

auto InlineCompletionItem::insertText() const
    -> std::variant<std::monostate, std::string, StringValue> {
  auto& value = (*repr_)["insertText"];

  std::variant<std::monostate, std::string, StringValue> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineCompletionItem::filterText() const -> std::optional<std::string> {
  if (!repr_->contains("filterText")) return std::nullopt;

  auto& value = (*repr_)["filterText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineCompletionItem::range() const -> std::optional<Range> {
  if (!repr_->contains("range")) return std::nullopt;

  auto& value = (*repr_)["range"];

  return Range(value);
}

auto InlineCompletionItem::command() const -> std::optional<Command> {
  if (!repr_->contains("command")) return std::nullopt;

  auto& value = (*repr_)["command"];

  return Command(value);
}

auto InlineCompletionItem::insertText(
    std::variant<std::monostate, std::string, StringValue> insertText)
    -> InlineCompletionItem& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string insertText) {
      repr_->emplace("insertText", std::move(insertText));
    }

    void operator()(StringValue insertText) {
      repr_->emplace("insertText", insertText);
    }
  } v{repr_};

  std::visit(v, insertText);

  return *this;
}

auto InlineCompletionItem::filterText(std::optional<std::string> filterText)
    -> InlineCompletionItem& {
  if (!filterText.has_value()) {
    repr_->erase("filterText");
    return *this;
  }
  repr_->emplace("filterText", std::move(filterText.value()));
  return *this;
}

auto InlineCompletionItem::range(std::optional<Range> range)
    -> InlineCompletionItem& {
  if (!range.has_value()) {
    repr_->erase("range");
    return *this;
  }
  repr_->emplace("range", range.value());
  return *this;
}

auto InlineCompletionItem::command(std::optional<Command> command)
    -> InlineCompletionItem& {
  if (!command.has_value()) {
    repr_->erase("command");
    return *this;
  }
  repr_->emplace("command", command.value());
  return *this;
}

InlineCompletionRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto InlineCompletionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineCompletionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InlineCompletionRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineCompletionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> InlineCompletionRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto InlineCompletionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> InlineCompletionRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "InlineCompletionRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto InlineCompletionRegistrationOptions::id(std::optional<std::string> id)
    -> InlineCompletionRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

TextDocumentContentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto TextDocumentContentParams::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentParams::uri(std::string uri)
    -> TextDocumentContentParams& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

TextDocumentContentResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("text")) return false;
  return true;
}

auto TextDocumentContentResult::text() const -> std::string {
  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentResult::text(std::string text)
    -> TextDocumentContentResult& {
  repr_->emplace("text", std::move(text));
  return *this;
}

TextDocumentContentRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("schemes")) return false;
  return true;
}

auto TextDocumentContentRegistrationOptions::schemes() const
    -> Vector<std::string> {
  auto& value = (*repr_)["schemes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto TextDocumentContentRegistrationOptions::id() const
    -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentRegistrationOptions::schemes(
    Vector<std::string> schemes) -> TextDocumentContentRegistrationOptions& {
  lsp_runtime_error(
      "TextDocumentContentRegistrationOptions::schemes: not implement yet");
  return *this;
}

auto TextDocumentContentRegistrationOptions::id(std::optional<std::string> id)
    -> TextDocumentContentRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

TextDocumentContentRefreshParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto TextDocumentContentRefreshParams::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentRefreshParams::uri(std::string uri)
    -> TextDocumentContentRefreshParams& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

RegistrationParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("registrations")) return false;
  return true;
}

auto RegistrationParams::registrations() const -> Vector<Registration> {
  auto& value = (*repr_)["registrations"];

  assert(value.is_array());
  return Vector<Registration>(value);
}

auto RegistrationParams::registrations(Vector<Registration> registrations)
    -> RegistrationParams& {
  lsp_runtime_error("RegistrationParams::registrations: not implement yet");
  return *this;
}

UnregistrationParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("unregisterations")) return false;
  return true;
}

auto UnregistrationParams::unregisterations() const -> Vector<Unregistration> {
  auto& value = (*repr_)["unregisterations"];

  assert(value.is_array());
  return Vector<Unregistration>(value);
}

auto UnregistrationParams::unregisterations(
    Vector<Unregistration> unregisterations) -> UnregistrationParams& {
  lsp_runtime_error(
      "UnregistrationParams::unregisterations: not implement yet");
  return *this;
}

InitializeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("processId")) return false;
  if (!repr_->contains("rootUri")) return false;
  if (!repr_->contains("capabilities")) return false;
  return true;
}

auto InitializeParams::processId() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  auto& value = (*repr_)["processId"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::clientInfo() const -> std::optional<ClientInfo> {
  if (!repr_->contains("clientInfo")) return std::nullopt;

  auto& value = (*repr_)["clientInfo"];

  return ClientInfo(value);
}

auto InitializeParams::locale() const -> std::optional<std::string> {
  if (!repr_->contains("locale")) return std::nullopt;

  auto& value = (*repr_)["locale"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InitializeParams::rootPath() const -> std::optional<
    std::variant<std::monostate, std::string, std::nullptr_t>> {
  if (!repr_->contains("rootPath")) return std::nullopt;

  auto& value = (*repr_)["rootPath"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::rootUri() const
    -> std::variant<std::monostate, std::string, std::nullptr_t> {
  auto& value = (*repr_)["rootUri"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::capabilities() const -> ClientCapabilities {
  auto& value = (*repr_)["capabilities"];

  return ClientCapabilities(value);
}

auto InitializeParams::initializationOptions() const -> std::optional<LSPAny> {
  if (!repr_->contains("initializationOptions")) return std::nullopt;

  auto& value = (*repr_)["initializationOptions"];

  assert(value.is_object());
  return LSPAny(value);
}

auto InitializeParams::trace() const -> std::optional<TraceValue> {
  if (!repr_->contains("trace")) return std::nullopt;

  auto& value = (*repr_)["trace"];

  lsp_runtime_error("InitializeParams::trace: not implement yet");
}

auto InitializeParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::workspaceFolders() const -> std::optional<
    std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>> {
  if (!repr_->contains("workspaceFolders")) return std::nullopt;

  auto& value = (*repr_)["workspaceFolders"];

  std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto InitializeParams::processId(
    std::variant<std::monostate, int, std::nullptr_t> processId)
    -> InitializeParams& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int processId) {
      repr_->emplace("processId", std::move(processId));
    }

    void operator()(std::nullptr_t processId) {
      repr_->emplace("processId", std::move(processId));
    }
  } v{repr_};

  std::visit(v, processId);

  return *this;
}

auto InitializeParams::clientInfo(std::optional<ClientInfo> clientInfo)
    -> InitializeParams& {
  if (!clientInfo.has_value()) {
    repr_->erase("clientInfo");
    return *this;
  }
  repr_->emplace("clientInfo", clientInfo.value());
  return *this;
}

auto InitializeParams::locale(std::optional<std::string> locale)
    -> InitializeParams& {
  if (!locale.has_value()) {
    repr_->erase("locale");
    return *this;
  }
  repr_->emplace("locale", std::move(locale.value()));
  return *this;
}

auto InitializeParams::rootPath(
    std::optional<std::variant<std::monostate, std::string, std::nullptr_t>>
        rootPath) -> InitializeParams& {
  if (!rootPath.has_value()) {
    repr_->erase("rootPath");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string rootPath) {
      repr_->emplace("rootPath", std::move(rootPath));
    }

    void operator()(std::nullptr_t rootPath) {
      repr_->emplace("rootPath", std::move(rootPath));
    }
  } v{repr_};

  std::visit(v, rootPath.value());

  return *this;
}

auto InitializeParams::rootUri(
    std::variant<std::monostate, std::string, std::nullptr_t> rootUri)
    -> InitializeParams& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string rootUri) {
      repr_->emplace("rootUri", std::move(rootUri));
    }

    void operator()(std::nullptr_t rootUri) {
      repr_->emplace("rootUri", std::move(rootUri));
    }
  } v{repr_};

  std::visit(v, rootUri);

  return *this;
}

auto InitializeParams::capabilities(ClientCapabilities capabilities)
    -> InitializeParams& {
  repr_->emplace("capabilities", capabilities);
  return *this;
}

auto InitializeParams::initializationOptions(
    std::optional<LSPAny> initializationOptions) -> InitializeParams& {
  if (!initializationOptions.has_value()) {
    repr_->erase("initializationOptions");
    return *this;
  }
  lsp_runtime_error(
      "InitializeParams::initializationOptions: not implement yet");
  return *this;
}

auto InitializeParams::trace(std::optional<TraceValue> trace)
    -> InitializeParams& {
  if (!trace.has_value()) {
    repr_->erase("trace");
    return *this;
  }
  lsp_runtime_error("InitializeParams::trace: not implement yet");
  return *this;
}

auto InitializeParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> InitializeParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("InitializeParams::workDoneToken: not implement yet");
  return *this;
}

auto InitializeParams::workspaceFolders(
    std::optional<
        std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>
        workspaceFolders) -> InitializeParams& {
  if (!workspaceFolders.has_value()) {
    repr_->erase("workspaceFolders");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(Vector<WorkspaceFolder> workspaceFolders) {
      lsp_runtime_error(
          "InitializeParams::workspaceFolders: not implement yet");
    }

    void operator()(std::nullptr_t workspaceFolders) {
      repr_->emplace("workspaceFolders", std::move(workspaceFolders));
    }
  } v{repr_};

  std::visit(v, workspaceFolders.value());

  return *this;
}

InitializeResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("capabilities")) return false;
  return true;
}

auto InitializeResult::capabilities() const -> ServerCapabilities {
  auto& value = (*repr_)["capabilities"];

  return ServerCapabilities(value);
}

auto InitializeResult::serverInfo() const -> std::optional<ServerInfo> {
  if (!repr_->contains("serverInfo")) return std::nullopt;

  auto& value = (*repr_)["serverInfo"];

  return ServerInfo(value);
}

auto InitializeResult::capabilities(ServerCapabilities capabilities)
    -> InitializeResult& {
  repr_->emplace("capabilities", capabilities);
  return *this;
}

auto InitializeResult::serverInfo(std::optional<ServerInfo> serverInfo)
    -> InitializeResult& {
  if (!serverInfo.has_value()) {
    repr_->erase("serverInfo");
    return *this;
  }
  repr_->emplace("serverInfo", serverInfo.value());
  return *this;
}

InitializeError::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("retry")) return false;
  return true;
}

auto InitializeError::retry() const -> bool {
  auto& value = (*repr_)["retry"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InitializeError::retry(bool retry) -> InitializeError& {
  repr_->emplace("retry", std::move(retry));
  return *this;
}

InitializedParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

DidChangeConfigurationParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("settings")) return false;
  return true;
}

auto DidChangeConfigurationParams::settings() const -> LSPAny {
  auto& value = (*repr_)["settings"];

  assert(value.is_object());
  return LSPAny(value);
}

auto DidChangeConfigurationParams::settings(LSPAny settings)
    -> DidChangeConfigurationParams& {
  lsp_runtime_error(
      "DidChangeConfigurationParams::settings: not implement yet");
  return *this;
}

DidChangeConfigurationRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DidChangeConfigurationRegistrationOptions::section() const
    -> std::optional<
        std::variant<std::monostate, std::string, Vector<std::string>>> {
  if (!repr_->contains("section")) return std::nullopt;

  auto& value = (*repr_)["section"];

  std::variant<std::monostate, std::string, Vector<std::string>> result;

  details::try_emplace(result, value);

  return result;
}

auto DidChangeConfigurationRegistrationOptions::section(
    std::optional<
        std::variant<std::monostate, std::string, Vector<std::string>>>
        section) -> DidChangeConfigurationRegistrationOptions& {
  if (!section.has_value()) {
    repr_->erase("section");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string section) {
      repr_->emplace("section", std::move(section));
    }

    void operator()(Vector<std::string> section) {
      lsp_runtime_error(
          "DidChangeConfigurationRegistrationOptions::section: not implement "
          "yet");
    }
  } v{repr_};

  std::visit(v, section.value());

  return *this;
}

ShowMessageParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("type")) return false;
  if (!repr_->contains("message")) return false;
  return true;
}

auto ShowMessageParams::type() const -> MessageType {
  auto& value = (*repr_)["type"];

  return MessageType(value);
}

auto ShowMessageParams::message() const -> std::string {
  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ShowMessageParams::type(MessageType type) -> ShowMessageParams& {
  repr_->emplace("type", static_cast<long>(type));
  return *this;
}

auto ShowMessageParams::message(std::string message) -> ShowMessageParams& {
  repr_->emplace("message", std::move(message));
  return *this;
}

ShowMessageRequestParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("type")) return false;
  if (!repr_->contains("message")) return false;
  return true;
}

auto ShowMessageRequestParams::type() const -> MessageType {
  auto& value = (*repr_)["type"];

  return MessageType(value);
}

auto ShowMessageRequestParams::message() const -> std::string {
  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ShowMessageRequestParams::actions() const
    -> std::optional<Vector<MessageActionItem>> {
  if (!repr_->contains("actions")) return std::nullopt;

  auto& value = (*repr_)["actions"];

  assert(value.is_array());
  return Vector<MessageActionItem>(value);
}

auto ShowMessageRequestParams::type(MessageType type)
    -> ShowMessageRequestParams& {
  repr_->emplace("type", static_cast<long>(type));
  return *this;
}

auto ShowMessageRequestParams::message(std::string message)
    -> ShowMessageRequestParams& {
  repr_->emplace("message", std::move(message));
  return *this;
}

auto ShowMessageRequestParams::actions(
    std::optional<Vector<MessageActionItem>> actions)
    -> ShowMessageRequestParams& {
  if (!actions.has_value()) {
    repr_->erase("actions");
    return *this;
  }
  lsp_runtime_error("ShowMessageRequestParams::actions: not implement yet");
  return *this;
}

MessageActionItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("title")) return false;
  return true;
}

auto MessageActionItem::title() const -> std::string {
  auto& value = (*repr_)["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MessageActionItem::title(std::string title) -> MessageActionItem& {
  repr_->emplace("title", std::move(title));
  return *this;
}

LogMessageParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("type")) return false;
  if (!repr_->contains("message")) return false;
  return true;
}

auto LogMessageParams::type() const -> MessageType {
  auto& value = (*repr_)["type"];

  return MessageType(value);
}

auto LogMessageParams::message() const -> std::string {
  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LogMessageParams::type(MessageType type) -> LogMessageParams& {
  repr_->emplace("type", static_cast<long>(type));
  return *this;
}

auto LogMessageParams::message(std::string message) -> LogMessageParams& {
  repr_->emplace("message", std::move(message));
  return *this;
}

DidOpenTextDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DidOpenTextDocumentParams::textDocument() const -> TextDocumentItem {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentItem(value);
}

auto DidOpenTextDocumentParams::textDocument(TextDocumentItem textDocument)
    -> DidOpenTextDocumentParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

DidChangeTextDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("contentChanges")) return false;
  return true;
}

auto DidChangeTextDocumentParams::textDocument() const
    -> VersionedTextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return VersionedTextDocumentIdentifier(value);
}

auto DidChangeTextDocumentParams::contentChanges() const
    -> Vector<TextDocumentContentChangeEvent> {
  auto& value = (*repr_)["contentChanges"];

  assert(value.is_array());
  return Vector<TextDocumentContentChangeEvent>(value);
}

auto DidChangeTextDocumentParams::textDocument(
    VersionedTextDocumentIdentifier textDocument)
    -> DidChangeTextDocumentParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DidChangeTextDocumentParams::contentChanges(
    Vector<TextDocumentContentChangeEvent> contentChanges)
    -> DidChangeTextDocumentParams& {
  lsp_runtime_error(
      "DidChangeTextDocumentParams::contentChanges: not implement yet");
  return *this;
}

TextDocumentChangeRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("syncKind")) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto TextDocumentChangeRegistrationOptions::syncKind() const
    -> TextDocumentSyncKind {
  auto& value = (*repr_)["syncKind"];

  return TextDocumentSyncKind(value);
}

auto TextDocumentChangeRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentChangeRegistrationOptions::syncKind(
    TextDocumentSyncKind syncKind) -> TextDocumentChangeRegistrationOptions& {
  repr_->emplace("syncKind", static_cast<long>(syncKind));
  return *this;
}

auto TextDocumentChangeRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TextDocumentChangeRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "TextDocumentChangeRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

DidCloseTextDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DidCloseTextDocumentParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DidCloseTextDocumentParams::textDocument(
    TextDocumentIdentifier textDocument) -> DidCloseTextDocumentParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

DidSaveTextDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DidSaveTextDocumentParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DidSaveTextDocumentParams::text() const -> std::optional<std::string> {
  if (!repr_->contains("text")) return std::nullopt;

  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DidSaveTextDocumentParams::textDocument(
    TextDocumentIdentifier textDocument) -> DidSaveTextDocumentParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DidSaveTextDocumentParams::text(std::optional<std::string> text)
    -> DidSaveTextDocumentParams& {
  if (!text.has_value()) {
    repr_->erase("text");
    return *this;
  }
  repr_->emplace("text", std::move(text.value()));
  return *this;
}

TextDocumentSaveRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto TextDocumentSaveRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentSaveRegistrationOptions::includeText() const
    -> std::optional<bool> {
  if (!repr_->contains("includeText")) return std::nullopt;

  auto& value = (*repr_)["includeText"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSaveRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> TextDocumentSaveRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "TextDocumentSaveRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto TextDocumentSaveRegistrationOptions::includeText(
    std::optional<bool> includeText) -> TextDocumentSaveRegistrationOptions& {
  if (!includeText.has_value()) {
    repr_->erase("includeText");
    return *this;
  }
  repr_->emplace("includeText", std::move(includeText.value()));
  return *this;
}

WillSaveTextDocumentParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("reason")) return false;
  return true;
}

auto WillSaveTextDocumentParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto WillSaveTextDocumentParams::reason() const -> TextDocumentSaveReason {
  auto& value = (*repr_)["reason"];

  return TextDocumentSaveReason(value);
}

auto WillSaveTextDocumentParams::textDocument(
    TextDocumentIdentifier textDocument) -> WillSaveTextDocumentParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto WillSaveTextDocumentParams::reason(TextDocumentSaveReason reason)
    -> WillSaveTextDocumentParams& {
  repr_->emplace("reason", static_cast<long>(reason));
  return *this;
}

TextEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("newText")) return false;
  return true;
}

auto TextEdit::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto TextEdit::newText() const -> std::string {
  auto& value = (*repr_)["newText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextEdit::range(Range range) -> TextEdit& {
  repr_->emplace("range", range);
  return *this;
}

auto TextEdit::newText(std::string newText) -> TextEdit& {
  repr_->emplace("newText", std::move(newText));
  return *this;
}

DidChangeWatchedFilesParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("changes")) return false;
  return true;
}

auto DidChangeWatchedFilesParams::changes() const -> Vector<FileEvent> {
  auto& value = (*repr_)["changes"];

  assert(value.is_array());
  return Vector<FileEvent>(value);
}

auto DidChangeWatchedFilesParams::changes(Vector<FileEvent> changes)
    -> DidChangeWatchedFilesParams& {
  lsp_runtime_error("DidChangeWatchedFilesParams::changes: not implement yet");
  return *this;
}

DidChangeWatchedFilesRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("watchers")) return false;
  return true;
}

auto DidChangeWatchedFilesRegistrationOptions::watchers() const
    -> Vector<FileSystemWatcher> {
  auto& value = (*repr_)["watchers"];

  assert(value.is_array());
  return Vector<FileSystemWatcher>(value);
}

auto DidChangeWatchedFilesRegistrationOptions::watchers(
    Vector<FileSystemWatcher> watchers)
    -> DidChangeWatchedFilesRegistrationOptions& {
  lsp_runtime_error(
      "DidChangeWatchedFilesRegistrationOptions::watchers: not implement yet");
  return *this;
}

PublishDiagnosticsParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("diagnostics")) return false;
  return true;
}

auto PublishDiagnosticsParams::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PublishDiagnosticsParams::version() const -> std::optional<int> {
  if (!repr_->contains("version")) return std::nullopt;

  auto& value = (*repr_)["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto PublishDiagnosticsParams::diagnostics() const -> Vector<Diagnostic> {
  auto& value = (*repr_)["diagnostics"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto PublishDiagnosticsParams::uri(std::string uri)
    -> PublishDiagnosticsParams& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto PublishDiagnosticsParams::version(std::optional<int> version)
    -> PublishDiagnosticsParams& {
  if (!version.has_value()) {
    repr_->erase("version");
    return *this;
  }
  repr_->emplace("version", std::move(version.value()));
  return *this;
}

auto PublishDiagnosticsParams::diagnostics(Vector<Diagnostic> diagnostics)
    -> PublishDiagnosticsParams& {
  lsp_runtime_error("PublishDiagnosticsParams::diagnostics: not implement yet");
  return *this;
}

CompletionParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto CompletionParams::context() const -> std::optional<CompletionContext> {
  if (!repr_->contains("context")) return std::nullopt;

  auto& value = (*repr_)["context"];

  return CompletionContext(value);
}

auto CompletionParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CompletionParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto CompletionParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionParams::context(std::optional<CompletionContext> context)
    -> CompletionParams& {
  if (!context.has_value()) {
    repr_->erase("context");
    return *this;
  }
  repr_->emplace("context", context.value());
  return *this;
}

auto CompletionParams::textDocument(TextDocumentIdentifier textDocument)
    -> CompletionParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto CompletionParams::position(Position position) -> CompletionParams& {
  repr_->emplace("position", position);
  return *this;
}

auto CompletionParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> CompletionParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("CompletionParams::workDoneToken: not implement yet");
  return *this;
}

auto CompletionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> CompletionParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("CompletionParams::partialResultToken: not implement yet");
  return *this;
}

CompletionItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("label")) return false;
  return true;
}

auto CompletionItem::label() const -> std::string {
  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::labelDetails() const
    -> std::optional<CompletionItemLabelDetails> {
  if (!repr_->contains("labelDetails")) return std::nullopt;

  auto& value = (*repr_)["labelDetails"];

  return CompletionItemLabelDetails(value);
}

auto CompletionItem::kind() const -> std::optional<CompletionItemKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  return CompletionItemKind(value);
}

auto CompletionItem::tags() const -> std::optional<Vector<CompletionItemTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<CompletionItemTag>(value);
}

auto CompletionItem::detail() const -> std::optional<std::string> {
  if (!repr_->contains("detail")) return std::nullopt;

  auto& value = (*repr_)["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::documentation() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_->contains("documentation")) return std::nullopt;

  auto& value = (*repr_)["documentation"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionItem::deprecated() const -> std::optional<bool> {
  if (!repr_->contains("deprecated")) return std::nullopt;

  auto& value = (*repr_)["deprecated"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionItem::preselect() const -> std::optional<bool> {
  if (!repr_->contains("preselect")) return std::nullopt;

  auto& value = (*repr_)["preselect"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionItem::sortText() const -> std::optional<std::string> {
  if (!repr_->contains("sortText")) return std::nullopt;

  auto& value = (*repr_)["sortText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::filterText() const -> std::optional<std::string> {
  if (!repr_->contains("filterText")) return std::nullopt;

  auto& value = (*repr_)["filterText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::insertText() const -> std::optional<std::string> {
  if (!repr_->contains("insertText")) return std::nullopt;

  auto& value = (*repr_)["insertText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::insertTextFormat() const
    -> std::optional<InsertTextFormat> {
  if (!repr_->contains("insertTextFormat")) return std::nullopt;

  auto& value = (*repr_)["insertTextFormat"];

  return InsertTextFormat(value);
}

auto CompletionItem::insertTextMode() const -> std::optional<InsertTextMode> {
  if (!repr_->contains("insertTextMode")) return std::nullopt;

  auto& value = (*repr_)["insertTextMode"];

  return InsertTextMode(value);
}

auto CompletionItem::textEdit() const -> std::optional<
    std::variant<std::monostate, TextEdit, InsertReplaceEdit>> {
  if (!repr_->contains("textEdit")) return std::nullopt;

  auto& value = (*repr_)["textEdit"];

  std::variant<std::monostate, TextEdit, InsertReplaceEdit> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionItem::textEditText() const -> std::optional<std::string> {
  if (!repr_->contains("textEditText")) return std::nullopt;

  auto& value = (*repr_)["textEditText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItem::additionalTextEdits() const
    -> std::optional<Vector<TextEdit>> {
  if (!repr_->contains("additionalTextEdits")) return std::nullopt;

  auto& value = (*repr_)["additionalTextEdits"];

  assert(value.is_array());
  return Vector<TextEdit>(value);
}

auto CompletionItem::commitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("commitCharacters")) return std::nullopt;

  auto& value = (*repr_)["commitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionItem::command() const -> std::optional<Command> {
  if (!repr_->contains("command")) return std::nullopt;

  auto& value = (*repr_)["command"];

  return Command(value);
}

auto CompletionItem::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CompletionItem::label(std::string label) -> CompletionItem& {
  repr_->emplace("label", std::move(label));
  return *this;
}

auto CompletionItem::labelDetails(
    std::optional<CompletionItemLabelDetails> labelDetails) -> CompletionItem& {
  if (!labelDetails.has_value()) {
    repr_->erase("labelDetails");
    return *this;
  }
  repr_->emplace("labelDetails", labelDetails.value());
  return *this;
}

auto CompletionItem::kind(std::optional<CompletionItemKind> kind)
    -> CompletionItem& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  repr_->emplace("kind", static_cast<long>(kind.value()));
  return *this;
}

auto CompletionItem::tags(std::optional<Vector<CompletionItemTag>> tags)
    -> CompletionItem& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("CompletionItem::tags: not implement yet");
  return *this;
}

auto CompletionItem::detail(std::optional<std::string> detail)
    -> CompletionItem& {
  if (!detail.has_value()) {
    repr_->erase("detail");
    return *this;
  }
  repr_->emplace("detail", std::move(detail.value()));
  return *this;
}

auto CompletionItem::documentation(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        documentation) -> CompletionItem& {
  if (!documentation.has_value()) {
    repr_->erase("documentation");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string documentation) {
      repr_->emplace("documentation", std::move(documentation));
    }

    void operator()(MarkupContent documentation) {
      repr_->emplace("documentation", documentation);
    }
  } v{repr_};

  std::visit(v, documentation.value());

  return *this;
}

auto CompletionItem::deprecated(std::optional<bool> deprecated)
    -> CompletionItem& {
  if (!deprecated.has_value()) {
    repr_->erase("deprecated");
    return *this;
  }
  repr_->emplace("deprecated", std::move(deprecated.value()));
  return *this;
}

auto CompletionItem::preselect(std::optional<bool> preselect)
    -> CompletionItem& {
  if (!preselect.has_value()) {
    repr_->erase("preselect");
    return *this;
  }
  repr_->emplace("preselect", std::move(preselect.value()));
  return *this;
}

auto CompletionItem::sortText(std::optional<std::string> sortText)
    -> CompletionItem& {
  if (!sortText.has_value()) {
    repr_->erase("sortText");
    return *this;
  }
  repr_->emplace("sortText", std::move(sortText.value()));
  return *this;
}

auto CompletionItem::filterText(std::optional<std::string> filterText)
    -> CompletionItem& {
  if (!filterText.has_value()) {
    repr_->erase("filterText");
    return *this;
  }
  repr_->emplace("filterText", std::move(filterText.value()));
  return *this;
}

auto CompletionItem::insertText(std::optional<std::string> insertText)
    -> CompletionItem& {
  if (!insertText.has_value()) {
    repr_->erase("insertText");
    return *this;
  }
  repr_->emplace("insertText", std::move(insertText.value()));
  return *this;
}

auto CompletionItem::insertTextFormat(
    std::optional<InsertTextFormat> insertTextFormat) -> CompletionItem& {
  if (!insertTextFormat.has_value()) {
    repr_->erase("insertTextFormat");
    return *this;
  }
  repr_->emplace("insertTextFormat",
                 static_cast<long>(insertTextFormat.value()));
  return *this;
}

auto CompletionItem::insertTextMode(
    std::optional<InsertTextMode> insertTextMode) -> CompletionItem& {
  if (!insertTextMode.has_value()) {
    repr_->erase("insertTextMode");
    return *this;
  }
  repr_->emplace("insertTextMode", static_cast<long>(insertTextMode.value()));
  return *this;
}

auto CompletionItem::textEdit(
    std::optional<std::variant<std::monostate, TextEdit, InsertReplaceEdit>>
        textEdit) -> CompletionItem& {
  if (!textEdit.has_value()) {
    repr_->erase("textEdit");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(TextEdit textEdit) { repr_->emplace("textEdit", textEdit); }

    void operator()(InsertReplaceEdit textEdit) {
      repr_->emplace("textEdit", textEdit);
    }
  } v{repr_};

  std::visit(v, textEdit.value());

  return *this;
}

auto CompletionItem::textEditText(std::optional<std::string> textEditText)
    -> CompletionItem& {
  if (!textEditText.has_value()) {
    repr_->erase("textEditText");
    return *this;
  }
  repr_->emplace("textEditText", std::move(textEditText.value()));
  return *this;
}

auto CompletionItem::additionalTextEdits(
    std::optional<Vector<TextEdit>> additionalTextEdits) -> CompletionItem& {
  if (!additionalTextEdits.has_value()) {
    repr_->erase("additionalTextEdits");
    return *this;
  }
  lsp_runtime_error("CompletionItem::additionalTextEdits: not implement yet");
  return *this;
}

auto CompletionItem::commitCharacters(
    std::optional<Vector<std::string>> commitCharacters) -> CompletionItem& {
  if (!commitCharacters.has_value()) {
    repr_->erase("commitCharacters");
    return *this;
  }
  lsp_runtime_error("CompletionItem::commitCharacters: not implement yet");
  return *this;
}

auto CompletionItem::command(std::optional<Command> command)
    -> CompletionItem& {
  if (!command.has_value()) {
    repr_->erase("command");
    return *this;
  }
  repr_->emplace("command", command.value());
  return *this;
}

auto CompletionItem::data(std::optional<LSPAny> data) -> CompletionItem& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("CompletionItem::data: not implement yet");
  return *this;
}

CompletionList::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("isIncomplete")) return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto CompletionList::isIncomplete() const -> bool {
  auto& value = (*repr_)["isIncomplete"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionList::itemDefaults() const
    -> std::optional<CompletionItemDefaults> {
  if (!repr_->contains("itemDefaults")) return std::nullopt;

  auto& value = (*repr_)["itemDefaults"];

  return CompletionItemDefaults(value);
}

auto CompletionList::applyKind() const
    -> std::optional<CompletionItemApplyKinds> {
  if (!repr_->contains("applyKind")) return std::nullopt;

  auto& value = (*repr_)["applyKind"];

  return CompletionItemApplyKinds(value);
}

auto CompletionList::items() const -> Vector<CompletionItem> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<CompletionItem>(value);
}

auto CompletionList::isIncomplete(bool isIncomplete) -> CompletionList& {
  repr_->emplace("isIncomplete", std::move(isIncomplete));
  return *this;
}

auto CompletionList::itemDefaults(
    std::optional<CompletionItemDefaults> itemDefaults) -> CompletionList& {
  if (!itemDefaults.has_value()) {
    repr_->erase("itemDefaults");
    return *this;
  }
  repr_->emplace("itemDefaults", itemDefaults.value());
  return *this;
}

auto CompletionList::applyKind(
    std::optional<CompletionItemApplyKinds> applyKind) -> CompletionList& {
  if (!applyKind.has_value()) {
    repr_->erase("applyKind");
    return *this;
  }
  repr_->emplace("applyKind", applyKind.value());
  return *this;
}

auto CompletionList::items(Vector<CompletionItem> items) -> CompletionList& {
  lsp_runtime_error("CompletionList::items: not implement yet");
  return *this;
}

CompletionRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto CompletionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionRegistrationOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("triggerCharacters")) return std::nullopt;

  auto& value = (*repr_)["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionRegistrationOptions::allCommitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("allCommitCharacters")) return std::nullopt;

  auto& value = (*repr_)["allCommitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionRegistrationOptions::completionItem() const
    -> std::optional<ServerCompletionItemOptions> {
  if (!repr_->contains("completionItem")) return std::nullopt;

  auto& value = (*repr_)["completionItem"];

  return ServerCompletionItemOptions(value);
}

auto CompletionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CompletionRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "CompletionRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto CompletionRegistrationOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> CompletionRegistrationOptions& {
  if (!triggerCharacters.has_value()) {
    repr_->erase("triggerCharacters");
    return *this;
  }
  lsp_runtime_error(
      "CompletionRegistrationOptions::triggerCharacters: not implement yet");
  return *this;
}

auto CompletionRegistrationOptions::allCommitCharacters(
    std::optional<Vector<std::string>> allCommitCharacters)
    -> CompletionRegistrationOptions& {
  if (!allCommitCharacters.has_value()) {
    repr_->erase("allCommitCharacters");
    return *this;
  }
  lsp_runtime_error(
      "CompletionRegistrationOptions::allCommitCharacters: not implement yet");
  return *this;
}

auto CompletionRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> CompletionRegistrationOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto CompletionRegistrationOptions::completionItem(
    std::optional<ServerCompletionItemOptions> completionItem)
    -> CompletionRegistrationOptions& {
  if (!completionItem.has_value()) {
    repr_->erase("completionItem");
    return *this;
  }
  repr_->emplace("completionItem", completionItem.value());
  return *this;
}

auto CompletionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CompletionRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

HoverParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto HoverParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto HoverParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto HoverParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto HoverParams::textDocument(TextDocumentIdentifier textDocument)
    -> HoverParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto HoverParams::position(Position position) -> HoverParams& {
  repr_->emplace("position", position);
  return *this;
}

auto HoverParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> HoverParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("HoverParams::workDoneToken: not implement yet");
  return *this;
}

Hover::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("contents")) return false;
  return true;
}

auto Hover::contents() const
    -> std::variant<std::monostate, MarkupContent, MarkedString,
                    Vector<MarkedString>> {
  auto& value = (*repr_)["contents"];

  std::variant<std::monostate, MarkupContent, MarkedString,
               Vector<MarkedString>>
      result;

  details::try_emplace(result, value);

  return result;
}

auto Hover::range() const -> std::optional<Range> {
  if (!repr_->contains("range")) return std::nullopt;

  auto& value = (*repr_)["range"];

  return Range(value);
}

auto Hover::contents(std::variant<std::monostate, MarkupContent, MarkedString,
                                  Vector<MarkedString>>
                         contents) -> Hover& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(MarkupContent contents) {
      repr_->emplace("contents", contents);
    }

    void operator()(MarkedString contents) {
      lsp_runtime_error("Hover::contents: not implement yet");
    }

    void operator()(Vector<MarkedString> contents) {
      lsp_runtime_error("Hover::contents: not implement yet");
    }
  } v{repr_};

  std::visit(v, contents);

  return *this;
}

auto Hover::range(std::optional<Range> range) -> Hover& {
  if (!range.has_value()) {
    repr_->erase("range");
    return *this;
  }
  repr_->emplace("range", range.value());
  return *this;
}

HoverRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto HoverRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto HoverRegistrationOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto HoverRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> HoverRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "HoverRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto HoverRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> HoverRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

SignatureHelpParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto SignatureHelpParams::context() const
    -> std::optional<SignatureHelpContext> {
  if (!repr_->contains("context")) return std::nullopt;

  auto& value = (*repr_)["context"];

  return SignatureHelpContext(value);
}

auto SignatureHelpParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto SignatureHelpParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto SignatureHelpParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelpParams::context(std::optional<SignatureHelpContext> context)
    -> SignatureHelpParams& {
  if (!context.has_value()) {
    repr_->erase("context");
    return *this;
  }
  repr_->emplace("context", context.value());
  return *this;
}

auto SignatureHelpParams::textDocument(TextDocumentIdentifier textDocument)
    -> SignatureHelpParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto SignatureHelpParams::position(Position position) -> SignatureHelpParams& {
  repr_->emplace("position", position);
  return *this;
}

auto SignatureHelpParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> SignatureHelpParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("SignatureHelpParams::workDoneToken: not implement yet");
  return *this;
}

SignatureHelp::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("signatures")) return false;
  return true;
}

auto SignatureHelp::signatures() const -> Vector<SignatureInformation> {
  auto& value = (*repr_)["signatures"];

  assert(value.is_array());
  return Vector<SignatureInformation>(value);
}

auto SignatureHelp::activeSignature() const -> std::optional<long> {
  if (!repr_->contains("activeSignature")) return std::nullopt;

  auto& value = (*repr_)["activeSignature"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto SignatureHelp::activeParameter() const
    -> std::optional<std::variant<std::monostate, long, std::nullptr_t>> {
  if (!repr_->contains("activeParameter")) return std::nullopt;

  auto& value = (*repr_)["activeParameter"];

  std::variant<std::monostate, long, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelp::signatures(Vector<SignatureInformation> signatures)
    -> SignatureHelp& {
  lsp_runtime_error("SignatureHelp::signatures: not implement yet");
  return *this;
}

auto SignatureHelp::activeSignature(std::optional<long> activeSignature)
    -> SignatureHelp& {
  if (!activeSignature.has_value()) {
    repr_->erase("activeSignature");
    return *this;
  }
  repr_->emplace("activeSignature", std::move(activeSignature.value()));
  return *this;
}

auto SignatureHelp::activeParameter(
    std::optional<std::variant<std::monostate, long, std::nullptr_t>>
        activeParameter) -> SignatureHelp& {
  if (!activeParameter.has_value()) {
    repr_->erase("activeParameter");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(long activeParameter) {
      repr_->emplace("activeParameter", std::move(activeParameter));
    }

    void operator()(std::nullptr_t activeParameter) {
      repr_->emplace("activeParameter", std::move(activeParameter));
    }
  } v{repr_};

  std::visit(v, activeParameter.value());

  return *this;
}

SignatureHelpRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto SignatureHelpRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureHelpRegistrationOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("triggerCharacters")) return std::nullopt;

  auto& value = (*repr_)["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpRegistrationOptions::retriggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("retriggerCharacters")) return std::nullopt;

  auto& value = (*repr_)["retriggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> SignatureHelpRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "SignatureHelpRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto SignatureHelpRegistrationOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> SignatureHelpRegistrationOptions& {
  if (!triggerCharacters.has_value()) {
    repr_->erase("triggerCharacters");
    return *this;
  }
  lsp_runtime_error(
      "SignatureHelpRegistrationOptions::triggerCharacters: not implement yet");
  return *this;
}

auto SignatureHelpRegistrationOptions::retriggerCharacters(
    std::optional<Vector<std::string>> retriggerCharacters)
    -> SignatureHelpRegistrationOptions& {
  if (!retriggerCharacters.has_value()) {
    repr_->erase("retriggerCharacters");
    return *this;
  }
  lsp_runtime_error(
      "SignatureHelpRegistrationOptions::retriggerCharacters: not implement "
      "yet");
  return *this;
}

auto SignatureHelpRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SignatureHelpRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DefinitionParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto DefinitionParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DefinitionParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto DefinitionParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionParams::textDocument(TextDocumentIdentifier textDocument)
    -> DefinitionParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DefinitionParams::position(Position position) -> DefinitionParams& {
  repr_->emplace("position", position);
  return *this;
}

auto DefinitionParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> DefinitionParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("DefinitionParams::workDoneToken: not implement yet");
  return *this;
}

auto DefinitionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DefinitionParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("DefinitionParams::partialResultToken: not implement yet");
  return *this;
}

DefinitionRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DefinitionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DefinitionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DefinitionRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DefinitionRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DefinitionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DefinitionRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

ReferenceParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("context")) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto ReferenceParams::context() const -> ReferenceContext {
  auto& value = (*repr_)["context"];

  return ReferenceContext(value);
}

auto ReferenceParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto ReferenceParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto ReferenceParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ReferenceParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ReferenceParams::context(ReferenceContext context) -> ReferenceParams& {
  repr_->emplace("context", context);
  return *this;
}

auto ReferenceParams::textDocument(TextDocumentIdentifier textDocument)
    -> ReferenceParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto ReferenceParams::position(Position position) -> ReferenceParams& {
  repr_->emplace("position", position);
  return *this;
}

auto ReferenceParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> ReferenceParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("ReferenceParams::workDoneToken: not implement yet");
  return *this;
}

auto ReferenceParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> ReferenceParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("ReferenceParams::partialResultToken: not implement yet");
  return *this;
}

ReferenceRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto ReferenceRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto ReferenceRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> ReferenceRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "ReferenceRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto ReferenceRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> ReferenceRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentHighlightParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto DocumentHighlightParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentHighlightParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto DocumentHighlightParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentHighlightParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentHighlightParams::position(Position position)
    -> DocumentHighlightParams& {
  repr_->emplace("position", position);
  return *this;
}

auto DocumentHighlightParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentHighlightParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentHighlightParams::workDoneToken: not implement yet");
  return *this;
}

auto DocumentHighlightParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken)
    -> DocumentHighlightParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentHighlightParams::partialResultToken: not implement yet");
  return *this;
}

DocumentHighlight::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto DocumentHighlight::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto DocumentHighlight::kind() const -> std::optional<DocumentHighlightKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  return DocumentHighlightKind(value);
}

auto DocumentHighlight::range(Range range) -> DocumentHighlight& {
  repr_->emplace("range", range);
  return *this;
}

auto DocumentHighlight::kind(std::optional<DocumentHighlightKind> kind)
    -> DocumentHighlight& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  repr_->emplace("kind", static_cast<long>(kind.value()));
  return *this;
}

DocumentHighlightRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DocumentHighlightRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentHighlightRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentHighlightRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentHighlightRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentHighlightRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentHighlightRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentHighlightRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentSymbolParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DocumentSymbolParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentSymbolParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentSymbolParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentSymbolParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentSymbolParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("DocumentSymbolParams::workDoneToken: not implement yet");
  return *this;
}

auto DocumentSymbolParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DocumentSymbolParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentSymbolParams::partialResultToken: not implement yet");
  return *this;
}

SymbolInformation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("location")) return false;
  if (!repr_->contains("name")) return false;
  if (!repr_->contains("kind")) return false;
  return true;
}

auto SymbolInformation::deprecated() const -> std::optional<bool> {
  if (!repr_->contains("deprecated")) return std::nullopt;

  auto& value = (*repr_)["deprecated"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SymbolInformation::location() const -> Location {
  auto& value = (*repr_)["location"];

  return Location(value);
}

auto SymbolInformation::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SymbolInformation::kind() const -> SymbolKind {
  auto& value = (*repr_)["kind"];

  return SymbolKind(value);
}

auto SymbolInformation::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto SymbolInformation::containerName() const -> std::optional<std::string> {
  if (!repr_->contains("containerName")) return std::nullopt;

  auto& value = (*repr_)["containerName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SymbolInformation::deprecated(std::optional<bool> deprecated)
    -> SymbolInformation& {
  if (!deprecated.has_value()) {
    repr_->erase("deprecated");
    return *this;
  }
  repr_->emplace("deprecated", std::move(deprecated.value()));
  return *this;
}

auto SymbolInformation::location(Location location) -> SymbolInformation& {
  repr_->emplace("location", location);
  return *this;
}

auto SymbolInformation::name(std::string name) -> SymbolInformation& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto SymbolInformation::kind(SymbolKind kind) -> SymbolInformation& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto SymbolInformation::tags(std::optional<Vector<SymbolTag>> tags)
    -> SymbolInformation& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("SymbolInformation::tags: not implement yet");
  return *this;
}

auto SymbolInformation::containerName(std::optional<std::string> containerName)
    -> SymbolInformation& {
  if (!containerName.has_value()) {
    repr_->erase("containerName");
    return *this;
  }
  repr_->emplace("containerName", std::move(containerName.value()));
  return *this;
}

DocumentSymbol::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("name")) return false;
  if (!repr_->contains("kind")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("selectionRange")) return false;
  return true;
}

auto DocumentSymbol::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbol::detail() const -> std::optional<std::string> {
  if (!repr_->contains("detail")) return std::nullopt;

  auto& value = (*repr_)["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbol::kind() const -> SymbolKind {
  auto& value = (*repr_)["kind"];

  return SymbolKind(value);
}

auto DocumentSymbol::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto DocumentSymbol::deprecated() const -> std::optional<bool> {
  if (!repr_->contains("deprecated")) return std::nullopt;

  auto& value = (*repr_)["deprecated"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbol::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto DocumentSymbol::selectionRange() const -> Range {
  auto& value = (*repr_)["selectionRange"];

  return Range(value);
}

auto DocumentSymbol::children() const -> std::optional<Vector<DocumentSymbol>> {
  if (!repr_->contains("children")) return std::nullopt;

  auto& value = (*repr_)["children"];

  assert(value.is_array());
  return Vector<DocumentSymbol>(value);
}

auto DocumentSymbol::name(std::string name) -> DocumentSymbol& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto DocumentSymbol::detail(std::optional<std::string> detail)
    -> DocumentSymbol& {
  if (!detail.has_value()) {
    repr_->erase("detail");
    return *this;
  }
  repr_->emplace("detail", std::move(detail.value()));
  return *this;
}

auto DocumentSymbol::kind(SymbolKind kind) -> DocumentSymbol& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto DocumentSymbol::tags(std::optional<Vector<SymbolTag>> tags)
    -> DocumentSymbol& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("DocumentSymbol::tags: not implement yet");
  return *this;
}

auto DocumentSymbol::deprecated(std::optional<bool> deprecated)
    -> DocumentSymbol& {
  if (!deprecated.has_value()) {
    repr_->erase("deprecated");
    return *this;
  }
  repr_->emplace("deprecated", std::move(deprecated.value()));
  return *this;
}

auto DocumentSymbol::range(Range range) -> DocumentSymbol& {
  repr_->emplace("range", range);
  return *this;
}

auto DocumentSymbol::selectionRange(Range selectionRange) -> DocumentSymbol& {
  repr_->emplace("selectionRange", selectionRange);
  return *this;
}

auto DocumentSymbol::children(std::optional<Vector<DocumentSymbol>> children)
    -> DocumentSymbol& {
  if (!children.has_value()) {
    repr_->erase("children");
    return *this;
  }
  lsp_runtime_error("DocumentSymbol::children: not implement yet");
  return *this;
}

DocumentSymbolRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DocumentSymbolRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentSymbolRegistrationOptions::label() const
    -> std::optional<std::string> {
  if (!repr_->contains("label")) return std::nullopt;

  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbolRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentSymbolRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentSymbolRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentSymbolRegistrationOptions::label(std::optional<std::string> label)
    -> DocumentSymbolRegistrationOptions& {
  if (!label.has_value()) {
    repr_->erase("label");
    return *this;
  }
  repr_->emplace("label", std::move(label.value()));
  return *this;
}

auto DocumentSymbolRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentSymbolRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

CodeActionParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("context")) return false;
  return true;
}

auto CodeActionParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CodeActionParams::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto CodeActionParams::context() const -> CodeActionContext {
  auto& value = (*repr_)["context"];

  return CodeActionContext(value);
}

auto CodeActionParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionParams::textDocument(TextDocumentIdentifier textDocument)
    -> CodeActionParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto CodeActionParams::range(Range range) -> CodeActionParams& {
  repr_->emplace("range", range);
  return *this;
}

auto CodeActionParams::context(CodeActionContext context) -> CodeActionParams& {
  repr_->emplace("context", context);
  return *this;
}

auto CodeActionParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> CodeActionParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("CodeActionParams::workDoneToken: not implement yet");
  return *this;
}

auto CodeActionParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> CodeActionParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("CodeActionParams::partialResultToken: not implement yet");
  return *this;
}

Command::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("title")) return false;
  if (!repr_->contains("command")) return false;
  return true;
}

auto Command::title() const -> std::string {
  auto& value = (*repr_)["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Command::tooltip() const -> std::optional<std::string> {
  if (!repr_->contains("tooltip")) return std::nullopt;

  auto& value = (*repr_)["tooltip"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Command::command() const -> std::string {
  auto& value = (*repr_)["command"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Command::arguments() const -> std::optional<Vector<LSPAny>> {
  if (!repr_->contains("arguments")) return std::nullopt;

  auto& value = (*repr_)["arguments"];

  assert(value.is_array());
  return Vector<LSPAny>(value);
}

auto Command::title(std::string title) -> Command& {
  repr_->emplace("title", std::move(title));
  return *this;
}

auto Command::tooltip(std::optional<std::string> tooltip) -> Command& {
  if (!tooltip.has_value()) {
    repr_->erase("tooltip");
    return *this;
  }
  repr_->emplace("tooltip", std::move(tooltip.value()));
  return *this;
}

auto Command::command(std::string command) -> Command& {
  repr_->emplace("command", std::move(command));
  return *this;
}

auto Command::arguments(std::optional<Vector<LSPAny>> arguments) -> Command& {
  if (!arguments.has_value()) {
    repr_->erase("arguments");
    return *this;
  }
  lsp_runtime_error("Command::arguments: not implement yet");
  return *this;
}

CodeAction::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("title")) return false;
  return true;
}

auto CodeAction::title() const -> std::string {
  auto& value = (*repr_)["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CodeAction::kind() const -> std::optional<CodeActionKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  lsp_runtime_error("CodeAction::kind: not implement yet");
}

auto CodeAction::diagnostics() const -> std::optional<Vector<Diagnostic>> {
  if (!repr_->contains("diagnostics")) return std::nullopt;

  auto& value = (*repr_)["diagnostics"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto CodeAction::isPreferred() const -> std::optional<bool> {
  if (!repr_->contains("isPreferred")) return std::nullopt;

  auto& value = (*repr_)["isPreferred"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeAction::disabled() const -> std::optional<CodeActionDisabled> {
  if (!repr_->contains("disabled")) return std::nullopt;

  auto& value = (*repr_)["disabled"];

  return CodeActionDisabled(value);
}

auto CodeAction::edit() const -> std::optional<WorkspaceEdit> {
  if (!repr_->contains("edit")) return std::nullopt;

  auto& value = (*repr_)["edit"];

  return WorkspaceEdit(value);
}

auto CodeAction::command() const -> std::optional<Command> {
  if (!repr_->contains("command")) return std::nullopt;

  auto& value = (*repr_)["command"];

  return Command(value);
}

auto CodeAction::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CodeAction::tags() const -> std::optional<Vector<CodeActionTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<CodeActionTag>(value);
}

auto CodeAction::title(std::string title) -> CodeAction& {
  repr_->emplace("title", std::move(title));
  return *this;
}

auto CodeAction::kind(std::optional<CodeActionKind> kind) -> CodeAction& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  lsp_runtime_error("CodeAction::kind: not implement yet");
  return *this;
}

auto CodeAction::diagnostics(std::optional<Vector<Diagnostic>> diagnostics)
    -> CodeAction& {
  if (!diagnostics.has_value()) {
    repr_->erase("diagnostics");
    return *this;
  }
  lsp_runtime_error("CodeAction::diagnostics: not implement yet");
  return *this;
}

auto CodeAction::isPreferred(std::optional<bool> isPreferred) -> CodeAction& {
  if (!isPreferred.has_value()) {
    repr_->erase("isPreferred");
    return *this;
  }
  repr_->emplace("isPreferred", std::move(isPreferred.value()));
  return *this;
}

auto CodeAction::disabled(std::optional<CodeActionDisabled> disabled)
    -> CodeAction& {
  if (!disabled.has_value()) {
    repr_->erase("disabled");
    return *this;
  }
  repr_->emplace("disabled", disabled.value());
  return *this;
}

auto CodeAction::edit(std::optional<WorkspaceEdit> edit) -> CodeAction& {
  if (!edit.has_value()) {
    repr_->erase("edit");
    return *this;
  }
  repr_->emplace("edit", edit.value());
  return *this;
}

auto CodeAction::command(std::optional<Command> command) -> CodeAction& {
  if (!command.has_value()) {
    repr_->erase("command");
    return *this;
  }
  repr_->emplace("command", command.value());
  return *this;
}

auto CodeAction::data(std::optional<LSPAny> data) -> CodeAction& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("CodeAction::data: not implement yet");
  return *this;
}

auto CodeAction::tags(std::optional<Vector<CodeActionTag>> tags)
    -> CodeAction& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("CodeAction::tags: not implement yet");
  return *this;
}

CodeActionRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto CodeActionRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CodeActionRegistrationOptions::codeActionKinds() const
    -> std::optional<Vector<CodeActionKind>> {
  if (!repr_->contains("codeActionKinds")) return std::nullopt;

  auto& value = (*repr_)["codeActionKinds"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto CodeActionRegistrationOptions::documentation() const
    -> std::optional<Vector<CodeActionKindDocumentation>> {
  if (!repr_->contains("documentation")) return std::nullopt;

  auto& value = (*repr_)["documentation"];

  assert(value.is_array());
  return Vector<CodeActionKindDocumentation>(value);
}

auto CodeActionRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CodeActionRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "CodeActionRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto CodeActionRegistrationOptions::codeActionKinds(
    std::optional<Vector<CodeActionKind>> codeActionKinds)
    -> CodeActionRegistrationOptions& {
  if (!codeActionKinds.has_value()) {
    repr_->erase("codeActionKinds");
    return *this;
  }
  lsp_runtime_error(
      "CodeActionRegistrationOptions::codeActionKinds: not implement yet");
  return *this;
}

auto CodeActionRegistrationOptions::documentation(
    std::optional<Vector<CodeActionKindDocumentation>> documentation)
    -> CodeActionRegistrationOptions& {
  if (!documentation.has_value()) {
    repr_->erase("documentation");
    return *this;
  }
  lsp_runtime_error(
      "CodeActionRegistrationOptions::documentation: not implement yet");
  return *this;
}

auto CodeActionRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> CodeActionRegistrationOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto CodeActionRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CodeActionRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

WorkspaceSymbolParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("query")) return false;
  return true;
}

auto WorkspaceSymbolParams::query() const -> std::string {
  auto& value = (*repr_)["query"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceSymbolParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbolParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbolParams::query(std::string query) -> WorkspaceSymbolParams& {
  repr_->emplace("query", std::move(query));
  return *this;
}

auto WorkspaceSymbolParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> WorkspaceSymbolParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("WorkspaceSymbolParams::workDoneToken: not implement yet");
  return *this;
}

auto WorkspaceSymbolParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> WorkspaceSymbolParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "WorkspaceSymbolParams::partialResultToken: not implement yet");
  return *this;
}

WorkspaceSymbol::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("location")) return false;
  if (!repr_->contains("name")) return false;
  if (!repr_->contains("kind")) return false;
  return true;
}

auto WorkspaceSymbol::location() const
    -> std::variant<std::monostate, Location, LocationUriOnly> {
  auto& value = (*repr_)["location"];

  std::variant<std::monostate, Location, LocationUriOnly> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceSymbol::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto WorkspaceSymbol::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceSymbol::kind() const -> SymbolKind {
  auto& value = (*repr_)["kind"];

  return SymbolKind(value);
}

auto WorkspaceSymbol::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto WorkspaceSymbol::containerName() const -> std::optional<std::string> {
  if (!repr_->contains("containerName")) return std::nullopt;

  auto& value = (*repr_)["containerName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceSymbol::location(
    std::variant<std::monostate, Location, LocationUriOnly> location)
    -> WorkspaceSymbol& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(Location location) { repr_->emplace("location", location); }

    void operator()(LocationUriOnly location) {
      repr_->emplace("location", location);
    }
  } v{repr_};

  std::visit(v, location);

  return *this;
}

auto WorkspaceSymbol::data(std::optional<LSPAny> data) -> WorkspaceSymbol& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("WorkspaceSymbol::data: not implement yet");
  return *this;
}

auto WorkspaceSymbol::name(std::string name) -> WorkspaceSymbol& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto WorkspaceSymbol::kind(SymbolKind kind) -> WorkspaceSymbol& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto WorkspaceSymbol::tags(std::optional<Vector<SymbolTag>> tags)
    -> WorkspaceSymbol& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("WorkspaceSymbol::tags: not implement yet");
  return *this;
}

auto WorkspaceSymbol::containerName(std::optional<std::string> containerName)
    -> WorkspaceSymbol& {
  if (!containerName.has_value()) {
    repr_->erase("containerName");
    return *this;
  }
  repr_->emplace("containerName", std::move(containerName.value()));
  return *this;
}

WorkspaceSymbolRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceSymbolRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider)
    -> WorkspaceSymbolRegistrationOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto WorkspaceSymbolRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> WorkspaceSymbolRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

CodeLensParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto CodeLensParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto CodeLensParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensParams::textDocument(TextDocumentIdentifier textDocument)
    -> CodeLensParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto CodeLensParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> CodeLensParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("CodeLensParams::workDoneToken: not implement yet");
  return *this;
}

auto CodeLensParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> CodeLensParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error("CodeLensParams::partialResultToken: not implement yet");
  return *this;
}

CodeLens::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto CodeLens::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto CodeLens::command() const -> std::optional<Command> {
  if (!repr_->contains("command")) return std::nullopt;

  auto& value = (*repr_)["command"];

  return Command(value);
}

auto CodeLens::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CodeLens::range(Range range) -> CodeLens& {
  repr_->emplace("range", range);
  return *this;
}

auto CodeLens::command(std::optional<Command> command) -> CodeLens& {
  if (!command.has_value()) {
    repr_->erase("command");
    return *this;
  }
  repr_->emplace("command", command.value());
  return *this;
}

auto CodeLens::data(std::optional<LSPAny> data) -> CodeLens& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("CodeLens::data: not implement yet");
  return *this;
}

CodeLensRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto CodeLensRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto CodeLensRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> CodeLensRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "CodeLensRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto CodeLensRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> CodeLensRegistrationOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto CodeLensRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CodeLensRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentLinkParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  return true;
}

auto DocumentLinkParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentLinkParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentLinkParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentLinkParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentLinkParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("DocumentLinkParams::workDoneToken: not implement yet");
  return *this;
}

auto DocumentLinkParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> DocumentLinkParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentLinkParams::partialResultToken: not implement yet");
  return *this;
}

DocumentLink::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto DocumentLink::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto DocumentLink::target() const -> std::optional<std::string> {
  if (!repr_->contains("target")) return std::nullopt;

  auto& value = (*repr_)["target"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentLink::tooltip() const -> std::optional<std::string> {
  if (!repr_->contains("tooltip")) return std::nullopt;

  auto& value = (*repr_)["tooltip"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentLink::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto DocumentLink::range(Range range) -> DocumentLink& {
  repr_->emplace("range", range);
  return *this;
}

auto DocumentLink::target(std::optional<std::string> target) -> DocumentLink& {
  if (!target.has_value()) {
    repr_->erase("target");
    return *this;
  }
  repr_->emplace("target", std::move(target.value()));
  return *this;
}

auto DocumentLink::tooltip(std::optional<std::string> tooltip)
    -> DocumentLink& {
  if (!tooltip.has_value()) {
    repr_->erase("tooltip");
    return *this;
  }
  repr_->emplace("tooltip", std::move(tooltip.value()));
  return *this;
}

auto DocumentLink::data(std::optional<LSPAny> data) -> DocumentLink& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("DocumentLink::data: not implement yet");
  return *this;
}

DocumentLinkRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DocumentLinkRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentLinkRegistrationOptions::resolveProvider() const
    -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentLinkRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentLinkRegistrationOptions::documentSelector: not implement "
          "yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentLinkRegistrationOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> DocumentLinkRegistrationOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto DocumentLinkRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentLinkRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentFormattingParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("options")) return false;
  return true;
}

auto DocumentFormattingParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentFormattingParams::options() const -> FormattingOptions {
  auto& value = (*repr_)["options"];

  return FormattingOptions(value);
}

auto DocumentFormattingParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentFormattingParams::textDocument(TextDocumentIdentifier textDocument)
    -> DocumentFormattingParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentFormattingParams::options(FormattingOptions options)
    -> DocumentFormattingParams& {
  repr_->emplace("options", options);
  return *this;
}

auto DocumentFormattingParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> DocumentFormattingParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentFormattingParams::workDoneToken: not implement yet");
  return *this;
}

DocumentFormattingRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DocumentFormattingRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentFormattingRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentFormattingRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentFormattingRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentFormattingRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentFormattingRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentFormattingRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentRangeFormattingParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("options")) return false;
  return true;
}

auto DocumentRangeFormattingParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentRangeFormattingParams::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto DocumentRangeFormattingParams::options() const -> FormattingOptions {
  auto& value = (*repr_)["options"];

  return FormattingOptions(value);
}

auto DocumentRangeFormattingParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangeFormattingParams::textDocument(
    TextDocumentIdentifier textDocument) -> DocumentRangeFormattingParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentRangeFormattingParams::range(Range range)
    -> DocumentRangeFormattingParams& {
  repr_->emplace("range", range);
  return *this;
}

auto DocumentRangeFormattingParams::options(FormattingOptions options)
    -> DocumentRangeFormattingParams& {
  repr_->emplace("options", options);
  return *this;
}

auto DocumentRangeFormattingParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> DocumentRangeFormattingParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentRangeFormattingParams::workDoneToken: not implement yet");
  return *this;
}

DocumentRangeFormattingRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto DocumentRangeFormattingRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangeFormattingRegistrationOptions::rangesSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("rangesSupport")) return std::nullopt;

  auto& value = (*repr_)["rangesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentRangeFormattingRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentRangeFormattingRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentRangeFormattingRegistrationOptions::rangesSupport(
    std::optional<bool> rangesSupport)
    -> DocumentRangeFormattingRegistrationOptions& {
  if (!rangesSupport.has_value()) {
    repr_->erase("rangesSupport");
    return *this;
  }
  repr_->emplace("rangesSupport", std::move(rangesSupport.value()));
  return *this;
}

auto DocumentRangeFormattingRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> DocumentRangeFormattingRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentRangesFormattingParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("ranges")) return false;
  if (!repr_->contains("options")) return false;
  return true;
}

auto DocumentRangesFormattingParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentRangesFormattingParams::ranges() const -> Vector<Range> {
  auto& value = (*repr_)["ranges"];

  assert(value.is_array());
  return Vector<Range>(value);
}

auto DocumentRangesFormattingParams::options() const -> FormattingOptions {
  auto& value = (*repr_)["options"];

  return FormattingOptions(value);
}

auto DocumentRangesFormattingParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentRangesFormattingParams::textDocument(
    TextDocumentIdentifier textDocument) -> DocumentRangesFormattingParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentRangesFormattingParams::ranges(Vector<Range> ranges)
    -> DocumentRangesFormattingParams& {
  lsp_runtime_error(
      "DocumentRangesFormattingParams::ranges: not implement yet");
  return *this;
}

auto DocumentRangesFormattingParams::options(FormattingOptions options)
    -> DocumentRangesFormattingParams& {
  repr_->emplace("options", options);
  return *this;
}

auto DocumentRangesFormattingParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken)
    -> DocumentRangesFormattingParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error(
      "DocumentRangesFormattingParams::workDoneToken: not implement yet");
  return *this;
}

DocumentOnTypeFormattingParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  if (!repr_->contains("ch")) return false;
  if (!repr_->contains("options")) return false;
  return true;
}

auto DocumentOnTypeFormattingParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto DocumentOnTypeFormattingParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto DocumentOnTypeFormattingParams::ch() const -> std::string {
  auto& value = (*repr_)["ch"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentOnTypeFormattingParams::options() const -> FormattingOptions {
  auto& value = (*repr_)["options"];

  return FormattingOptions(value);
}

auto DocumentOnTypeFormattingParams::textDocument(
    TextDocumentIdentifier textDocument) -> DocumentOnTypeFormattingParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto DocumentOnTypeFormattingParams::position(Position position)
    -> DocumentOnTypeFormattingParams& {
  repr_->emplace("position", position);
  return *this;
}

auto DocumentOnTypeFormattingParams::ch(std::string ch)
    -> DocumentOnTypeFormattingParams& {
  repr_->emplace("ch", std::move(ch));
  return *this;
}

auto DocumentOnTypeFormattingParams::options(FormattingOptions options)
    -> DocumentOnTypeFormattingParams& {
  repr_->emplace("options", options);
  return *this;
}

DocumentOnTypeFormattingRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  if (!repr_->contains("firstTriggerCharacter")) return false;
  return true;
}

auto DocumentOnTypeFormattingRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto DocumentOnTypeFormattingRegistrationOptions::firstTriggerCharacter() const
    -> std::string {
  auto& value = (*repr_)["firstTriggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentOnTypeFormattingRegistrationOptions::moreTriggerCharacter() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("moreTriggerCharacter")) return std::nullopt;

  auto& value = (*repr_)["moreTriggerCharacter"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto DocumentOnTypeFormattingRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> DocumentOnTypeFormattingRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "DocumentOnTypeFormattingRegistrationOptions::documentSelector: not "
          "implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto DocumentOnTypeFormattingRegistrationOptions::firstTriggerCharacter(
    std::string firstTriggerCharacter)
    -> DocumentOnTypeFormattingRegistrationOptions& {
  repr_->emplace("firstTriggerCharacter", std::move(firstTriggerCharacter));
  return *this;
}

auto DocumentOnTypeFormattingRegistrationOptions::moreTriggerCharacter(
    std::optional<Vector<std::string>> moreTriggerCharacter)
    -> DocumentOnTypeFormattingRegistrationOptions& {
  if (!moreTriggerCharacter.has_value()) {
    repr_->erase("moreTriggerCharacter");
    return *this;
  }
  lsp_runtime_error(
      "DocumentOnTypeFormattingRegistrationOptions::moreTriggerCharacter: not "
      "implement yet");
  return *this;
}

RenameParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  if (!repr_->contains("newName")) return false;
  return true;
}

auto RenameParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto RenameParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto RenameParams::newName() const -> std::string {
  auto& value = (*repr_)["newName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto RenameParams::textDocument(TextDocumentIdentifier textDocument)
    -> RenameParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto RenameParams::position(Position position) -> RenameParams& {
  repr_->emplace("position", position);
  return *this;
}

auto RenameParams::newName(std::string newName) -> RenameParams& {
  repr_->emplace("newName", std::move(newName));
  return *this;
}

auto RenameParams::workDoneToken(std::optional<ProgressToken> workDoneToken)
    -> RenameParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("RenameParams::workDoneToken: not implement yet");
  return *this;
}

RenameRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("documentSelector")) return false;
  return true;
}

auto RenameRegistrationOptions::documentSelector() const
    -> std::variant<std::monostate, DocumentSelector, std::nullptr_t> {
  auto& value = (*repr_)["documentSelector"];

  std::variant<std::monostate, DocumentSelector, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto RenameRegistrationOptions::prepareProvider() const -> std::optional<bool> {
  if (!repr_->contains("prepareProvider")) return std::nullopt;

  auto& value = (*repr_)["prepareProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameRegistrationOptions::documentSelector(
    std::variant<std::monostate, DocumentSelector, std::nullptr_t>
        documentSelector) -> RenameRegistrationOptions& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DocumentSelector documentSelector) {
      lsp_runtime_error(
          "RenameRegistrationOptions::documentSelector: not implement yet");
    }

    void operator()(std::nullptr_t documentSelector) {
      repr_->emplace("documentSelector", std::move(documentSelector));
    }
  } v{repr_};

  std::visit(v, documentSelector);

  return *this;
}

auto RenameRegistrationOptions::prepareProvider(
    std::optional<bool> prepareProvider) -> RenameRegistrationOptions& {
  if (!prepareProvider.has_value()) {
    repr_->erase("prepareProvider");
    return *this;
  }
  repr_->emplace("prepareProvider", std::move(prepareProvider.value()));
  return *this;
}

auto RenameRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> RenameRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

PrepareRenameParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto PrepareRenameParams::textDocument() const -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto PrepareRenameParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto PrepareRenameParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto PrepareRenameParams::textDocument(TextDocumentIdentifier textDocument)
    -> PrepareRenameParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto PrepareRenameParams::position(Position position) -> PrepareRenameParams& {
  repr_->emplace("position", position);
  return *this;
}

auto PrepareRenameParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> PrepareRenameParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("PrepareRenameParams::workDoneToken: not implement yet");
  return *this;
}

ExecuteCommandParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("command")) return false;
  return true;
}

auto ExecuteCommandParams::command() const -> std::string {
  auto& value = (*repr_)["command"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ExecuteCommandParams::arguments() const -> std::optional<Vector<LSPAny>> {
  if (!repr_->contains("arguments")) return std::nullopt;

  auto& value = (*repr_)["arguments"];

  assert(value.is_array());
  return Vector<LSPAny>(value);
}

auto ExecuteCommandParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ExecuteCommandParams::command(std::string command)
    -> ExecuteCommandParams& {
  repr_->emplace("command", std::move(command));
  return *this;
}

auto ExecuteCommandParams::arguments(std::optional<Vector<LSPAny>> arguments)
    -> ExecuteCommandParams& {
  if (!arguments.has_value()) {
    repr_->erase("arguments");
    return *this;
  }
  lsp_runtime_error("ExecuteCommandParams::arguments: not implement yet");
  return *this;
}

auto ExecuteCommandParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> ExecuteCommandParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("ExecuteCommandParams::workDoneToken: not implement yet");
  return *this;
}

ExecuteCommandRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("commands")) return false;
  return true;
}

auto ExecuteCommandRegistrationOptions::commands() const
    -> Vector<std::string> {
  auto& value = (*repr_)["commands"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ExecuteCommandRegistrationOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecuteCommandRegistrationOptions::commands(Vector<std::string> commands)
    -> ExecuteCommandRegistrationOptions& {
  lsp_runtime_error(
      "ExecuteCommandRegistrationOptions::commands: not implement yet");
  return *this;
}

auto ExecuteCommandRegistrationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress)
    -> ExecuteCommandRegistrationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

ApplyWorkspaceEditParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("edit")) return false;
  return true;
}

auto ApplyWorkspaceEditParams::label() const -> std::optional<std::string> {
  if (!repr_->contains("label")) return std::nullopt;

  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ApplyWorkspaceEditParams::edit() const -> WorkspaceEdit {
  auto& value = (*repr_)["edit"];

  return WorkspaceEdit(value);
}

auto ApplyWorkspaceEditParams::metadata() const
    -> std::optional<WorkspaceEditMetadata> {
  if (!repr_->contains("metadata")) return std::nullopt;

  auto& value = (*repr_)["metadata"];

  return WorkspaceEditMetadata(value);
}

auto ApplyWorkspaceEditParams::label(std::optional<std::string> label)
    -> ApplyWorkspaceEditParams& {
  if (!label.has_value()) {
    repr_->erase("label");
    return *this;
  }
  repr_->emplace("label", std::move(label.value()));
  return *this;
}

auto ApplyWorkspaceEditParams::edit(WorkspaceEdit edit)
    -> ApplyWorkspaceEditParams& {
  repr_->emplace("edit", edit);
  return *this;
}

auto ApplyWorkspaceEditParams::metadata(
    std::optional<WorkspaceEditMetadata> metadata)
    -> ApplyWorkspaceEditParams& {
  if (!metadata.has_value()) {
    repr_->erase("metadata");
    return *this;
  }
  repr_->emplace("metadata", metadata.value());
  return *this;
}

ApplyWorkspaceEditResult::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("applied")) return false;
  return true;
}

auto ApplyWorkspaceEditResult::applied() const -> bool {
  auto& value = (*repr_)["applied"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ApplyWorkspaceEditResult::failureReason() const
    -> std::optional<std::string> {
  if (!repr_->contains("failureReason")) return std::nullopt;

  auto& value = (*repr_)["failureReason"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ApplyWorkspaceEditResult::failedChange() const -> std::optional<long> {
  if (!repr_->contains("failedChange")) return std::nullopt;

  auto& value = (*repr_)["failedChange"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto ApplyWorkspaceEditResult::applied(bool applied)
    -> ApplyWorkspaceEditResult& {
  repr_->emplace("applied", std::move(applied));
  return *this;
}

auto ApplyWorkspaceEditResult::failureReason(
    std::optional<std::string> failureReason) -> ApplyWorkspaceEditResult& {
  if (!failureReason.has_value()) {
    repr_->erase("failureReason");
    return *this;
  }
  repr_->emplace("failureReason", std::move(failureReason.value()));
  return *this;
}

auto ApplyWorkspaceEditResult::failedChange(std::optional<long> failedChange)
    -> ApplyWorkspaceEditResult& {
  if (!failedChange.has_value()) {
    repr_->erase("failedChange");
    return *this;
  }
  repr_->emplace("failedChange", std::move(failedChange.value()));
  return *this;
}

WorkDoneProgressBegin::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "begin") return false;
  if (!repr_->contains("title")) return false;
  return true;
}

auto WorkDoneProgressBegin::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressBegin::title() const -> std::string {
  auto& value = (*repr_)["title"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressBegin::cancellable() const -> std::optional<bool> {
  if (!repr_->contains("cancellable")) return std::nullopt;

  auto& value = (*repr_)["cancellable"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkDoneProgressBegin::message() const -> std::optional<std::string> {
  if (!repr_->contains("message")) return std::nullopt;

  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressBegin::percentage() const -> std::optional<long> {
  if (!repr_->contains("percentage")) return std::nullopt;

  auto& value = (*repr_)["percentage"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto WorkDoneProgressBegin::kind(std::string kind) -> WorkDoneProgressBegin& {
  lsp_runtime_error("WorkDoneProgressBegin::kind: not implement yet");
  return *this;
}

auto WorkDoneProgressBegin::title(std::string title) -> WorkDoneProgressBegin& {
  repr_->emplace("title", std::move(title));
  return *this;
}

auto WorkDoneProgressBegin::cancellable(std::optional<bool> cancellable)
    -> WorkDoneProgressBegin& {
  if (!cancellable.has_value()) {
    repr_->erase("cancellable");
    return *this;
  }
  repr_->emplace("cancellable", std::move(cancellable.value()));
  return *this;
}

auto WorkDoneProgressBegin::message(std::optional<std::string> message)
    -> WorkDoneProgressBegin& {
  if (!message.has_value()) {
    repr_->erase("message");
    return *this;
  }
  repr_->emplace("message", std::move(message.value()));
  return *this;
}

auto WorkDoneProgressBegin::percentage(std::optional<long> percentage)
    -> WorkDoneProgressBegin& {
  if (!percentage.has_value()) {
    repr_->erase("percentage");
    return *this;
  }
  repr_->emplace("percentage", std::move(percentage.value()));
  return *this;
}

WorkDoneProgressReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "report") return false;
  return true;
}

auto WorkDoneProgressReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressReport::cancellable() const -> std::optional<bool> {
  if (!repr_->contains("cancellable")) return std::nullopt;

  auto& value = (*repr_)["cancellable"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkDoneProgressReport::message() const -> std::optional<std::string> {
  if (!repr_->contains("message")) return std::nullopt;

  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressReport::percentage() const -> std::optional<long> {
  if (!repr_->contains("percentage")) return std::nullopt;

  auto& value = (*repr_)["percentage"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto WorkDoneProgressReport::kind(std::string kind) -> WorkDoneProgressReport& {
  lsp_runtime_error("WorkDoneProgressReport::kind: not implement yet");
  return *this;
}

auto WorkDoneProgressReport::cancellable(std::optional<bool> cancellable)
    -> WorkDoneProgressReport& {
  if (!cancellable.has_value()) {
    repr_->erase("cancellable");
    return *this;
  }
  repr_->emplace("cancellable", std::move(cancellable.value()));
  return *this;
}

auto WorkDoneProgressReport::message(std::optional<std::string> message)
    -> WorkDoneProgressReport& {
  if (!message.has_value()) {
    repr_->erase("message");
    return *this;
  }
  repr_->emplace("message", std::move(message.value()));
  return *this;
}

auto WorkDoneProgressReport::percentage(std::optional<long> percentage)
    -> WorkDoneProgressReport& {
  if (!percentage.has_value()) {
    repr_->erase("percentage");
    return *this;
  }
  repr_->emplace("percentage", std::move(percentage.value()));
  return *this;
}

WorkDoneProgressEnd::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "end") return false;
  return true;
}

auto WorkDoneProgressEnd::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressEnd::message() const -> std::optional<std::string> {
  if (!repr_->contains("message")) return std::nullopt;

  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkDoneProgressEnd::kind(std::string kind) -> WorkDoneProgressEnd& {
  lsp_runtime_error("WorkDoneProgressEnd::kind: not implement yet");
  return *this;
}

auto WorkDoneProgressEnd::message(std::optional<std::string> message)
    -> WorkDoneProgressEnd& {
  if (!message.has_value()) {
    repr_->erase("message");
    return *this;
  }
  repr_->emplace("message", std::move(message.value()));
  return *this;
}

SetTraceParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto SetTraceParams::value() const -> TraceValue {
  auto& value = (*repr_)["value"];

  lsp_runtime_error("SetTraceParams::value: not implement yet");
}

auto SetTraceParams::value(TraceValue value) -> SetTraceParams& {
  lsp_runtime_error("SetTraceParams::value: not implement yet");
  return *this;
}

LogTraceParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("message")) return false;
  return true;
}

auto LogTraceParams::message() const -> std::string {
  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LogTraceParams::verbose() const -> std::optional<std::string> {
  if (!repr_->contains("verbose")) return std::nullopt;

  auto& value = (*repr_)["verbose"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LogTraceParams::message(std::string message) -> LogTraceParams& {
  repr_->emplace("message", std::move(message));
  return *this;
}

auto LogTraceParams::verbose(std::optional<std::string> verbose)
    -> LogTraceParams& {
  if (!verbose.has_value()) {
    repr_->erase("verbose");
    return *this;
  }
  repr_->emplace("verbose", std::move(verbose.value()));
  return *this;
}

CancelParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("id")) return false;
  return true;
}

auto CancelParams::id() const
    -> std::variant<std::monostate, int, std::string> {
  auto& value = (*repr_)["id"];

  std::variant<std::monostate, int, std::string> result;

  details::try_emplace(result, value);

  return result;
}

auto CancelParams::id(std::variant<std::monostate, int, std::string> id)
    -> CancelParams& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int id) { repr_->emplace("id", std::move(id)); }

    void operator()(std::string id) { repr_->emplace("id", std::move(id)); }
  } v{repr_};

  std::visit(v, id);

  return *this;
}

ProgressParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("token")) return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto ProgressParams::token() const -> ProgressToken {
  auto& value = (*repr_)["token"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto ProgressParams::value() const -> LSPAny {
  auto& value = (*repr_)["value"];

  assert(value.is_object());
  return LSPAny(value);
}

auto ProgressParams::token(ProgressToken token) -> ProgressParams& {
  lsp_runtime_error("ProgressParams::token: not implement yet");
  return *this;
}

auto ProgressParams::value(LSPAny value) -> ProgressParams& {
  lsp_runtime_error("ProgressParams::value: not implement yet");
  return *this;
}

TextDocumentPositionParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("position")) return false;
  return true;
}

auto TextDocumentPositionParams::textDocument() const
    -> TextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return TextDocumentIdentifier(value);
}

auto TextDocumentPositionParams::position() const -> Position {
  auto& value = (*repr_)["position"];

  return Position(value);
}

auto TextDocumentPositionParams::textDocument(
    TextDocumentIdentifier textDocument) -> TextDocumentPositionParams& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto TextDocumentPositionParams::position(Position position)
    -> TextDocumentPositionParams& {
  repr_->emplace("position", position);
  return *this;
}

WorkDoneProgressParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkDoneProgressParams::workDoneToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto WorkDoneProgressParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> WorkDoneProgressParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("WorkDoneProgressParams::workDoneToken: not implement yet");
  return *this;
}

PartialResultParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto PartialResultParams::partialResultToken() const
    -> std::optional<ProgressToken> {
  if (!repr_->contains("partialResultToken")) return std::nullopt;

  auto& value = (*repr_)["partialResultToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto PartialResultParams::partialResultToken(
    std::optional<ProgressToken> partialResultToken) -> PartialResultParams& {
  if (!partialResultToken.has_value()) {
    repr_->erase("partialResultToken");
    return *this;
  }
  lsp_runtime_error(
      "PartialResultParams::partialResultToken: not implement yet");
  return *this;
}

LocationLink::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("targetUri")) return false;
  if (!repr_->contains("targetRange")) return false;
  if (!repr_->contains("targetSelectionRange")) return false;
  return true;
}

auto LocationLink::originSelectionRange() const -> std::optional<Range> {
  if (!repr_->contains("originSelectionRange")) return std::nullopt;

  auto& value = (*repr_)["originSelectionRange"];

  return Range(value);
}

auto LocationLink::targetUri() const -> std::string {
  auto& value = (*repr_)["targetUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LocationLink::targetRange() const -> Range {
  auto& value = (*repr_)["targetRange"];

  return Range(value);
}

auto LocationLink::targetSelectionRange() const -> Range {
  auto& value = (*repr_)["targetSelectionRange"];

  return Range(value);
}

auto LocationLink::originSelectionRange(
    std::optional<Range> originSelectionRange) -> LocationLink& {
  if (!originSelectionRange.has_value()) {
    repr_->erase("originSelectionRange");
    return *this;
  }
  repr_->emplace("originSelectionRange", originSelectionRange.value());
  return *this;
}

auto LocationLink::targetUri(std::string targetUri) -> LocationLink& {
  repr_->emplace("targetUri", std::move(targetUri));
  return *this;
}

auto LocationLink::targetRange(Range targetRange) -> LocationLink& {
  repr_->emplace("targetRange", targetRange);
  return *this;
}

auto LocationLink::targetSelectionRange(Range targetSelectionRange)
    -> LocationLink& {
  repr_->emplace("targetSelectionRange", targetSelectionRange);
  return *this;
}

Range::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("start")) return false;
  if (!repr_->contains("end")) return false;
  return true;
}

auto Range::start() const -> Position {
  auto& value = (*repr_)["start"];

  return Position(value);
}

auto Range::end() const -> Position {
  auto& value = (*repr_)["end"];

  return Position(value);
}

auto Range::start(Position start) -> Range& {
  repr_->emplace("start", start);
  return *this;
}

auto Range::end(Position end) -> Range& {
  repr_->emplace("end", end);
  return *this;
}

ImplementationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ImplementationOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> ImplementationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

StaticRegistrationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto StaticRegistrationOptions::id() const -> std::optional<std::string> {
  if (!repr_->contains("id")) return std::nullopt;

  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto StaticRegistrationOptions::id(std::optional<std::string> id)
    -> StaticRegistrationOptions& {
  if (!id.has_value()) {
    repr_->erase("id");
    return *this;
  }
  repr_->emplace("id", std::move(id.value()));
  return *this;
}

TypeDefinitionOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TypeDefinitionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> TypeDefinitionOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

WorkspaceFoldersChangeEvent::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("added")) return false;
  if (!repr_->contains("removed")) return false;
  return true;
}

auto WorkspaceFoldersChangeEvent::added() const -> Vector<WorkspaceFolder> {
  auto& value = (*repr_)["added"];

  assert(value.is_array());
  return Vector<WorkspaceFolder>(value);
}

auto WorkspaceFoldersChangeEvent::removed() const -> Vector<WorkspaceFolder> {
  auto& value = (*repr_)["removed"];

  assert(value.is_array());
  return Vector<WorkspaceFolder>(value);
}

auto WorkspaceFoldersChangeEvent::added(Vector<WorkspaceFolder> added)
    -> WorkspaceFoldersChangeEvent& {
  lsp_runtime_error("WorkspaceFoldersChangeEvent::added: not implement yet");
  return *this;
}

auto WorkspaceFoldersChangeEvent::removed(Vector<WorkspaceFolder> removed)
    -> WorkspaceFoldersChangeEvent& {
  lsp_runtime_error("WorkspaceFoldersChangeEvent::removed: not implement yet");
  return *this;
}

ConfigurationItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ConfigurationItem::scopeUri() const -> std::optional<std::string> {
  if (!repr_->contains("scopeUri")) return std::nullopt;

  auto& value = (*repr_)["scopeUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ConfigurationItem::section() const -> std::optional<std::string> {
  if (!repr_->contains("section")) return std::nullopt;

  auto& value = (*repr_)["section"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ConfigurationItem::scopeUri(std::optional<std::string> scopeUri)
    -> ConfigurationItem& {
  if (!scopeUri.has_value()) {
    repr_->erase("scopeUri");
    return *this;
  }
  repr_->emplace("scopeUri", std::move(scopeUri.value()));
  return *this;
}

auto ConfigurationItem::section(std::optional<std::string> section)
    -> ConfigurationItem& {
  if (!section.has_value()) {
    repr_->erase("section");
    return *this;
  }
  repr_->emplace("section", std::move(section.value()));
  return *this;
}

TextDocumentIdentifier::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto TextDocumentIdentifier::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentIdentifier::uri(std::string uri) -> TextDocumentIdentifier& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

Color::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("red")) return false;
  if (!repr_->contains("green")) return false;
  if (!repr_->contains("blue")) return false;
  if (!repr_->contains("alpha")) return false;
  return true;
}

auto Color::red() const -> double {
  auto& value = (*repr_)["red"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::green() const -> double {
  auto& value = (*repr_)["green"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::blue() const -> double {
  auto& value = (*repr_)["blue"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::alpha() const -> double {
  auto& value = (*repr_)["alpha"];

  assert(value.is_number());
  return value.get<double>();
}

auto Color::red(double red) -> Color& {
  repr_->emplace("red", std::move(red));
  return *this;
}

auto Color::green(double green) -> Color& {
  repr_->emplace("green", std::move(green));
  return *this;
}

auto Color::blue(double blue) -> Color& {
  repr_->emplace("blue", std::move(blue));
  return *this;
}

auto Color::alpha(double alpha) -> Color& {
  repr_->emplace("alpha", std::move(alpha));
  return *this;
}

DocumentColorOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentColorOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentColorOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentColorOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

FoldingRangeOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto FoldingRangeOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> FoldingRangeOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DeclarationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DeclarationOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DeclarationOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

Position::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("line")) return false;
  if (!repr_->contains("character")) return false;
  return true;
}

auto Position::line() const -> long {
  auto& value = (*repr_)["line"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto Position::character() const -> long {
  auto& value = (*repr_)["character"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto Position::line(long line) -> Position& {
  repr_->emplace("line", std::move(line));
  return *this;
}

auto Position::character(long character) -> Position& {
  repr_->emplace("character", std::move(character));
  return *this;
}

SelectionRangeOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SelectionRangeOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SelectionRangeOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SelectionRangeOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

CallHierarchyOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CallHierarchyOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CallHierarchyOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> CallHierarchyOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

SemanticTokensOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("legend")) return false;
  return true;
}

auto SemanticTokensOptions::legend() const -> SemanticTokensLegend {
  auto& value = (*repr_)["legend"];

  return SemanticTokensLegend(value);
}

auto SemanticTokensOptions::range() const
    -> std::optional<std::variant<std::monostate, bool, json>> {
  if (!repr_->contains("range")) return std::nullopt;

  auto& value = (*repr_)["range"];

  std::variant<std::monostate, bool, json> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensOptions::full() const -> std::optional<
    std::variant<std::monostate, bool, SemanticTokensFullDelta>> {
  if (!repr_->contains("full")) return std::nullopt;

  auto& value = (*repr_)["full"];

  std::variant<std::monostate, bool, SemanticTokensFullDelta> result;

  details::try_emplace(result, value);

  return result;
}

auto SemanticTokensOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensOptions::legend(SemanticTokensLegend legend)
    -> SemanticTokensOptions& {
  repr_->emplace("legend", legend);
  return *this;
}

auto SemanticTokensOptions::range(
    std::optional<std::variant<std::monostate, bool, json>> range)
    -> SemanticTokensOptions& {
  if (!range.has_value()) {
    repr_->erase("range");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool range) { repr_->emplace("range", std::move(range)); }

    void operator()(json range) {
      lsp_runtime_error("SemanticTokensOptions::range: not implement yet");
    }
  } v{repr_};

  std::visit(v, range.value());

  return *this;
}

auto SemanticTokensOptions::full(
    std::optional<std::variant<std::monostate, bool, SemanticTokensFullDelta>>
        full) -> SemanticTokensOptions& {
  if (!full.has_value()) {
    repr_->erase("full");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool full) { repr_->emplace("full", std::move(full)); }

    void operator()(SemanticTokensFullDelta full) {
      repr_->emplace("full", full);
    }
  } v{repr_};

  std::visit(v, full.value());

  return *this;
}

auto SemanticTokensOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SemanticTokensOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

SemanticTokensEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("start")) return false;
  if (!repr_->contains("deleteCount")) return false;
  return true;
}

auto SemanticTokensEdit::start() const -> long {
  auto& value = (*repr_)["start"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto SemanticTokensEdit::deleteCount() const -> long {
  auto& value = (*repr_)["deleteCount"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto SemanticTokensEdit::data() const -> std::optional<Vector<long>> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_array());
  return Vector<long>(value);
}

auto SemanticTokensEdit::start(long start) -> SemanticTokensEdit& {
  repr_->emplace("start", std::move(start));
  return *this;
}

auto SemanticTokensEdit::deleteCount(long deleteCount) -> SemanticTokensEdit& {
  repr_->emplace("deleteCount", std::move(deleteCount));
  return *this;
}

auto SemanticTokensEdit::data(std::optional<Vector<long>> data)
    -> SemanticTokensEdit& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("SemanticTokensEdit::data: not implement yet");
  return *this;
}

LinkedEditingRangeOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto LinkedEditingRangeOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto LinkedEditingRangeOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> LinkedEditingRangeOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

FileCreate::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto FileCreate::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileCreate::uri(std::string uri) -> FileCreate& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

TextDocumentEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("textDocument")) return false;
  if (!repr_->contains("edits")) return false;
  return true;
}

auto TextDocumentEdit::textDocument() const
    -> OptionalVersionedTextDocumentIdentifier {
  auto& value = (*repr_)["textDocument"];

  return OptionalVersionedTextDocumentIdentifier(value);
}

auto TextDocumentEdit::edits() const
    -> Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                           SnippetTextEdit>> {
  auto& value = (*repr_)["edits"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                             SnippetTextEdit>>(value);
}

auto TextDocumentEdit::textDocument(
    OptionalVersionedTextDocumentIdentifier textDocument) -> TextDocumentEdit& {
  repr_->emplace("textDocument", textDocument);
  return *this;
}

auto TextDocumentEdit::edits(
    Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                        SnippetTextEdit>>
        edits) -> TextDocumentEdit& {
  lsp_runtime_error("TextDocumentEdit::edits: not implement yet");
  return *this;
}

CreateFile::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "create") return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto CreateFile::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CreateFile::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CreateFile::options() const -> std::optional<CreateFileOptions> {
  if (!repr_->contains("options")) return std::nullopt;

  auto& value = (*repr_)["options"];

  return CreateFileOptions(value);
}

auto CreateFile::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_->contains("annotationId")) return std::nullopt;

  auto& value = (*repr_)["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CreateFile::kind(std::string kind) -> CreateFile& {
  lsp_runtime_error("CreateFile::kind: not implement yet");
  return *this;
}

auto CreateFile::uri(std::string uri) -> CreateFile& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto CreateFile::options(std::optional<CreateFileOptions> options)
    -> CreateFile& {
  if (!options.has_value()) {
    repr_->erase("options");
    return *this;
  }
  repr_->emplace("options", options.value());
  return *this;
}

auto CreateFile::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId) -> CreateFile& {
  if (!annotationId.has_value()) {
    repr_->erase("annotationId");
    return *this;
  }
  lsp_runtime_error("CreateFile::annotationId: not implement yet");
  return *this;
}

RenameFile::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "rename") return false;
  if (!repr_->contains("oldUri")) return false;
  if (!repr_->contains("newUri")) return false;
  return true;
}

auto RenameFile::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::oldUri() const -> std::string {
  auto& value = (*repr_)["oldUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::newUri() const -> std::string {
  auto& value = (*repr_)["newUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::options() const -> std::optional<RenameFileOptions> {
  if (!repr_->contains("options")) return std::nullopt;

  auto& value = (*repr_)["options"];

  return RenameFileOptions(value);
}

auto RenameFile::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_->contains("annotationId")) return std::nullopt;

  auto& value = (*repr_)["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RenameFile::kind(std::string kind) -> RenameFile& {
  lsp_runtime_error("RenameFile::kind: not implement yet");
  return *this;
}

auto RenameFile::oldUri(std::string oldUri) -> RenameFile& {
  repr_->emplace("oldUri", std::move(oldUri));
  return *this;
}

auto RenameFile::newUri(std::string newUri) -> RenameFile& {
  repr_->emplace("newUri", std::move(newUri));
  return *this;
}

auto RenameFile::options(std::optional<RenameFileOptions> options)
    -> RenameFile& {
  if (!options.has_value()) {
    repr_->erase("options");
    return *this;
  }
  repr_->emplace("options", options.value());
  return *this;
}

auto RenameFile::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId) -> RenameFile& {
  if (!annotationId.has_value()) {
    repr_->erase("annotationId");
    return *this;
  }
  lsp_runtime_error("RenameFile::annotationId: not implement yet");
  return *this;
}

DeleteFile::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "delete") return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto DeleteFile::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeleteFile::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeleteFile::options() const -> std::optional<DeleteFileOptions> {
  if (!repr_->contains("options")) return std::nullopt;

  auto& value = (*repr_)["options"];

  return DeleteFileOptions(value);
}

auto DeleteFile::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_->contains("annotationId")) return std::nullopt;

  auto& value = (*repr_)["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DeleteFile::kind(std::string kind) -> DeleteFile& {
  lsp_runtime_error("DeleteFile::kind: not implement yet");
  return *this;
}

auto DeleteFile::uri(std::string uri) -> DeleteFile& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto DeleteFile::options(std::optional<DeleteFileOptions> options)
    -> DeleteFile& {
  if (!options.has_value()) {
    repr_->erase("options");
    return *this;
  }
  repr_->emplace("options", options.value());
  return *this;
}

auto DeleteFile::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId) -> DeleteFile& {
  if (!annotationId.has_value()) {
    repr_->erase("annotationId");
    return *this;
  }
  lsp_runtime_error("DeleteFile::annotationId: not implement yet");
  return *this;
}

ChangeAnnotation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("label")) return false;
  return true;
}

auto ChangeAnnotation::label() const -> std::string {
  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ChangeAnnotation::needsConfirmation() const -> std::optional<bool> {
  if (!repr_->contains("needsConfirmation")) return std::nullopt;

  auto& value = (*repr_)["needsConfirmation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ChangeAnnotation::description() const -> std::optional<std::string> {
  if (!repr_->contains("description")) return std::nullopt;

  auto& value = (*repr_)["description"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ChangeAnnotation::label(std::string label) -> ChangeAnnotation& {
  repr_->emplace("label", std::move(label));
  return *this;
}

auto ChangeAnnotation::needsConfirmation(std::optional<bool> needsConfirmation)
    -> ChangeAnnotation& {
  if (!needsConfirmation.has_value()) {
    repr_->erase("needsConfirmation");
    return *this;
  }
  repr_->emplace("needsConfirmation", std::move(needsConfirmation.value()));
  return *this;
}

auto ChangeAnnotation::description(std::optional<std::string> description)
    -> ChangeAnnotation& {
  if (!description.has_value()) {
    repr_->erase("description");
    return *this;
  }
  repr_->emplace("description", std::move(description.value()));
  return *this;
}

FileOperationFilter::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("pattern")) return false;
  return true;
}

auto FileOperationFilter::scheme() const -> std::optional<std::string> {
  if (!repr_->contains("scheme")) return std::nullopt;

  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileOperationFilter::pattern() const -> FileOperationPattern {
  auto& value = (*repr_)["pattern"];

  return FileOperationPattern(value);
}

auto FileOperationFilter::scheme(std::optional<std::string> scheme)
    -> FileOperationFilter& {
  if (!scheme.has_value()) {
    repr_->erase("scheme");
    return *this;
  }
  repr_->emplace("scheme", std::move(scheme.value()));
  return *this;
}

auto FileOperationFilter::pattern(FileOperationPattern pattern)
    -> FileOperationFilter& {
  repr_->emplace("pattern", pattern);
  return *this;
}

FileRename::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("oldUri")) return false;
  if (!repr_->contains("newUri")) return false;
  return true;
}

auto FileRename::oldUri() const -> std::string {
  auto& value = (*repr_)["oldUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileRename::newUri() const -> std::string {
  auto& value = (*repr_)["newUri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileRename::oldUri(std::string oldUri) -> FileRename& {
  repr_->emplace("oldUri", std::move(oldUri));
  return *this;
}

auto FileRename::newUri(std::string newUri) -> FileRename& {
  repr_->emplace("newUri", std::move(newUri));
  return *this;
}

FileDelete::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto FileDelete::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileDelete::uri(std::string uri) -> FileDelete& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

MonikerOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto MonikerOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto MonikerOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> MonikerOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

TypeHierarchyOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TypeHierarchyOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeHierarchyOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> TypeHierarchyOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

InlineValueContext::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("frameId")) return false;
  if (!repr_->contains("stoppedLocation")) return false;
  return true;
}

auto InlineValueContext::frameId() const -> int {
  auto& value = (*repr_)["frameId"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto InlineValueContext::stoppedLocation() const -> Range {
  auto& value = (*repr_)["stoppedLocation"];

  return Range(value);
}

auto InlineValueContext::frameId(int frameId) -> InlineValueContext& {
  repr_->emplace("frameId", std::move(frameId));
  return *this;
}

auto InlineValueContext::stoppedLocation(Range stoppedLocation)
    -> InlineValueContext& {
  repr_->emplace("stoppedLocation", stoppedLocation);
  return *this;
}

InlineValueText::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("text")) return false;
  return true;
}

auto InlineValueText::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto InlineValueText::text() const -> std::string {
  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueText::range(Range range) -> InlineValueText& {
  repr_->emplace("range", range);
  return *this;
}

auto InlineValueText::text(std::string text) -> InlineValueText& {
  repr_->emplace("text", std::move(text));
  return *this;
}

InlineValueVariableLookup::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("caseSensitiveLookup")) return false;
  return true;
}

auto InlineValueVariableLookup::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto InlineValueVariableLookup::variableName() const
    -> std::optional<std::string> {
  if (!repr_->contains("variableName")) return std::nullopt;

  auto& value = (*repr_)["variableName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueVariableLookup::caseSensitiveLookup() const -> bool {
  auto& value = (*repr_)["caseSensitiveLookup"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueVariableLookup::range(Range range)
    -> InlineValueVariableLookup& {
  repr_->emplace("range", range);
  return *this;
}

auto InlineValueVariableLookup::variableName(
    std::optional<std::string> variableName) -> InlineValueVariableLookup& {
  if (!variableName.has_value()) {
    repr_->erase("variableName");
    return *this;
  }
  repr_->emplace("variableName", std::move(variableName.value()));
  return *this;
}

auto InlineValueVariableLookup::caseSensitiveLookup(bool caseSensitiveLookup)
    -> InlineValueVariableLookup& {
  repr_->emplace("caseSensitiveLookup", std::move(caseSensitiveLookup));
  return *this;
}

InlineValueEvaluatableExpression::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  return true;
}

auto InlineValueEvaluatableExpression::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto InlineValueEvaluatableExpression::expression() const
    -> std::optional<std::string> {
  if (!repr_->contains("expression")) return std::nullopt;

  auto& value = (*repr_)["expression"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlineValueEvaluatableExpression::range(Range range)
    -> InlineValueEvaluatableExpression& {
  repr_->emplace("range", range);
  return *this;
}

auto InlineValueEvaluatableExpression::expression(
    std::optional<std::string> expression)
    -> InlineValueEvaluatableExpression& {
  if (!expression.has_value()) {
    repr_->erase("expression");
    return *this;
  }
  repr_->emplace("expression", std::move(expression.value()));
  return *this;
}

InlineValueOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlineValueOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> InlineValueOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

InlayHintLabelPart::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto InlayHintLabelPart::value() const -> std::string {
  auto& value = (*repr_)["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InlayHintLabelPart::tooltip() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_->contains("tooltip")) return std::nullopt;

  auto& value = (*repr_)["tooltip"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto InlayHintLabelPart::location() const -> std::optional<Location> {
  if (!repr_->contains("location")) return std::nullopt;

  auto& value = (*repr_)["location"];

  return Location(value);
}

auto InlayHintLabelPart::command() const -> std::optional<Command> {
  if (!repr_->contains("command")) return std::nullopt;

  auto& value = (*repr_)["command"];

  return Command(value);
}

auto InlayHintLabelPart::value(std::string value) -> InlayHintLabelPart& {
  repr_->emplace("value", std::move(value));
  return *this;
}

auto InlayHintLabelPart::tooltip(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        tooltip) -> InlayHintLabelPart& {
  if (!tooltip.has_value()) {
    repr_->erase("tooltip");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string tooltip) {
      repr_->emplace("tooltip", std::move(tooltip));
    }

    void operator()(MarkupContent tooltip) {
      repr_->emplace("tooltip", tooltip);
    }
  } v{repr_};

  std::visit(v, tooltip.value());

  return *this;
}

auto InlayHintLabelPart::location(std::optional<Location> location)
    -> InlayHintLabelPart& {
  if (!location.has_value()) {
    repr_->erase("location");
    return *this;
  }
  repr_->emplace("location", location.value());
  return *this;
}

auto InlayHintLabelPart::command(std::optional<Command> command)
    -> InlayHintLabelPart& {
  if (!command.has_value()) {
    repr_->erase("command");
    return *this;
  }
  repr_->emplace("command", command.value());
  return *this;
}

MarkupContent::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto MarkupContent::kind() const -> MarkupKind {
  auto& value = (*repr_)["kind"];

  lsp_runtime_error("MarkupContent::kind: not implement yet");
}

auto MarkupContent::value() const -> std::string {
  auto& value = (*repr_)["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkupContent::kind(MarkupKind kind) -> MarkupContent& {
  lsp_runtime_error("MarkupContent::kind: not implement yet");
  return *this;
}

auto MarkupContent::value(std::string value) -> MarkupContent& {
  repr_->emplace("value", std::move(value));
  return *this;
}

InlayHintOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlayHintOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> InlayHintOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto InlayHintOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> InlayHintOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

RelatedFullDocumentDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "full") return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto RelatedFullDocumentDiagnosticReport::relatedDocuments() const
    -> std::optional<Map<
        std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                  UnchangedDocumentDiagnosticReport>>> {
  if (!repr_->contains("relatedDocuments")) return std::nullopt;

  auto& value = (*repr_)["relatedDocuments"];

  assert(value.is_object());
  return Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>(value);
}

auto RelatedFullDocumentDiagnosticReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedFullDocumentDiagnosticReport::resultId() const
    -> std::optional<std::string> {
  if (!repr_->contains("resultId")) return std::nullopt;

  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedFullDocumentDiagnosticReport::items() const -> Vector<Diagnostic> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto RelatedFullDocumentDiagnosticReport::relatedDocuments(
    std::optional<Map<std::string,
                      std::variant<std::monostate, FullDocumentDiagnosticReport,
                                   UnchangedDocumentDiagnosticReport>>>
        relatedDocuments) -> RelatedFullDocumentDiagnosticReport& {
  if (!relatedDocuments.has_value()) {
    repr_->erase("relatedDocuments");
    return *this;
  }
  lsp_runtime_error(
      "RelatedFullDocumentDiagnosticReport::relatedDocuments: not implement "
      "yet");
  return *this;
}

auto RelatedFullDocumentDiagnosticReport::kind(std::string kind)
    -> RelatedFullDocumentDiagnosticReport& {
  lsp_runtime_error(
      "RelatedFullDocumentDiagnosticReport::kind: not implement yet");
  return *this;
}

auto RelatedFullDocumentDiagnosticReport::resultId(
    std::optional<std::string> resultId)
    -> RelatedFullDocumentDiagnosticReport& {
  if (!resultId.has_value()) {
    repr_->erase("resultId");
    return *this;
  }
  repr_->emplace("resultId", std::move(resultId.value()));
  return *this;
}

auto RelatedFullDocumentDiagnosticReport::items(Vector<Diagnostic> items)
    -> RelatedFullDocumentDiagnosticReport& {
  lsp_runtime_error(
      "RelatedFullDocumentDiagnosticReport::items: not implement yet");
  return *this;
}

RelatedUnchangedDocumentDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "unchanged") return false;
  if (!repr_->contains("resultId")) return false;
  return true;
}

auto RelatedUnchangedDocumentDiagnosticReport::relatedDocuments() const
    -> std::optional<Map<
        std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                  UnchangedDocumentDiagnosticReport>>> {
  if (!repr_->contains("relatedDocuments")) return std::nullopt;

  auto& value = (*repr_)["relatedDocuments"];

  assert(value.is_object());
  return Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>(value);
}

auto RelatedUnchangedDocumentDiagnosticReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedUnchangedDocumentDiagnosticReport::resultId() const -> std::string {
  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelatedUnchangedDocumentDiagnosticReport::relatedDocuments(
    std::optional<Map<std::string,
                      std::variant<std::monostate, FullDocumentDiagnosticReport,
                                   UnchangedDocumentDiagnosticReport>>>
        relatedDocuments) -> RelatedUnchangedDocumentDiagnosticReport& {
  if (!relatedDocuments.has_value()) {
    repr_->erase("relatedDocuments");
    return *this;
  }
  lsp_runtime_error(
      "RelatedUnchangedDocumentDiagnosticReport::relatedDocuments: not "
      "implement yet");
  return *this;
}

auto RelatedUnchangedDocumentDiagnosticReport::kind(std::string kind)
    -> RelatedUnchangedDocumentDiagnosticReport& {
  lsp_runtime_error(
      "RelatedUnchangedDocumentDiagnosticReport::kind: not implement yet");
  return *this;
}

auto RelatedUnchangedDocumentDiagnosticReport::resultId(std::string resultId)
    -> RelatedUnchangedDocumentDiagnosticReport& {
  repr_->emplace("resultId", std::move(resultId));
  return *this;
}

FullDocumentDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "full") return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto FullDocumentDiagnosticReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FullDocumentDiagnosticReport::resultId() const
    -> std::optional<std::string> {
  if (!repr_->contains("resultId")) return std::nullopt;

  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FullDocumentDiagnosticReport::items() const -> Vector<Diagnostic> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto FullDocumentDiagnosticReport::kind(std::string kind)
    -> FullDocumentDiagnosticReport& {
  lsp_runtime_error("FullDocumentDiagnosticReport::kind: not implement yet");
  return *this;
}

auto FullDocumentDiagnosticReport::resultId(std::optional<std::string> resultId)
    -> FullDocumentDiagnosticReport& {
  if (!resultId.has_value()) {
    repr_->erase("resultId");
    return *this;
  }
  repr_->emplace("resultId", std::move(resultId.value()));
  return *this;
}

auto FullDocumentDiagnosticReport::items(Vector<Diagnostic> items)
    -> FullDocumentDiagnosticReport& {
  lsp_runtime_error("FullDocumentDiagnosticReport::items: not implement yet");
  return *this;
}

UnchangedDocumentDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "unchanged") return false;
  if (!repr_->contains("resultId")) return false;
  return true;
}

auto UnchangedDocumentDiagnosticReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto UnchangedDocumentDiagnosticReport::resultId() const -> std::string {
  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto UnchangedDocumentDiagnosticReport::kind(std::string kind)
    -> UnchangedDocumentDiagnosticReport& {
  lsp_runtime_error(
      "UnchangedDocumentDiagnosticReport::kind: not implement yet");
  return *this;
}

auto UnchangedDocumentDiagnosticReport::resultId(std::string resultId)
    -> UnchangedDocumentDiagnosticReport& {
  repr_->emplace("resultId", std::move(resultId));
  return *this;
}

DiagnosticOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("interFileDependencies")) return false;
  if (!repr_->contains("workspaceDiagnostics")) return false;
  return true;
}

auto DiagnosticOptions::identifier() const -> std::optional<std::string> {
  if (!repr_->contains("identifier")) return std::nullopt;

  auto& value = (*repr_)["identifier"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticOptions::interFileDependencies() const -> bool {
  auto& value = (*repr_)["interFileDependencies"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticOptions::workspaceDiagnostics() const -> bool {
  auto& value = (*repr_)["workspaceDiagnostics"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticOptions::identifier(std::optional<std::string> identifier)
    -> DiagnosticOptions& {
  if (!identifier.has_value()) {
    repr_->erase("identifier");
    return *this;
  }
  repr_->emplace("identifier", std::move(identifier.value()));
  return *this;
}

auto DiagnosticOptions::interFileDependencies(bool interFileDependencies)
    -> DiagnosticOptions& {
  repr_->emplace("interFileDependencies", std::move(interFileDependencies));
  return *this;
}

auto DiagnosticOptions::workspaceDiagnostics(bool workspaceDiagnostics)
    -> DiagnosticOptions& {
  repr_->emplace("workspaceDiagnostics", std::move(workspaceDiagnostics));
  return *this;
}

auto DiagnosticOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DiagnosticOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

PreviousResultId::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto PreviousResultId::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PreviousResultId::value() const -> std::string {
  auto& value = (*repr_)["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PreviousResultId::uri(std::string uri) -> PreviousResultId& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto PreviousResultId::value(std::string value) -> PreviousResultId& {
  repr_->emplace("value", std::move(value));
  return *this;
}

NotebookDocument::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("notebookType")) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("cells")) return false;
  return true;
}

auto NotebookDocument::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocument::notebookType() const -> std::string {
  auto& value = (*repr_)["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocument::version() const -> int {
  auto& value = (*repr_)["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto NotebookDocument::metadata() const -> std::optional<LSPObject> {
  if (!repr_->contains("metadata")) return std::nullopt;

  auto& value = (*repr_)["metadata"];

  assert(value.is_object());
  return LSPObject(value);
}

auto NotebookDocument::cells() const -> Vector<NotebookCell> {
  auto& value = (*repr_)["cells"];

  assert(value.is_array());
  return Vector<NotebookCell>(value);
}

auto NotebookDocument::uri(std::string uri) -> NotebookDocument& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto NotebookDocument::notebookType(std::string notebookType)
    -> NotebookDocument& {
  repr_->emplace("notebookType", std::move(notebookType));
  return *this;
}

auto NotebookDocument::version(int version) -> NotebookDocument& {
  repr_->emplace("version", std::move(version));
  return *this;
}

auto NotebookDocument::metadata(std::optional<LSPObject> metadata)
    -> NotebookDocument& {
  if (!metadata.has_value()) {
    repr_->erase("metadata");
    return *this;
  }
  lsp_runtime_error("NotebookDocument::metadata: not implement yet");
  return *this;
}

auto NotebookDocument::cells(Vector<NotebookCell> cells) -> NotebookDocument& {
  lsp_runtime_error("NotebookDocument::cells: not implement yet");
  return *this;
}

TextDocumentItem::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("languageId")) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("text")) return false;
  return true;
}

auto TextDocumentItem::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentItem::languageId() const -> LanguageKind {
  auto& value = (*repr_)["languageId"];

  lsp_runtime_error("TextDocumentItem::languageId: not implement yet");
}

auto TextDocumentItem::version() const -> int {
  auto& value = (*repr_)["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto TextDocumentItem::text() const -> std::string {
  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentItem::uri(std::string uri) -> TextDocumentItem& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto TextDocumentItem::languageId(LanguageKind languageId)
    -> TextDocumentItem& {
  lsp_runtime_error("TextDocumentItem::languageId: not implement yet");
  return *this;
}

auto TextDocumentItem::version(int version) -> TextDocumentItem& {
  repr_->emplace("version", std::move(version));
  return *this;
}

auto TextDocumentItem::text(std::string text) -> TextDocumentItem& {
  repr_->emplace("text", std::move(text));
  return *this;
}

NotebookDocumentSyncOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookSelector")) return false;
  return true;
}

auto NotebookDocumentSyncOptions::notebookSelector() const
    -> Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                           NotebookDocumentFilterWithCells>> {
  auto& value = (*repr_)["notebookSelector"];

  assert(value.is_array());
  return Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>(value);
}

auto NotebookDocumentSyncOptions::save() const -> std::optional<bool> {
  if (!repr_->contains("save")) return std::nullopt;

  auto& value = (*repr_)["save"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncOptions::notebookSelector(
    Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                        NotebookDocumentFilterWithCells>>
        notebookSelector) -> NotebookDocumentSyncOptions& {
  lsp_runtime_error(
      "NotebookDocumentSyncOptions::notebookSelector: not implement yet");
  return *this;
}

auto NotebookDocumentSyncOptions::save(std::optional<bool> save)
    -> NotebookDocumentSyncOptions& {
  if (!save.has_value()) {
    repr_->erase("save");
    return *this;
  }
  repr_->emplace("save", std::move(save.value()));
  return *this;
}

VersionedNotebookDocumentIdentifier::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto VersionedNotebookDocumentIdentifier::version() const -> int {
  auto& value = (*repr_)["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto VersionedNotebookDocumentIdentifier::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto VersionedNotebookDocumentIdentifier::version(int version)
    -> VersionedNotebookDocumentIdentifier& {
  repr_->emplace("version", std::move(version));
  return *this;
}

auto VersionedNotebookDocumentIdentifier::uri(std::string uri)
    -> VersionedNotebookDocumentIdentifier& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

NotebookDocumentChangeEvent::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto NotebookDocumentChangeEvent::metadata() const -> std::optional<LSPObject> {
  if (!repr_->contains("metadata")) return std::nullopt;

  auto& value = (*repr_)["metadata"];

  assert(value.is_object());
  return LSPObject(value);
}

auto NotebookDocumentChangeEvent::cells() const
    -> std::optional<NotebookDocumentCellChanges> {
  if (!repr_->contains("cells")) return std::nullopt;

  auto& value = (*repr_)["cells"];

  return NotebookDocumentCellChanges(value);
}

auto NotebookDocumentChangeEvent::metadata(std::optional<LSPObject> metadata)
    -> NotebookDocumentChangeEvent& {
  if (!metadata.has_value()) {
    repr_->erase("metadata");
    return *this;
  }
  lsp_runtime_error("NotebookDocumentChangeEvent::metadata: not implement yet");
  return *this;
}

auto NotebookDocumentChangeEvent::cells(
    std::optional<NotebookDocumentCellChanges> cells)
    -> NotebookDocumentChangeEvent& {
  if (!cells.has_value()) {
    repr_->erase("cells");
    return *this;
  }
  repr_->emplace("cells", cells.value());
  return *this;
}

NotebookDocumentIdentifier::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto NotebookDocumentIdentifier::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentIdentifier::uri(std::string uri)
    -> NotebookDocumentIdentifier& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

InlineCompletionContext::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("triggerKind")) return false;
  return true;
}

auto InlineCompletionContext::triggerKind() const
    -> InlineCompletionTriggerKind {
  auto& value = (*repr_)["triggerKind"];

  return InlineCompletionTriggerKind(value);
}

auto InlineCompletionContext::selectedCompletionInfo() const
    -> std::optional<SelectedCompletionInfo> {
  if (!repr_->contains("selectedCompletionInfo")) return std::nullopt;

  auto& value = (*repr_)["selectedCompletionInfo"];

  return SelectedCompletionInfo(value);
}

auto InlineCompletionContext::triggerKind(
    InlineCompletionTriggerKind triggerKind) -> InlineCompletionContext& {
  repr_->emplace("triggerKind", static_cast<long>(triggerKind));
  return *this;
}

auto InlineCompletionContext::selectedCompletionInfo(
    std::optional<SelectedCompletionInfo> selectedCompletionInfo)
    -> InlineCompletionContext& {
  if (!selectedCompletionInfo.has_value()) {
    repr_->erase("selectedCompletionInfo");
    return *this;
  }
  repr_->emplace("selectedCompletionInfo", selectedCompletionInfo.value());
  return *this;
}

StringValue::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "snippet") return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto StringValue::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto StringValue::value() const -> std::string {
  auto& value = (*repr_)["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto StringValue::kind(std::string kind) -> StringValue& {
  lsp_runtime_error("StringValue::kind: not implement yet");
  return *this;
}

auto StringValue::value(std::string value) -> StringValue& {
  repr_->emplace("value", std::move(value));
  return *this;
}

InlineCompletionOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlineCompletionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineCompletionOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> InlineCompletionOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

TextDocumentContentOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("schemes")) return false;
  return true;
}

auto TextDocumentContentOptions::schemes() const -> Vector<std::string> {
  auto& value = (*repr_)["schemes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto TextDocumentContentOptions::schemes(Vector<std::string> schemes)
    -> TextDocumentContentOptions& {
  lsp_runtime_error("TextDocumentContentOptions::schemes: not implement yet");
  return *this;
}

Registration::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("id")) return false;
  if (!repr_->contains("method")) return false;
  return true;
}

auto Registration::id() const -> std::string {
  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Registration::method() const -> std::string {
  auto& value = (*repr_)["method"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Registration::registerOptions() const -> std::optional<LSPAny> {
  if (!repr_->contains("registerOptions")) return std::nullopt;

  auto& value = (*repr_)["registerOptions"];

  assert(value.is_object());
  return LSPAny(value);
}

auto Registration::id(std::string id) -> Registration& {
  repr_->emplace("id", std::move(id));
  return *this;
}

auto Registration::method(std::string method) -> Registration& {
  repr_->emplace("method", std::move(method));
  return *this;
}

auto Registration::registerOptions(std::optional<LSPAny> registerOptions)
    -> Registration& {
  if (!registerOptions.has_value()) {
    repr_->erase("registerOptions");
    return *this;
  }
  lsp_runtime_error("Registration::registerOptions: not implement yet");
  return *this;
}

Unregistration::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("id")) return false;
  if (!repr_->contains("method")) return false;
  return true;
}

auto Unregistration::id() const -> std::string {
  auto& value = (*repr_)["id"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Unregistration::method() const -> std::string {
  auto& value = (*repr_)["method"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Unregistration::id(std::string id) -> Unregistration& {
  repr_->emplace("id", std::move(id));
  return *this;
}

auto Unregistration::method(std::string method) -> Unregistration& {
  repr_->emplace("method", std::move(method));
  return *this;
}

_InitializeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("processId")) return false;
  if (!repr_->contains("rootUri")) return false;
  if (!repr_->contains("capabilities")) return false;
  return true;
}

auto _InitializeParams::processId() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  auto& value = (*repr_)["processId"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::clientInfo() const -> std::optional<ClientInfo> {
  if (!repr_->contains("clientInfo")) return std::nullopt;

  auto& value = (*repr_)["clientInfo"];

  return ClientInfo(value);
}

auto _InitializeParams::locale() const -> std::optional<std::string> {
  if (!repr_->contains("locale")) return std::nullopt;

  auto& value = (*repr_)["locale"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto _InitializeParams::rootPath() const -> std::optional<
    std::variant<std::monostate, std::string, std::nullptr_t>> {
  if (!repr_->contains("rootPath")) return std::nullopt;

  auto& value = (*repr_)["rootPath"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::rootUri() const
    -> std::variant<std::monostate, std::string, std::nullptr_t> {
  auto& value = (*repr_)["rootUri"];

  std::variant<std::monostate, std::string, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::capabilities() const -> ClientCapabilities {
  auto& value = (*repr_)["capabilities"];

  return ClientCapabilities(value);
}

auto _InitializeParams::initializationOptions() const -> std::optional<LSPAny> {
  if (!repr_->contains("initializationOptions")) return std::nullopt;

  auto& value = (*repr_)["initializationOptions"];

  assert(value.is_object());
  return LSPAny(value);
}

auto _InitializeParams::trace() const -> std::optional<TraceValue> {
  if (!repr_->contains("trace")) return std::nullopt;

  auto& value = (*repr_)["trace"];

  lsp_runtime_error("_InitializeParams::trace: not implement yet");
}

auto _InitializeParams::workDoneToken() const -> std::optional<ProgressToken> {
  if (!repr_->contains("workDoneToken")) return std::nullopt;

  auto& value = (*repr_)["workDoneToken"];

  ProgressToken result;

  details::try_emplace(result, value);

  return result;
}

auto _InitializeParams::processId(
    std::variant<std::monostate, int, std::nullptr_t> processId)
    -> _InitializeParams& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int processId) {
      repr_->emplace("processId", std::move(processId));
    }

    void operator()(std::nullptr_t processId) {
      repr_->emplace("processId", std::move(processId));
    }
  } v{repr_};

  std::visit(v, processId);

  return *this;
}

auto _InitializeParams::clientInfo(std::optional<ClientInfo> clientInfo)
    -> _InitializeParams& {
  if (!clientInfo.has_value()) {
    repr_->erase("clientInfo");
    return *this;
  }
  repr_->emplace("clientInfo", clientInfo.value());
  return *this;
}

auto _InitializeParams::locale(std::optional<std::string> locale)
    -> _InitializeParams& {
  if (!locale.has_value()) {
    repr_->erase("locale");
    return *this;
  }
  repr_->emplace("locale", std::move(locale.value()));
  return *this;
}

auto _InitializeParams::rootPath(
    std::optional<std::variant<std::monostate, std::string, std::nullptr_t>>
        rootPath) -> _InitializeParams& {
  if (!rootPath.has_value()) {
    repr_->erase("rootPath");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string rootPath) {
      repr_->emplace("rootPath", std::move(rootPath));
    }

    void operator()(std::nullptr_t rootPath) {
      repr_->emplace("rootPath", std::move(rootPath));
    }
  } v{repr_};

  std::visit(v, rootPath.value());

  return *this;
}

auto _InitializeParams::rootUri(
    std::variant<std::monostate, std::string, std::nullptr_t> rootUri)
    -> _InitializeParams& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string rootUri) {
      repr_->emplace("rootUri", std::move(rootUri));
    }

    void operator()(std::nullptr_t rootUri) {
      repr_->emplace("rootUri", std::move(rootUri));
    }
  } v{repr_};

  std::visit(v, rootUri);

  return *this;
}

auto _InitializeParams::capabilities(ClientCapabilities capabilities)
    -> _InitializeParams& {
  repr_->emplace("capabilities", capabilities);
  return *this;
}

auto _InitializeParams::initializationOptions(
    std::optional<LSPAny> initializationOptions) -> _InitializeParams& {
  if (!initializationOptions.has_value()) {
    repr_->erase("initializationOptions");
    return *this;
  }
  lsp_runtime_error(
      "_InitializeParams::initializationOptions: not implement yet");
  return *this;
}

auto _InitializeParams::trace(std::optional<TraceValue> trace)
    -> _InitializeParams& {
  if (!trace.has_value()) {
    repr_->erase("trace");
    return *this;
  }
  lsp_runtime_error("_InitializeParams::trace: not implement yet");
  return *this;
}

auto _InitializeParams::workDoneToken(
    std::optional<ProgressToken> workDoneToken) -> _InitializeParams& {
  if (!workDoneToken.has_value()) {
    repr_->erase("workDoneToken");
    return *this;
  }
  lsp_runtime_error("_InitializeParams::workDoneToken: not implement yet");
  return *this;
}

WorkspaceFoldersInitializeParams::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceFoldersInitializeParams::workspaceFolders() const
    -> std::optional<
        std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>> {
  if (!repr_->contains("workspaceFolders")) return std::nullopt;

  auto& value = (*repr_)["workspaceFolders"];

  std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFoldersInitializeParams::workspaceFolders(
    std::optional<
        std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>
        workspaceFolders) -> WorkspaceFoldersInitializeParams& {
  if (!workspaceFolders.has_value()) {
    repr_->erase("workspaceFolders");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(Vector<WorkspaceFolder> workspaceFolders) {
      lsp_runtime_error(
          "WorkspaceFoldersInitializeParams::workspaceFolders: not implement "
          "yet");
    }

    void operator()(std::nullptr_t workspaceFolders) {
      repr_->emplace("workspaceFolders", std::move(workspaceFolders));
    }
  } v{repr_};

  std::visit(v, workspaceFolders.value());

  return *this;
}

ServerCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ServerCapabilities::positionEncoding() const
    -> std::optional<PositionEncodingKind> {
  if (!repr_->contains("positionEncoding")) return std::nullopt;

  auto& value = (*repr_)["positionEncoding"];

  lsp_runtime_error("ServerCapabilities::positionEncoding: not implement yet");
}

auto ServerCapabilities::textDocumentSync() const
    -> std::optional<std::variant<std::monostate, TextDocumentSyncOptions,
                                  TextDocumentSyncKind>> {
  if (!repr_->contains("textDocumentSync")) return std::nullopt;

  auto& value = (*repr_)["textDocumentSync"];

  std::variant<std::monostate, TextDocumentSyncOptions, TextDocumentSyncKind>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::notebookDocumentSync() const
    -> std::optional<std::variant<std::monostate, NotebookDocumentSyncOptions,
                                  NotebookDocumentSyncRegistrationOptions>> {
  if (!repr_->contains("notebookDocumentSync")) return std::nullopt;

  auto& value = (*repr_)["notebookDocumentSync"];

  std::variant<std::monostate, NotebookDocumentSyncOptions,
               NotebookDocumentSyncRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::completionProvider() const
    -> std::optional<CompletionOptions> {
  if (!repr_->contains("completionProvider")) return std::nullopt;

  auto& value = (*repr_)["completionProvider"];

  return CompletionOptions(value);
}

auto ServerCapabilities::hoverProvider() const
    -> std::optional<std::variant<std::monostate, bool, HoverOptions>> {
  if (!repr_->contains("hoverProvider")) return std::nullopt;

  auto& value = (*repr_)["hoverProvider"];

  std::variant<std::monostate, bool, HoverOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::signatureHelpProvider() const
    -> std::optional<SignatureHelpOptions> {
  if (!repr_->contains("signatureHelpProvider")) return std::nullopt;

  auto& value = (*repr_)["signatureHelpProvider"];

  return SignatureHelpOptions(value);
}

auto ServerCapabilities::declarationProvider() const
    -> std::optional<std::variant<std::monostate, bool, DeclarationOptions,
                                  DeclarationRegistrationOptions>> {
  if (!repr_->contains("declarationProvider")) return std::nullopt;

  auto& value = (*repr_)["declarationProvider"];

  std::variant<std::monostate, bool, DeclarationOptions,
               DeclarationRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::definitionProvider() const
    -> std::optional<std::variant<std::monostate, bool, DefinitionOptions>> {
  if (!repr_->contains("definitionProvider")) return std::nullopt;

  auto& value = (*repr_)["definitionProvider"];

  std::variant<std::monostate, bool, DefinitionOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::typeDefinitionProvider() const
    -> std::optional<std::variant<std::monostate, bool, TypeDefinitionOptions,
                                  TypeDefinitionRegistrationOptions>> {
  if (!repr_->contains("typeDefinitionProvider")) return std::nullopt;

  auto& value = (*repr_)["typeDefinitionProvider"];

  std::variant<std::monostate, bool, TypeDefinitionOptions,
               TypeDefinitionRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::implementationProvider() const
    -> std::optional<std::variant<std::monostate, bool, ImplementationOptions,
                                  ImplementationRegistrationOptions>> {
  if (!repr_->contains("implementationProvider")) return std::nullopt;

  auto& value = (*repr_)["implementationProvider"];

  std::variant<std::monostate, bool, ImplementationOptions,
               ImplementationRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::referencesProvider() const
    -> std::optional<std::variant<std::monostate, bool, ReferenceOptions>> {
  if (!repr_->contains("referencesProvider")) return std::nullopt;

  auto& value = (*repr_)["referencesProvider"];

  std::variant<std::monostate, bool, ReferenceOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentHighlightProvider() const -> std::optional<
    std::variant<std::monostate, bool, DocumentHighlightOptions>> {
  if (!repr_->contains("documentHighlightProvider")) return std::nullopt;

  auto& value = (*repr_)["documentHighlightProvider"];

  std::variant<std::monostate, bool, DocumentHighlightOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentSymbolProvider() const -> std::optional<
    std::variant<std::monostate, bool, DocumentSymbolOptions>> {
  if (!repr_->contains("documentSymbolProvider")) return std::nullopt;

  auto& value = (*repr_)["documentSymbolProvider"];

  std::variant<std::monostate, bool, DocumentSymbolOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::codeActionProvider() const
    -> std::optional<std::variant<std::monostate, bool, CodeActionOptions>> {
  if (!repr_->contains("codeActionProvider")) return std::nullopt;

  auto& value = (*repr_)["codeActionProvider"];

  std::variant<std::monostate, bool, CodeActionOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::codeLensProvider() const
    -> std::optional<CodeLensOptions> {
  if (!repr_->contains("codeLensProvider")) return std::nullopt;

  auto& value = (*repr_)["codeLensProvider"];

  return CodeLensOptions(value);
}

auto ServerCapabilities::documentLinkProvider() const
    -> std::optional<DocumentLinkOptions> {
  if (!repr_->contains("documentLinkProvider")) return std::nullopt;

  auto& value = (*repr_)["documentLinkProvider"];

  return DocumentLinkOptions(value);
}

auto ServerCapabilities::colorProvider() const
    -> std::optional<std::variant<std::monostate, bool, DocumentColorOptions,
                                  DocumentColorRegistrationOptions>> {
  if (!repr_->contains("colorProvider")) return std::nullopt;

  auto& value = (*repr_)["colorProvider"];

  std::variant<std::monostate, bool, DocumentColorOptions,
               DocumentColorRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::workspaceSymbolProvider() const -> std::optional<
    std::variant<std::monostate, bool, WorkspaceSymbolOptions>> {
  if (!repr_->contains("workspaceSymbolProvider")) return std::nullopt;

  auto& value = (*repr_)["workspaceSymbolProvider"];

  std::variant<std::monostate, bool, WorkspaceSymbolOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentFormattingProvider() const -> std::optional<
    std::variant<std::monostate, bool, DocumentFormattingOptions>> {
  if (!repr_->contains("documentFormattingProvider")) return std::nullopt;

  auto& value = (*repr_)["documentFormattingProvider"];

  std::variant<std::monostate, bool, DocumentFormattingOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentRangeFormattingProvider() const
    -> std::optional<
        std::variant<std::monostate, bool, DocumentRangeFormattingOptions>> {
  if (!repr_->contains("documentRangeFormattingProvider")) return std::nullopt;

  auto& value = (*repr_)["documentRangeFormattingProvider"];

  std::variant<std::monostate, bool, DocumentRangeFormattingOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::documentOnTypeFormattingProvider() const
    -> std::optional<DocumentOnTypeFormattingOptions> {
  if (!repr_->contains("documentOnTypeFormattingProvider")) return std::nullopt;

  auto& value = (*repr_)["documentOnTypeFormattingProvider"];

  return DocumentOnTypeFormattingOptions(value);
}

auto ServerCapabilities::renameProvider() const
    -> std::optional<std::variant<std::monostate, bool, RenameOptions>> {
  if (!repr_->contains("renameProvider")) return std::nullopt;

  auto& value = (*repr_)["renameProvider"];

  std::variant<std::monostate, bool, RenameOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::foldingRangeProvider() const
    -> std::optional<std::variant<std::monostate, bool, FoldingRangeOptions,
                                  FoldingRangeRegistrationOptions>> {
  if (!repr_->contains("foldingRangeProvider")) return std::nullopt;

  auto& value = (*repr_)["foldingRangeProvider"];

  std::variant<std::monostate, bool, FoldingRangeOptions,
               FoldingRangeRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::selectionRangeProvider() const
    -> std::optional<std::variant<std::monostate, bool, SelectionRangeOptions,
                                  SelectionRangeRegistrationOptions>> {
  if (!repr_->contains("selectionRangeProvider")) return std::nullopt;

  auto& value = (*repr_)["selectionRangeProvider"];

  std::variant<std::monostate, bool, SelectionRangeOptions,
               SelectionRangeRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::executeCommandProvider() const
    -> std::optional<ExecuteCommandOptions> {
  if (!repr_->contains("executeCommandProvider")) return std::nullopt;

  auto& value = (*repr_)["executeCommandProvider"];

  return ExecuteCommandOptions(value);
}

auto ServerCapabilities::callHierarchyProvider() const
    -> std::optional<std::variant<std::monostate, bool, CallHierarchyOptions,
                                  CallHierarchyRegistrationOptions>> {
  if (!repr_->contains("callHierarchyProvider")) return std::nullopt;

  auto& value = (*repr_)["callHierarchyProvider"];

  std::variant<std::monostate, bool, CallHierarchyOptions,
               CallHierarchyRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::linkedEditingRangeProvider() const -> std::optional<
    std::variant<std::monostate, bool, LinkedEditingRangeOptions,
                 LinkedEditingRangeRegistrationOptions>> {
  if (!repr_->contains("linkedEditingRangeProvider")) return std::nullopt;

  auto& value = (*repr_)["linkedEditingRangeProvider"];

  std::variant<std::monostate, bool, LinkedEditingRangeOptions,
               LinkedEditingRangeRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::semanticTokensProvider() const
    -> std::optional<std::variant<std::monostate, SemanticTokensOptions,
                                  SemanticTokensRegistrationOptions>> {
  if (!repr_->contains("semanticTokensProvider")) return std::nullopt;

  auto& value = (*repr_)["semanticTokensProvider"];

  std::variant<std::monostate, SemanticTokensOptions,
               SemanticTokensRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::monikerProvider() const
    -> std::optional<std::variant<std::monostate, bool, MonikerOptions,
                                  MonikerRegistrationOptions>> {
  if (!repr_->contains("monikerProvider")) return std::nullopt;

  auto& value = (*repr_)["monikerProvider"];

  std::variant<std::monostate, bool, MonikerOptions, MonikerRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::typeHierarchyProvider() const
    -> std::optional<std::variant<std::monostate, bool, TypeHierarchyOptions,
                                  TypeHierarchyRegistrationOptions>> {
  if (!repr_->contains("typeHierarchyProvider")) return std::nullopt;

  auto& value = (*repr_)["typeHierarchyProvider"];

  std::variant<std::monostate, bool, TypeHierarchyOptions,
               TypeHierarchyRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::inlineValueProvider() const
    -> std::optional<std::variant<std::monostate, bool, InlineValueOptions,
                                  InlineValueRegistrationOptions>> {
  if (!repr_->contains("inlineValueProvider")) return std::nullopt;

  auto& value = (*repr_)["inlineValueProvider"];

  std::variant<std::monostate, bool, InlineValueOptions,
               InlineValueRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::inlayHintProvider() const
    -> std::optional<std::variant<std::monostate, bool, InlayHintOptions,
                                  InlayHintRegistrationOptions>> {
  if (!repr_->contains("inlayHintProvider")) return std::nullopt;

  auto& value = (*repr_)["inlayHintProvider"];

  std::variant<std::monostate, bool, InlayHintOptions,
               InlayHintRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::diagnosticProvider() const
    -> std::optional<std::variant<std::monostate, DiagnosticOptions,
                                  DiagnosticRegistrationOptions>> {
  if (!repr_->contains("diagnosticProvider")) return std::nullopt;

  auto& value = (*repr_)["diagnosticProvider"];

  std::variant<std::monostate, DiagnosticOptions, DiagnosticRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::inlineCompletionProvider() const -> std::optional<
    std::variant<std::monostate, bool, InlineCompletionOptions>> {
  if (!repr_->contains("inlineCompletionProvider")) return std::nullopt;

  auto& value = (*repr_)["inlineCompletionProvider"];

  std::variant<std::monostate, bool, InlineCompletionOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto ServerCapabilities::workspace() const -> std::optional<WorkspaceOptions> {
  if (!repr_->contains("workspace")) return std::nullopt;

  auto& value = (*repr_)["workspace"];

  return WorkspaceOptions(value);
}

auto ServerCapabilities::experimental() const -> std::optional<LSPAny> {
  if (!repr_->contains("experimental")) return std::nullopt;

  auto& value = (*repr_)["experimental"];

  assert(value.is_object());
  return LSPAny(value);
}

auto ServerCapabilities::positionEncoding(
    std::optional<PositionEncodingKind> positionEncoding)
    -> ServerCapabilities& {
  if (!positionEncoding.has_value()) {
    repr_->erase("positionEncoding");
    return *this;
  }
  lsp_runtime_error("ServerCapabilities::positionEncoding: not implement yet");
  return *this;
}

auto ServerCapabilities::textDocumentSync(
    std::optional<std::variant<std::monostate, TextDocumentSyncOptions,
                               TextDocumentSyncKind>>
        textDocumentSync) -> ServerCapabilities& {
  if (!textDocumentSync.has_value()) {
    repr_->erase("textDocumentSync");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(TextDocumentSyncOptions textDocumentSync) {
      repr_->emplace("textDocumentSync", textDocumentSync);
    }

    void operator()(TextDocumentSyncKind textDocumentSync) {
      repr_->emplace("textDocumentSync", static_cast<long>(textDocumentSync));
    }
  } v{repr_};

  std::visit(v, textDocumentSync.value());

  return *this;
}

auto ServerCapabilities::notebookDocumentSync(
    std::optional<std::variant<std::monostate, NotebookDocumentSyncOptions,
                               NotebookDocumentSyncRegistrationOptions>>
        notebookDocumentSync) -> ServerCapabilities& {
  if (!notebookDocumentSync.has_value()) {
    repr_->erase("notebookDocumentSync");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(NotebookDocumentSyncOptions notebookDocumentSync) {
      repr_->emplace("notebookDocumentSync", notebookDocumentSync);
    }

    void operator()(
        NotebookDocumentSyncRegistrationOptions notebookDocumentSync) {
      repr_->emplace("notebookDocumentSync", notebookDocumentSync);
    }
  } v{repr_};

  std::visit(v, notebookDocumentSync.value());

  return *this;
}

auto ServerCapabilities::completionProvider(
    std::optional<CompletionOptions> completionProvider)
    -> ServerCapabilities& {
  if (!completionProvider.has_value()) {
    repr_->erase("completionProvider");
    return *this;
  }
  repr_->emplace("completionProvider", completionProvider.value());
  return *this;
}

auto ServerCapabilities::hoverProvider(
    std::optional<std::variant<std::monostate, bool, HoverOptions>>
        hoverProvider) -> ServerCapabilities& {
  if (!hoverProvider.has_value()) {
    repr_->erase("hoverProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool hoverProvider) {
      repr_->emplace("hoverProvider", std::move(hoverProvider));
    }

    void operator()(HoverOptions hoverProvider) {
      repr_->emplace("hoverProvider", hoverProvider);
    }
  } v{repr_};

  std::visit(v, hoverProvider.value());

  return *this;
}

auto ServerCapabilities::signatureHelpProvider(
    std::optional<SignatureHelpOptions> signatureHelpProvider)
    -> ServerCapabilities& {
  if (!signatureHelpProvider.has_value()) {
    repr_->erase("signatureHelpProvider");
    return *this;
  }
  repr_->emplace("signatureHelpProvider", signatureHelpProvider.value());
  return *this;
}

auto ServerCapabilities::declarationProvider(
    std::optional<std::variant<std::monostate, bool, DeclarationOptions,
                               DeclarationRegistrationOptions>>
        declarationProvider) -> ServerCapabilities& {
  if (!declarationProvider.has_value()) {
    repr_->erase("declarationProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool declarationProvider) {
      repr_->emplace("declarationProvider", std::move(declarationProvider));
    }

    void operator()(DeclarationOptions declarationProvider) {
      repr_->emplace("declarationProvider", declarationProvider);
    }

    void operator()(DeclarationRegistrationOptions declarationProvider) {
      repr_->emplace("declarationProvider", declarationProvider);
    }
  } v{repr_};

  std::visit(v, declarationProvider.value());

  return *this;
}

auto ServerCapabilities::definitionProvider(
    std::optional<std::variant<std::monostate, bool, DefinitionOptions>>
        definitionProvider) -> ServerCapabilities& {
  if (!definitionProvider.has_value()) {
    repr_->erase("definitionProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool definitionProvider) {
      repr_->emplace("definitionProvider", std::move(definitionProvider));
    }

    void operator()(DefinitionOptions definitionProvider) {
      repr_->emplace("definitionProvider", definitionProvider);
    }
  } v{repr_};

  std::visit(v, definitionProvider.value());

  return *this;
}

auto ServerCapabilities::typeDefinitionProvider(
    std::optional<std::variant<std::monostate, bool, TypeDefinitionOptions,
                               TypeDefinitionRegistrationOptions>>
        typeDefinitionProvider) -> ServerCapabilities& {
  if (!typeDefinitionProvider.has_value()) {
    repr_->erase("typeDefinitionProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool typeDefinitionProvider) {
      repr_->emplace("typeDefinitionProvider",
                     std::move(typeDefinitionProvider));
    }

    void operator()(TypeDefinitionOptions typeDefinitionProvider) {
      repr_->emplace("typeDefinitionProvider", typeDefinitionProvider);
    }

    void operator()(TypeDefinitionRegistrationOptions typeDefinitionProvider) {
      repr_->emplace("typeDefinitionProvider", typeDefinitionProvider);
    }
  } v{repr_};

  std::visit(v, typeDefinitionProvider.value());

  return *this;
}

auto ServerCapabilities::implementationProvider(
    std::optional<std::variant<std::monostate, bool, ImplementationOptions,
                               ImplementationRegistrationOptions>>
        implementationProvider) -> ServerCapabilities& {
  if (!implementationProvider.has_value()) {
    repr_->erase("implementationProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool implementationProvider) {
      repr_->emplace("implementationProvider",
                     std::move(implementationProvider));
    }

    void operator()(ImplementationOptions implementationProvider) {
      repr_->emplace("implementationProvider", implementationProvider);
    }

    void operator()(ImplementationRegistrationOptions implementationProvider) {
      repr_->emplace("implementationProvider", implementationProvider);
    }
  } v{repr_};

  std::visit(v, implementationProvider.value());

  return *this;
}

auto ServerCapabilities::referencesProvider(
    std::optional<std::variant<std::monostate, bool, ReferenceOptions>>
        referencesProvider) -> ServerCapabilities& {
  if (!referencesProvider.has_value()) {
    repr_->erase("referencesProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool referencesProvider) {
      repr_->emplace("referencesProvider", std::move(referencesProvider));
    }

    void operator()(ReferenceOptions referencesProvider) {
      repr_->emplace("referencesProvider", referencesProvider);
    }
  } v{repr_};

  std::visit(v, referencesProvider.value());

  return *this;
}

auto ServerCapabilities::documentHighlightProvider(
    std::optional<std::variant<std::monostate, bool, DocumentHighlightOptions>>
        documentHighlightProvider) -> ServerCapabilities& {
  if (!documentHighlightProvider.has_value()) {
    repr_->erase("documentHighlightProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool documentHighlightProvider) {
      repr_->emplace("documentHighlightProvider",
                     std::move(documentHighlightProvider));
    }

    void operator()(DocumentHighlightOptions documentHighlightProvider) {
      repr_->emplace("documentHighlightProvider", documentHighlightProvider);
    }
  } v{repr_};

  std::visit(v, documentHighlightProvider.value());

  return *this;
}

auto ServerCapabilities::documentSymbolProvider(
    std::optional<std::variant<std::monostate, bool, DocumentSymbolOptions>>
        documentSymbolProvider) -> ServerCapabilities& {
  if (!documentSymbolProvider.has_value()) {
    repr_->erase("documentSymbolProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool documentSymbolProvider) {
      repr_->emplace("documentSymbolProvider",
                     std::move(documentSymbolProvider));
    }

    void operator()(DocumentSymbolOptions documentSymbolProvider) {
      repr_->emplace("documentSymbolProvider", documentSymbolProvider);
    }
  } v{repr_};

  std::visit(v, documentSymbolProvider.value());

  return *this;
}

auto ServerCapabilities::codeActionProvider(
    std::optional<std::variant<std::monostate, bool, CodeActionOptions>>
        codeActionProvider) -> ServerCapabilities& {
  if (!codeActionProvider.has_value()) {
    repr_->erase("codeActionProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool codeActionProvider) {
      repr_->emplace("codeActionProvider", std::move(codeActionProvider));
    }

    void operator()(CodeActionOptions codeActionProvider) {
      repr_->emplace("codeActionProvider", codeActionProvider);
    }
  } v{repr_};

  std::visit(v, codeActionProvider.value());

  return *this;
}

auto ServerCapabilities::codeLensProvider(
    std::optional<CodeLensOptions> codeLensProvider) -> ServerCapabilities& {
  if (!codeLensProvider.has_value()) {
    repr_->erase("codeLensProvider");
    return *this;
  }
  repr_->emplace("codeLensProvider", codeLensProvider.value());
  return *this;
}

auto ServerCapabilities::documentLinkProvider(
    std::optional<DocumentLinkOptions> documentLinkProvider)
    -> ServerCapabilities& {
  if (!documentLinkProvider.has_value()) {
    repr_->erase("documentLinkProvider");
    return *this;
  }
  repr_->emplace("documentLinkProvider", documentLinkProvider.value());
  return *this;
}

auto ServerCapabilities::colorProvider(
    std::optional<std::variant<std::monostate, bool, DocumentColorOptions,
                               DocumentColorRegistrationOptions>>
        colorProvider) -> ServerCapabilities& {
  if (!colorProvider.has_value()) {
    repr_->erase("colorProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool colorProvider) {
      repr_->emplace("colorProvider", std::move(colorProvider));
    }

    void operator()(DocumentColorOptions colorProvider) {
      repr_->emplace("colorProvider", colorProvider);
    }

    void operator()(DocumentColorRegistrationOptions colorProvider) {
      repr_->emplace("colorProvider", colorProvider);
    }
  } v{repr_};

  std::visit(v, colorProvider.value());

  return *this;
}

auto ServerCapabilities::workspaceSymbolProvider(
    std::optional<std::variant<std::monostate, bool, WorkspaceSymbolOptions>>
        workspaceSymbolProvider) -> ServerCapabilities& {
  if (!workspaceSymbolProvider.has_value()) {
    repr_->erase("workspaceSymbolProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool workspaceSymbolProvider) {
      repr_->emplace("workspaceSymbolProvider",
                     std::move(workspaceSymbolProvider));
    }

    void operator()(WorkspaceSymbolOptions workspaceSymbolProvider) {
      repr_->emplace("workspaceSymbolProvider", workspaceSymbolProvider);
    }
  } v{repr_};

  std::visit(v, workspaceSymbolProvider.value());

  return *this;
}

auto ServerCapabilities::documentFormattingProvider(
    std::optional<std::variant<std::monostate, bool, DocumentFormattingOptions>>
        documentFormattingProvider) -> ServerCapabilities& {
  if (!documentFormattingProvider.has_value()) {
    repr_->erase("documentFormattingProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool documentFormattingProvider) {
      repr_->emplace("documentFormattingProvider",
                     std::move(documentFormattingProvider));
    }

    void operator()(DocumentFormattingOptions documentFormattingProvider) {
      repr_->emplace("documentFormattingProvider", documentFormattingProvider);
    }
  } v{repr_};

  std::visit(v, documentFormattingProvider.value());

  return *this;
}

auto ServerCapabilities::documentRangeFormattingProvider(
    std::optional<
        std::variant<std::monostate, bool, DocumentRangeFormattingOptions>>
        documentRangeFormattingProvider) -> ServerCapabilities& {
  if (!documentRangeFormattingProvider.has_value()) {
    repr_->erase("documentRangeFormattingProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool documentRangeFormattingProvider) {
      repr_->emplace("documentRangeFormattingProvider",
                     std::move(documentRangeFormattingProvider));
    }

    void operator()(
        DocumentRangeFormattingOptions documentRangeFormattingProvider) {
      repr_->emplace("documentRangeFormattingProvider",
                     documentRangeFormattingProvider);
    }
  } v{repr_};

  std::visit(v, documentRangeFormattingProvider.value());

  return *this;
}

auto ServerCapabilities::documentOnTypeFormattingProvider(
    std::optional<DocumentOnTypeFormattingOptions>
        documentOnTypeFormattingProvider) -> ServerCapabilities& {
  if (!documentOnTypeFormattingProvider.has_value()) {
    repr_->erase("documentOnTypeFormattingProvider");
    return *this;
  }
  repr_->emplace("documentOnTypeFormattingProvider",
                 documentOnTypeFormattingProvider.value());
  return *this;
}

auto ServerCapabilities::renameProvider(
    std::optional<std::variant<std::monostate, bool, RenameOptions>>
        renameProvider) -> ServerCapabilities& {
  if (!renameProvider.has_value()) {
    repr_->erase("renameProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool renameProvider) {
      repr_->emplace("renameProvider", std::move(renameProvider));
    }

    void operator()(RenameOptions renameProvider) {
      repr_->emplace("renameProvider", renameProvider);
    }
  } v{repr_};

  std::visit(v, renameProvider.value());

  return *this;
}

auto ServerCapabilities::foldingRangeProvider(
    std::optional<std::variant<std::monostate, bool, FoldingRangeOptions,
                               FoldingRangeRegistrationOptions>>
        foldingRangeProvider) -> ServerCapabilities& {
  if (!foldingRangeProvider.has_value()) {
    repr_->erase("foldingRangeProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool foldingRangeProvider) {
      repr_->emplace("foldingRangeProvider", std::move(foldingRangeProvider));
    }

    void operator()(FoldingRangeOptions foldingRangeProvider) {
      repr_->emplace("foldingRangeProvider", foldingRangeProvider);
    }

    void operator()(FoldingRangeRegistrationOptions foldingRangeProvider) {
      repr_->emplace("foldingRangeProvider", foldingRangeProvider);
    }
  } v{repr_};

  std::visit(v, foldingRangeProvider.value());

  return *this;
}

auto ServerCapabilities::selectionRangeProvider(
    std::optional<std::variant<std::monostate, bool, SelectionRangeOptions,
                               SelectionRangeRegistrationOptions>>
        selectionRangeProvider) -> ServerCapabilities& {
  if (!selectionRangeProvider.has_value()) {
    repr_->erase("selectionRangeProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool selectionRangeProvider) {
      repr_->emplace("selectionRangeProvider",
                     std::move(selectionRangeProvider));
    }

    void operator()(SelectionRangeOptions selectionRangeProvider) {
      repr_->emplace("selectionRangeProvider", selectionRangeProvider);
    }

    void operator()(SelectionRangeRegistrationOptions selectionRangeProvider) {
      repr_->emplace("selectionRangeProvider", selectionRangeProvider);
    }
  } v{repr_};

  std::visit(v, selectionRangeProvider.value());

  return *this;
}

auto ServerCapabilities::executeCommandProvider(
    std::optional<ExecuteCommandOptions> executeCommandProvider)
    -> ServerCapabilities& {
  if (!executeCommandProvider.has_value()) {
    repr_->erase("executeCommandProvider");
    return *this;
  }
  repr_->emplace("executeCommandProvider", executeCommandProvider.value());
  return *this;
}

auto ServerCapabilities::callHierarchyProvider(
    std::optional<std::variant<std::monostate, bool, CallHierarchyOptions,
                               CallHierarchyRegistrationOptions>>
        callHierarchyProvider) -> ServerCapabilities& {
  if (!callHierarchyProvider.has_value()) {
    repr_->erase("callHierarchyProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool callHierarchyProvider) {
      repr_->emplace("callHierarchyProvider", std::move(callHierarchyProvider));
    }

    void operator()(CallHierarchyOptions callHierarchyProvider) {
      repr_->emplace("callHierarchyProvider", callHierarchyProvider);
    }

    void operator()(CallHierarchyRegistrationOptions callHierarchyProvider) {
      repr_->emplace("callHierarchyProvider", callHierarchyProvider);
    }
  } v{repr_};

  std::visit(v, callHierarchyProvider.value());

  return *this;
}

auto ServerCapabilities::linkedEditingRangeProvider(
    std::optional<std::variant<std::monostate, bool, LinkedEditingRangeOptions,
                               LinkedEditingRangeRegistrationOptions>>
        linkedEditingRangeProvider) -> ServerCapabilities& {
  if (!linkedEditingRangeProvider.has_value()) {
    repr_->erase("linkedEditingRangeProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool linkedEditingRangeProvider) {
      repr_->emplace("linkedEditingRangeProvider",
                     std::move(linkedEditingRangeProvider));
    }

    void operator()(LinkedEditingRangeOptions linkedEditingRangeProvider) {
      repr_->emplace("linkedEditingRangeProvider", linkedEditingRangeProvider);
    }

    void operator()(
        LinkedEditingRangeRegistrationOptions linkedEditingRangeProvider) {
      repr_->emplace("linkedEditingRangeProvider", linkedEditingRangeProvider);
    }
  } v{repr_};

  std::visit(v, linkedEditingRangeProvider.value());

  return *this;
}

auto ServerCapabilities::semanticTokensProvider(
    std::optional<std::variant<std::monostate, SemanticTokensOptions,
                               SemanticTokensRegistrationOptions>>
        semanticTokensProvider) -> ServerCapabilities& {
  if (!semanticTokensProvider.has_value()) {
    repr_->erase("semanticTokensProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(SemanticTokensOptions semanticTokensProvider) {
      repr_->emplace("semanticTokensProvider", semanticTokensProvider);
    }

    void operator()(SemanticTokensRegistrationOptions semanticTokensProvider) {
      repr_->emplace("semanticTokensProvider", semanticTokensProvider);
    }
  } v{repr_};

  std::visit(v, semanticTokensProvider.value());

  return *this;
}

auto ServerCapabilities::monikerProvider(
    std::optional<std::variant<std::monostate, bool, MonikerOptions,
                               MonikerRegistrationOptions>>
        monikerProvider) -> ServerCapabilities& {
  if (!monikerProvider.has_value()) {
    repr_->erase("monikerProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool monikerProvider) {
      repr_->emplace("monikerProvider", std::move(monikerProvider));
    }

    void operator()(MonikerOptions monikerProvider) {
      repr_->emplace("monikerProvider", monikerProvider);
    }

    void operator()(MonikerRegistrationOptions monikerProvider) {
      repr_->emplace("monikerProvider", monikerProvider);
    }
  } v{repr_};

  std::visit(v, monikerProvider.value());

  return *this;
}

auto ServerCapabilities::typeHierarchyProvider(
    std::optional<std::variant<std::monostate, bool, TypeHierarchyOptions,
                               TypeHierarchyRegistrationOptions>>
        typeHierarchyProvider) -> ServerCapabilities& {
  if (!typeHierarchyProvider.has_value()) {
    repr_->erase("typeHierarchyProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool typeHierarchyProvider) {
      repr_->emplace("typeHierarchyProvider", std::move(typeHierarchyProvider));
    }

    void operator()(TypeHierarchyOptions typeHierarchyProvider) {
      repr_->emplace("typeHierarchyProvider", typeHierarchyProvider);
    }

    void operator()(TypeHierarchyRegistrationOptions typeHierarchyProvider) {
      repr_->emplace("typeHierarchyProvider", typeHierarchyProvider);
    }
  } v{repr_};

  std::visit(v, typeHierarchyProvider.value());

  return *this;
}

auto ServerCapabilities::inlineValueProvider(
    std::optional<std::variant<std::monostate, bool, InlineValueOptions,
                               InlineValueRegistrationOptions>>
        inlineValueProvider) -> ServerCapabilities& {
  if (!inlineValueProvider.has_value()) {
    repr_->erase("inlineValueProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool inlineValueProvider) {
      repr_->emplace("inlineValueProvider", std::move(inlineValueProvider));
    }

    void operator()(InlineValueOptions inlineValueProvider) {
      repr_->emplace("inlineValueProvider", inlineValueProvider);
    }

    void operator()(InlineValueRegistrationOptions inlineValueProvider) {
      repr_->emplace("inlineValueProvider", inlineValueProvider);
    }
  } v{repr_};

  std::visit(v, inlineValueProvider.value());

  return *this;
}

auto ServerCapabilities::inlayHintProvider(
    std::optional<std::variant<std::monostate, bool, InlayHintOptions,
                               InlayHintRegistrationOptions>>
        inlayHintProvider) -> ServerCapabilities& {
  if (!inlayHintProvider.has_value()) {
    repr_->erase("inlayHintProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool inlayHintProvider) {
      repr_->emplace("inlayHintProvider", std::move(inlayHintProvider));
    }

    void operator()(InlayHintOptions inlayHintProvider) {
      repr_->emplace("inlayHintProvider", inlayHintProvider);
    }

    void operator()(InlayHintRegistrationOptions inlayHintProvider) {
      repr_->emplace("inlayHintProvider", inlayHintProvider);
    }
  } v{repr_};

  std::visit(v, inlayHintProvider.value());

  return *this;
}

auto ServerCapabilities::diagnosticProvider(
    std::optional<std::variant<std::monostate, DiagnosticOptions,
                               DiagnosticRegistrationOptions>>
        diagnosticProvider) -> ServerCapabilities& {
  if (!diagnosticProvider.has_value()) {
    repr_->erase("diagnosticProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(DiagnosticOptions diagnosticProvider) {
      repr_->emplace("diagnosticProvider", diagnosticProvider);
    }

    void operator()(DiagnosticRegistrationOptions diagnosticProvider) {
      repr_->emplace("diagnosticProvider", diagnosticProvider);
    }
  } v{repr_};

  std::visit(v, diagnosticProvider.value());

  return *this;
}

auto ServerCapabilities::inlineCompletionProvider(
    std::optional<std::variant<std::monostate, bool, InlineCompletionOptions>>
        inlineCompletionProvider) -> ServerCapabilities& {
  if (!inlineCompletionProvider.has_value()) {
    repr_->erase("inlineCompletionProvider");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool inlineCompletionProvider) {
      repr_->emplace("inlineCompletionProvider",
                     std::move(inlineCompletionProvider));
    }

    void operator()(InlineCompletionOptions inlineCompletionProvider) {
      repr_->emplace("inlineCompletionProvider", inlineCompletionProvider);
    }
  } v{repr_};

  std::visit(v, inlineCompletionProvider.value());

  return *this;
}

auto ServerCapabilities::workspace(std::optional<WorkspaceOptions> workspace)
    -> ServerCapabilities& {
  if (!workspace.has_value()) {
    repr_->erase("workspace");
    return *this;
  }
  repr_->emplace("workspace", workspace.value());
  return *this;
}

auto ServerCapabilities::experimental(std::optional<LSPAny> experimental)
    -> ServerCapabilities& {
  if (!experimental.has_value()) {
    repr_->erase("experimental");
    return *this;
  }
  lsp_runtime_error("ServerCapabilities::experimental: not implement yet");
  return *this;
}

ServerInfo::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("name")) return false;
  return true;
}

auto ServerInfo::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ServerInfo::version() const -> std::optional<std::string> {
  if (!repr_->contains("version")) return std::nullopt;

  auto& value = (*repr_)["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ServerInfo::name(std::string name) -> ServerInfo& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto ServerInfo::version(std::optional<std::string> version) -> ServerInfo& {
  if (!version.has_value()) {
    repr_->erase("version");
    return *this;
  }
  repr_->emplace("version", std::move(version.value()));
  return *this;
}

VersionedTextDocumentIdentifier::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto VersionedTextDocumentIdentifier::version() const -> int {
  auto& value = (*repr_)["version"];

  assert(value.is_number_integer());
  return value.get<int>();
}

auto VersionedTextDocumentIdentifier::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto VersionedTextDocumentIdentifier::version(int version)
    -> VersionedTextDocumentIdentifier& {
  repr_->emplace("version", std::move(version));
  return *this;
}

auto VersionedTextDocumentIdentifier::uri(std::string uri)
    -> VersionedTextDocumentIdentifier& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

SaveOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SaveOptions::includeText() const -> std::optional<bool> {
  if (!repr_->contains("includeText")) return std::nullopt;

  auto& value = (*repr_)["includeText"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SaveOptions::includeText(std::optional<bool> includeText) -> SaveOptions& {
  if (!includeText.has_value()) {
    repr_->erase("includeText");
    return *this;
  }
  repr_->emplace("includeText", std::move(includeText.value()));
  return *this;
}

FileEvent::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("type")) return false;
  return true;
}

auto FileEvent::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileEvent::type() const -> FileChangeType {
  auto& value = (*repr_)["type"];

  return FileChangeType(value);
}

auto FileEvent::uri(std::string uri) -> FileEvent& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto FileEvent::type(FileChangeType type) -> FileEvent& {
  repr_->emplace("type", static_cast<long>(type));
  return *this;
}

FileSystemWatcher::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("globPattern")) return false;
  return true;
}

auto FileSystemWatcher::globPattern() const -> GlobPattern {
  auto& value = (*repr_)["globPattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto FileSystemWatcher::kind() const -> std::optional<WatchKind> {
  if (!repr_->contains("kind")) return std::nullopt;

  auto& value = (*repr_)["kind"];

  return WatchKind(value);
}

auto FileSystemWatcher::globPattern(GlobPattern globPattern)
    -> FileSystemWatcher& {
  lsp_runtime_error("FileSystemWatcher::globPattern: not implement yet");
  return *this;
}

auto FileSystemWatcher::kind(std::optional<WatchKind> kind)
    -> FileSystemWatcher& {
  if (!kind.has_value()) {
    repr_->erase("kind");
    return *this;
  }
  repr_->emplace("kind", static_cast<long>(kind.value()));
  return *this;
}

Diagnostic::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("message")) return false;
  return true;
}

auto Diagnostic::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto Diagnostic::severity() const -> std::optional<DiagnosticSeverity> {
  if (!repr_->contains("severity")) return std::nullopt;

  auto& value = (*repr_)["severity"];

  return DiagnosticSeverity(value);
}

auto Diagnostic::code() const
    -> std::optional<std::variant<std::monostate, int, std::string>> {
  if (!repr_->contains("code")) return std::nullopt;

  auto& value = (*repr_)["code"];

  std::variant<std::monostate, int, std::string> result;

  details::try_emplace(result, value);

  return result;
}

auto Diagnostic::codeDescription() const -> std::optional<CodeDescription> {
  if (!repr_->contains("codeDescription")) return std::nullopt;

  auto& value = (*repr_)["codeDescription"];

  return CodeDescription(value);
}

auto Diagnostic::source() const -> std::optional<std::string> {
  if (!repr_->contains("source")) return std::nullopt;

  auto& value = (*repr_)["source"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Diagnostic::message() const -> std::string {
  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto Diagnostic::tags() const -> std::optional<Vector<DiagnosticTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<DiagnosticTag>(value);
}

auto Diagnostic::relatedInformation() const
    -> std::optional<Vector<DiagnosticRelatedInformation>> {
  if (!repr_->contains("relatedInformation")) return std::nullopt;

  auto& value = (*repr_)["relatedInformation"];

  assert(value.is_array());
  return Vector<DiagnosticRelatedInformation>(value);
}

auto Diagnostic::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto Diagnostic::range(Range range) -> Diagnostic& {
  repr_->emplace("range", range);
  return *this;
}

auto Diagnostic::severity(std::optional<DiagnosticSeverity> severity)
    -> Diagnostic& {
  if (!severity.has_value()) {
    repr_->erase("severity");
    return *this;
  }
  repr_->emplace("severity", static_cast<long>(severity.value()));
  return *this;
}

auto Diagnostic::code(
    std::optional<std::variant<std::monostate, int, std::string>> code)
    -> Diagnostic& {
  if (!code.has_value()) {
    repr_->erase("code");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int code) { repr_->emplace("code", std::move(code)); }

    void operator()(std::string code) {
      repr_->emplace("code", std::move(code));
    }
  } v{repr_};

  std::visit(v, code.value());

  return *this;
}

auto Diagnostic::codeDescription(std::optional<CodeDescription> codeDescription)
    -> Diagnostic& {
  if (!codeDescription.has_value()) {
    repr_->erase("codeDescription");
    return *this;
  }
  repr_->emplace("codeDescription", codeDescription.value());
  return *this;
}

auto Diagnostic::source(std::optional<std::string> source) -> Diagnostic& {
  if (!source.has_value()) {
    repr_->erase("source");
    return *this;
  }
  repr_->emplace("source", std::move(source.value()));
  return *this;
}

auto Diagnostic::message(std::string message) -> Diagnostic& {
  repr_->emplace("message", std::move(message));
  return *this;
}

auto Diagnostic::tags(std::optional<Vector<DiagnosticTag>> tags)
    -> Diagnostic& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("Diagnostic::tags: not implement yet");
  return *this;
}

auto Diagnostic::relatedInformation(
    std::optional<Vector<DiagnosticRelatedInformation>> relatedInformation)
    -> Diagnostic& {
  if (!relatedInformation.has_value()) {
    repr_->erase("relatedInformation");
    return *this;
  }
  lsp_runtime_error("Diagnostic::relatedInformation: not implement yet");
  return *this;
}

auto Diagnostic::data(std::optional<LSPAny> data) -> Diagnostic& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("Diagnostic::data: not implement yet");
  return *this;
}

CompletionContext::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("triggerKind")) return false;
  return true;
}

auto CompletionContext::triggerKind() const -> CompletionTriggerKind {
  auto& value = (*repr_)["triggerKind"];

  return CompletionTriggerKind(value);
}

auto CompletionContext::triggerCharacter() const -> std::optional<std::string> {
  if (!repr_->contains("triggerCharacter")) return std::nullopt;

  auto& value = (*repr_)["triggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionContext::triggerKind(CompletionTriggerKind triggerKind)
    -> CompletionContext& {
  repr_->emplace("triggerKind", static_cast<long>(triggerKind));
  return *this;
}

auto CompletionContext::triggerCharacter(
    std::optional<std::string> triggerCharacter) -> CompletionContext& {
  if (!triggerCharacter.has_value()) {
    repr_->erase("triggerCharacter");
    return *this;
  }
  repr_->emplace("triggerCharacter", std::move(triggerCharacter.value()));
  return *this;
}

CompletionItemLabelDetails::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CompletionItemLabelDetails::detail() const -> std::optional<std::string> {
  if (!repr_->contains("detail")) return std::nullopt;

  auto& value = (*repr_)["detail"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItemLabelDetails::description() const
    -> std::optional<std::string> {
  if (!repr_->contains("description")) return std::nullopt;

  auto& value = (*repr_)["description"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CompletionItemLabelDetails::detail(std::optional<std::string> detail)
    -> CompletionItemLabelDetails& {
  if (!detail.has_value()) {
    repr_->erase("detail");
    return *this;
  }
  repr_->emplace("detail", std::move(detail.value()));
  return *this;
}

auto CompletionItemLabelDetails::description(
    std::optional<std::string> description) -> CompletionItemLabelDetails& {
  if (!description.has_value()) {
    repr_->erase("description");
    return *this;
  }
  repr_->emplace("description", std::move(description.value()));
  return *this;
}

InsertReplaceEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("newText")) return false;
  if (!repr_->contains("insert")) return false;
  if (!repr_->contains("replace")) return false;
  return true;
}

auto InsertReplaceEdit::newText() const -> std::string {
  auto& value = (*repr_)["newText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto InsertReplaceEdit::insert() const -> Range {
  auto& value = (*repr_)["insert"];

  return Range(value);
}

auto InsertReplaceEdit::replace() const -> Range {
  auto& value = (*repr_)["replace"];

  return Range(value);
}

auto InsertReplaceEdit::newText(std::string newText) -> InsertReplaceEdit& {
  repr_->emplace("newText", std::move(newText));
  return *this;
}

auto InsertReplaceEdit::insert(Range insert) -> InsertReplaceEdit& {
  repr_->emplace("insert", insert);
  return *this;
}

auto InsertReplaceEdit::replace(Range replace) -> InsertReplaceEdit& {
  repr_->emplace("replace", replace);
  return *this;
}

CompletionItemDefaults::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CompletionItemDefaults::commitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("commitCharacters")) return std::nullopt;

  auto& value = (*repr_)["commitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionItemDefaults::editRange() const -> std::optional<
    std::variant<std::monostate, Range, EditRangeWithInsertReplace>> {
  if (!repr_->contains("editRange")) return std::nullopt;

  auto& value = (*repr_)["editRange"];

  std::variant<std::monostate, Range, EditRangeWithInsertReplace> result;

  details::try_emplace(result, value);

  return result;
}

auto CompletionItemDefaults::insertTextFormat() const
    -> std::optional<InsertTextFormat> {
  if (!repr_->contains("insertTextFormat")) return std::nullopt;

  auto& value = (*repr_)["insertTextFormat"];

  return InsertTextFormat(value);
}

auto CompletionItemDefaults::insertTextMode() const
    -> std::optional<InsertTextMode> {
  if (!repr_->contains("insertTextMode")) return std::nullopt;

  auto& value = (*repr_)["insertTextMode"];

  return InsertTextMode(value);
}

auto CompletionItemDefaults::data() const -> std::optional<LSPAny> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_object());
  return LSPAny(value);
}

auto CompletionItemDefaults::commitCharacters(
    std::optional<Vector<std::string>> commitCharacters)
    -> CompletionItemDefaults& {
  if (!commitCharacters.has_value()) {
    repr_->erase("commitCharacters");
    return *this;
  }
  lsp_runtime_error(
      "CompletionItemDefaults::commitCharacters: not implement yet");
  return *this;
}

auto CompletionItemDefaults::editRange(
    std::optional<
        std::variant<std::monostate, Range, EditRangeWithInsertReplace>>
        editRange) -> CompletionItemDefaults& {
  if (!editRange.has_value()) {
    repr_->erase("editRange");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(Range editRange) { repr_->emplace("editRange", editRange); }

    void operator()(EditRangeWithInsertReplace editRange) {
      repr_->emplace("editRange", editRange);
    }
  } v{repr_};

  std::visit(v, editRange.value());

  return *this;
}

auto CompletionItemDefaults::insertTextFormat(
    std::optional<InsertTextFormat> insertTextFormat)
    -> CompletionItemDefaults& {
  if (!insertTextFormat.has_value()) {
    repr_->erase("insertTextFormat");
    return *this;
  }
  repr_->emplace("insertTextFormat",
                 static_cast<long>(insertTextFormat.value()));
  return *this;
}

auto CompletionItemDefaults::insertTextMode(
    std::optional<InsertTextMode> insertTextMode) -> CompletionItemDefaults& {
  if (!insertTextMode.has_value()) {
    repr_->erase("insertTextMode");
    return *this;
  }
  repr_->emplace("insertTextMode", static_cast<long>(insertTextMode.value()));
  return *this;
}

auto CompletionItemDefaults::data(std::optional<LSPAny> data)
    -> CompletionItemDefaults& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("CompletionItemDefaults::data: not implement yet");
  return *this;
}

CompletionItemApplyKinds::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CompletionItemApplyKinds::commitCharacters() const
    -> std::optional<ApplyKind> {
  if (!repr_->contains("commitCharacters")) return std::nullopt;

  auto& value = (*repr_)["commitCharacters"];

  return ApplyKind(value);
}

auto CompletionItemApplyKinds::data() const -> std::optional<ApplyKind> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  return ApplyKind(value);
}

auto CompletionItemApplyKinds::commitCharacters(
    std::optional<ApplyKind> commitCharacters) -> CompletionItemApplyKinds& {
  if (!commitCharacters.has_value()) {
    repr_->erase("commitCharacters");
    return *this;
  }
  repr_->emplace("commitCharacters",
                 static_cast<long>(commitCharacters.value()));
  return *this;
}

auto CompletionItemApplyKinds::data(std::optional<ApplyKind> data)
    -> CompletionItemApplyKinds& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  repr_->emplace("data", static_cast<long>(data.value()));
  return *this;
}

CompletionOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CompletionOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("triggerCharacters")) return std::nullopt;

  auto& value = (*repr_)["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionOptions::allCommitCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("allCommitCharacters")) return std::nullopt;

  auto& value = (*repr_)["allCommitCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionOptions::completionItem() const
    -> std::optional<ServerCompletionItemOptions> {
  if (!repr_->contains("completionItem")) return std::nullopt;

  auto& value = (*repr_)["completionItem"];

  return ServerCompletionItemOptions(value);
}

auto CompletionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> CompletionOptions& {
  if (!triggerCharacters.has_value()) {
    repr_->erase("triggerCharacters");
    return *this;
  }
  lsp_runtime_error("CompletionOptions::triggerCharacters: not implement yet");
  return *this;
}

auto CompletionOptions::allCommitCharacters(
    std::optional<Vector<std::string>> allCommitCharacters)
    -> CompletionOptions& {
  if (!allCommitCharacters.has_value()) {
    repr_->erase("allCommitCharacters");
    return *this;
  }
  lsp_runtime_error(
      "CompletionOptions::allCommitCharacters: not implement yet");
  return *this;
}

auto CompletionOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> CompletionOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto CompletionOptions::completionItem(
    std::optional<ServerCompletionItemOptions> completionItem)
    -> CompletionOptions& {
  if (!completionItem.has_value()) {
    repr_->erase("completionItem");
    return *this;
  }
  repr_->emplace("completionItem", completionItem.value());
  return *this;
}

auto CompletionOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> CompletionOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

HoverOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto HoverOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto HoverOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> HoverOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

SignatureHelpContext::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("triggerKind")) return false;
  if (!repr_->contains("isRetrigger")) return false;
  return true;
}

auto SignatureHelpContext::triggerKind() const -> SignatureHelpTriggerKind {
  auto& value = (*repr_)["triggerKind"];

  return SignatureHelpTriggerKind(value);
}

auto SignatureHelpContext::triggerCharacter() const
    -> std::optional<std::string> {
  if (!repr_->contains("triggerCharacter")) return std::nullopt;

  auto& value = (*repr_)["triggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SignatureHelpContext::isRetrigger() const -> bool {
  auto& value = (*repr_)["isRetrigger"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpContext::activeSignatureHelp() const
    -> std::optional<SignatureHelp> {
  if (!repr_->contains("activeSignatureHelp")) return std::nullopt;

  auto& value = (*repr_)["activeSignatureHelp"];

  return SignatureHelp(value);
}

auto SignatureHelpContext::triggerKind(SignatureHelpTriggerKind triggerKind)
    -> SignatureHelpContext& {
  repr_->emplace("triggerKind", static_cast<long>(triggerKind));
  return *this;
}

auto SignatureHelpContext::triggerCharacter(
    std::optional<std::string> triggerCharacter) -> SignatureHelpContext& {
  if (!triggerCharacter.has_value()) {
    repr_->erase("triggerCharacter");
    return *this;
  }
  repr_->emplace("triggerCharacter", std::move(triggerCharacter.value()));
  return *this;
}

auto SignatureHelpContext::isRetrigger(bool isRetrigger)
    -> SignatureHelpContext& {
  repr_->emplace("isRetrigger", std::move(isRetrigger));
  return *this;
}

auto SignatureHelpContext::activeSignatureHelp(
    std::optional<SignatureHelp> activeSignatureHelp) -> SignatureHelpContext& {
  if (!activeSignatureHelp.has_value()) {
    repr_->erase("activeSignatureHelp");
    return *this;
  }
  repr_->emplace("activeSignatureHelp", activeSignatureHelp.value());
  return *this;
}

SignatureInformation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("label")) return false;
  return true;
}

auto SignatureInformation::label() const -> std::string {
  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SignatureInformation::documentation() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_->contains("documentation")) return std::nullopt;

  auto& value = (*repr_)["documentation"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureInformation::parameters() const
    -> std::optional<Vector<ParameterInformation>> {
  if (!repr_->contains("parameters")) return std::nullopt;

  auto& value = (*repr_)["parameters"];

  assert(value.is_array());
  return Vector<ParameterInformation>(value);
}

auto SignatureInformation::activeParameter() const
    -> std::optional<std::variant<std::monostate, long, std::nullptr_t>> {
  if (!repr_->contains("activeParameter")) return std::nullopt;

  auto& value = (*repr_)["activeParameter"];

  std::variant<std::monostate, long, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto SignatureInformation::label(std::string label) -> SignatureInformation& {
  repr_->emplace("label", std::move(label));
  return *this;
}

auto SignatureInformation::documentation(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        documentation) -> SignatureInformation& {
  if (!documentation.has_value()) {
    repr_->erase("documentation");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string documentation) {
      repr_->emplace("documentation", std::move(documentation));
    }

    void operator()(MarkupContent documentation) {
      repr_->emplace("documentation", documentation);
    }
  } v{repr_};

  std::visit(v, documentation.value());

  return *this;
}

auto SignatureInformation::parameters(
    std::optional<Vector<ParameterInformation>> parameters)
    -> SignatureInformation& {
  if (!parameters.has_value()) {
    repr_->erase("parameters");
    return *this;
  }
  lsp_runtime_error("SignatureInformation::parameters: not implement yet");
  return *this;
}

auto SignatureInformation::activeParameter(
    std::optional<std::variant<std::monostate, long, std::nullptr_t>>
        activeParameter) -> SignatureInformation& {
  if (!activeParameter.has_value()) {
    repr_->erase("activeParameter");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(long activeParameter) {
      repr_->emplace("activeParameter", std::move(activeParameter));
    }

    void operator()(std::nullptr_t activeParameter) {
      repr_->emplace("activeParameter", std::move(activeParameter));
    }
  } v{repr_};

  std::visit(v, activeParameter.value());

  return *this;
}

SignatureHelpOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SignatureHelpOptions::triggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("triggerCharacters")) return std::nullopt;

  auto& value = (*repr_)["triggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpOptions::retriggerCharacters() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("retriggerCharacters")) return std::nullopt;

  auto& value = (*repr_)["retriggerCharacters"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SignatureHelpOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpOptions::triggerCharacters(
    std::optional<Vector<std::string>> triggerCharacters)
    -> SignatureHelpOptions& {
  if (!triggerCharacters.has_value()) {
    repr_->erase("triggerCharacters");
    return *this;
  }
  lsp_runtime_error(
      "SignatureHelpOptions::triggerCharacters: not implement yet");
  return *this;
}

auto SignatureHelpOptions::retriggerCharacters(
    std::optional<Vector<std::string>> retriggerCharacters)
    -> SignatureHelpOptions& {
  if (!retriggerCharacters.has_value()) {
    repr_->erase("retriggerCharacters");
    return *this;
  }
  lsp_runtime_error(
      "SignatureHelpOptions::retriggerCharacters: not implement yet");
  return *this;
}

auto SignatureHelpOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> SignatureHelpOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DefinitionOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DefinitionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DefinitionOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

ReferenceContext::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("includeDeclaration")) return false;
  return true;
}

auto ReferenceContext::includeDeclaration() const -> bool {
  auto& value = (*repr_)["includeDeclaration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceContext::includeDeclaration(bool includeDeclaration)
    -> ReferenceContext& {
  repr_->emplace("includeDeclaration", std::move(includeDeclaration));
  return *this;
}

ReferenceOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ReferenceOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> ReferenceOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentHighlightOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentHighlightOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentHighlightOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentHighlightOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

BaseSymbolInformation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("name")) return false;
  if (!repr_->contains("kind")) return false;
  return true;
}

auto BaseSymbolInformation::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto BaseSymbolInformation::kind() const -> SymbolKind {
  auto& value = (*repr_)["kind"];

  return SymbolKind(value);
}

auto BaseSymbolInformation::tags() const -> std::optional<Vector<SymbolTag>> {
  if (!repr_->contains("tags")) return std::nullopt;

  auto& value = (*repr_)["tags"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto BaseSymbolInformation::containerName() const
    -> std::optional<std::string> {
  if (!repr_->contains("containerName")) return std::nullopt;

  auto& value = (*repr_)["containerName"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto BaseSymbolInformation::name(std::string name) -> BaseSymbolInformation& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto BaseSymbolInformation::kind(SymbolKind kind) -> BaseSymbolInformation& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto BaseSymbolInformation::tags(std::optional<Vector<SymbolTag>> tags)
    -> BaseSymbolInformation& {
  if (!tags.has_value()) {
    repr_->erase("tags");
    return *this;
  }
  lsp_runtime_error("BaseSymbolInformation::tags: not implement yet");
  return *this;
}

auto BaseSymbolInformation::containerName(
    std::optional<std::string> containerName) -> BaseSymbolInformation& {
  if (!containerName.has_value()) {
    repr_->erase("containerName");
    return *this;
  }
  repr_->emplace("containerName", std::move(containerName.value()));
  return *this;
}

DocumentSymbolOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentSymbolOptions::label() const -> std::optional<std::string> {
  if (!repr_->contains("label")) return std::nullopt;

  auto& value = (*repr_)["label"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentSymbolOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolOptions::label(std::optional<std::string> label)
    -> DocumentSymbolOptions& {
  if (!label.has_value()) {
    repr_->erase("label");
    return *this;
  }
  repr_->emplace("label", std::move(label.value()));
  return *this;
}

auto DocumentSymbolOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentSymbolOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

CodeActionContext::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("diagnostics")) return false;
  return true;
}

auto CodeActionContext::diagnostics() const -> Vector<Diagnostic> {
  auto& value = (*repr_)["diagnostics"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto CodeActionContext::only() const -> std::optional<Vector<CodeActionKind>> {
  if (!repr_->contains("only")) return std::nullopt;

  auto& value = (*repr_)["only"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto CodeActionContext::triggerKind() const
    -> std::optional<CodeActionTriggerKind> {
  if (!repr_->contains("triggerKind")) return std::nullopt;

  auto& value = (*repr_)["triggerKind"];

  return CodeActionTriggerKind(value);
}

auto CodeActionContext::diagnostics(Vector<Diagnostic> diagnostics)
    -> CodeActionContext& {
  lsp_runtime_error("CodeActionContext::diagnostics: not implement yet");
  return *this;
}

auto CodeActionContext::only(std::optional<Vector<CodeActionKind>> only)
    -> CodeActionContext& {
  if (!only.has_value()) {
    repr_->erase("only");
    return *this;
  }
  lsp_runtime_error("CodeActionContext::only: not implement yet");
  return *this;
}

auto CodeActionContext::triggerKind(
    std::optional<CodeActionTriggerKind> triggerKind) -> CodeActionContext& {
  if (!triggerKind.has_value()) {
    repr_->erase("triggerKind");
    return *this;
  }
  repr_->emplace("triggerKind", static_cast<long>(triggerKind.value()));
  return *this;
}

CodeActionDisabled::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("reason")) return false;
  return true;
}

auto CodeActionDisabled::reason() const -> std::string {
  auto& value = (*repr_)["reason"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CodeActionDisabled::reason(std::string reason) -> CodeActionDisabled& {
  repr_->emplace("reason", std::move(reason));
  return *this;
}

CodeActionOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CodeActionOptions::codeActionKinds() const
    -> std::optional<Vector<CodeActionKind>> {
  if (!repr_->contains("codeActionKinds")) return std::nullopt;

  auto& value = (*repr_)["codeActionKinds"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto CodeActionOptions::documentation() const
    -> std::optional<Vector<CodeActionKindDocumentation>> {
  if (!repr_->contains("documentation")) return std::nullopt;

  auto& value = (*repr_)["documentation"];

  assert(value.is_array());
  return Vector<CodeActionKindDocumentation>(value);
}

auto CodeActionOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionOptions::codeActionKinds(
    std::optional<Vector<CodeActionKind>> codeActionKinds)
    -> CodeActionOptions& {
  if (!codeActionKinds.has_value()) {
    repr_->erase("codeActionKinds");
    return *this;
  }
  lsp_runtime_error("CodeActionOptions::codeActionKinds: not implement yet");
  return *this;
}

auto CodeActionOptions::documentation(
    std::optional<Vector<CodeActionKindDocumentation>> documentation)
    -> CodeActionOptions& {
  if (!documentation.has_value()) {
    repr_->erase("documentation");
    return *this;
  }
  lsp_runtime_error("CodeActionOptions::documentation: not implement yet");
  return *this;
}

auto CodeActionOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> CodeActionOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto CodeActionOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> CodeActionOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

LocationUriOnly::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto LocationUriOnly::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto LocationUriOnly::uri(std::string uri) -> LocationUriOnly& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

WorkspaceSymbolOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceSymbolOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolOptions::resolveProvider(
    std::optional<bool> resolveProvider) -> WorkspaceSymbolOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto WorkspaceSymbolOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> WorkspaceSymbolOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

CodeLensOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CodeLensOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> CodeLensOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto CodeLensOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> CodeLensOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentLinkOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentLinkOptions::resolveProvider() const -> std::optional<bool> {
  if (!repr_->contains("resolveProvider")) return std::nullopt;

  auto& value = (*repr_)["resolveProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkOptions::resolveProvider(std::optional<bool> resolveProvider)
    -> DocumentLinkOptions& {
  if (!resolveProvider.has_value()) {
    repr_->erase("resolveProvider");
    return *this;
  }
  repr_->emplace("resolveProvider", std::move(resolveProvider.value()));
  return *this;
}

auto DocumentLinkOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> DocumentLinkOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

FormattingOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("tabSize")) return false;
  if (!repr_->contains("insertSpaces")) return false;
  return true;
}

auto FormattingOptions::tabSize() const -> long {
  auto& value = (*repr_)["tabSize"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FormattingOptions::insertSpaces() const -> bool {
  auto& value = (*repr_)["insertSpaces"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::trimTrailingWhitespace() const -> std::optional<bool> {
  if (!repr_->contains("trimTrailingWhitespace")) return std::nullopt;

  auto& value = (*repr_)["trimTrailingWhitespace"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::insertFinalNewline() const -> std::optional<bool> {
  if (!repr_->contains("insertFinalNewline")) return std::nullopt;

  auto& value = (*repr_)["insertFinalNewline"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::trimFinalNewlines() const -> std::optional<bool> {
  if (!repr_->contains("trimFinalNewlines")) return std::nullopt;

  auto& value = (*repr_)["trimFinalNewlines"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FormattingOptions::tabSize(long tabSize) -> FormattingOptions& {
  repr_->emplace("tabSize", std::move(tabSize));
  return *this;
}

auto FormattingOptions::insertSpaces(bool insertSpaces) -> FormattingOptions& {
  repr_->emplace("insertSpaces", std::move(insertSpaces));
  return *this;
}

auto FormattingOptions::trimTrailingWhitespace(
    std::optional<bool> trimTrailingWhitespace) -> FormattingOptions& {
  if (!trimTrailingWhitespace.has_value()) {
    repr_->erase("trimTrailingWhitespace");
    return *this;
  }
  repr_->emplace("trimTrailingWhitespace",
                 std::move(trimTrailingWhitespace.value()));
  return *this;
}

auto FormattingOptions::insertFinalNewline(
    std::optional<bool> insertFinalNewline) -> FormattingOptions& {
  if (!insertFinalNewline.has_value()) {
    repr_->erase("insertFinalNewline");
    return *this;
  }
  repr_->emplace("insertFinalNewline", std::move(insertFinalNewline.value()));
  return *this;
}

auto FormattingOptions::trimFinalNewlines(std::optional<bool> trimFinalNewlines)
    -> FormattingOptions& {
  if (!trimFinalNewlines.has_value()) {
    repr_->erase("trimFinalNewlines");
    return *this;
  }
  repr_->emplace("trimFinalNewlines", std::move(trimFinalNewlines.value()));
  return *this;
}

DocumentFormattingOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentFormattingOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentFormattingOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentFormattingOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentRangeFormattingOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentRangeFormattingOptions::rangesSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("rangesSupport")) return std::nullopt;

  auto& value = (*repr_)["rangesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingOptions::workDoneProgress() const
    -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingOptions::rangesSupport(
    std::optional<bool> rangesSupport) -> DocumentRangeFormattingOptions& {
  if (!rangesSupport.has_value()) {
    repr_->erase("rangesSupport");
    return *this;
  }
  repr_->emplace("rangesSupport", std::move(rangesSupport.value()));
  return *this;
}

auto DocumentRangeFormattingOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> DocumentRangeFormattingOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

DocumentOnTypeFormattingOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("firstTriggerCharacter")) return false;
  return true;
}

auto DocumentOnTypeFormattingOptions::firstTriggerCharacter() const
    -> std::string {
  auto& value = (*repr_)["firstTriggerCharacter"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DocumentOnTypeFormattingOptions::moreTriggerCharacter() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("moreTriggerCharacter")) return std::nullopt;

  auto& value = (*repr_)["moreTriggerCharacter"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto DocumentOnTypeFormattingOptions::firstTriggerCharacter(
    std::string firstTriggerCharacter) -> DocumentOnTypeFormattingOptions& {
  repr_->emplace("firstTriggerCharacter", std::move(firstTriggerCharacter));
  return *this;
}

auto DocumentOnTypeFormattingOptions::moreTriggerCharacter(
    std::optional<Vector<std::string>> moreTriggerCharacter)
    -> DocumentOnTypeFormattingOptions& {
  if (!moreTriggerCharacter.has_value()) {
    repr_->erase("moreTriggerCharacter");
    return *this;
  }
  lsp_runtime_error(
      "DocumentOnTypeFormattingOptions::moreTriggerCharacter: not implement "
      "yet");
  return *this;
}

RenameOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto RenameOptions::prepareProvider() const -> std::optional<bool> {
  if (!repr_->contains("prepareProvider")) return std::nullopt;

  auto& value = (*repr_)["prepareProvider"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameOptions::prepareProvider(std::optional<bool> prepareProvider)
    -> RenameOptions& {
  if (!prepareProvider.has_value()) {
    repr_->erase("prepareProvider");
    return *this;
  }
  repr_->emplace("prepareProvider", std::move(prepareProvider.value()));
  return *this;
}

auto RenameOptions::workDoneProgress(std::optional<bool> workDoneProgress)
    -> RenameOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

PrepareRenamePlaceholder::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("placeholder")) return false;
  return true;
}

auto PrepareRenamePlaceholder::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto PrepareRenamePlaceholder::placeholder() const -> std::string {
  auto& value = (*repr_)["placeholder"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto PrepareRenamePlaceholder::range(Range range) -> PrepareRenamePlaceholder& {
  repr_->emplace("range", range);
  return *this;
}

auto PrepareRenamePlaceholder::placeholder(std::string placeholder)
    -> PrepareRenamePlaceholder& {
  repr_->emplace("placeholder", std::move(placeholder));
  return *this;
}

PrepareRenameDefaultBehavior::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("defaultBehavior")) return false;
  return true;
}

auto PrepareRenameDefaultBehavior::defaultBehavior() const -> bool {
  auto& value = (*repr_)["defaultBehavior"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PrepareRenameDefaultBehavior::defaultBehavior(bool defaultBehavior)
    -> PrepareRenameDefaultBehavior& {
  repr_->emplace("defaultBehavior", std::move(defaultBehavior));
  return *this;
}

ExecuteCommandOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("commands")) return false;
  return true;
}

auto ExecuteCommandOptions::commands() const -> Vector<std::string> {
  auto& value = (*repr_)["commands"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ExecuteCommandOptions::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecuteCommandOptions::commands(Vector<std::string> commands)
    -> ExecuteCommandOptions& {
  lsp_runtime_error("ExecuteCommandOptions::commands: not implement yet");
  return *this;
}

auto ExecuteCommandOptions::workDoneProgress(
    std::optional<bool> workDoneProgress) -> ExecuteCommandOptions& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

WorkspaceEditMetadata::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceEditMetadata::isRefactoring() const -> std::optional<bool> {
  if (!repr_->contains("isRefactoring")) return std::nullopt;

  auto& value = (*repr_)["isRefactoring"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditMetadata::isRefactoring(std::optional<bool> isRefactoring)
    -> WorkspaceEditMetadata& {
  if (!isRefactoring.has_value()) {
    repr_->erase("isRefactoring");
    return *this;
  }
  repr_->emplace("isRefactoring", std::move(isRefactoring.value()));
  return *this;
}

SemanticTokensLegend::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("tokenTypes")) return false;
  if (!repr_->contains("tokenModifiers")) return false;
  return true;
}

auto SemanticTokensLegend::tokenTypes() const -> Vector<std::string> {
  auto& value = (*repr_)["tokenTypes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensLegend::tokenModifiers() const -> Vector<std::string> {
  auto& value = (*repr_)["tokenModifiers"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensLegend::tokenTypes(Vector<std::string> tokenTypes)
    -> SemanticTokensLegend& {
  lsp_runtime_error("SemanticTokensLegend::tokenTypes: not implement yet");
  return *this;
}

auto SemanticTokensLegend::tokenModifiers(Vector<std::string> tokenModifiers)
    -> SemanticTokensLegend& {
  lsp_runtime_error("SemanticTokensLegend::tokenModifiers: not implement yet");
  return *this;
}

SemanticTokensFullDelta::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SemanticTokensFullDelta::delta() const -> std::optional<bool> {
  if (!repr_->contains("delta")) return std::nullopt;

  auto& value = (*repr_)["delta"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensFullDelta::delta(std::optional<bool> delta)
    -> SemanticTokensFullDelta& {
  if (!delta.has_value()) {
    repr_->erase("delta");
    return *this;
  }
  repr_->emplace("delta", std::move(delta.value()));
  return *this;
}

OptionalVersionedTextDocumentIdentifier::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("uri")) return false;
  return true;
}

auto OptionalVersionedTextDocumentIdentifier::version() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  auto& value = (*repr_)["version"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto OptionalVersionedTextDocumentIdentifier::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto OptionalVersionedTextDocumentIdentifier::version(
    std::variant<std::monostate, int, std::nullptr_t> version)
    -> OptionalVersionedTextDocumentIdentifier& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int version) {
      repr_->emplace("version", std::move(version));
    }

    void operator()(std::nullptr_t version) {
      repr_->emplace("version", std::move(version));
    }
  } v{repr_};

  std::visit(v, version);

  return *this;
}

auto OptionalVersionedTextDocumentIdentifier::uri(std::string uri)
    -> OptionalVersionedTextDocumentIdentifier& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

AnnotatedTextEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("annotationId")) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("newText")) return false;
  return true;
}

auto AnnotatedTextEdit::annotationId() const -> ChangeAnnotationIdentifier {
  auto& value = (*repr_)["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto AnnotatedTextEdit::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto AnnotatedTextEdit::newText() const -> std::string {
  auto& value = (*repr_)["newText"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto AnnotatedTextEdit::annotationId(ChangeAnnotationIdentifier annotationId)
    -> AnnotatedTextEdit& {
  lsp_runtime_error("AnnotatedTextEdit::annotationId: not implement yet");
  return *this;
}

auto AnnotatedTextEdit::range(Range range) -> AnnotatedTextEdit& {
  repr_->emplace("range", range);
  return *this;
}

auto AnnotatedTextEdit::newText(std::string newText) -> AnnotatedTextEdit& {
  repr_->emplace("newText", std::move(newText));
  return *this;
}

SnippetTextEdit::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("snippet")) return false;
  return true;
}

auto SnippetTextEdit::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto SnippetTextEdit::snippet() const -> StringValue {
  auto& value = (*repr_)["snippet"];

  return StringValue(value);
}

auto SnippetTextEdit::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_->contains("annotationId")) return std::nullopt;

  auto& value = (*repr_)["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SnippetTextEdit::range(Range range) -> SnippetTextEdit& {
  repr_->emplace("range", range);
  return *this;
}

auto SnippetTextEdit::snippet(StringValue snippet) -> SnippetTextEdit& {
  repr_->emplace("snippet", snippet);
  return *this;
}

auto SnippetTextEdit::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId)
    -> SnippetTextEdit& {
  if (!annotationId.has_value()) {
    repr_->erase("annotationId");
    return *this;
  }
  lsp_runtime_error("SnippetTextEdit::annotationId: not implement yet");
  return *this;
}

ResourceOperation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  return true;
}

auto ResourceOperation::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ResourceOperation::annotationId() const
    -> std::optional<ChangeAnnotationIdentifier> {
  if (!repr_->contains("annotationId")) return std::nullopt;

  auto& value = (*repr_)["annotationId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ResourceOperation::kind(std::string kind) -> ResourceOperation& {
  repr_->emplace("kind", std::move(kind));
  return *this;
}

auto ResourceOperation::annotationId(
    std::optional<ChangeAnnotationIdentifier> annotationId)
    -> ResourceOperation& {
  if (!annotationId.has_value()) {
    repr_->erase("annotationId");
    return *this;
  }
  lsp_runtime_error("ResourceOperation::annotationId: not implement yet");
  return *this;
}

CreateFileOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CreateFileOptions::overwrite() const -> std::optional<bool> {
  if (!repr_->contains("overwrite")) return std::nullopt;

  auto& value = (*repr_)["overwrite"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CreateFileOptions::ignoreIfExists() const -> std::optional<bool> {
  if (!repr_->contains("ignoreIfExists")) return std::nullopt;

  auto& value = (*repr_)["ignoreIfExists"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CreateFileOptions::overwrite(std::optional<bool> overwrite)
    -> CreateFileOptions& {
  if (!overwrite.has_value()) {
    repr_->erase("overwrite");
    return *this;
  }
  repr_->emplace("overwrite", std::move(overwrite.value()));
  return *this;
}

auto CreateFileOptions::ignoreIfExists(std::optional<bool> ignoreIfExists)
    -> CreateFileOptions& {
  if (!ignoreIfExists.has_value()) {
    repr_->erase("ignoreIfExists");
    return *this;
  }
  repr_->emplace("ignoreIfExists", std::move(ignoreIfExists.value()));
  return *this;
}

RenameFileOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto RenameFileOptions::overwrite() const -> std::optional<bool> {
  if (!repr_->contains("overwrite")) return std::nullopt;

  auto& value = (*repr_)["overwrite"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameFileOptions::ignoreIfExists() const -> std::optional<bool> {
  if (!repr_->contains("ignoreIfExists")) return std::nullopt;

  auto& value = (*repr_)["ignoreIfExists"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameFileOptions::overwrite(std::optional<bool> overwrite)
    -> RenameFileOptions& {
  if (!overwrite.has_value()) {
    repr_->erase("overwrite");
    return *this;
  }
  repr_->emplace("overwrite", std::move(overwrite.value()));
  return *this;
}

auto RenameFileOptions::ignoreIfExists(std::optional<bool> ignoreIfExists)
    -> RenameFileOptions& {
  if (!ignoreIfExists.has_value()) {
    repr_->erase("ignoreIfExists");
    return *this;
  }
  repr_->emplace("ignoreIfExists", std::move(ignoreIfExists.value()));
  return *this;
}

DeleteFileOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DeleteFileOptions::recursive() const -> std::optional<bool> {
  if (!repr_->contains("recursive")) return std::nullopt;

  auto& value = (*repr_)["recursive"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeleteFileOptions::ignoreIfNotExists() const -> std::optional<bool> {
  if (!repr_->contains("ignoreIfNotExists")) return std::nullopt;

  auto& value = (*repr_)["ignoreIfNotExists"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeleteFileOptions::recursive(std::optional<bool> recursive)
    -> DeleteFileOptions& {
  if (!recursive.has_value()) {
    repr_->erase("recursive");
    return *this;
  }
  repr_->emplace("recursive", std::move(recursive.value()));
  return *this;
}

auto DeleteFileOptions::ignoreIfNotExists(std::optional<bool> ignoreIfNotExists)
    -> DeleteFileOptions& {
  if (!ignoreIfNotExists.has_value()) {
    repr_->erase("ignoreIfNotExists");
    return *this;
  }
  repr_->emplace("ignoreIfNotExists", std::move(ignoreIfNotExists.value()));
  return *this;
}

FileOperationPattern::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("glob")) return false;
  return true;
}

auto FileOperationPattern::glob() const -> std::string {
  auto& value = (*repr_)["glob"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto FileOperationPattern::matches() const
    -> std::optional<FileOperationPatternKind> {
  if (!repr_->contains("matches")) return std::nullopt;

  auto& value = (*repr_)["matches"];

  lsp_runtime_error("FileOperationPattern::matches: not implement yet");
}

auto FileOperationPattern::options() const
    -> std::optional<FileOperationPatternOptions> {
  if (!repr_->contains("options")) return std::nullopt;

  auto& value = (*repr_)["options"];

  return FileOperationPatternOptions(value);
}

auto FileOperationPattern::glob(std::string glob) -> FileOperationPattern& {
  repr_->emplace("glob", std::move(glob));
  return *this;
}

auto FileOperationPattern::matches(
    std::optional<FileOperationPatternKind> matches) -> FileOperationPattern& {
  if (!matches.has_value()) {
    repr_->erase("matches");
    return *this;
  }
  lsp_runtime_error("FileOperationPattern::matches: not implement yet");
  return *this;
}

auto FileOperationPattern::options(
    std::optional<FileOperationPatternOptions> options)
    -> FileOperationPattern& {
  if (!options.has_value()) {
    repr_->erase("options");
    return *this;
  }
  repr_->emplace("options", options.value());
  return *this;
}

WorkspaceFullDocumentDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "full") return false;
  if (!repr_->contains("items")) return false;
  return true;
}

auto WorkspaceFullDocumentDiagnosticReport::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFullDocumentDiagnosticReport::version() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  auto& value = (*repr_)["version"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFullDocumentDiagnosticReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFullDocumentDiagnosticReport::resultId() const
    -> std::optional<std::string> {
  if (!repr_->contains("resultId")) return std::nullopt;

  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceFullDocumentDiagnosticReport::items() const
    -> Vector<Diagnostic> {
  auto& value = (*repr_)["items"];

  assert(value.is_array());
  return Vector<Diagnostic>(value);
}

auto WorkspaceFullDocumentDiagnosticReport::uri(std::string uri)
    -> WorkspaceFullDocumentDiagnosticReport& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::version(
    std::variant<std::monostate, int, std::nullptr_t> version)
    -> WorkspaceFullDocumentDiagnosticReport& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int version) {
      repr_->emplace("version", std::move(version));
    }

    void operator()(std::nullptr_t version) {
      repr_->emplace("version", std::move(version));
    }
  } v{repr_};

  std::visit(v, version);

  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::kind(std::string kind)
    -> WorkspaceFullDocumentDiagnosticReport& {
  lsp_runtime_error(
      "WorkspaceFullDocumentDiagnosticReport::kind: not implement yet");
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::resultId(
    std::optional<std::string> resultId)
    -> WorkspaceFullDocumentDiagnosticReport& {
  if (!resultId.has_value()) {
    repr_->erase("resultId");
    return *this;
  }
  repr_->emplace("resultId", std::move(resultId.value()));
  return *this;
}

auto WorkspaceFullDocumentDiagnosticReport::items(Vector<Diagnostic> items)
    -> WorkspaceFullDocumentDiagnosticReport& {
  lsp_runtime_error(
      "WorkspaceFullDocumentDiagnosticReport::items: not implement yet");
  return *this;
}

WorkspaceUnchangedDocumentDiagnosticReport::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("uri")) return false;
  if (!repr_->contains("version")) return false;
  if (!repr_->contains("kind")) return false;
  if ((*repr_)["kind"] != "unchanged") return false;
  if (!repr_->contains("resultId")) return false;
  return true;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::uri() const -> std::string {
  auto& value = (*repr_)["uri"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceUnchangedDocumentDiagnosticReport::version() const
    -> std::variant<std::monostate, int, std::nullptr_t> {
  auto& value = (*repr_)["version"];

  std::variant<std::monostate, int, std::nullptr_t> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::kind() const -> std::string {
  auto& value = (*repr_)["kind"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceUnchangedDocumentDiagnosticReport::resultId() const
    -> std::string {
  auto& value = (*repr_)["resultId"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto WorkspaceUnchangedDocumentDiagnosticReport::uri(std::string uri)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  repr_->emplace("uri", std::move(uri));
  return *this;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::version(
    std::variant<std::monostate, int, std::nullptr_t> version)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(int version) {
      repr_->emplace("version", std::move(version));
    }

    void operator()(std::nullptr_t version) {
      repr_->emplace("version", std::move(version));
    }
  } v{repr_};

  std::visit(v, version);

  return *this;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::kind(std::string kind)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  lsp_runtime_error(
      "WorkspaceUnchangedDocumentDiagnosticReport::kind: not implement yet");
  return *this;
}

auto WorkspaceUnchangedDocumentDiagnosticReport::resultId(std::string resultId)
    -> WorkspaceUnchangedDocumentDiagnosticReport& {
  repr_->emplace("resultId", std::move(resultId));
  return *this;
}

NotebookCell::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if (!repr_->contains("document")) return false;
  return true;
}

auto NotebookCell::kind() const -> NotebookCellKind {
  auto& value = (*repr_)["kind"];

  return NotebookCellKind(value);
}

auto NotebookCell::document() const -> std::string {
  auto& value = (*repr_)["document"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookCell::metadata() const -> std::optional<LSPObject> {
  if (!repr_->contains("metadata")) return std::nullopt;

  auto& value = (*repr_)["metadata"];

  assert(value.is_object());
  return LSPObject(value);
}

auto NotebookCell::executionSummary() const -> std::optional<ExecutionSummary> {
  if (!repr_->contains("executionSummary")) return std::nullopt;

  auto& value = (*repr_)["executionSummary"];

  return ExecutionSummary(value);
}

auto NotebookCell::kind(NotebookCellKind kind) -> NotebookCell& {
  repr_->emplace("kind", static_cast<long>(kind));
  return *this;
}

auto NotebookCell::document(std::string document) -> NotebookCell& {
  repr_->emplace("document", std::move(document));
  return *this;
}

auto NotebookCell::metadata(std::optional<LSPObject> metadata)
    -> NotebookCell& {
  if (!metadata.has_value()) {
    repr_->erase("metadata");
    return *this;
  }
  lsp_runtime_error("NotebookCell::metadata: not implement yet");
  return *this;
}

auto NotebookCell::executionSummary(
    std::optional<ExecutionSummary> executionSummary) -> NotebookCell& {
  if (!executionSummary.has_value()) {
    repr_->erase("executionSummary");
    return *this;
  }
  repr_->emplace("executionSummary", executionSummary.value());
  return *this;
}

NotebookDocumentFilterWithNotebook::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebook")) return false;
  return true;
}

auto NotebookDocumentFilterWithNotebook::notebook() const
    -> std::variant<std::monostate, std::string, NotebookDocumentFilter> {
  auto& value = (*repr_)["notebook"];

  std::variant<std::monostate, std::string, NotebookDocumentFilter> result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterWithNotebook::cells() const
    -> std::optional<Vector<NotebookCellLanguage>> {
  if (!repr_->contains("cells")) return std::nullopt;

  auto& value = (*repr_)["cells"];

  assert(value.is_array());
  return Vector<NotebookCellLanguage>(value);
}

auto NotebookDocumentFilterWithNotebook::notebook(
    std::variant<std::monostate, std::string, NotebookDocumentFilter> notebook)
    -> NotebookDocumentFilterWithNotebook& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string notebook) {
      repr_->emplace("notebook", std::move(notebook));
    }

    void operator()(NotebookDocumentFilter notebook) {
      lsp_runtime_error(
          "NotebookDocumentFilterWithNotebook::notebook: not implement yet");
    }
  } v{repr_};

  std::visit(v, notebook);

  return *this;
}

auto NotebookDocumentFilterWithNotebook::cells(
    std::optional<Vector<NotebookCellLanguage>> cells)
    -> NotebookDocumentFilterWithNotebook& {
  if (!cells.has_value()) {
    repr_->erase("cells");
    return *this;
  }
  lsp_runtime_error(
      "NotebookDocumentFilterWithNotebook::cells: not implement yet");
  return *this;
}

NotebookDocumentFilterWithCells::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("cells")) return false;
  return true;
}

auto NotebookDocumentFilterWithCells::notebook() const -> std::optional<
    std::variant<std::monostate, std::string, NotebookDocumentFilter>> {
  if (!repr_->contains("notebook")) return std::nullopt;

  auto& value = (*repr_)["notebook"];

  std::variant<std::monostate, std::string, NotebookDocumentFilter> result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterWithCells::cells() const
    -> Vector<NotebookCellLanguage> {
  auto& value = (*repr_)["cells"];

  assert(value.is_array());
  return Vector<NotebookCellLanguage>(value);
}

auto NotebookDocumentFilterWithCells::notebook(
    std::optional<
        std::variant<std::monostate, std::string, NotebookDocumentFilter>>
        notebook) -> NotebookDocumentFilterWithCells& {
  if (!notebook.has_value()) {
    repr_->erase("notebook");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string notebook) {
      repr_->emplace("notebook", std::move(notebook));
    }

    void operator()(NotebookDocumentFilter notebook) {
      lsp_runtime_error(
          "NotebookDocumentFilterWithCells::notebook: not implement yet");
    }
  } v{repr_};

  std::visit(v, notebook.value());

  return *this;
}

auto NotebookDocumentFilterWithCells::cells(Vector<NotebookCellLanguage> cells)
    -> NotebookDocumentFilterWithCells& {
  lsp_runtime_error(
      "NotebookDocumentFilterWithCells::cells: not implement yet");
  return *this;
}

NotebookDocumentCellChanges::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto NotebookDocumentCellChanges::structure() const
    -> std::optional<NotebookDocumentCellChangeStructure> {
  if (!repr_->contains("structure")) return std::nullopt;

  auto& value = (*repr_)["structure"];

  return NotebookDocumentCellChangeStructure(value);
}

auto NotebookDocumentCellChanges::data() const
    -> std::optional<Vector<NotebookCell>> {
  if (!repr_->contains("data")) return std::nullopt;

  auto& value = (*repr_)["data"];

  assert(value.is_array());
  return Vector<NotebookCell>(value);
}

auto NotebookDocumentCellChanges::textContent() const
    -> std::optional<Vector<NotebookDocumentCellContentChanges>> {
  if (!repr_->contains("textContent")) return std::nullopt;

  auto& value = (*repr_)["textContent"];

  assert(value.is_array());
  return Vector<NotebookDocumentCellContentChanges>(value);
}

auto NotebookDocumentCellChanges::structure(
    std::optional<NotebookDocumentCellChangeStructure> structure)
    -> NotebookDocumentCellChanges& {
  if (!structure.has_value()) {
    repr_->erase("structure");
    return *this;
  }
  repr_->emplace("structure", structure.value());
  return *this;
}

auto NotebookDocumentCellChanges::data(std::optional<Vector<NotebookCell>> data)
    -> NotebookDocumentCellChanges& {
  if (!data.has_value()) {
    repr_->erase("data");
    return *this;
  }
  lsp_runtime_error("NotebookDocumentCellChanges::data: not implement yet");
  return *this;
}

auto NotebookDocumentCellChanges::textContent(
    std::optional<Vector<NotebookDocumentCellContentChanges>> textContent)
    -> NotebookDocumentCellChanges& {
  if (!textContent.has_value()) {
    repr_->erase("textContent");
    return *this;
  }
  lsp_runtime_error(
      "NotebookDocumentCellChanges::textContent: not implement yet");
  return *this;
}

SelectedCompletionInfo::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("text")) return false;
  return true;
}

auto SelectedCompletionInfo::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto SelectedCompletionInfo::text() const -> std::string {
  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto SelectedCompletionInfo::range(Range range) -> SelectedCompletionInfo& {
  repr_->emplace("range", range);
  return *this;
}

auto SelectedCompletionInfo::text(std::string text) -> SelectedCompletionInfo& {
  repr_->emplace("text", std::move(text));
  return *this;
}

ClientInfo::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("name")) return false;
  return true;
}

auto ClientInfo::name() const -> std::string {
  auto& value = (*repr_)["name"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ClientInfo::version() const -> std::optional<std::string> {
  if (!repr_->contains("version")) return std::nullopt;

  auto& value = (*repr_)["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto ClientInfo::name(std::string name) -> ClientInfo& {
  repr_->emplace("name", std::move(name));
  return *this;
}

auto ClientInfo::version(std::optional<std::string> version) -> ClientInfo& {
  if (!version.has_value()) {
    repr_->erase("version");
    return *this;
  }
  repr_->emplace("version", std::move(version.value()));
  return *this;
}

ClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientCapabilities::workspace() const
    -> std::optional<WorkspaceClientCapabilities> {
  if (!repr_->contains("workspace")) return std::nullopt;

  auto& value = (*repr_)["workspace"];

  return WorkspaceClientCapabilities(value);
}

auto ClientCapabilities::textDocument() const
    -> std::optional<TextDocumentClientCapabilities> {
  if (!repr_->contains("textDocument")) return std::nullopt;

  auto& value = (*repr_)["textDocument"];

  return TextDocumentClientCapabilities(value);
}

auto ClientCapabilities::notebookDocument() const
    -> std::optional<NotebookDocumentClientCapabilities> {
  if (!repr_->contains("notebookDocument")) return std::nullopt;

  auto& value = (*repr_)["notebookDocument"];

  return NotebookDocumentClientCapabilities(value);
}

auto ClientCapabilities::window() const
    -> std::optional<WindowClientCapabilities> {
  if (!repr_->contains("window")) return std::nullopt;

  auto& value = (*repr_)["window"];

  return WindowClientCapabilities(value);
}

auto ClientCapabilities::general() const
    -> std::optional<GeneralClientCapabilities> {
  if (!repr_->contains("general")) return std::nullopt;

  auto& value = (*repr_)["general"];

  return GeneralClientCapabilities(value);
}

auto ClientCapabilities::experimental() const -> std::optional<LSPAny> {
  if (!repr_->contains("experimental")) return std::nullopt;

  auto& value = (*repr_)["experimental"];

  assert(value.is_object());
  return LSPAny(value);
}

auto ClientCapabilities::workspace(
    std::optional<WorkspaceClientCapabilities> workspace)
    -> ClientCapabilities& {
  if (!workspace.has_value()) {
    repr_->erase("workspace");
    return *this;
  }
  repr_->emplace("workspace", workspace.value());
  return *this;
}

auto ClientCapabilities::textDocument(
    std::optional<TextDocumentClientCapabilities> textDocument)
    -> ClientCapabilities& {
  if (!textDocument.has_value()) {
    repr_->erase("textDocument");
    return *this;
  }
  repr_->emplace("textDocument", textDocument.value());
  return *this;
}

auto ClientCapabilities::notebookDocument(
    std::optional<NotebookDocumentClientCapabilities> notebookDocument)
    -> ClientCapabilities& {
  if (!notebookDocument.has_value()) {
    repr_->erase("notebookDocument");
    return *this;
  }
  repr_->emplace("notebookDocument", notebookDocument.value());
  return *this;
}

auto ClientCapabilities::window(std::optional<WindowClientCapabilities> window)
    -> ClientCapabilities& {
  if (!window.has_value()) {
    repr_->erase("window");
    return *this;
  }
  repr_->emplace("window", window.value());
  return *this;
}

auto ClientCapabilities::general(
    std::optional<GeneralClientCapabilities> general) -> ClientCapabilities& {
  if (!general.has_value()) {
    repr_->erase("general");
    return *this;
  }
  repr_->emplace("general", general.value());
  return *this;
}

auto ClientCapabilities::experimental(std::optional<LSPAny> experimental)
    -> ClientCapabilities& {
  if (!experimental.has_value()) {
    repr_->erase("experimental");
    return *this;
  }
  lsp_runtime_error("ClientCapabilities::experimental: not implement yet");
  return *this;
}

TextDocumentSyncOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TextDocumentSyncOptions::openClose() const -> std::optional<bool> {
  if (!repr_->contains("openClose")) return std::nullopt;

  auto& value = (*repr_)["openClose"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncOptions::change() const
    -> std::optional<TextDocumentSyncKind> {
  if (!repr_->contains("change")) return std::nullopt;

  auto& value = (*repr_)["change"];

  return TextDocumentSyncKind(value);
}

auto TextDocumentSyncOptions::willSave() const -> std::optional<bool> {
  if (!repr_->contains("willSave")) return std::nullopt;

  auto& value = (*repr_)["willSave"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncOptions::willSaveWaitUntil() const -> std::optional<bool> {
  if (!repr_->contains("willSaveWaitUntil")) return std::nullopt;

  auto& value = (*repr_)["willSaveWaitUntil"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncOptions::save() const
    -> std::optional<std::variant<std::monostate, bool, SaveOptions>> {
  if (!repr_->contains("save")) return std::nullopt;

  auto& value = (*repr_)["save"];

  std::variant<std::monostate, bool, SaveOptions> result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentSyncOptions::openClose(std::optional<bool> openClose)
    -> TextDocumentSyncOptions& {
  if (!openClose.has_value()) {
    repr_->erase("openClose");
    return *this;
  }
  repr_->emplace("openClose", std::move(openClose.value()));
  return *this;
}

auto TextDocumentSyncOptions::change(std::optional<TextDocumentSyncKind> change)
    -> TextDocumentSyncOptions& {
  if (!change.has_value()) {
    repr_->erase("change");
    return *this;
  }
  repr_->emplace("change", static_cast<long>(change.value()));
  return *this;
}

auto TextDocumentSyncOptions::willSave(std::optional<bool> willSave)
    -> TextDocumentSyncOptions& {
  if (!willSave.has_value()) {
    repr_->erase("willSave");
    return *this;
  }
  repr_->emplace("willSave", std::move(willSave.value()));
  return *this;
}

auto TextDocumentSyncOptions::willSaveWaitUntil(
    std::optional<bool> willSaveWaitUntil) -> TextDocumentSyncOptions& {
  if (!willSaveWaitUntil.has_value()) {
    repr_->erase("willSaveWaitUntil");
    return *this;
  }
  repr_->emplace("willSaveWaitUntil", std::move(willSaveWaitUntil.value()));
  return *this;
}

auto TextDocumentSyncOptions::save(
    std::optional<std::variant<std::monostate, bool, SaveOptions>> save)
    -> TextDocumentSyncOptions& {
  if (!save.has_value()) {
    repr_->erase("save");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool save) { repr_->emplace("save", std::move(save)); }

    void operator()(SaveOptions save) { repr_->emplace("save", save); }
  } v{repr_};

  std::visit(v, save.value());

  return *this;
}

WorkspaceOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceOptions::workspaceFolders() const
    -> std::optional<WorkspaceFoldersServerCapabilities> {
  if (!repr_->contains("workspaceFolders")) return std::nullopt;

  auto& value = (*repr_)["workspaceFolders"];

  return WorkspaceFoldersServerCapabilities(value);
}

auto WorkspaceOptions::fileOperations() const
    -> std::optional<FileOperationOptions> {
  if (!repr_->contains("fileOperations")) return std::nullopt;

  auto& value = (*repr_)["fileOperations"];

  return FileOperationOptions(value);
}

auto WorkspaceOptions::textDocumentContent() const
    -> std::optional<std::variant<std::monostate, TextDocumentContentOptions,
                                  TextDocumentContentRegistrationOptions>> {
  if (!repr_->contains("textDocumentContent")) return std::nullopt;

  auto& value = (*repr_)["textDocumentContent"];

  std::variant<std::monostate, TextDocumentContentOptions,
               TextDocumentContentRegistrationOptions>
      result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceOptions::workspaceFolders(
    std::optional<WorkspaceFoldersServerCapabilities> workspaceFolders)
    -> WorkspaceOptions& {
  if (!workspaceFolders.has_value()) {
    repr_->erase("workspaceFolders");
    return *this;
  }
  repr_->emplace("workspaceFolders", workspaceFolders.value());
  return *this;
}

auto WorkspaceOptions::fileOperations(
    std::optional<FileOperationOptions> fileOperations) -> WorkspaceOptions& {
  if (!fileOperations.has_value()) {
    repr_->erase("fileOperations");
    return *this;
  }
  repr_->emplace("fileOperations", fileOperations.value());
  return *this;
}

auto WorkspaceOptions::textDocumentContent(
    std::optional<std::variant<std::monostate, TextDocumentContentOptions,
                               TextDocumentContentRegistrationOptions>>
        textDocumentContent) -> WorkspaceOptions& {
  if (!textDocumentContent.has_value()) {
    repr_->erase("textDocumentContent");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(TextDocumentContentOptions textDocumentContent) {
      repr_->emplace("textDocumentContent", textDocumentContent);
    }

    void operator()(
        TextDocumentContentRegistrationOptions textDocumentContent) {
      repr_->emplace("textDocumentContent", textDocumentContent);
    }
  } v{repr_};

  std::visit(v, textDocumentContent.value());

  return *this;
}

TextDocumentContentChangePartial::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("range")) return false;
  if (!repr_->contains("text")) return false;
  return true;
}

auto TextDocumentContentChangePartial::range() const -> Range {
  auto& value = (*repr_)["range"];

  return Range(value);
}

auto TextDocumentContentChangePartial::rangeLength() const
    -> std::optional<long> {
  if (!repr_->contains("rangeLength")) return std::nullopt;

  auto& value = (*repr_)["rangeLength"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto TextDocumentContentChangePartial::text() const -> std::string {
  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentChangePartial::range(Range range)
    -> TextDocumentContentChangePartial& {
  repr_->emplace("range", range);
  return *this;
}

auto TextDocumentContentChangePartial::rangeLength(
    std::optional<long> rangeLength) -> TextDocumentContentChangePartial& {
  if (!rangeLength.has_value()) {
    repr_->erase("rangeLength");
    return *this;
  }
  repr_->emplace("rangeLength", std::move(rangeLength.value()));
  return *this;
}

auto TextDocumentContentChangePartial::text(std::string text)
    -> TextDocumentContentChangePartial& {
  repr_->emplace("text", std::move(text));
  return *this;
}

TextDocumentContentChangeWholeDocument::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("text")) return false;
  return true;
}

auto TextDocumentContentChangeWholeDocument::text() const -> std::string {
  auto& value = (*repr_)["text"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentContentChangeWholeDocument::text(std::string text)
    -> TextDocumentContentChangeWholeDocument& {
  repr_->emplace("text", std::move(text));
  return *this;
}

CodeDescription::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("href")) return false;
  return true;
}

auto CodeDescription::href() const -> std::string {
  auto& value = (*repr_)["href"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto CodeDescription::href(std::string href) -> CodeDescription& {
  repr_->emplace("href", std::move(href));
  return *this;
}

DiagnosticRelatedInformation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("location")) return false;
  if (!repr_->contains("message")) return false;
  return true;
}

auto DiagnosticRelatedInformation::location() const -> Location {
  auto& value = (*repr_)["location"];

  return Location(value);
}

auto DiagnosticRelatedInformation::message() const -> std::string {
  auto& value = (*repr_)["message"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto DiagnosticRelatedInformation::location(Location location)
    -> DiagnosticRelatedInformation& {
  repr_->emplace("location", location);
  return *this;
}

auto DiagnosticRelatedInformation::message(std::string message)
    -> DiagnosticRelatedInformation& {
  repr_->emplace("message", std::move(message));
  return *this;
}

EditRangeWithInsertReplace::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("insert")) return false;
  if (!repr_->contains("replace")) return false;
  return true;
}

auto EditRangeWithInsertReplace::insert() const -> Range {
  auto& value = (*repr_)["insert"];

  return Range(value);
}

auto EditRangeWithInsertReplace::replace() const -> Range {
  auto& value = (*repr_)["replace"];

  return Range(value);
}

auto EditRangeWithInsertReplace::insert(Range insert)
    -> EditRangeWithInsertReplace& {
  repr_->emplace("insert", insert);
  return *this;
}

auto EditRangeWithInsertReplace::replace(Range replace)
    -> EditRangeWithInsertReplace& {
  repr_->emplace("replace", replace);
  return *this;
}

ServerCompletionItemOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ServerCompletionItemOptions::labelDetailsSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("labelDetailsSupport")) return std::nullopt;

  auto& value = (*repr_)["labelDetailsSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ServerCompletionItemOptions::labelDetailsSupport(
    std::optional<bool> labelDetailsSupport) -> ServerCompletionItemOptions& {
  if (!labelDetailsSupport.has_value()) {
    repr_->erase("labelDetailsSupport");
    return *this;
  }
  repr_->emplace("labelDetailsSupport", std::move(labelDetailsSupport.value()));
  return *this;
}

MarkedStringWithLanguage::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("language")) return false;
  if (!repr_->contains("value")) return false;
  return true;
}

auto MarkedStringWithLanguage::language() const -> std::string {
  auto& value = (*repr_)["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkedStringWithLanguage::value() const -> std::string {
  auto& value = (*repr_)["value"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkedStringWithLanguage::language(std::string language)
    -> MarkedStringWithLanguage& {
  repr_->emplace("language", std::move(language));
  return *this;
}

auto MarkedStringWithLanguage::value(std::string value)
    -> MarkedStringWithLanguage& {
  repr_->emplace("value", std::move(value));
  return *this;
}

ParameterInformation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("label")) return false;
  return true;
}

auto ParameterInformation::label() const
    -> std::variant<std::monostate, std::string, std::tuple<long, long>> {
  auto& value = (*repr_)["label"];

  std::variant<std::monostate, std::string, std::tuple<long, long>> result;

  details::try_emplace(result, value);

  return result;
}

auto ParameterInformation::documentation() const
    -> std::optional<std::variant<std::monostate, std::string, MarkupContent>> {
  if (!repr_->contains("documentation")) return std::nullopt;

  auto& value = (*repr_)["documentation"];

  std::variant<std::monostate, std::string, MarkupContent> result;

  details::try_emplace(result, value);

  return result;
}

auto ParameterInformation::label(
    std::variant<std::monostate, std::string, std::tuple<long, long>> label)
    -> ParameterInformation& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string label) {
      repr_->emplace("label", std::move(label));
    }

    void operator()(std::tuple<long, long> label) {
      lsp_runtime_error("ParameterInformation::label: not implement yet");
    }
  } v{repr_};

  std::visit(v, label);

  return *this;
}

auto ParameterInformation::documentation(
    std::optional<std::variant<std::monostate, std::string, MarkupContent>>
        documentation) -> ParameterInformation& {
  if (!documentation.has_value()) {
    repr_->erase("documentation");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string documentation) {
      repr_->emplace("documentation", std::move(documentation));
    }

    void operator()(MarkupContent documentation) {
      repr_->emplace("documentation", documentation);
    }
  } v{repr_};

  std::visit(v, documentation.value());

  return *this;
}

CodeActionKindDocumentation::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("kind")) return false;
  if (!repr_->contains("command")) return false;
  return true;
}

auto CodeActionKindDocumentation::kind() const -> CodeActionKind {
  auto& value = (*repr_)["kind"];

  lsp_runtime_error("CodeActionKindDocumentation::kind: not implement yet");
}

auto CodeActionKindDocumentation::command() const -> Command {
  auto& value = (*repr_)["command"];

  return Command(value);
}

auto CodeActionKindDocumentation::kind(CodeActionKind kind)
    -> CodeActionKindDocumentation& {
  lsp_runtime_error("CodeActionKindDocumentation::kind: not implement yet");
  return *this;
}

auto CodeActionKindDocumentation::command(Command command)
    -> CodeActionKindDocumentation& {
  repr_->emplace("command", command);
  return *this;
}

NotebookCellTextDocumentFilter::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebook")) return false;
  return true;
}

auto NotebookCellTextDocumentFilter::notebook() const
    -> std::variant<std::monostate, std::string, NotebookDocumentFilter> {
  auto& value = (*repr_)["notebook"];

  std::variant<std::monostate, std::string, NotebookDocumentFilter> result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookCellTextDocumentFilter::language() const
    -> std::optional<std::string> {
  if (!repr_->contains("language")) return std::nullopt;

  auto& value = (*repr_)["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookCellTextDocumentFilter::notebook(
    std::variant<std::monostate, std::string, NotebookDocumentFilter> notebook)
    -> NotebookCellTextDocumentFilter& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string notebook) {
      repr_->emplace("notebook", std::move(notebook));
    }

    void operator()(NotebookDocumentFilter notebook) {
      lsp_runtime_error(
          "NotebookCellTextDocumentFilter::notebook: not implement yet");
    }
  } v{repr_};

  std::visit(v, notebook);

  return *this;
}

auto NotebookCellTextDocumentFilter::language(
    std::optional<std::string> language) -> NotebookCellTextDocumentFilter& {
  if (!language.has_value()) {
    repr_->erase("language");
    return *this;
  }
  repr_->emplace("language", std::move(language.value()));
  return *this;
}

FileOperationPatternOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto FileOperationPatternOptions::ignoreCase() const -> std::optional<bool> {
  if (!repr_->contains("ignoreCase")) return std::nullopt;

  auto& value = (*repr_)["ignoreCase"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationPatternOptions::ignoreCase(std::optional<bool> ignoreCase)
    -> FileOperationPatternOptions& {
  if (!ignoreCase.has_value()) {
    repr_->erase("ignoreCase");
    return *this;
  }
  repr_->emplace("ignoreCase", std::move(ignoreCase.value()));
  return *this;
}

ExecutionSummary::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("executionOrder")) return false;
  return true;
}

auto ExecutionSummary::executionOrder() const -> long {
  auto& value = (*repr_)["executionOrder"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto ExecutionSummary::success() const -> std::optional<bool> {
  if (!repr_->contains("success")) return std::nullopt;

  auto& value = (*repr_)["success"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecutionSummary::executionOrder(long executionOrder)
    -> ExecutionSummary& {
  repr_->emplace("executionOrder", std::move(executionOrder));
  return *this;
}

auto ExecutionSummary::success(std::optional<bool> success)
    -> ExecutionSummary& {
  if (!success.has_value()) {
    repr_->erase("success");
    return *this;
  }
  repr_->emplace("success", std::move(success.value()));
  return *this;
}

NotebookCellLanguage::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("language")) return false;
  return true;
}

auto NotebookCellLanguage::language() const -> std::string {
  auto& value = (*repr_)["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookCellLanguage::language(std::string language)
    -> NotebookCellLanguage& {
  repr_->emplace("language", std::move(language));
  return *this;
}

NotebookDocumentCellChangeStructure::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("array")) return false;
  return true;
}

auto NotebookDocumentCellChangeStructure::array() const
    -> NotebookCellArrayChange {
  auto& value = (*repr_)["array"];

  return NotebookCellArrayChange(value);
}

auto NotebookDocumentCellChangeStructure::didOpen() const
    -> std::optional<Vector<TextDocumentItem>> {
  if (!repr_->contains("didOpen")) return std::nullopt;

  auto& value = (*repr_)["didOpen"];

  assert(value.is_array());
  return Vector<TextDocumentItem>(value);
}

auto NotebookDocumentCellChangeStructure::didClose() const
    -> std::optional<Vector<TextDocumentIdentifier>> {
  if (!repr_->contains("didClose")) return std::nullopt;

  auto& value = (*repr_)["didClose"];

  assert(value.is_array());
  return Vector<TextDocumentIdentifier>(value);
}

auto NotebookDocumentCellChangeStructure::array(NotebookCellArrayChange array)
    -> NotebookDocumentCellChangeStructure& {
  repr_->emplace("array", array);
  return *this;
}

auto NotebookDocumentCellChangeStructure::didOpen(
    std::optional<Vector<TextDocumentItem>> didOpen)
    -> NotebookDocumentCellChangeStructure& {
  if (!didOpen.has_value()) {
    repr_->erase("didOpen");
    return *this;
  }
  lsp_runtime_error(
      "NotebookDocumentCellChangeStructure::didOpen: not implement yet");
  return *this;
}

auto NotebookDocumentCellChangeStructure::didClose(
    std::optional<Vector<TextDocumentIdentifier>> didClose)
    -> NotebookDocumentCellChangeStructure& {
  if (!didClose.has_value()) {
    repr_->erase("didClose");
    return *this;
  }
  lsp_runtime_error(
      "NotebookDocumentCellChangeStructure::didClose: not implement yet");
  return *this;
}

NotebookDocumentCellContentChanges::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("document")) return false;
  if (!repr_->contains("changes")) return false;
  return true;
}

auto NotebookDocumentCellContentChanges::document() const
    -> VersionedTextDocumentIdentifier {
  auto& value = (*repr_)["document"];

  return VersionedTextDocumentIdentifier(value);
}

auto NotebookDocumentCellContentChanges::changes() const
    -> Vector<TextDocumentContentChangeEvent> {
  auto& value = (*repr_)["changes"];

  assert(value.is_array());
  return Vector<TextDocumentContentChangeEvent>(value);
}

auto NotebookDocumentCellContentChanges::document(
    VersionedTextDocumentIdentifier document)
    -> NotebookDocumentCellContentChanges& {
  repr_->emplace("document", document);
  return *this;
}

auto NotebookDocumentCellContentChanges::changes(
    Vector<TextDocumentContentChangeEvent> changes)
    -> NotebookDocumentCellContentChanges& {
  lsp_runtime_error(
      "NotebookDocumentCellContentChanges::changes: not implement yet");
  return *this;
}

WorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceClientCapabilities::applyEdit() const -> std::optional<bool> {
  if (!repr_->contains("applyEdit")) return std::nullopt;

  auto& value = (*repr_)["applyEdit"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceClientCapabilities::workspaceEdit() const
    -> std::optional<WorkspaceEditClientCapabilities> {
  if (!repr_->contains("workspaceEdit")) return std::nullopt;

  auto& value = (*repr_)["workspaceEdit"];

  return WorkspaceEditClientCapabilities(value);
}

auto WorkspaceClientCapabilities::didChangeConfiguration() const
    -> std::optional<DidChangeConfigurationClientCapabilities> {
  if (!repr_->contains("didChangeConfiguration")) return std::nullopt;

  auto& value = (*repr_)["didChangeConfiguration"];

  return DidChangeConfigurationClientCapabilities(value);
}

auto WorkspaceClientCapabilities::didChangeWatchedFiles() const
    -> std::optional<DidChangeWatchedFilesClientCapabilities> {
  if (!repr_->contains("didChangeWatchedFiles")) return std::nullopt;

  auto& value = (*repr_)["didChangeWatchedFiles"];

  return DidChangeWatchedFilesClientCapabilities(value);
}

auto WorkspaceClientCapabilities::symbol() const
    -> std::optional<WorkspaceSymbolClientCapabilities> {
  if (!repr_->contains("symbol")) return std::nullopt;

  auto& value = (*repr_)["symbol"];

  return WorkspaceSymbolClientCapabilities(value);
}

auto WorkspaceClientCapabilities::executeCommand() const
    -> std::optional<ExecuteCommandClientCapabilities> {
  if (!repr_->contains("executeCommand")) return std::nullopt;

  auto& value = (*repr_)["executeCommand"];

  return ExecuteCommandClientCapabilities(value);
}

auto WorkspaceClientCapabilities::workspaceFolders() const
    -> std::optional<bool> {
  if (!repr_->contains("workspaceFolders")) return std::nullopt;

  auto& value = (*repr_)["workspaceFolders"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceClientCapabilities::configuration() const -> std::optional<bool> {
  if (!repr_->contains("configuration")) return std::nullopt;

  auto& value = (*repr_)["configuration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceClientCapabilities::semanticTokens() const
    -> std::optional<SemanticTokensWorkspaceClientCapabilities> {
  if (!repr_->contains("semanticTokens")) return std::nullopt;

  auto& value = (*repr_)["semanticTokens"];

  return SemanticTokensWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::codeLens() const
    -> std::optional<CodeLensWorkspaceClientCapabilities> {
  if (!repr_->contains("codeLens")) return std::nullopt;

  auto& value = (*repr_)["codeLens"];

  return CodeLensWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::fileOperations() const
    -> std::optional<FileOperationClientCapabilities> {
  if (!repr_->contains("fileOperations")) return std::nullopt;

  auto& value = (*repr_)["fileOperations"];

  return FileOperationClientCapabilities(value);
}

auto WorkspaceClientCapabilities::inlineValue() const
    -> std::optional<InlineValueWorkspaceClientCapabilities> {
  if (!repr_->contains("inlineValue")) return std::nullopt;

  auto& value = (*repr_)["inlineValue"];

  return InlineValueWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::inlayHint() const
    -> std::optional<InlayHintWorkspaceClientCapabilities> {
  if (!repr_->contains("inlayHint")) return std::nullopt;

  auto& value = (*repr_)["inlayHint"];

  return InlayHintWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::diagnostics() const
    -> std::optional<DiagnosticWorkspaceClientCapabilities> {
  if (!repr_->contains("diagnostics")) return std::nullopt;

  auto& value = (*repr_)["diagnostics"];

  return DiagnosticWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::foldingRange() const
    -> std::optional<FoldingRangeWorkspaceClientCapabilities> {
  if (!repr_->contains("foldingRange")) return std::nullopt;

  auto& value = (*repr_)["foldingRange"];

  return FoldingRangeWorkspaceClientCapabilities(value);
}

auto WorkspaceClientCapabilities::textDocumentContent() const
    -> std::optional<TextDocumentContentClientCapabilities> {
  if (!repr_->contains("textDocumentContent")) return std::nullopt;

  auto& value = (*repr_)["textDocumentContent"];

  return TextDocumentContentClientCapabilities(value);
}

auto WorkspaceClientCapabilities::applyEdit(std::optional<bool> applyEdit)
    -> WorkspaceClientCapabilities& {
  if (!applyEdit.has_value()) {
    repr_->erase("applyEdit");
    return *this;
  }
  repr_->emplace("applyEdit", std::move(applyEdit.value()));
  return *this;
}

auto WorkspaceClientCapabilities::workspaceEdit(
    std::optional<WorkspaceEditClientCapabilities> workspaceEdit)
    -> WorkspaceClientCapabilities& {
  if (!workspaceEdit.has_value()) {
    repr_->erase("workspaceEdit");
    return *this;
  }
  repr_->emplace("workspaceEdit", workspaceEdit.value());
  return *this;
}

auto WorkspaceClientCapabilities::didChangeConfiguration(
    std::optional<DidChangeConfigurationClientCapabilities>
        didChangeConfiguration) -> WorkspaceClientCapabilities& {
  if (!didChangeConfiguration.has_value()) {
    repr_->erase("didChangeConfiguration");
    return *this;
  }
  repr_->emplace("didChangeConfiguration", didChangeConfiguration.value());
  return *this;
}

auto WorkspaceClientCapabilities::didChangeWatchedFiles(
    std::optional<DidChangeWatchedFilesClientCapabilities>
        didChangeWatchedFiles) -> WorkspaceClientCapabilities& {
  if (!didChangeWatchedFiles.has_value()) {
    repr_->erase("didChangeWatchedFiles");
    return *this;
  }
  repr_->emplace("didChangeWatchedFiles", didChangeWatchedFiles.value());
  return *this;
}

auto WorkspaceClientCapabilities::symbol(
    std::optional<WorkspaceSymbolClientCapabilities> symbol)
    -> WorkspaceClientCapabilities& {
  if (!symbol.has_value()) {
    repr_->erase("symbol");
    return *this;
  }
  repr_->emplace("symbol", symbol.value());
  return *this;
}

auto WorkspaceClientCapabilities::executeCommand(
    std::optional<ExecuteCommandClientCapabilities> executeCommand)
    -> WorkspaceClientCapabilities& {
  if (!executeCommand.has_value()) {
    repr_->erase("executeCommand");
    return *this;
  }
  repr_->emplace("executeCommand", executeCommand.value());
  return *this;
}

auto WorkspaceClientCapabilities::workspaceFolders(
    std::optional<bool> workspaceFolders) -> WorkspaceClientCapabilities& {
  if (!workspaceFolders.has_value()) {
    repr_->erase("workspaceFolders");
    return *this;
  }
  repr_->emplace("workspaceFolders", std::move(workspaceFolders.value()));
  return *this;
}

auto WorkspaceClientCapabilities::configuration(
    std::optional<bool> configuration) -> WorkspaceClientCapabilities& {
  if (!configuration.has_value()) {
    repr_->erase("configuration");
    return *this;
  }
  repr_->emplace("configuration", std::move(configuration.value()));
  return *this;
}

auto WorkspaceClientCapabilities::semanticTokens(
    std::optional<SemanticTokensWorkspaceClientCapabilities> semanticTokens)
    -> WorkspaceClientCapabilities& {
  if (!semanticTokens.has_value()) {
    repr_->erase("semanticTokens");
    return *this;
  }
  repr_->emplace("semanticTokens", semanticTokens.value());
  return *this;
}

auto WorkspaceClientCapabilities::codeLens(
    std::optional<CodeLensWorkspaceClientCapabilities> codeLens)
    -> WorkspaceClientCapabilities& {
  if (!codeLens.has_value()) {
    repr_->erase("codeLens");
    return *this;
  }
  repr_->emplace("codeLens", codeLens.value());
  return *this;
}

auto WorkspaceClientCapabilities::fileOperations(
    std::optional<FileOperationClientCapabilities> fileOperations)
    -> WorkspaceClientCapabilities& {
  if (!fileOperations.has_value()) {
    repr_->erase("fileOperations");
    return *this;
  }
  repr_->emplace("fileOperations", fileOperations.value());
  return *this;
}

auto WorkspaceClientCapabilities::inlineValue(
    std::optional<InlineValueWorkspaceClientCapabilities> inlineValue)
    -> WorkspaceClientCapabilities& {
  if (!inlineValue.has_value()) {
    repr_->erase("inlineValue");
    return *this;
  }
  repr_->emplace("inlineValue", inlineValue.value());
  return *this;
}

auto WorkspaceClientCapabilities::inlayHint(
    std::optional<InlayHintWorkspaceClientCapabilities> inlayHint)
    -> WorkspaceClientCapabilities& {
  if (!inlayHint.has_value()) {
    repr_->erase("inlayHint");
    return *this;
  }
  repr_->emplace("inlayHint", inlayHint.value());
  return *this;
}

auto WorkspaceClientCapabilities::diagnostics(
    std::optional<DiagnosticWorkspaceClientCapabilities> diagnostics)
    -> WorkspaceClientCapabilities& {
  if (!diagnostics.has_value()) {
    repr_->erase("diagnostics");
    return *this;
  }
  repr_->emplace("diagnostics", diagnostics.value());
  return *this;
}

auto WorkspaceClientCapabilities::foldingRange(
    std::optional<FoldingRangeWorkspaceClientCapabilities> foldingRange)
    -> WorkspaceClientCapabilities& {
  if (!foldingRange.has_value()) {
    repr_->erase("foldingRange");
    return *this;
  }
  repr_->emplace("foldingRange", foldingRange.value());
  return *this;
}

auto WorkspaceClientCapabilities::textDocumentContent(
    std::optional<TextDocumentContentClientCapabilities> textDocumentContent)
    -> WorkspaceClientCapabilities& {
  if (!textDocumentContent.has_value()) {
    repr_->erase("textDocumentContent");
    return *this;
  }
  repr_->emplace("textDocumentContent", textDocumentContent.value());
  return *this;
}

TextDocumentClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TextDocumentClientCapabilities::synchronization() const
    -> std::optional<TextDocumentSyncClientCapabilities> {
  if (!repr_->contains("synchronization")) return std::nullopt;

  auto& value = (*repr_)["synchronization"];

  return TextDocumentSyncClientCapabilities(value);
}

auto TextDocumentClientCapabilities::filters() const
    -> std::optional<TextDocumentFilterClientCapabilities> {
  if (!repr_->contains("filters")) return std::nullopt;

  auto& value = (*repr_)["filters"];

  return TextDocumentFilterClientCapabilities(value);
}

auto TextDocumentClientCapabilities::completion() const
    -> std::optional<CompletionClientCapabilities> {
  if (!repr_->contains("completion")) return std::nullopt;

  auto& value = (*repr_)["completion"];

  return CompletionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::hover() const
    -> std::optional<HoverClientCapabilities> {
  if (!repr_->contains("hover")) return std::nullopt;

  auto& value = (*repr_)["hover"];

  return HoverClientCapabilities(value);
}

auto TextDocumentClientCapabilities::signatureHelp() const
    -> std::optional<SignatureHelpClientCapabilities> {
  if (!repr_->contains("signatureHelp")) return std::nullopt;

  auto& value = (*repr_)["signatureHelp"];

  return SignatureHelpClientCapabilities(value);
}

auto TextDocumentClientCapabilities::declaration() const
    -> std::optional<DeclarationClientCapabilities> {
  if (!repr_->contains("declaration")) return std::nullopt;

  auto& value = (*repr_)["declaration"];

  return DeclarationClientCapabilities(value);
}

auto TextDocumentClientCapabilities::definition() const
    -> std::optional<DefinitionClientCapabilities> {
  if (!repr_->contains("definition")) return std::nullopt;

  auto& value = (*repr_)["definition"];

  return DefinitionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::typeDefinition() const
    -> std::optional<TypeDefinitionClientCapabilities> {
  if (!repr_->contains("typeDefinition")) return std::nullopt;

  auto& value = (*repr_)["typeDefinition"];

  return TypeDefinitionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::implementation() const
    -> std::optional<ImplementationClientCapabilities> {
  if (!repr_->contains("implementation")) return std::nullopt;

  auto& value = (*repr_)["implementation"];

  return ImplementationClientCapabilities(value);
}

auto TextDocumentClientCapabilities::references() const
    -> std::optional<ReferenceClientCapabilities> {
  if (!repr_->contains("references")) return std::nullopt;

  auto& value = (*repr_)["references"];

  return ReferenceClientCapabilities(value);
}

auto TextDocumentClientCapabilities::documentHighlight() const
    -> std::optional<DocumentHighlightClientCapabilities> {
  if (!repr_->contains("documentHighlight")) return std::nullopt;

  auto& value = (*repr_)["documentHighlight"];

  return DocumentHighlightClientCapabilities(value);
}

auto TextDocumentClientCapabilities::documentSymbol() const
    -> std::optional<DocumentSymbolClientCapabilities> {
  if (!repr_->contains("documentSymbol")) return std::nullopt;

  auto& value = (*repr_)["documentSymbol"];

  return DocumentSymbolClientCapabilities(value);
}

auto TextDocumentClientCapabilities::codeAction() const
    -> std::optional<CodeActionClientCapabilities> {
  if (!repr_->contains("codeAction")) return std::nullopt;

  auto& value = (*repr_)["codeAction"];

  return CodeActionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::codeLens() const
    -> std::optional<CodeLensClientCapabilities> {
  if (!repr_->contains("codeLens")) return std::nullopt;

  auto& value = (*repr_)["codeLens"];

  return CodeLensClientCapabilities(value);
}

auto TextDocumentClientCapabilities::documentLink() const
    -> std::optional<DocumentLinkClientCapabilities> {
  if (!repr_->contains("documentLink")) return std::nullopt;

  auto& value = (*repr_)["documentLink"];

  return DocumentLinkClientCapabilities(value);
}

auto TextDocumentClientCapabilities::colorProvider() const
    -> std::optional<DocumentColorClientCapabilities> {
  if (!repr_->contains("colorProvider")) return std::nullopt;

  auto& value = (*repr_)["colorProvider"];

  return DocumentColorClientCapabilities(value);
}

auto TextDocumentClientCapabilities::formatting() const
    -> std::optional<DocumentFormattingClientCapabilities> {
  if (!repr_->contains("formatting")) return std::nullopt;

  auto& value = (*repr_)["formatting"];

  return DocumentFormattingClientCapabilities(value);
}

auto TextDocumentClientCapabilities::rangeFormatting() const
    -> std::optional<DocumentRangeFormattingClientCapabilities> {
  if (!repr_->contains("rangeFormatting")) return std::nullopt;

  auto& value = (*repr_)["rangeFormatting"];

  return DocumentRangeFormattingClientCapabilities(value);
}

auto TextDocumentClientCapabilities::onTypeFormatting() const
    -> std::optional<DocumentOnTypeFormattingClientCapabilities> {
  if (!repr_->contains("onTypeFormatting")) return std::nullopt;

  auto& value = (*repr_)["onTypeFormatting"];

  return DocumentOnTypeFormattingClientCapabilities(value);
}

auto TextDocumentClientCapabilities::rename() const
    -> std::optional<RenameClientCapabilities> {
  if (!repr_->contains("rename")) return std::nullopt;

  auto& value = (*repr_)["rename"];

  return RenameClientCapabilities(value);
}

auto TextDocumentClientCapabilities::foldingRange() const
    -> std::optional<FoldingRangeClientCapabilities> {
  if (!repr_->contains("foldingRange")) return std::nullopt;

  auto& value = (*repr_)["foldingRange"];

  return FoldingRangeClientCapabilities(value);
}

auto TextDocumentClientCapabilities::selectionRange() const
    -> std::optional<SelectionRangeClientCapabilities> {
  if (!repr_->contains("selectionRange")) return std::nullopt;

  auto& value = (*repr_)["selectionRange"];

  return SelectionRangeClientCapabilities(value);
}

auto TextDocumentClientCapabilities::publishDiagnostics() const
    -> std::optional<PublishDiagnosticsClientCapabilities> {
  if (!repr_->contains("publishDiagnostics")) return std::nullopt;

  auto& value = (*repr_)["publishDiagnostics"];

  return PublishDiagnosticsClientCapabilities(value);
}

auto TextDocumentClientCapabilities::callHierarchy() const
    -> std::optional<CallHierarchyClientCapabilities> {
  if (!repr_->contains("callHierarchy")) return std::nullopt;

  auto& value = (*repr_)["callHierarchy"];

  return CallHierarchyClientCapabilities(value);
}

auto TextDocumentClientCapabilities::semanticTokens() const
    -> std::optional<SemanticTokensClientCapabilities> {
  if (!repr_->contains("semanticTokens")) return std::nullopt;

  auto& value = (*repr_)["semanticTokens"];

  return SemanticTokensClientCapabilities(value);
}

auto TextDocumentClientCapabilities::linkedEditingRange() const
    -> std::optional<LinkedEditingRangeClientCapabilities> {
  if (!repr_->contains("linkedEditingRange")) return std::nullopt;

  auto& value = (*repr_)["linkedEditingRange"];

  return LinkedEditingRangeClientCapabilities(value);
}

auto TextDocumentClientCapabilities::moniker() const
    -> std::optional<MonikerClientCapabilities> {
  if (!repr_->contains("moniker")) return std::nullopt;

  auto& value = (*repr_)["moniker"];

  return MonikerClientCapabilities(value);
}

auto TextDocumentClientCapabilities::typeHierarchy() const
    -> std::optional<TypeHierarchyClientCapabilities> {
  if (!repr_->contains("typeHierarchy")) return std::nullopt;

  auto& value = (*repr_)["typeHierarchy"];

  return TypeHierarchyClientCapabilities(value);
}

auto TextDocumentClientCapabilities::inlineValue() const
    -> std::optional<InlineValueClientCapabilities> {
  if (!repr_->contains("inlineValue")) return std::nullopt;

  auto& value = (*repr_)["inlineValue"];

  return InlineValueClientCapabilities(value);
}

auto TextDocumentClientCapabilities::inlayHint() const
    -> std::optional<InlayHintClientCapabilities> {
  if (!repr_->contains("inlayHint")) return std::nullopt;

  auto& value = (*repr_)["inlayHint"];

  return InlayHintClientCapabilities(value);
}

auto TextDocumentClientCapabilities::diagnostic() const
    -> std::optional<DiagnosticClientCapabilities> {
  if (!repr_->contains("diagnostic")) return std::nullopt;

  auto& value = (*repr_)["diagnostic"];

  return DiagnosticClientCapabilities(value);
}

auto TextDocumentClientCapabilities::inlineCompletion() const
    -> std::optional<InlineCompletionClientCapabilities> {
  if (!repr_->contains("inlineCompletion")) return std::nullopt;

  auto& value = (*repr_)["inlineCompletion"];

  return InlineCompletionClientCapabilities(value);
}

auto TextDocumentClientCapabilities::synchronization(
    std::optional<TextDocumentSyncClientCapabilities> synchronization)
    -> TextDocumentClientCapabilities& {
  if (!synchronization.has_value()) {
    repr_->erase("synchronization");
    return *this;
  }
  repr_->emplace("synchronization", synchronization.value());
  return *this;
}

auto TextDocumentClientCapabilities::filters(
    std::optional<TextDocumentFilterClientCapabilities> filters)
    -> TextDocumentClientCapabilities& {
  if (!filters.has_value()) {
    repr_->erase("filters");
    return *this;
  }
  repr_->emplace("filters", filters.value());
  return *this;
}

auto TextDocumentClientCapabilities::completion(
    std::optional<CompletionClientCapabilities> completion)
    -> TextDocumentClientCapabilities& {
  if (!completion.has_value()) {
    repr_->erase("completion");
    return *this;
  }
  repr_->emplace("completion", completion.value());
  return *this;
}

auto TextDocumentClientCapabilities::hover(
    std::optional<HoverClientCapabilities> hover)
    -> TextDocumentClientCapabilities& {
  if (!hover.has_value()) {
    repr_->erase("hover");
    return *this;
  }
  repr_->emplace("hover", hover.value());
  return *this;
}

auto TextDocumentClientCapabilities::signatureHelp(
    std::optional<SignatureHelpClientCapabilities> signatureHelp)
    -> TextDocumentClientCapabilities& {
  if (!signatureHelp.has_value()) {
    repr_->erase("signatureHelp");
    return *this;
  }
  repr_->emplace("signatureHelp", signatureHelp.value());
  return *this;
}

auto TextDocumentClientCapabilities::declaration(
    std::optional<DeclarationClientCapabilities> declaration)
    -> TextDocumentClientCapabilities& {
  if (!declaration.has_value()) {
    repr_->erase("declaration");
    return *this;
  }
  repr_->emplace("declaration", declaration.value());
  return *this;
}

auto TextDocumentClientCapabilities::definition(
    std::optional<DefinitionClientCapabilities> definition)
    -> TextDocumentClientCapabilities& {
  if (!definition.has_value()) {
    repr_->erase("definition");
    return *this;
  }
  repr_->emplace("definition", definition.value());
  return *this;
}

auto TextDocumentClientCapabilities::typeDefinition(
    std::optional<TypeDefinitionClientCapabilities> typeDefinition)
    -> TextDocumentClientCapabilities& {
  if (!typeDefinition.has_value()) {
    repr_->erase("typeDefinition");
    return *this;
  }
  repr_->emplace("typeDefinition", typeDefinition.value());
  return *this;
}

auto TextDocumentClientCapabilities::implementation(
    std::optional<ImplementationClientCapabilities> implementation)
    -> TextDocumentClientCapabilities& {
  if (!implementation.has_value()) {
    repr_->erase("implementation");
    return *this;
  }
  repr_->emplace("implementation", implementation.value());
  return *this;
}

auto TextDocumentClientCapabilities::references(
    std::optional<ReferenceClientCapabilities> references)
    -> TextDocumentClientCapabilities& {
  if (!references.has_value()) {
    repr_->erase("references");
    return *this;
  }
  repr_->emplace("references", references.value());
  return *this;
}

auto TextDocumentClientCapabilities::documentHighlight(
    std::optional<DocumentHighlightClientCapabilities> documentHighlight)
    -> TextDocumentClientCapabilities& {
  if (!documentHighlight.has_value()) {
    repr_->erase("documentHighlight");
    return *this;
  }
  repr_->emplace("documentHighlight", documentHighlight.value());
  return *this;
}

auto TextDocumentClientCapabilities::documentSymbol(
    std::optional<DocumentSymbolClientCapabilities> documentSymbol)
    -> TextDocumentClientCapabilities& {
  if (!documentSymbol.has_value()) {
    repr_->erase("documentSymbol");
    return *this;
  }
  repr_->emplace("documentSymbol", documentSymbol.value());
  return *this;
}

auto TextDocumentClientCapabilities::codeAction(
    std::optional<CodeActionClientCapabilities> codeAction)
    -> TextDocumentClientCapabilities& {
  if (!codeAction.has_value()) {
    repr_->erase("codeAction");
    return *this;
  }
  repr_->emplace("codeAction", codeAction.value());
  return *this;
}

auto TextDocumentClientCapabilities::codeLens(
    std::optional<CodeLensClientCapabilities> codeLens)
    -> TextDocumentClientCapabilities& {
  if (!codeLens.has_value()) {
    repr_->erase("codeLens");
    return *this;
  }
  repr_->emplace("codeLens", codeLens.value());
  return *this;
}

auto TextDocumentClientCapabilities::documentLink(
    std::optional<DocumentLinkClientCapabilities> documentLink)
    -> TextDocumentClientCapabilities& {
  if (!documentLink.has_value()) {
    repr_->erase("documentLink");
    return *this;
  }
  repr_->emplace("documentLink", documentLink.value());
  return *this;
}

auto TextDocumentClientCapabilities::colorProvider(
    std::optional<DocumentColorClientCapabilities> colorProvider)
    -> TextDocumentClientCapabilities& {
  if (!colorProvider.has_value()) {
    repr_->erase("colorProvider");
    return *this;
  }
  repr_->emplace("colorProvider", colorProvider.value());
  return *this;
}

auto TextDocumentClientCapabilities::formatting(
    std::optional<DocumentFormattingClientCapabilities> formatting)
    -> TextDocumentClientCapabilities& {
  if (!formatting.has_value()) {
    repr_->erase("formatting");
    return *this;
  }
  repr_->emplace("formatting", formatting.value());
  return *this;
}

auto TextDocumentClientCapabilities::rangeFormatting(
    std::optional<DocumentRangeFormattingClientCapabilities> rangeFormatting)
    -> TextDocumentClientCapabilities& {
  if (!rangeFormatting.has_value()) {
    repr_->erase("rangeFormatting");
    return *this;
  }
  repr_->emplace("rangeFormatting", rangeFormatting.value());
  return *this;
}

auto TextDocumentClientCapabilities::onTypeFormatting(
    std::optional<DocumentOnTypeFormattingClientCapabilities> onTypeFormatting)
    -> TextDocumentClientCapabilities& {
  if (!onTypeFormatting.has_value()) {
    repr_->erase("onTypeFormatting");
    return *this;
  }
  repr_->emplace("onTypeFormatting", onTypeFormatting.value());
  return *this;
}

auto TextDocumentClientCapabilities::rename(
    std::optional<RenameClientCapabilities> rename)
    -> TextDocumentClientCapabilities& {
  if (!rename.has_value()) {
    repr_->erase("rename");
    return *this;
  }
  repr_->emplace("rename", rename.value());
  return *this;
}

auto TextDocumentClientCapabilities::foldingRange(
    std::optional<FoldingRangeClientCapabilities> foldingRange)
    -> TextDocumentClientCapabilities& {
  if (!foldingRange.has_value()) {
    repr_->erase("foldingRange");
    return *this;
  }
  repr_->emplace("foldingRange", foldingRange.value());
  return *this;
}

auto TextDocumentClientCapabilities::selectionRange(
    std::optional<SelectionRangeClientCapabilities> selectionRange)
    -> TextDocumentClientCapabilities& {
  if (!selectionRange.has_value()) {
    repr_->erase("selectionRange");
    return *this;
  }
  repr_->emplace("selectionRange", selectionRange.value());
  return *this;
}

auto TextDocumentClientCapabilities::publishDiagnostics(
    std::optional<PublishDiagnosticsClientCapabilities> publishDiagnostics)
    -> TextDocumentClientCapabilities& {
  if (!publishDiagnostics.has_value()) {
    repr_->erase("publishDiagnostics");
    return *this;
  }
  repr_->emplace("publishDiagnostics", publishDiagnostics.value());
  return *this;
}

auto TextDocumentClientCapabilities::callHierarchy(
    std::optional<CallHierarchyClientCapabilities> callHierarchy)
    -> TextDocumentClientCapabilities& {
  if (!callHierarchy.has_value()) {
    repr_->erase("callHierarchy");
    return *this;
  }
  repr_->emplace("callHierarchy", callHierarchy.value());
  return *this;
}

auto TextDocumentClientCapabilities::semanticTokens(
    std::optional<SemanticTokensClientCapabilities> semanticTokens)
    -> TextDocumentClientCapabilities& {
  if (!semanticTokens.has_value()) {
    repr_->erase("semanticTokens");
    return *this;
  }
  repr_->emplace("semanticTokens", semanticTokens.value());
  return *this;
}

auto TextDocumentClientCapabilities::linkedEditingRange(
    std::optional<LinkedEditingRangeClientCapabilities> linkedEditingRange)
    -> TextDocumentClientCapabilities& {
  if (!linkedEditingRange.has_value()) {
    repr_->erase("linkedEditingRange");
    return *this;
  }
  repr_->emplace("linkedEditingRange", linkedEditingRange.value());
  return *this;
}

auto TextDocumentClientCapabilities::moniker(
    std::optional<MonikerClientCapabilities> moniker)
    -> TextDocumentClientCapabilities& {
  if (!moniker.has_value()) {
    repr_->erase("moniker");
    return *this;
  }
  repr_->emplace("moniker", moniker.value());
  return *this;
}

auto TextDocumentClientCapabilities::typeHierarchy(
    std::optional<TypeHierarchyClientCapabilities> typeHierarchy)
    -> TextDocumentClientCapabilities& {
  if (!typeHierarchy.has_value()) {
    repr_->erase("typeHierarchy");
    return *this;
  }
  repr_->emplace("typeHierarchy", typeHierarchy.value());
  return *this;
}

auto TextDocumentClientCapabilities::inlineValue(
    std::optional<InlineValueClientCapabilities> inlineValue)
    -> TextDocumentClientCapabilities& {
  if (!inlineValue.has_value()) {
    repr_->erase("inlineValue");
    return *this;
  }
  repr_->emplace("inlineValue", inlineValue.value());
  return *this;
}

auto TextDocumentClientCapabilities::inlayHint(
    std::optional<InlayHintClientCapabilities> inlayHint)
    -> TextDocumentClientCapabilities& {
  if (!inlayHint.has_value()) {
    repr_->erase("inlayHint");
    return *this;
  }
  repr_->emplace("inlayHint", inlayHint.value());
  return *this;
}

auto TextDocumentClientCapabilities::diagnostic(
    std::optional<DiagnosticClientCapabilities> diagnostic)
    -> TextDocumentClientCapabilities& {
  if (!diagnostic.has_value()) {
    repr_->erase("diagnostic");
    return *this;
  }
  repr_->emplace("diagnostic", diagnostic.value());
  return *this;
}

auto TextDocumentClientCapabilities::inlineCompletion(
    std::optional<InlineCompletionClientCapabilities> inlineCompletion)
    -> TextDocumentClientCapabilities& {
  if (!inlineCompletion.has_value()) {
    repr_->erase("inlineCompletion");
    return *this;
  }
  repr_->emplace("inlineCompletion", inlineCompletion.value());
  return *this;
}

NotebookDocumentClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("synchronization")) return false;
  return true;
}

auto NotebookDocumentClientCapabilities::synchronization() const
    -> NotebookDocumentSyncClientCapabilities {
  auto& value = (*repr_)["synchronization"];

  return NotebookDocumentSyncClientCapabilities(value);
}

auto NotebookDocumentClientCapabilities::synchronization(
    NotebookDocumentSyncClientCapabilities synchronization)
    -> NotebookDocumentClientCapabilities& {
  repr_->emplace("synchronization", synchronization);
  return *this;
}

WindowClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WindowClientCapabilities::workDoneProgress() const -> std::optional<bool> {
  if (!repr_->contains("workDoneProgress")) return std::nullopt;

  auto& value = (*repr_)["workDoneProgress"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WindowClientCapabilities::showMessage() const
    -> std::optional<ShowMessageRequestClientCapabilities> {
  if (!repr_->contains("showMessage")) return std::nullopt;

  auto& value = (*repr_)["showMessage"];

  return ShowMessageRequestClientCapabilities(value);
}

auto WindowClientCapabilities::showDocument() const
    -> std::optional<ShowDocumentClientCapabilities> {
  if (!repr_->contains("showDocument")) return std::nullopt;

  auto& value = (*repr_)["showDocument"];

  return ShowDocumentClientCapabilities(value);
}

auto WindowClientCapabilities::workDoneProgress(
    std::optional<bool> workDoneProgress) -> WindowClientCapabilities& {
  if (!workDoneProgress.has_value()) {
    repr_->erase("workDoneProgress");
    return *this;
  }
  repr_->emplace("workDoneProgress", std::move(workDoneProgress.value()));
  return *this;
}

auto WindowClientCapabilities::showMessage(
    std::optional<ShowMessageRequestClientCapabilities> showMessage)
    -> WindowClientCapabilities& {
  if (!showMessage.has_value()) {
    repr_->erase("showMessage");
    return *this;
  }
  repr_->emplace("showMessage", showMessage.value());
  return *this;
}

auto WindowClientCapabilities::showDocument(
    std::optional<ShowDocumentClientCapabilities> showDocument)
    -> WindowClientCapabilities& {
  if (!showDocument.has_value()) {
    repr_->erase("showDocument");
    return *this;
  }
  repr_->emplace("showDocument", showDocument.value());
  return *this;
}

GeneralClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto GeneralClientCapabilities::staleRequestSupport() const
    -> std::optional<StaleRequestSupportOptions> {
  if (!repr_->contains("staleRequestSupport")) return std::nullopt;

  auto& value = (*repr_)["staleRequestSupport"];

  return StaleRequestSupportOptions(value);
}

auto GeneralClientCapabilities::regularExpressions() const
    -> std::optional<RegularExpressionsClientCapabilities> {
  if (!repr_->contains("regularExpressions")) return std::nullopt;

  auto& value = (*repr_)["regularExpressions"];

  return RegularExpressionsClientCapabilities(value);
}

auto GeneralClientCapabilities::markdown() const
    -> std::optional<MarkdownClientCapabilities> {
  if (!repr_->contains("markdown")) return std::nullopt;

  auto& value = (*repr_)["markdown"];

  return MarkdownClientCapabilities(value);
}

auto GeneralClientCapabilities::positionEncodings() const
    -> std::optional<Vector<PositionEncodingKind>> {
  if (!repr_->contains("positionEncodings")) return std::nullopt;

  auto& value = (*repr_)["positionEncodings"];

  assert(value.is_array());
  return Vector<PositionEncodingKind>(value);
}

auto GeneralClientCapabilities::staleRequestSupport(
    std::optional<StaleRequestSupportOptions> staleRequestSupport)
    -> GeneralClientCapabilities& {
  if (!staleRequestSupport.has_value()) {
    repr_->erase("staleRequestSupport");
    return *this;
  }
  repr_->emplace("staleRequestSupport", staleRequestSupport.value());
  return *this;
}

auto GeneralClientCapabilities::regularExpressions(
    std::optional<RegularExpressionsClientCapabilities> regularExpressions)
    -> GeneralClientCapabilities& {
  if (!regularExpressions.has_value()) {
    repr_->erase("regularExpressions");
    return *this;
  }
  repr_->emplace("regularExpressions", regularExpressions.value());
  return *this;
}

auto GeneralClientCapabilities::markdown(
    std::optional<MarkdownClientCapabilities> markdown)
    -> GeneralClientCapabilities& {
  if (!markdown.has_value()) {
    repr_->erase("markdown");
    return *this;
  }
  repr_->emplace("markdown", markdown.value());
  return *this;
}

auto GeneralClientCapabilities::positionEncodings(
    std::optional<Vector<PositionEncodingKind>> positionEncodings)
    -> GeneralClientCapabilities& {
  if (!positionEncodings.has_value()) {
    repr_->erase("positionEncodings");
    return *this;
  }
  lsp_runtime_error(
      "GeneralClientCapabilities::positionEncodings: not implement yet");
  return *this;
}

WorkspaceFoldersServerCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceFoldersServerCapabilities::supported() const
    -> std::optional<bool> {
  if (!repr_->contains("supported")) return std::nullopt;

  auto& value = (*repr_)["supported"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceFoldersServerCapabilities::changeNotifications() const
    -> std::optional<std::variant<std::monostate, std::string, bool>> {
  if (!repr_->contains("changeNotifications")) return std::nullopt;

  auto& value = (*repr_)["changeNotifications"];

  std::variant<std::monostate, std::string, bool> result;

  details::try_emplace(result, value);

  return result;
}

auto WorkspaceFoldersServerCapabilities::supported(
    std::optional<bool> supported) -> WorkspaceFoldersServerCapabilities& {
  if (!supported.has_value()) {
    repr_->erase("supported");
    return *this;
  }
  repr_->emplace("supported", std::move(supported.value()));
  return *this;
}

auto WorkspaceFoldersServerCapabilities::changeNotifications(
    std::optional<std::variant<std::monostate, std::string, bool>>
        changeNotifications) -> WorkspaceFoldersServerCapabilities& {
  if (!changeNotifications.has_value()) {
    repr_->erase("changeNotifications");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(std::string changeNotifications) {
      repr_->emplace("changeNotifications", std::move(changeNotifications));
    }

    void operator()(bool changeNotifications) {
      repr_->emplace("changeNotifications", std::move(changeNotifications));
    }
  } v{repr_};

  std::visit(v, changeNotifications.value());

  return *this;
}

FileOperationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto FileOperationOptions::didCreate() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_->contains("didCreate")) return std::nullopt;

  auto& value = (*repr_)["didCreate"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::willCreate() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_->contains("willCreate")) return std::nullopt;

  auto& value = (*repr_)["willCreate"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::didRename() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_->contains("didRename")) return std::nullopt;

  auto& value = (*repr_)["didRename"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::willRename() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_->contains("willRename")) return std::nullopt;

  auto& value = (*repr_)["willRename"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::didDelete() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_->contains("didDelete")) return std::nullopt;

  auto& value = (*repr_)["didDelete"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::willDelete() const
    -> std::optional<FileOperationRegistrationOptions> {
  if (!repr_->contains("willDelete")) return std::nullopt;

  auto& value = (*repr_)["willDelete"];

  return FileOperationRegistrationOptions(value);
}

auto FileOperationOptions::didCreate(
    std::optional<FileOperationRegistrationOptions> didCreate)
    -> FileOperationOptions& {
  if (!didCreate.has_value()) {
    repr_->erase("didCreate");
    return *this;
  }
  repr_->emplace("didCreate", didCreate.value());
  return *this;
}

auto FileOperationOptions::willCreate(
    std::optional<FileOperationRegistrationOptions> willCreate)
    -> FileOperationOptions& {
  if (!willCreate.has_value()) {
    repr_->erase("willCreate");
    return *this;
  }
  repr_->emplace("willCreate", willCreate.value());
  return *this;
}

auto FileOperationOptions::didRename(
    std::optional<FileOperationRegistrationOptions> didRename)
    -> FileOperationOptions& {
  if (!didRename.has_value()) {
    repr_->erase("didRename");
    return *this;
  }
  repr_->emplace("didRename", didRename.value());
  return *this;
}

auto FileOperationOptions::willRename(
    std::optional<FileOperationRegistrationOptions> willRename)
    -> FileOperationOptions& {
  if (!willRename.has_value()) {
    repr_->erase("willRename");
    return *this;
  }
  repr_->emplace("willRename", willRename.value());
  return *this;
}

auto FileOperationOptions::didDelete(
    std::optional<FileOperationRegistrationOptions> didDelete)
    -> FileOperationOptions& {
  if (!didDelete.has_value()) {
    repr_->erase("didDelete");
    return *this;
  }
  repr_->emplace("didDelete", didDelete.value());
  return *this;
}

auto FileOperationOptions::willDelete(
    std::optional<FileOperationRegistrationOptions> willDelete)
    -> FileOperationOptions& {
  if (!willDelete.has_value()) {
    repr_->erase("willDelete");
    return *this;
  }
  repr_->emplace("willDelete", willDelete.value());
  return *this;
}

RelativePattern::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("baseUri")) return false;
  if (!repr_->contains("pattern")) return false;
  return true;
}

auto RelativePattern::baseUri() const
    -> std::variant<std::monostate, WorkspaceFolder, std::string> {
  auto& value = (*repr_)["baseUri"];

  std::variant<std::monostate, WorkspaceFolder, std::string> result;

  details::try_emplace(result, value);

  return result;
}

auto RelativePattern::pattern() const -> Pattern {
  auto& value = (*repr_)["pattern"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RelativePattern::baseUri(
    std::variant<std::monostate, WorkspaceFolder, std::string> baseUri)
    -> RelativePattern& {
  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(WorkspaceFolder baseUri) {
      repr_->emplace("baseUri", baseUri);
    }

    void operator()(std::string baseUri) {
      repr_->emplace("baseUri", std::move(baseUri));
    }
  } v{repr_};

  std::visit(v, baseUri);

  return *this;
}

auto RelativePattern::pattern(Pattern pattern) -> RelativePattern& {
  lsp_runtime_error("RelativePattern::pattern: not implement yet");
  return *this;
}

TextDocumentFilterLanguage::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("language")) return false;
  return true;
}

auto TextDocumentFilterLanguage::language() const -> std::string {
  auto& value = (*repr_)["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterLanguage::scheme() const -> std::optional<std::string> {
  if (!repr_->contains("scheme")) return std::nullopt;

  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterLanguage::pattern() const -> std::optional<GlobPattern> {
  if (!repr_->contains("pattern")) return std::nullopt;

  auto& value = (*repr_)["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentFilterLanguage::language(std::string language)
    -> TextDocumentFilterLanguage& {
  repr_->emplace("language", std::move(language));
  return *this;
}

auto TextDocumentFilterLanguage::scheme(std::optional<std::string> scheme)
    -> TextDocumentFilterLanguage& {
  if (!scheme.has_value()) {
    repr_->erase("scheme");
    return *this;
  }
  repr_->emplace("scheme", std::move(scheme.value()));
  return *this;
}

auto TextDocumentFilterLanguage::pattern(std::optional<GlobPattern> pattern)
    -> TextDocumentFilterLanguage& {
  if (!pattern.has_value()) {
    repr_->erase("pattern");
    return *this;
  }
  lsp_runtime_error("TextDocumentFilterLanguage::pattern: not implement yet");
  return *this;
}

TextDocumentFilterScheme::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("scheme")) return false;
  return true;
}

auto TextDocumentFilterScheme::language() const -> std::optional<std::string> {
  if (!repr_->contains("language")) return std::nullopt;

  auto& value = (*repr_)["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterScheme::scheme() const -> std::string {
  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterScheme::pattern() const -> std::optional<GlobPattern> {
  if (!repr_->contains("pattern")) return std::nullopt;

  auto& value = (*repr_)["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentFilterScheme::language(std::optional<std::string> language)
    -> TextDocumentFilterScheme& {
  if (!language.has_value()) {
    repr_->erase("language");
    return *this;
  }
  repr_->emplace("language", std::move(language.value()));
  return *this;
}

auto TextDocumentFilterScheme::scheme(std::string scheme)
    -> TextDocumentFilterScheme& {
  repr_->emplace("scheme", std::move(scheme));
  return *this;
}

auto TextDocumentFilterScheme::pattern(std::optional<GlobPattern> pattern)
    -> TextDocumentFilterScheme& {
  if (!pattern.has_value()) {
    repr_->erase("pattern");
    return *this;
  }
  lsp_runtime_error("TextDocumentFilterScheme::pattern: not implement yet");
  return *this;
}

TextDocumentFilterPattern::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("pattern")) return false;
  return true;
}

auto TextDocumentFilterPattern::language() const -> std::optional<std::string> {
  if (!repr_->contains("language")) return std::nullopt;

  auto& value = (*repr_)["language"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterPattern::scheme() const -> std::optional<std::string> {
  if (!repr_->contains("scheme")) return std::nullopt;

  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto TextDocumentFilterPattern::pattern() const -> GlobPattern {
  auto& value = (*repr_)["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto TextDocumentFilterPattern::language(std::optional<std::string> language)
    -> TextDocumentFilterPattern& {
  if (!language.has_value()) {
    repr_->erase("language");
    return *this;
  }
  repr_->emplace("language", std::move(language.value()));
  return *this;
}

auto TextDocumentFilterPattern::scheme(std::optional<std::string> scheme)
    -> TextDocumentFilterPattern& {
  if (!scheme.has_value()) {
    repr_->erase("scheme");
    return *this;
  }
  repr_->emplace("scheme", std::move(scheme.value()));
  return *this;
}

auto TextDocumentFilterPattern::pattern(GlobPattern pattern)
    -> TextDocumentFilterPattern& {
  lsp_runtime_error("TextDocumentFilterPattern::pattern: not implement yet");
  return *this;
}

NotebookDocumentFilterNotebookType::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("notebookType")) return false;
  return true;
}

auto NotebookDocumentFilterNotebookType::notebookType() const -> std::string {
  auto& value = (*repr_)["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterNotebookType::scheme() const
    -> std::optional<std::string> {
  if (!repr_->contains("scheme")) return std::nullopt;

  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterNotebookType::pattern() const
    -> std::optional<GlobPattern> {
  if (!repr_->contains("pattern")) return std::nullopt;

  auto& value = (*repr_)["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterNotebookType::notebookType(std::string notebookType)
    -> NotebookDocumentFilterNotebookType& {
  repr_->emplace("notebookType", std::move(notebookType));
  return *this;
}

auto NotebookDocumentFilterNotebookType::scheme(
    std::optional<std::string> scheme) -> NotebookDocumentFilterNotebookType& {
  if (!scheme.has_value()) {
    repr_->erase("scheme");
    return *this;
  }
  repr_->emplace("scheme", std::move(scheme.value()));
  return *this;
}

auto NotebookDocumentFilterNotebookType::pattern(
    std::optional<GlobPattern> pattern) -> NotebookDocumentFilterNotebookType& {
  if (!pattern.has_value()) {
    repr_->erase("pattern");
    return *this;
  }
  lsp_runtime_error(
      "NotebookDocumentFilterNotebookType::pattern: not implement yet");
  return *this;
}

NotebookDocumentFilterScheme::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("scheme")) return false;
  return true;
}

auto NotebookDocumentFilterScheme::notebookType() const
    -> std::optional<std::string> {
  if (!repr_->contains("notebookType")) return std::nullopt;

  auto& value = (*repr_)["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterScheme::scheme() const -> std::string {
  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterScheme::pattern() const
    -> std::optional<GlobPattern> {
  if (!repr_->contains("pattern")) return std::nullopt;

  auto& value = (*repr_)["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterScheme::notebookType(
    std::optional<std::string> notebookType) -> NotebookDocumentFilterScheme& {
  if (!notebookType.has_value()) {
    repr_->erase("notebookType");
    return *this;
  }
  repr_->emplace("notebookType", std::move(notebookType.value()));
  return *this;
}

auto NotebookDocumentFilterScheme::scheme(std::string scheme)
    -> NotebookDocumentFilterScheme& {
  repr_->emplace("scheme", std::move(scheme));
  return *this;
}

auto NotebookDocumentFilterScheme::pattern(std::optional<GlobPattern> pattern)
    -> NotebookDocumentFilterScheme& {
  if (!pattern.has_value()) {
    repr_->erase("pattern");
    return *this;
  }
  lsp_runtime_error("NotebookDocumentFilterScheme::pattern: not implement yet");
  return *this;
}

NotebookDocumentFilterPattern::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("pattern")) return false;
  return true;
}

auto NotebookDocumentFilterPattern::notebookType() const
    -> std::optional<std::string> {
  if (!repr_->contains("notebookType")) return std::nullopt;

  auto& value = (*repr_)["notebookType"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterPattern::scheme() const
    -> std::optional<std::string> {
  if (!repr_->contains("scheme")) return std::nullopt;

  auto& value = (*repr_)["scheme"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto NotebookDocumentFilterPattern::pattern() const -> GlobPattern {
  auto& value = (*repr_)["pattern"];

  GlobPattern result;

  details::try_emplace(result, value);

  return result;
}

auto NotebookDocumentFilterPattern::notebookType(
    std::optional<std::string> notebookType) -> NotebookDocumentFilterPattern& {
  if (!notebookType.has_value()) {
    repr_->erase("notebookType");
    return *this;
  }
  repr_->emplace("notebookType", std::move(notebookType.value()));
  return *this;
}

auto NotebookDocumentFilterPattern::scheme(std::optional<std::string> scheme)
    -> NotebookDocumentFilterPattern& {
  if (!scheme.has_value()) {
    repr_->erase("scheme");
    return *this;
  }
  repr_->emplace("scheme", std::move(scheme.value()));
  return *this;
}

auto NotebookDocumentFilterPattern::pattern(GlobPattern pattern)
    -> NotebookDocumentFilterPattern& {
  lsp_runtime_error(
      "NotebookDocumentFilterPattern::pattern: not implement yet");
  return *this;
}

NotebookCellArrayChange::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("start")) return false;
  if (!repr_->contains("deleteCount")) return false;
  return true;
}

auto NotebookCellArrayChange::start() const -> long {
  auto& value = (*repr_)["start"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto NotebookCellArrayChange::deleteCount() const -> long {
  auto& value = (*repr_)["deleteCount"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto NotebookCellArrayChange::cells() const
    -> std::optional<Vector<NotebookCell>> {
  if (!repr_->contains("cells")) return std::nullopt;

  auto& value = (*repr_)["cells"];

  assert(value.is_array());
  return Vector<NotebookCell>(value);
}

auto NotebookCellArrayChange::start(long start) -> NotebookCellArrayChange& {
  repr_->emplace("start", std::move(start));
  return *this;
}

auto NotebookCellArrayChange::deleteCount(long deleteCount)
    -> NotebookCellArrayChange& {
  repr_->emplace("deleteCount", std::move(deleteCount));
  return *this;
}

auto NotebookCellArrayChange::cells(std::optional<Vector<NotebookCell>> cells)
    -> NotebookCellArrayChange& {
  if (!cells.has_value()) {
    repr_->erase("cells");
    return *this;
  }
  lsp_runtime_error("NotebookCellArrayChange::cells: not implement yet");
  return *this;
}

WorkspaceEditClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceEditClientCapabilities::documentChanges() const
    -> std::optional<bool> {
  if (!repr_->contains("documentChanges")) return std::nullopt;

  auto& value = (*repr_)["documentChanges"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::resourceOperations() const
    -> std::optional<Vector<ResourceOperationKind>> {
  if (!repr_->contains("resourceOperations")) return std::nullopt;

  auto& value = (*repr_)["resourceOperations"];

  assert(value.is_array());
  return Vector<ResourceOperationKind>(value);
}

auto WorkspaceEditClientCapabilities::failureHandling() const
    -> std::optional<FailureHandlingKind> {
  if (!repr_->contains("failureHandling")) return std::nullopt;

  auto& value = (*repr_)["failureHandling"];

  lsp_runtime_error(
      "WorkspaceEditClientCapabilities::failureHandling: not implement yet");
}

auto WorkspaceEditClientCapabilities::normalizesLineEndings() const
    -> std::optional<bool> {
  if (!repr_->contains("normalizesLineEndings")) return std::nullopt;

  auto& value = (*repr_)["normalizesLineEndings"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::changeAnnotationSupport() const
    -> std::optional<ChangeAnnotationsSupportOptions> {
  if (!repr_->contains("changeAnnotationSupport")) return std::nullopt;

  auto& value = (*repr_)["changeAnnotationSupport"];

  return ChangeAnnotationsSupportOptions(value);
}

auto WorkspaceEditClientCapabilities::metadataSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("metadataSupport")) return std::nullopt;

  auto& value = (*repr_)["metadataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::snippetEditSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("snippetEditSupport")) return std::nullopt;

  auto& value = (*repr_)["snippetEditSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceEditClientCapabilities::documentChanges(
    std::optional<bool> documentChanges) -> WorkspaceEditClientCapabilities& {
  if (!documentChanges.has_value()) {
    repr_->erase("documentChanges");
    return *this;
  }
  repr_->emplace("documentChanges", std::move(documentChanges.value()));
  return *this;
}

auto WorkspaceEditClientCapabilities::resourceOperations(
    std::optional<Vector<ResourceOperationKind>> resourceOperations)
    -> WorkspaceEditClientCapabilities& {
  if (!resourceOperations.has_value()) {
    repr_->erase("resourceOperations");
    return *this;
  }
  lsp_runtime_error(
      "WorkspaceEditClientCapabilities::resourceOperations: not implement yet");
  return *this;
}

auto WorkspaceEditClientCapabilities::failureHandling(
    std::optional<FailureHandlingKind> failureHandling)
    -> WorkspaceEditClientCapabilities& {
  if (!failureHandling.has_value()) {
    repr_->erase("failureHandling");
    return *this;
  }
  lsp_runtime_error(
      "WorkspaceEditClientCapabilities::failureHandling: not implement yet");
  return *this;
}

auto WorkspaceEditClientCapabilities::normalizesLineEndings(
    std::optional<bool> normalizesLineEndings)
    -> WorkspaceEditClientCapabilities& {
  if (!normalizesLineEndings.has_value()) {
    repr_->erase("normalizesLineEndings");
    return *this;
  }
  repr_->emplace("normalizesLineEndings",
                 std::move(normalizesLineEndings.value()));
  return *this;
}

auto WorkspaceEditClientCapabilities::changeAnnotationSupport(
    std::optional<ChangeAnnotationsSupportOptions> changeAnnotationSupport)
    -> WorkspaceEditClientCapabilities& {
  if (!changeAnnotationSupport.has_value()) {
    repr_->erase("changeAnnotationSupport");
    return *this;
  }
  repr_->emplace("changeAnnotationSupport", changeAnnotationSupport.value());
  return *this;
}

auto WorkspaceEditClientCapabilities::metadataSupport(
    std::optional<bool> metadataSupport) -> WorkspaceEditClientCapabilities& {
  if (!metadataSupport.has_value()) {
    repr_->erase("metadataSupport");
    return *this;
  }
  repr_->emplace("metadataSupport", std::move(metadataSupport.value()));
  return *this;
}

auto WorkspaceEditClientCapabilities::snippetEditSupport(
    std::optional<bool> snippetEditSupport)
    -> WorkspaceEditClientCapabilities& {
  if (!snippetEditSupport.has_value()) {
    repr_->erase("snippetEditSupport");
    return *this;
  }
  repr_->emplace("snippetEditSupport", std::move(snippetEditSupport.value()));
  return *this;
}

DidChangeConfigurationClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DidChangeConfigurationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DidChangeConfigurationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DidChangeConfigurationClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

DidChangeWatchedFilesClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DidChangeWatchedFilesClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DidChangeWatchedFilesClientCapabilities::relativePatternSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("relativePatternSupport")) return std::nullopt;

  auto& value = (*repr_)["relativePatternSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DidChangeWatchedFilesClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DidChangeWatchedFilesClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DidChangeWatchedFilesClientCapabilities::relativePatternSupport(
    std::optional<bool> relativePatternSupport)
    -> DidChangeWatchedFilesClientCapabilities& {
  if (!relativePatternSupport.has_value()) {
    repr_->erase("relativePatternSupport");
    return *this;
  }
  repr_->emplace("relativePatternSupport",
                 std::move(relativePatternSupport.value()));
  return *this;
}

WorkspaceSymbolClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto WorkspaceSymbolClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto WorkspaceSymbolClientCapabilities::symbolKind() const
    -> std::optional<ClientSymbolKindOptions> {
  if (!repr_->contains("symbolKind")) return std::nullopt;

  auto& value = (*repr_)["symbolKind"];

  return ClientSymbolKindOptions(value);
}

auto WorkspaceSymbolClientCapabilities::tagSupport() const
    -> std::optional<ClientSymbolTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return ClientSymbolTagOptions(value);
}

auto WorkspaceSymbolClientCapabilities::resolveSupport() const
    -> std::optional<ClientSymbolResolveOptions> {
  if (!repr_->contains("resolveSupport")) return std::nullopt;

  auto& value = (*repr_)["resolveSupport"];

  return ClientSymbolResolveOptions(value);
}

auto WorkspaceSymbolClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> WorkspaceSymbolClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto WorkspaceSymbolClientCapabilities::symbolKind(
    std::optional<ClientSymbolKindOptions> symbolKind)
    -> WorkspaceSymbolClientCapabilities& {
  if (!symbolKind.has_value()) {
    repr_->erase("symbolKind");
    return *this;
  }
  repr_->emplace("symbolKind", symbolKind.value());
  return *this;
}

auto WorkspaceSymbolClientCapabilities::tagSupport(
    std::optional<ClientSymbolTagOptions> tagSupport)
    -> WorkspaceSymbolClientCapabilities& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

auto WorkspaceSymbolClientCapabilities::resolveSupport(
    std::optional<ClientSymbolResolveOptions> resolveSupport)
    -> WorkspaceSymbolClientCapabilities& {
  if (!resolveSupport.has_value()) {
    repr_->erase("resolveSupport");
    return *this;
  }
  repr_->emplace("resolveSupport", resolveSupport.value());
  return *this;
}

ExecuteCommandClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ExecuteCommandClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ExecuteCommandClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> ExecuteCommandClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

SemanticTokensWorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SemanticTokensWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("refreshSupport")) return std::nullopt;

  auto& value = (*repr_)["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> SemanticTokensWorkspaceClientCapabilities& {
  if (!refreshSupport.has_value()) {
    repr_->erase("refreshSupport");
    return *this;
  }
  repr_->emplace("refreshSupport", std::move(refreshSupport.value()));
  return *this;
}

CodeLensWorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CodeLensWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("refreshSupport")) return std::nullopt;

  auto& value = (*repr_)["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> CodeLensWorkspaceClientCapabilities& {
  if (!refreshSupport.has_value()) {
    repr_->erase("refreshSupport");
    return *this;
  }
  repr_->emplace("refreshSupport", std::move(refreshSupport.value()));
  return *this;
}

FileOperationClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto FileOperationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::didCreate() const -> std::optional<bool> {
  if (!repr_->contains("didCreate")) return std::nullopt;

  auto& value = (*repr_)["didCreate"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::willCreate() const
    -> std::optional<bool> {
  if (!repr_->contains("willCreate")) return std::nullopt;

  auto& value = (*repr_)["willCreate"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::didRename() const -> std::optional<bool> {
  if (!repr_->contains("didRename")) return std::nullopt;

  auto& value = (*repr_)["didRename"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::willRename() const
    -> std::optional<bool> {
  if (!repr_->contains("willRename")) return std::nullopt;

  auto& value = (*repr_)["willRename"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::didDelete() const -> std::optional<bool> {
  if (!repr_->contains("didDelete")) return std::nullopt;

  auto& value = (*repr_)["didDelete"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::willDelete() const
    -> std::optional<bool> {
  if (!repr_->contains("willDelete")) return std::nullopt;

  auto& value = (*repr_)["willDelete"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FileOperationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> FileOperationClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto FileOperationClientCapabilities::didCreate(std::optional<bool> didCreate)
    -> FileOperationClientCapabilities& {
  if (!didCreate.has_value()) {
    repr_->erase("didCreate");
    return *this;
  }
  repr_->emplace("didCreate", std::move(didCreate.value()));
  return *this;
}

auto FileOperationClientCapabilities::willCreate(std::optional<bool> willCreate)
    -> FileOperationClientCapabilities& {
  if (!willCreate.has_value()) {
    repr_->erase("willCreate");
    return *this;
  }
  repr_->emplace("willCreate", std::move(willCreate.value()));
  return *this;
}

auto FileOperationClientCapabilities::didRename(std::optional<bool> didRename)
    -> FileOperationClientCapabilities& {
  if (!didRename.has_value()) {
    repr_->erase("didRename");
    return *this;
  }
  repr_->emplace("didRename", std::move(didRename.value()));
  return *this;
}

auto FileOperationClientCapabilities::willRename(std::optional<bool> willRename)
    -> FileOperationClientCapabilities& {
  if (!willRename.has_value()) {
    repr_->erase("willRename");
    return *this;
  }
  repr_->emplace("willRename", std::move(willRename.value()));
  return *this;
}

auto FileOperationClientCapabilities::didDelete(std::optional<bool> didDelete)
    -> FileOperationClientCapabilities& {
  if (!didDelete.has_value()) {
    repr_->erase("didDelete");
    return *this;
  }
  repr_->emplace("didDelete", std::move(didDelete.value()));
  return *this;
}

auto FileOperationClientCapabilities::willDelete(std::optional<bool> willDelete)
    -> FileOperationClientCapabilities& {
  if (!willDelete.has_value()) {
    repr_->erase("willDelete");
    return *this;
  }
  repr_->emplace("willDelete", std::move(willDelete.value()));
  return *this;
}

InlineValueWorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlineValueWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("refreshSupport")) return std::nullopt;

  auto& value = (*repr_)["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> InlineValueWorkspaceClientCapabilities& {
  if (!refreshSupport.has_value()) {
    repr_->erase("refreshSupport");
    return *this;
  }
  repr_->emplace("refreshSupport", std::move(refreshSupport.value()));
  return *this;
}

InlayHintWorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlayHintWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("refreshSupport")) return std::nullopt;

  auto& value = (*repr_)["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> InlayHintWorkspaceClientCapabilities& {
  if (!refreshSupport.has_value()) {
    repr_->erase("refreshSupport");
    return *this;
  }
  repr_->emplace("refreshSupport", std::move(refreshSupport.value()));
  return *this;
}

DiagnosticWorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DiagnosticWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("refreshSupport")) return std::nullopt;

  auto& value = (*repr_)["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> DiagnosticWorkspaceClientCapabilities& {
  if (!refreshSupport.has_value()) {
    repr_->erase("refreshSupport");
    return *this;
  }
  repr_->emplace("refreshSupport", std::move(refreshSupport.value()));
  return *this;
}

FoldingRangeWorkspaceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto FoldingRangeWorkspaceClientCapabilities::refreshSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("refreshSupport")) return std::nullopt;

  auto& value = (*repr_)["refreshSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeWorkspaceClientCapabilities::refreshSupport(
    std::optional<bool> refreshSupport)
    -> FoldingRangeWorkspaceClientCapabilities& {
  if (!refreshSupport.has_value()) {
    repr_->erase("refreshSupport");
    return *this;
  }
  repr_->emplace("refreshSupport", std::move(refreshSupport.value()));
  return *this;
}

TextDocumentContentClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TextDocumentContentClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentContentClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TextDocumentContentClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

TextDocumentSyncClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TextDocumentSyncClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::willSave() const
    -> std::optional<bool> {
  if (!repr_->contains("willSave")) return std::nullopt;

  auto& value = (*repr_)["willSave"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::willSaveWaitUntil() const
    -> std::optional<bool> {
  if (!repr_->contains("willSaveWaitUntil")) return std::nullopt;

  auto& value = (*repr_)["willSaveWaitUntil"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::didSave() const
    -> std::optional<bool> {
  if (!repr_->contains("didSave")) return std::nullopt;

  auto& value = (*repr_)["didSave"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentSyncClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TextDocumentSyncClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto TextDocumentSyncClientCapabilities::willSave(std::optional<bool> willSave)
    -> TextDocumentSyncClientCapabilities& {
  if (!willSave.has_value()) {
    repr_->erase("willSave");
    return *this;
  }
  repr_->emplace("willSave", std::move(willSave.value()));
  return *this;
}

auto TextDocumentSyncClientCapabilities::willSaveWaitUntil(
    std::optional<bool> willSaveWaitUntil)
    -> TextDocumentSyncClientCapabilities& {
  if (!willSaveWaitUntil.has_value()) {
    repr_->erase("willSaveWaitUntil");
    return *this;
  }
  repr_->emplace("willSaveWaitUntil", std::move(willSaveWaitUntil.value()));
  return *this;
}

auto TextDocumentSyncClientCapabilities::didSave(std::optional<bool> didSave)
    -> TextDocumentSyncClientCapabilities& {
  if (!didSave.has_value()) {
    repr_->erase("didSave");
    return *this;
  }
  repr_->emplace("didSave", std::move(didSave.value()));
  return *this;
}

TextDocumentFilterClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TextDocumentFilterClientCapabilities::relativePatternSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("relativePatternSupport")) return std::nullopt;

  auto& value = (*repr_)["relativePatternSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TextDocumentFilterClientCapabilities::relativePatternSupport(
    std::optional<bool> relativePatternSupport)
    -> TextDocumentFilterClientCapabilities& {
  if (!relativePatternSupport.has_value()) {
    repr_->erase("relativePatternSupport");
    return *this;
  }
  repr_->emplace("relativePatternSupport",
                 std::move(relativePatternSupport.value()));
  return *this;
}

CompletionClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CompletionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionClientCapabilities::completionItem() const
    -> std::optional<ClientCompletionItemOptions> {
  if (!repr_->contains("completionItem")) return std::nullopt;

  auto& value = (*repr_)["completionItem"];

  return ClientCompletionItemOptions(value);
}

auto CompletionClientCapabilities::completionItemKind() const
    -> std::optional<ClientCompletionItemOptionsKind> {
  if (!repr_->contains("completionItemKind")) return std::nullopt;

  auto& value = (*repr_)["completionItemKind"];

  return ClientCompletionItemOptionsKind(value);
}

auto CompletionClientCapabilities::insertTextMode() const
    -> std::optional<InsertTextMode> {
  if (!repr_->contains("insertTextMode")) return std::nullopt;

  auto& value = (*repr_)["insertTextMode"];

  return InsertTextMode(value);
}

auto CompletionClientCapabilities::contextSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("contextSupport")) return std::nullopt;

  auto& value = (*repr_)["contextSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionClientCapabilities::completionList() const
    -> std::optional<CompletionListCapabilities> {
  if (!repr_->contains("completionList")) return std::nullopt;

  auto& value = (*repr_)["completionList"];

  return CompletionListCapabilities(value);
}

auto CompletionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> CompletionClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto CompletionClientCapabilities::completionItem(
    std::optional<ClientCompletionItemOptions> completionItem)
    -> CompletionClientCapabilities& {
  if (!completionItem.has_value()) {
    repr_->erase("completionItem");
    return *this;
  }
  repr_->emplace("completionItem", completionItem.value());
  return *this;
}

auto CompletionClientCapabilities::completionItemKind(
    std::optional<ClientCompletionItemOptionsKind> completionItemKind)
    -> CompletionClientCapabilities& {
  if (!completionItemKind.has_value()) {
    repr_->erase("completionItemKind");
    return *this;
  }
  repr_->emplace("completionItemKind", completionItemKind.value());
  return *this;
}

auto CompletionClientCapabilities::insertTextMode(
    std::optional<InsertTextMode> insertTextMode)
    -> CompletionClientCapabilities& {
  if (!insertTextMode.has_value()) {
    repr_->erase("insertTextMode");
    return *this;
  }
  repr_->emplace("insertTextMode", static_cast<long>(insertTextMode.value()));
  return *this;
}

auto CompletionClientCapabilities::contextSupport(
    std::optional<bool> contextSupport) -> CompletionClientCapabilities& {
  if (!contextSupport.has_value()) {
    repr_->erase("contextSupport");
    return *this;
  }
  repr_->emplace("contextSupport", std::move(contextSupport.value()));
  return *this;
}

auto CompletionClientCapabilities::completionList(
    std::optional<CompletionListCapabilities> completionList)
    -> CompletionClientCapabilities& {
  if (!completionList.has_value()) {
    repr_->erase("completionList");
    return *this;
  }
  repr_->emplace("completionList", completionList.value());
  return *this;
}

HoverClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto HoverClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto HoverClientCapabilities::contentFormat() const
    -> std::optional<Vector<MarkupKind>> {
  if (!repr_->contains("contentFormat")) return std::nullopt;

  auto& value = (*repr_)["contentFormat"];

  assert(value.is_array());
  return Vector<MarkupKind>(value);
}

auto HoverClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> HoverClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto HoverClientCapabilities::contentFormat(
    std::optional<Vector<MarkupKind>> contentFormat)
    -> HoverClientCapabilities& {
  if (!contentFormat.has_value()) {
    repr_->erase("contentFormat");
    return *this;
  }
  lsp_runtime_error(
      "HoverClientCapabilities::contentFormat: not implement yet");
  return *this;
}

SignatureHelpClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SignatureHelpClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpClientCapabilities::signatureInformation() const
    -> std::optional<ClientSignatureInformationOptions> {
  if (!repr_->contains("signatureInformation")) return std::nullopt;

  auto& value = (*repr_)["signatureInformation"];

  return ClientSignatureInformationOptions(value);
}

auto SignatureHelpClientCapabilities::contextSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("contextSupport")) return std::nullopt;

  auto& value = (*repr_)["contextSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SignatureHelpClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> SignatureHelpClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto SignatureHelpClientCapabilities::signatureInformation(
    std::optional<ClientSignatureInformationOptions> signatureInformation)
    -> SignatureHelpClientCapabilities& {
  if (!signatureInformation.has_value()) {
    repr_->erase("signatureInformation");
    return *this;
  }
  repr_->emplace("signatureInformation", signatureInformation.value());
  return *this;
}

auto SignatureHelpClientCapabilities::contextSupport(
    std::optional<bool> contextSupport) -> SignatureHelpClientCapabilities& {
  if (!contextSupport.has_value()) {
    repr_->erase("contextSupport");
    return *this;
  }
  repr_->emplace("contextSupport", std::move(contextSupport.value()));
  return *this;
}

DeclarationClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DeclarationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationClientCapabilities::linkSupport() const -> std::optional<bool> {
  if (!repr_->contains("linkSupport")) return std::nullopt;

  auto& value = (*repr_)["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DeclarationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> DeclarationClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DeclarationClientCapabilities::linkSupport(std::optional<bool> linkSupport)
    -> DeclarationClientCapabilities& {
  if (!linkSupport.has_value()) {
    repr_->erase("linkSupport");
    return *this;
  }
  repr_->emplace("linkSupport", std::move(linkSupport.value()));
  return *this;
}

DefinitionClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DefinitionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionClientCapabilities::linkSupport() const -> std::optional<bool> {
  if (!repr_->contains("linkSupport")) return std::nullopt;

  auto& value = (*repr_)["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DefinitionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> DefinitionClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DefinitionClientCapabilities::linkSupport(std::optional<bool> linkSupport)
    -> DefinitionClientCapabilities& {
  if (!linkSupport.has_value()) {
    repr_->erase("linkSupport");
    return *this;
  }
  repr_->emplace("linkSupport", std::move(linkSupport.value()));
  return *this;
}

TypeDefinitionClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TypeDefinitionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionClientCapabilities::linkSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("linkSupport")) return std::nullopt;

  auto& value = (*repr_)["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeDefinitionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TypeDefinitionClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto TypeDefinitionClientCapabilities::linkSupport(
    std::optional<bool> linkSupport) -> TypeDefinitionClientCapabilities& {
  if (!linkSupport.has_value()) {
    repr_->erase("linkSupport");
    return *this;
  }
  repr_->emplace("linkSupport", std::move(linkSupport.value()));
  return *this;
}

ImplementationClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ImplementationClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationClientCapabilities::linkSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("linkSupport")) return std::nullopt;

  auto& value = (*repr_)["linkSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ImplementationClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> ImplementationClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto ImplementationClientCapabilities::linkSupport(
    std::optional<bool> linkSupport) -> ImplementationClientCapabilities& {
  if (!linkSupport.has_value()) {
    repr_->erase("linkSupport");
    return *this;
  }
  repr_->emplace("linkSupport", std::move(linkSupport.value()));
  return *this;
}

ReferenceClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ReferenceClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ReferenceClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> ReferenceClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

DocumentHighlightClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentHighlightClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentHighlightClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentHighlightClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

DocumentSymbolClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentSymbolClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolClientCapabilities::symbolKind() const
    -> std::optional<ClientSymbolKindOptions> {
  if (!repr_->contains("symbolKind")) return std::nullopt;

  auto& value = (*repr_)["symbolKind"];

  return ClientSymbolKindOptions(value);
}

auto DocumentSymbolClientCapabilities::hierarchicalDocumentSymbolSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("hierarchicalDocumentSymbolSupport"))
    return std::nullopt;

  auto& value = (*repr_)["hierarchicalDocumentSymbolSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolClientCapabilities::tagSupport() const
    -> std::optional<ClientSymbolTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return ClientSymbolTagOptions(value);
}

auto DocumentSymbolClientCapabilities::labelSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("labelSupport")) return std::nullopt;

  auto& value = (*repr_)["labelSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentSymbolClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentSymbolClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DocumentSymbolClientCapabilities::symbolKind(
    std::optional<ClientSymbolKindOptions> symbolKind)
    -> DocumentSymbolClientCapabilities& {
  if (!symbolKind.has_value()) {
    repr_->erase("symbolKind");
    return *this;
  }
  repr_->emplace("symbolKind", symbolKind.value());
  return *this;
}

auto DocumentSymbolClientCapabilities::hierarchicalDocumentSymbolSupport(
    std::optional<bool> hierarchicalDocumentSymbolSupport)
    -> DocumentSymbolClientCapabilities& {
  if (!hierarchicalDocumentSymbolSupport.has_value()) {
    repr_->erase("hierarchicalDocumentSymbolSupport");
    return *this;
  }
  repr_->emplace("hierarchicalDocumentSymbolSupport",
                 std::move(hierarchicalDocumentSymbolSupport.value()));
  return *this;
}

auto DocumentSymbolClientCapabilities::tagSupport(
    std::optional<ClientSymbolTagOptions> tagSupport)
    -> DocumentSymbolClientCapabilities& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

auto DocumentSymbolClientCapabilities::labelSupport(
    std::optional<bool> labelSupport) -> DocumentSymbolClientCapabilities& {
  if (!labelSupport.has_value()) {
    repr_->erase("labelSupport");
    return *this;
  }
  repr_->emplace("labelSupport", std::move(labelSupport.value()));
  return *this;
}

CodeActionClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CodeActionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::codeActionLiteralSupport() const
    -> std::optional<ClientCodeActionLiteralOptions> {
  if (!repr_->contains("codeActionLiteralSupport")) return std::nullopt;

  auto& value = (*repr_)["codeActionLiteralSupport"];

  return ClientCodeActionLiteralOptions(value);
}

auto CodeActionClientCapabilities::isPreferredSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("isPreferredSupport")) return std::nullopt;

  auto& value = (*repr_)["isPreferredSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::disabledSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("disabledSupport")) return std::nullopt;

  auto& value = (*repr_)["disabledSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::dataSupport() const -> std::optional<bool> {
  if (!repr_->contains("dataSupport")) return std::nullopt;

  auto& value = (*repr_)["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::resolveSupport() const
    -> std::optional<ClientCodeActionResolveOptions> {
  if (!repr_->contains("resolveSupport")) return std::nullopt;

  auto& value = (*repr_)["resolveSupport"];

  return ClientCodeActionResolveOptions(value);
}

auto CodeActionClientCapabilities::honorsChangeAnnotations() const
    -> std::optional<bool> {
  if (!repr_->contains("honorsChangeAnnotations")) return std::nullopt;

  auto& value = (*repr_)["honorsChangeAnnotations"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::documentationSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("documentationSupport")) return std::nullopt;

  auto& value = (*repr_)["documentationSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeActionClientCapabilities::tagSupport() const
    -> std::optional<CodeActionTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return CodeActionTagOptions(value);
}

auto CodeActionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> CodeActionClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto CodeActionClientCapabilities::codeActionLiteralSupport(
    std::optional<ClientCodeActionLiteralOptions> codeActionLiteralSupport)
    -> CodeActionClientCapabilities& {
  if (!codeActionLiteralSupport.has_value()) {
    repr_->erase("codeActionLiteralSupport");
    return *this;
  }
  repr_->emplace("codeActionLiteralSupport", codeActionLiteralSupport.value());
  return *this;
}

auto CodeActionClientCapabilities::isPreferredSupport(
    std::optional<bool> isPreferredSupport) -> CodeActionClientCapabilities& {
  if (!isPreferredSupport.has_value()) {
    repr_->erase("isPreferredSupport");
    return *this;
  }
  repr_->emplace("isPreferredSupport", std::move(isPreferredSupport.value()));
  return *this;
}

auto CodeActionClientCapabilities::disabledSupport(
    std::optional<bool> disabledSupport) -> CodeActionClientCapabilities& {
  if (!disabledSupport.has_value()) {
    repr_->erase("disabledSupport");
    return *this;
  }
  repr_->emplace("disabledSupport", std::move(disabledSupport.value()));
  return *this;
}

auto CodeActionClientCapabilities::dataSupport(std::optional<bool> dataSupport)
    -> CodeActionClientCapabilities& {
  if (!dataSupport.has_value()) {
    repr_->erase("dataSupport");
    return *this;
  }
  repr_->emplace("dataSupport", std::move(dataSupport.value()));
  return *this;
}

auto CodeActionClientCapabilities::resolveSupport(
    std::optional<ClientCodeActionResolveOptions> resolveSupport)
    -> CodeActionClientCapabilities& {
  if (!resolveSupport.has_value()) {
    repr_->erase("resolveSupport");
    return *this;
  }
  repr_->emplace("resolveSupport", resolveSupport.value());
  return *this;
}

auto CodeActionClientCapabilities::honorsChangeAnnotations(
    std::optional<bool> honorsChangeAnnotations)
    -> CodeActionClientCapabilities& {
  if (!honorsChangeAnnotations.has_value()) {
    repr_->erase("honorsChangeAnnotations");
    return *this;
  }
  repr_->emplace("honorsChangeAnnotations",
                 std::move(honorsChangeAnnotations.value()));
  return *this;
}

auto CodeActionClientCapabilities::documentationSupport(
    std::optional<bool> documentationSupport) -> CodeActionClientCapabilities& {
  if (!documentationSupport.has_value()) {
    repr_->erase("documentationSupport");
    return *this;
  }
  repr_->emplace("documentationSupport",
                 std::move(documentationSupport.value()));
  return *this;
}

auto CodeActionClientCapabilities::tagSupport(
    std::optional<CodeActionTagOptions> tagSupport)
    -> CodeActionClientCapabilities& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

CodeLensClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CodeLensClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CodeLensClientCapabilities::resolveSupport() const
    -> std::optional<ClientCodeLensResolveOptions> {
  if (!repr_->contains("resolveSupport")) return std::nullopt;

  auto& value = (*repr_)["resolveSupport"];

  return ClientCodeLensResolveOptions(value);
}

auto CodeLensClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> CodeLensClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto CodeLensClientCapabilities::resolveSupport(
    std::optional<ClientCodeLensResolveOptions> resolveSupport)
    -> CodeLensClientCapabilities& {
  if (!resolveSupport.has_value()) {
    repr_->erase("resolveSupport");
    return *this;
  }
  repr_->emplace("resolveSupport", resolveSupport.value());
  return *this;
}

DocumentLinkClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentLinkClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkClientCapabilities::tooltipSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("tooltipSupport")) return std::nullopt;

  auto& value = (*repr_)["tooltipSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentLinkClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentLinkClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DocumentLinkClientCapabilities::tooltipSupport(
    std::optional<bool> tooltipSupport) -> DocumentLinkClientCapabilities& {
  if (!tooltipSupport.has_value()) {
    repr_->erase("tooltipSupport");
    return *this;
  }
  repr_->emplace("tooltipSupport", std::move(tooltipSupport.value()));
  return *this;
}

DocumentColorClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentColorClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentColorClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentColorClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

DocumentFormattingClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentFormattingClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentFormattingClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentFormattingClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

DocumentRangeFormattingClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentRangeFormattingClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingClientCapabilities::rangesSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("rangesSupport")) return std::nullopt;

  auto& value = (*repr_)["rangesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentRangeFormattingClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentRangeFormattingClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DocumentRangeFormattingClientCapabilities::rangesSupport(
    std::optional<bool> rangesSupport)
    -> DocumentRangeFormattingClientCapabilities& {
  if (!rangesSupport.has_value()) {
    repr_->erase("rangesSupport");
    return *this;
  }
  repr_->emplace("rangesSupport", std::move(rangesSupport.value()));
  return *this;
}

DocumentOnTypeFormattingClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DocumentOnTypeFormattingClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DocumentOnTypeFormattingClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> DocumentOnTypeFormattingClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

RenameClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto RenameClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameClientCapabilities::prepareSupport() const -> std::optional<bool> {
  if (!repr_->contains("prepareSupport")) return std::nullopt;

  auto& value = (*repr_)["prepareSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameClientCapabilities::prepareSupportDefaultBehavior() const
    -> std::optional<PrepareSupportDefaultBehavior> {
  if (!repr_->contains("prepareSupportDefaultBehavior")) return std::nullopt;

  auto& value = (*repr_)["prepareSupportDefaultBehavior"];

  return PrepareSupportDefaultBehavior(value);
}

auto RenameClientCapabilities::honorsChangeAnnotations() const
    -> std::optional<bool> {
  if (!repr_->contains("honorsChangeAnnotations")) return std::nullopt;

  auto& value = (*repr_)["honorsChangeAnnotations"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto RenameClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> RenameClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto RenameClientCapabilities::prepareSupport(
    std::optional<bool> prepareSupport) -> RenameClientCapabilities& {
  if (!prepareSupport.has_value()) {
    repr_->erase("prepareSupport");
    return *this;
  }
  repr_->emplace("prepareSupport", std::move(prepareSupport.value()));
  return *this;
}

auto RenameClientCapabilities::prepareSupportDefaultBehavior(
    std::optional<PrepareSupportDefaultBehavior> prepareSupportDefaultBehavior)
    -> RenameClientCapabilities& {
  if (!prepareSupportDefaultBehavior.has_value()) {
    repr_->erase("prepareSupportDefaultBehavior");
    return *this;
  }
  repr_->emplace("prepareSupportDefaultBehavior",
                 static_cast<long>(prepareSupportDefaultBehavior.value()));
  return *this;
}

auto RenameClientCapabilities::honorsChangeAnnotations(
    std::optional<bool> honorsChangeAnnotations) -> RenameClientCapabilities& {
  if (!honorsChangeAnnotations.has_value()) {
    repr_->erase("honorsChangeAnnotations");
    return *this;
  }
  repr_->emplace("honorsChangeAnnotations",
                 std::move(honorsChangeAnnotations.value()));
  return *this;
}

FoldingRangeClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto FoldingRangeClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeClientCapabilities::rangeLimit() const -> std::optional<long> {
  if (!repr_->contains("rangeLimit")) return std::nullopt;

  auto& value = (*repr_)["rangeLimit"];

  assert(value.is_number_integer());
  return value.get<long>();
}

auto FoldingRangeClientCapabilities::lineFoldingOnly() const
    -> std::optional<bool> {
  if (!repr_->contains("lineFoldingOnly")) return std::nullopt;

  auto& value = (*repr_)["lineFoldingOnly"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto FoldingRangeClientCapabilities::foldingRangeKind() const
    -> std::optional<ClientFoldingRangeKindOptions> {
  if (!repr_->contains("foldingRangeKind")) return std::nullopt;

  auto& value = (*repr_)["foldingRangeKind"];

  return ClientFoldingRangeKindOptions(value);
}

auto FoldingRangeClientCapabilities::foldingRange() const
    -> std::optional<ClientFoldingRangeOptions> {
  if (!repr_->contains("foldingRange")) return std::nullopt;

  auto& value = (*repr_)["foldingRange"];

  return ClientFoldingRangeOptions(value);
}

auto FoldingRangeClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> FoldingRangeClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto FoldingRangeClientCapabilities::rangeLimit(std::optional<long> rangeLimit)
    -> FoldingRangeClientCapabilities& {
  if (!rangeLimit.has_value()) {
    repr_->erase("rangeLimit");
    return *this;
  }
  repr_->emplace("rangeLimit", std::move(rangeLimit.value()));
  return *this;
}

auto FoldingRangeClientCapabilities::lineFoldingOnly(
    std::optional<bool> lineFoldingOnly) -> FoldingRangeClientCapabilities& {
  if (!lineFoldingOnly.has_value()) {
    repr_->erase("lineFoldingOnly");
    return *this;
  }
  repr_->emplace("lineFoldingOnly", std::move(lineFoldingOnly.value()));
  return *this;
}

auto FoldingRangeClientCapabilities::foldingRangeKind(
    std::optional<ClientFoldingRangeKindOptions> foldingRangeKind)
    -> FoldingRangeClientCapabilities& {
  if (!foldingRangeKind.has_value()) {
    repr_->erase("foldingRangeKind");
    return *this;
  }
  repr_->emplace("foldingRangeKind", foldingRangeKind.value());
  return *this;
}

auto FoldingRangeClientCapabilities::foldingRange(
    std::optional<ClientFoldingRangeOptions> foldingRange)
    -> FoldingRangeClientCapabilities& {
  if (!foldingRange.has_value()) {
    repr_->erase("foldingRange");
    return *this;
  }
  repr_->emplace("foldingRange", foldingRange.value());
  return *this;
}

SelectionRangeClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto SelectionRangeClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SelectionRangeClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> SelectionRangeClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

PublishDiagnosticsClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto PublishDiagnosticsClientCapabilities::versionSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("versionSupport")) return std::nullopt;

  auto& value = (*repr_)["versionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::relatedInformation() const
    -> std::optional<bool> {
  if (!repr_->contains("relatedInformation")) return std::nullopt;

  auto& value = (*repr_)["relatedInformation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::tagSupport() const
    -> std::optional<ClientDiagnosticsTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return ClientDiagnosticsTagOptions(value);
}

auto PublishDiagnosticsClientCapabilities::codeDescriptionSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("codeDescriptionSupport")) return std::nullopt;

  auto& value = (*repr_)["codeDescriptionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::dataSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("dataSupport")) return std::nullopt;

  auto& value = (*repr_)["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto PublishDiagnosticsClientCapabilities::versionSupport(
    std::optional<bool> versionSupport)
    -> PublishDiagnosticsClientCapabilities& {
  if (!versionSupport.has_value()) {
    repr_->erase("versionSupport");
    return *this;
  }
  repr_->emplace("versionSupport", std::move(versionSupport.value()));
  return *this;
}

auto PublishDiagnosticsClientCapabilities::relatedInformation(
    std::optional<bool> relatedInformation)
    -> PublishDiagnosticsClientCapabilities& {
  if (!relatedInformation.has_value()) {
    repr_->erase("relatedInformation");
    return *this;
  }
  repr_->emplace("relatedInformation", std::move(relatedInformation.value()));
  return *this;
}

auto PublishDiagnosticsClientCapabilities::tagSupport(
    std::optional<ClientDiagnosticsTagOptions> tagSupport)
    -> PublishDiagnosticsClientCapabilities& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

auto PublishDiagnosticsClientCapabilities::codeDescriptionSupport(
    std::optional<bool> codeDescriptionSupport)
    -> PublishDiagnosticsClientCapabilities& {
  if (!codeDescriptionSupport.has_value()) {
    repr_->erase("codeDescriptionSupport");
    return *this;
  }
  repr_->emplace("codeDescriptionSupport",
                 std::move(codeDescriptionSupport.value()));
  return *this;
}

auto PublishDiagnosticsClientCapabilities::dataSupport(
    std::optional<bool> dataSupport) -> PublishDiagnosticsClientCapabilities& {
  if (!dataSupport.has_value()) {
    repr_->erase("dataSupport");
    return *this;
  }
  repr_->emplace("dataSupport", std::move(dataSupport.value()));
  return *this;
}

CallHierarchyClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CallHierarchyClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CallHierarchyClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> CallHierarchyClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

SemanticTokensClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("requests")) return false;
  if (!repr_->contains("tokenTypes")) return false;
  if (!repr_->contains("tokenModifiers")) return false;
  if (!repr_->contains("formats")) return false;
  return true;
}

auto SemanticTokensClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::requests() const
    -> ClientSemanticTokensRequestOptions {
  auto& value = (*repr_)["requests"];

  return ClientSemanticTokensRequestOptions(value);
}

auto SemanticTokensClientCapabilities::tokenTypes() const
    -> Vector<std::string> {
  auto& value = (*repr_)["tokenTypes"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensClientCapabilities::tokenModifiers() const
    -> Vector<std::string> {
  auto& value = (*repr_)["tokenModifiers"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto SemanticTokensClientCapabilities::formats() const -> Vector<TokenFormat> {
  auto& value = (*repr_)["formats"];

  assert(value.is_array());
  return Vector<TokenFormat>(value);
}

auto SemanticTokensClientCapabilities::overlappingTokenSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("overlappingTokenSupport")) return std::nullopt;

  auto& value = (*repr_)["overlappingTokenSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::multilineTokenSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("multilineTokenSupport")) return std::nullopt;

  auto& value = (*repr_)["multilineTokenSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::serverCancelSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("serverCancelSupport")) return std::nullopt;

  auto& value = (*repr_)["serverCancelSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::augmentsSyntaxTokens() const
    -> std::optional<bool> {
  if (!repr_->contains("augmentsSyntaxTokens")) return std::nullopt;

  auto& value = (*repr_)["augmentsSyntaxTokens"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto SemanticTokensClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> SemanticTokensClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto SemanticTokensClientCapabilities::requests(
    ClientSemanticTokensRequestOptions requests)
    -> SemanticTokensClientCapabilities& {
  repr_->emplace("requests", requests);
  return *this;
}

auto SemanticTokensClientCapabilities::tokenTypes(
    Vector<std::string> tokenTypes) -> SemanticTokensClientCapabilities& {
  lsp_runtime_error(
      "SemanticTokensClientCapabilities::tokenTypes: not implement yet");
  return *this;
}

auto SemanticTokensClientCapabilities::tokenModifiers(
    Vector<std::string> tokenModifiers) -> SemanticTokensClientCapabilities& {
  lsp_runtime_error(
      "SemanticTokensClientCapabilities::tokenModifiers: not implement yet");
  return *this;
}

auto SemanticTokensClientCapabilities::formats(Vector<TokenFormat> formats)
    -> SemanticTokensClientCapabilities& {
  lsp_runtime_error(
      "SemanticTokensClientCapabilities::formats: not implement yet");
  return *this;
}

auto SemanticTokensClientCapabilities::overlappingTokenSupport(
    std::optional<bool> overlappingTokenSupport)
    -> SemanticTokensClientCapabilities& {
  if (!overlappingTokenSupport.has_value()) {
    repr_->erase("overlappingTokenSupport");
    return *this;
  }
  repr_->emplace("overlappingTokenSupport",
                 std::move(overlappingTokenSupport.value()));
  return *this;
}

auto SemanticTokensClientCapabilities::multilineTokenSupport(
    std::optional<bool> multilineTokenSupport)
    -> SemanticTokensClientCapabilities& {
  if (!multilineTokenSupport.has_value()) {
    repr_->erase("multilineTokenSupport");
    return *this;
  }
  repr_->emplace("multilineTokenSupport",
                 std::move(multilineTokenSupport.value()));
  return *this;
}

auto SemanticTokensClientCapabilities::serverCancelSupport(
    std::optional<bool> serverCancelSupport)
    -> SemanticTokensClientCapabilities& {
  if (!serverCancelSupport.has_value()) {
    repr_->erase("serverCancelSupport");
    return *this;
  }
  repr_->emplace("serverCancelSupport", std::move(serverCancelSupport.value()));
  return *this;
}

auto SemanticTokensClientCapabilities::augmentsSyntaxTokens(
    std::optional<bool> augmentsSyntaxTokens)
    -> SemanticTokensClientCapabilities& {
  if (!augmentsSyntaxTokens.has_value()) {
    repr_->erase("augmentsSyntaxTokens");
    return *this;
  }
  repr_->emplace("augmentsSyntaxTokens",
                 std::move(augmentsSyntaxTokens.value()));
  return *this;
}

LinkedEditingRangeClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto LinkedEditingRangeClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto LinkedEditingRangeClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> LinkedEditingRangeClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

MonikerClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto MonikerClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto MonikerClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> MonikerClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

TypeHierarchyClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto TypeHierarchyClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto TypeHierarchyClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> TypeHierarchyClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

InlineValueClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlineValueClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineValueClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> InlineValueClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

InlayHintClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlayHintClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlayHintClientCapabilities::resolveSupport() const
    -> std::optional<ClientInlayHintResolveOptions> {
  if (!repr_->contains("resolveSupport")) return std::nullopt;

  auto& value = (*repr_)["resolveSupport"];

  return ClientInlayHintResolveOptions(value);
}

auto InlayHintClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> InlayHintClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto InlayHintClientCapabilities::resolveSupport(
    std::optional<ClientInlayHintResolveOptions> resolveSupport)
    -> InlayHintClientCapabilities& {
  if (!resolveSupport.has_value()) {
    repr_->erase("resolveSupport");
    return *this;
  }
  repr_->emplace("resolveSupport", resolveSupport.value());
  return *this;
}

DiagnosticClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DiagnosticClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::relatedDocumentSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("relatedDocumentSupport")) return std::nullopt;

  auto& value = (*repr_)["relatedDocumentSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::relatedInformation() const
    -> std::optional<bool> {
  if (!repr_->contains("relatedInformation")) return std::nullopt;

  auto& value = (*repr_)["relatedInformation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::tagSupport() const
    -> std::optional<ClientDiagnosticsTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return ClientDiagnosticsTagOptions(value);
}

auto DiagnosticClientCapabilities::codeDescriptionSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("codeDescriptionSupport")) return std::nullopt;

  auto& value = (*repr_)["codeDescriptionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::dataSupport() const -> std::optional<bool> {
  if (!repr_->contains("dataSupport")) return std::nullopt;

  auto& value = (*repr_)["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration) -> DiagnosticClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto DiagnosticClientCapabilities::relatedDocumentSupport(
    std::optional<bool> relatedDocumentSupport)
    -> DiagnosticClientCapabilities& {
  if (!relatedDocumentSupport.has_value()) {
    repr_->erase("relatedDocumentSupport");
    return *this;
  }
  repr_->emplace("relatedDocumentSupport",
                 std::move(relatedDocumentSupport.value()));
  return *this;
}

auto DiagnosticClientCapabilities::relatedInformation(
    std::optional<bool> relatedInformation) -> DiagnosticClientCapabilities& {
  if (!relatedInformation.has_value()) {
    repr_->erase("relatedInformation");
    return *this;
  }
  repr_->emplace("relatedInformation", std::move(relatedInformation.value()));
  return *this;
}

auto DiagnosticClientCapabilities::tagSupport(
    std::optional<ClientDiagnosticsTagOptions> tagSupport)
    -> DiagnosticClientCapabilities& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

auto DiagnosticClientCapabilities::codeDescriptionSupport(
    std::optional<bool> codeDescriptionSupport)
    -> DiagnosticClientCapabilities& {
  if (!codeDescriptionSupport.has_value()) {
    repr_->erase("codeDescriptionSupport");
    return *this;
  }
  repr_->emplace("codeDescriptionSupport",
                 std::move(codeDescriptionSupport.value()));
  return *this;
}

auto DiagnosticClientCapabilities::dataSupport(std::optional<bool> dataSupport)
    -> DiagnosticClientCapabilities& {
  if (!dataSupport.has_value()) {
    repr_->erase("dataSupport");
    return *this;
  }
  repr_->emplace("dataSupport", std::move(dataSupport.value()));
  return *this;
}

InlineCompletionClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto InlineCompletionClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto InlineCompletionClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> InlineCompletionClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

NotebookDocumentSyncClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto NotebookDocumentSyncClientCapabilities::dynamicRegistration() const
    -> std::optional<bool> {
  if (!repr_->contains("dynamicRegistration")) return std::nullopt;

  auto& value = (*repr_)["dynamicRegistration"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncClientCapabilities::executionSummarySupport() const
    -> std::optional<bool> {
  if (!repr_->contains("executionSummarySupport")) return std::nullopt;

  auto& value = (*repr_)["executionSummarySupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto NotebookDocumentSyncClientCapabilities::dynamicRegistration(
    std::optional<bool> dynamicRegistration)
    -> NotebookDocumentSyncClientCapabilities& {
  if (!dynamicRegistration.has_value()) {
    repr_->erase("dynamicRegistration");
    return *this;
  }
  repr_->emplace("dynamicRegistration", std::move(dynamicRegistration.value()));
  return *this;
}

auto NotebookDocumentSyncClientCapabilities::executionSummarySupport(
    std::optional<bool> executionSummarySupport)
    -> NotebookDocumentSyncClientCapabilities& {
  if (!executionSummarySupport.has_value()) {
    repr_->erase("executionSummarySupport");
    return *this;
  }
  repr_->emplace("executionSummarySupport",
                 std::move(executionSummarySupport.value()));
  return *this;
}

ShowMessageRequestClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ShowMessageRequestClientCapabilities::messageActionItem() const
    -> std::optional<ClientShowMessageActionItemOptions> {
  if (!repr_->contains("messageActionItem")) return std::nullopt;

  auto& value = (*repr_)["messageActionItem"];

  return ClientShowMessageActionItemOptions(value);
}

auto ShowMessageRequestClientCapabilities::messageActionItem(
    std::optional<ClientShowMessageActionItemOptions> messageActionItem)
    -> ShowMessageRequestClientCapabilities& {
  if (!messageActionItem.has_value()) {
    repr_->erase("messageActionItem");
    return *this;
  }
  repr_->emplace("messageActionItem", messageActionItem.value());
  return *this;
}

ShowDocumentClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("support")) return false;
  return true;
}

auto ShowDocumentClientCapabilities::support() const -> bool {
  auto& value = (*repr_)["support"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ShowDocumentClientCapabilities::support(bool support)
    -> ShowDocumentClientCapabilities& {
  repr_->emplace("support", std::move(support));
  return *this;
}

StaleRequestSupportOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("cancel")) return false;
  if (!repr_->contains("retryOnContentModified")) return false;
  return true;
}

auto StaleRequestSupportOptions::cancel() const -> bool {
  auto& value = (*repr_)["cancel"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto StaleRequestSupportOptions::retryOnContentModified() const
    -> Vector<std::string> {
  auto& value = (*repr_)["retryOnContentModified"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto StaleRequestSupportOptions::cancel(bool cancel)
    -> StaleRequestSupportOptions& {
  repr_->emplace("cancel", std::move(cancel));
  return *this;
}

auto StaleRequestSupportOptions::retryOnContentModified(
    Vector<std::string> retryOnContentModified) -> StaleRequestSupportOptions& {
  lsp_runtime_error(
      "StaleRequestSupportOptions::retryOnContentModified: not implement yet");
  return *this;
}

RegularExpressionsClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("engine")) return false;
  return true;
}

auto RegularExpressionsClientCapabilities::engine() const
    -> RegularExpressionEngineKind {
  auto& value = (*repr_)["engine"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RegularExpressionsClientCapabilities::version() const
    -> std::optional<std::string> {
  if (!repr_->contains("version")) return std::nullopt;

  auto& value = (*repr_)["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto RegularExpressionsClientCapabilities::engine(
    RegularExpressionEngineKind engine)
    -> RegularExpressionsClientCapabilities& {
  lsp_runtime_error(
      "RegularExpressionsClientCapabilities::engine: not implement yet");
  return *this;
}

auto RegularExpressionsClientCapabilities::version(
    std::optional<std::string> version)
    -> RegularExpressionsClientCapabilities& {
  if (!version.has_value()) {
    repr_->erase("version");
    return *this;
  }
  repr_->emplace("version", std::move(version.value()));
  return *this;
}

MarkdownClientCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("parser")) return false;
  return true;
}

auto MarkdownClientCapabilities::parser() const -> std::string {
  auto& value = (*repr_)["parser"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkdownClientCapabilities::version() const -> std::optional<std::string> {
  if (!repr_->contains("version")) return std::nullopt;

  auto& value = (*repr_)["version"];

  assert(value.is_string());
  return value.get<std::string>();
}

auto MarkdownClientCapabilities::allowedTags() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("allowedTags")) return std::nullopt;

  auto& value = (*repr_)["allowedTags"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto MarkdownClientCapabilities::parser(std::string parser)
    -> MarkdownClientCapabilities& {
  repr_->emplace("parser", std::move(parser));
  return *this;
}

auto MarkdownClientCapabilities::version(std::optional<std::string> version)
    -> MarkdownClientCapabilities& {
  if (!version.has_value()) {
    repr_->erase("version");
    return *this;
  }
  repr_->emplace("version", std::move(version.value()));
  return *this;
}

auto MarkdownClientCapabilities::allowedTags(
    std::optional<Vector<std::string>> allowedTags)
    -> MarkdownClientCapabilities& {
  if (!allowedTags.has_value()) {
    repr_->erase("allowedTags");
    return *this;
  }
  lsp_runtime_error(
      "MarkdownClientCapabilities::allowedTags: not implement yet");
  return *this;
}

ChangeAnnotationsSupportOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ChangeAnnotationsSupportOptions::groupsOnLabel() const
    -> std::optional<bool> {
  if (!repr_->contains("groupsOnLabel")) return std::nullopt;

  auto& value = (*repr_)["groupsOnLabel"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ChangeAnnotationsSupportOptions::groupsOnLabel(
    std::optional<bool> groupsOnLabel) -> ChangeAnnotationsSupportOptions& {
  if (!groupsOnLabel.has_value()) {
    repr_->erase("groupsOnLabel");
    return *this;
  }
  repr_->emplace("groupsOnLabel", std::move(groupsOnLabel.value()));
  return *this;
}

ClientSymbolKindOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientSymbolKindOptions::valueSet() const
    -> std::optional<Vector<SymbolKind>> {
  if (!repr_->contains("valueSet")) return std::nullopt;

  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<SymbolKind>(value);
}

auto ClientSymbolKindOptions::valueSet(
    std::optional<Vector<SymbolKind>> valueSet) -> ClientSymbolKindOptions& {
  if (!valueSet.has_value()) {
    repr_->erase("valueSet");
    return *this;
  }
  lsp_runtime_error("ClientSymbolKindOptions::valueSet: not implement yet");
  return *this;
}

ClientSymbolTagOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("valueSet")) return false;
  return true;
}

auto ClientSymbolTagOptions::valueSet() const -> Vector<SymbolTag> {
  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<SymbolTag>(value);
}

auto ClientSymbolTagOptions::valueSet(Vector<SymbolTag> valueSet)
    -> ClientSymbolTagOptions& {
  lsp_runtime_error("ClientSymbolTagOptions::valueSet: not implement yet");
  return *this;
}

ClientSymbolResolveOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("properties")) return false;
  return true;
}

auto ClientSymbolResolveOptions::properties() const -> Vector<std::string> {
  auto& value = (*repr_)["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientSymbolResolveOptions::properties(Vector<std::string> properties)
    -> ClientSymbolResolveOptions& {
  lsp_runtime_error(
      "ClientSymbolResolveOptions::properties: not implement yet");
  return *this;
}

ClientCompletionItemOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientCompletionItemOptions::snippetSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("snippetSupport")) return std::nullopt;

  auto& value = (*repr_)["snippetSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::commitCharactersSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("commitCharactersSupport")) return std::nullopt;

  auto& value = (*repr_)["commitCharactersSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::documentationFormat() const
    -> std::optional<Vector<MarkupKind>> {
  if (!repr_->contains("documentationFormat")) return std::nullopt;

  auto& value = (*repr_)["documentationFormat"];

  assert(value.is_array());
  return Vector<MarkupKind>(value);
}

auto ClientCompletionItemOptions::deprecatedSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("deprecatedSupport")) return std::nullopt;

  auto& value = (*repr_)["deprecatedSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::preselectSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("preselectSupport")) return std::nullopt;

  auto& value = (*repr_)["preselectSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::tagSupport() const
    -> std::optional<CompletionItemTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return CompletionItemTagOptions(value);
}

auto ClientCompletionItemOptions::insertReplaceSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("insertReplaceSupport")) return std::nullopt;

  auto& value = (*repr_)["insertReplaceSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::resolveSupport() const
    -> std::optional<ClientCompletionItemResolveOptions> {
  if (!repr_->contains("resolveSupport")) return std::nullopt;

  auto& value = (*repr_)["resolveSupport"];

  return ClientCompletionItemResolveOptions(value);
}

auto ClientCompletionItemOptions::insertTextModeSupport() const
    -> std::optional<ClientCompletionItemInsertTextModeOptions> {
  if (!repr_->contains("insertTextModeSupport")) return std::nullopt;

  auto& value = (*repr_)["insertTextModeSupport"];

  return ClientCompletionItemInsertTextModeOptions(value);
}

auto ClientCompletionItemOptions::labelDetailsSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("labelDetailsSupport")) return std::nullopt;

  auto& value = (*repr_)["labelDetailsSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientCompletionItemOptions::snippetSupport(
    std::optional<bool> snippetSupport) -> ClientCompletionItemOptions& {
  if (!snippetSupport.has_value()) {
    repr_->erase("snippetSupport");
    return *this;
  }
  repr_->emplace("snippetSupport", std::move(snippetSupport.value()));
  return *this;
}

auto ClientCompletionItemOptions::commitCharactersSupport(
    std::optional<bool> commitCharactersSupport)
    -> ClientCompletionItemOptions& {
  if (!commitCharactersSupport.has_value()) {
    repr_->erase("commitCharactersSupport");
    return *this;
  }
  repr_->emplace("commitCharactersSupport",
                 std::move(commitCharactersSupport.value()));
  return *this;
}

auto ClientCompletionItemOptions::documentationFormat(
    std::optional<Vector<MarkupKind>> documentationFormat)
    -> ClientCompletionItemOptions& {
  if (!documentationFormat.has_value()) {
    repr_->erase("documentationFormat");
    return *this;
  }
  lsp_runtime_error(
      "ClientCompletionItemOptions::documentationFormat: not implement yet");
  return *this;
}

auto ClientCompletionItemOptions::deprecatedSupport(
    std::optional<bool> deprecatedSupport) -> ClientCompletionItemOptions& {
  if (!deprecatedSupport.has_value()) {
    repr_->erase("deprecatedSupport");
    return *this;
  }
  repr_->emplace("deprecatedSupport", std::move(deprecatedSupport.value()));
  return *this;
}

auto ClientCompletionItemOptions::preselectSupport(
    std::optional<bool> preselectSupport) -> ClientCompletionItemOptions& {
  if (!preselectSupport.has_value()) {
    repr_->erase("preselectSupport");
    return *this;
  }
  repr_->emplace("preselectSupport", std::move(preselectSupport.value()));
  return *this;
}

auto ClientCompletionItemOptions::tagSupport(
    std::optional<CompletionItemTagOptions> tagSupport)
    -> ClientCompletionItemOptions& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

auto ClientCompletionItemOptions::insertReplaceSupport(
    std::optional<bool> insertReplaceSupport) -> ClientCompletionItemOptions& {
  if (!insertReplaceSupport.has_value()) {
    repr_->erase("insertReplaceSupport");
    return *this;
  }
  repr_->emplace("insertReplaceSupport",
                 std::move(insertReplaceSupport.value()));
  return *this;
}

auto ClientCompletionItemOptions::resolveSupport(
    std::optional<ClientCompletionItemResolveOptions> resolveSupport)
    -> ClientCompletionItemOptions& {
  if (!resolveSupport.has_value()) {
    repr_->erase("resolveSupport");
    return *this;
  }
  repr_->emplace("resolveSupport", resolveSupport.value());
  return *this;
}

auto ClientCompletionItemOptions::insertTextModeSupport(
    std::optional<ClientCompletionItemInsertTextModeOptions>
        insertTextModeSupport) -> ClientCompletionItemOptions& {
  if (!insertTextModeSupport.has_value()) {
    repr_->erase("insertTextModeSupport");
    return *this;
  }
  repr_->emplace("insertTextModeSupport", insertTextModeSupport.value());
  return *this;
}

auto ClientCompletionItemOptions::labelDetailsSupport(
    std::optional<bool> labelDetailsSupport) -> ClientCompletionItemOptions& {
  if (!labelDetailsSupport.has_value()) {
    repr_->erase("labelDetailsSupport");
    return *this;
  }
  repr_->emplace("labelDetailsSupport", std::move(labelDetailsSupport.value()));
  return *this;
}

ClientCompletionItemOptionsKind::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientCompletionItemOptionsKind::valueSet() const
    -> std::optional<Vector<CompletionItemKind>> {
  if (!repr_->contains("valueSet")) return std::nullopt;

  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<CompletionItemKind>(value);
}

auto ClientCompletionItemOptionsKind::valueSet(
    std::optional<Vector<CompletionItemKind>> valueSet)
    -> ClientCompletionItemOptionsKind& {
  if (!valueSet.has_value()) {
    repr_->erase("valueSet");
    return *this;
  }
  lsp_runtime_error(
      "ClientCompletionItemOptionsKind::valueSet: not implement yet");
  return *this;
}

CompletionListCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto CompletionListCapabilities::itemDefaults() const
    -> std::optional<Vector<std::string>> {
  if (!repr_->contains("itemDefaults")) return std::nullopt;

  auto& value = (*repr_)["itemDefaults"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto CompletionListCapabilities::applyKindSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("applyKindSupport")) return std::nullopt;

  auto& value = (*repr_)["applyKindSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto CompletionListCapabilities::itemDefaults(
    std::optional<Vector<std::string>> itemDefaults)
    -> CompletionListCapabilities& {
  if (!itemDefaults.has_value()) {
    repr_->erase("itemDefaults");
    return *this;
  }
  lsp_runtime_error(
      "CompletionListCapabilities::itemDefaults: not implement yet");
  return *this;
}

auto CompletionListCapabilities::applyKindSupport(
    std::optional<bool> applyKindSupport) -> CompletionListCapabilities& {
  if (!applyKindSupport.has_value()) {
    repr_->erase("applyKindSupport");
    return *this;
  }
  repr_->emplace("applyKindSupport", std::move(applyKindSupport.value()));
  return *this;
}

ClientSignatureInformationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientSignatureInformationOptions::documentationFormat() const
    -> std::optional<Vector<MarkupKind>> {
  if (!repr_->contains("documentationFormat")) return std::nullopt;

  auto& value = (*repr_)["documentationFormat"];

  assert(value.is_array());
  return Vector<MarkupKind>(value);
}

auto ClientSignatureInformationOptions::parameterInformation() const
    -> std::optional<ClientSignatureParameterInformationOptions> {
  if (!repr_->contains("parameterInformation")) return std::nullopt;

  auto& value = (*repr_)["parameterInformation"];

  return ClientSignatureParameterInformationOptions(value);
}

auto ClientSignatureInformationOptions::activeParameterSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("activeParameterSupport")) return std::nullopt;

  auto& value = (*repr_)["activeParameterSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSignatureInformationOptions::noActiveParameterSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("noActiveParameterSupport")) return std::nullopt;

  auto& value = (*repr_)["noActiveParameterSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSignatureInformationOptions::documentationFormat(
    std::optional<Vector<MarkupKind>> documentationFormat)
    -> ClientSignatureInformationOptions& {
  if (!documentationFormat.has_value()) {
    repr_->erase("documentationFormat");
    return *this;
  }
  lsp_runtime_error(
      "ClientSignatureInformationOptions::documentationFormat: not implement "
      "yet");
  return *this;
}

auto ClientSignatureInformationOptions::parameterInformation(
    std::optional<ClientSignatureParameterInformationOptions>
        parameterInformation) -> ClientSignatureInformationOptions& {
  if (!parameterInformation.has_value()) {
    repr_->erase("parameterInformation");
    return *this;
  }
  repr_->emplace("parameterInformation", parameterInformation.value());
  return *this;
}

auto ClientSignatureInformationOptions::activeParameterSupport(
    std::optional<bool> activeParameterSupport)
    -> ClientSignatureInformationOptions& {
  if (!activeParameterSupport.has_value()) {
    repr_->erase("activeParameterSupport");
    return *this;
  }
  repr_->emplace("activeParameterSupport",
                 std::move(activeParameterSupport.value()));
  return *this;
}

auto ClientSignatureInformationOptions::noActiveParameterSupport(
    std::optional<bool> noActiveParameterSupport)
    -> ClientSignatureInformationOptions& {
  if (!noActiveParameterSupport.has_value()) {
    repr_->erase("noActiveParameterSupport");
    return *this;
  }
  repr_->emplace("noActiveParameterSupport",
                 std::move(noActiveParameterSupport.value()));
  return *this;
}

ClientCodeActionLiteralOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("codeActionKind")) return false;
  return true;
}

auto ClientCodeActionLiteralOptions::codeActionKind() const
    -> ClientCodeActionKindOptions {
  auto& value = (*repr_)["codeActionKind"];

  return ClientCodeActionKindOptions(value);
}

auto ClientCodeActionLiteralOptions::codeActionKind(
    ClientCodeActionKindOptions codeActionKind)
    -> ClientCodeActionLiteralOptions& {
  repr_->emplace("codeActionKind", codeActionKind);
  return *this;
}

ClientCodeActionResolveOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("properties")) return false;
  return true;
}

auto ClientCodeActionResolveOptions::properties() const -> Vector<std::string> {
  auto& value = (*repr_)["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientCodeActionResolveOptions::properties(Vector<std::string> properties)
    -> ClientCodeActionResolveOptions& {
  lsp_runtime_error(
      "ClientCodeActionResolveOptions::properties: not implement yet");
  return *this;
}

CodeActionTagOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("valueSet")) return false;
  return true;
}

auto CodeActionTagOptions::valueSet() const -> Vector<CodeActionTag> {
  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<CodeActionTag>(value);
}

auto CodeActionTagOptions::valueSet(Vector<CodeActionTag> valueSet)
    -> CodeActionTagOptions& {
  lsp_runtime_error("CodeActionTagOptions::valueSet: not implement yet");
  return *this;
}

ClientCodeLensResolveOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("properties")) return false;
  return true;
}

auto ClientCodeLensResolveOptions::properties() const -> Vector<std::string> {
  auto& value = (*repr_)["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientCodeLensResolveOptions::properties(Vector<std::string> properties)
    -> ClientCodeLensResolveOptions& {
  lsp_runtime_error(
      "ClientCodeLensResolveOptions::properties: not implement yet");
  return *this;
}

ClientFoldingRangeKindOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientFoldingRangeKindOptions::valueSet() const
    -> std::optional<Vector<FoldingRangeKind>> {
  if (!repr_->contains("valueSet")) return std::nullopt;

  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<FoldingRangeKind>(value);
}

auto ClientFoldingRangeKindOptions::valueSet(
    std::optional<Vector<FoldingRangeKind>> valueSet)
    -> ClientFoldingRangeKindOptions& {
  if (!valueSet.has_value()) {
    repr_->erase("valueSet");
    return *this;
  }
  lsp_runtime_error(
      "ClientFoldingRangeKindOptions::valueSet: not implement yet");
  return *this;
}

ClientFoldingRangeOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientFoldingRangeOptions::collapsedText() const -> std::optional<bool> {
  if (!repr_->contains("collapsedText")) return std::nullopt;

  auto& value = (*repr_)["collapsedText"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientFoldingRangeOptions::collapsedText(std::optional<bool> collapsedText)
    -> ClientFoldingRangeOptions& {
  if (!collapsedText.has_value()) {
    repr_->erase("collapsedText");
    return *this;
  }
  repr_->emplace("collapsedText", std::move(collapsedText.value()));
  return *this;
}

DiagnosticsCapabilities::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto DiagnosticsCapabilities::relatedInformation() const
    -> std::optional<bool> {
  if (!repr_->contains("relatedInformation")) return std::nullopt;

  auto& value = (*repr_)["relatedInformation"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticsCapabilities::tagSupport() const
    -> std::optional<ClientDiagnosticsTagOptions> {
  if (!repr_->contains("tagSupport")) return std::nullopt;

  auto& value = (*repr_)["tagSupport"];

  return ClientDiagnosticsTagOptions(value);
}

auto DiagnosticsCapabilities::codeDescriptionSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("codeDescriptionSupport")) return std::nullopt;

  auto& value = (*repr_)["codeDescriptionSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticsCapabilities::dataSupport() const -> std::optional<bool> {
  if (!repr_->contains("dataSupport")) return std::nullopt;

  auto& value = (*repr_)["dataSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto DiagnosticsCapabilities::relatedInformation(
    std::optional<bool> relatedInformation) -> DiagnosticsCapabilities& {
  if (!relatedInformation.has_value()) {
    repr_->erase("relatedInformation");
    return *this;
  }
  repr_->emplace("relatedInformation", std::move(relatedInformation.value()));
  return *this;
}

auto DiagnosticsCapabilities::tagSupport(
    std::optional<ClientDiagnosticsTagOptions> tagSupport)
    -> DiagnosticsCapabilities& {
  if (!tagSupport.has_value()) {
    repr_->erase("tagSupport");
    return *this;
  }
  repr_->emplace("tagSupport", tagSupport.value());
  return *this;
}

auto DiagnosticsCapabilities::codeDescriptionSupport(
    std::optional<bool> codeDescriptionSupport) -> DiagnosticsCapabilities& {
  if (!codeDescriptionSupport.has_value()) {
    repr_->erase("codeDescriptionSupport");
    return *this;
  }
  repr_->emplace("codeDescriptionSupport",
                 std::move(codeDescriptionSupport.value()));
  return *this;
}

auto DiagnosticsCapabilities::dataSupport(std::optional<bool> dataSupport)
    -> DiagnosticsCapabilities& {
  if (!dataSupport.has_value()) {
    repr_->erase("dataSupport");
    return *this;
  }
  repr_->emplace("dataSupport", std::move(dataSupport.value()));
  return *this;
}

ClientSemanticTokensRequestOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientSemanticTokensRequestOptions::range() const
    -> std::optional<std::variant<std::monostate, bool, json>> {
  if (!repr_->contains("range")) return std::nullopt;

  auto& value = (*repr_)["range"];

  std::variant<std::monostate, bool, json> result;

  details::try_emplace(result, value);

  return result;
}

auto ClientSemanticTokensRequestOptions::full() const -> std::optional<
    std::variant<std::monostate, bool, ClientSemanticTokensRequestFullDelta>> {
  if (!repr_->contains("full")) return std::nullopt;

  auto& value = (*repr_)["full"];

  std::variant<std::monostate, bool, ClientSemanticTokensRequestFullDelta>
      result;

  details::try_emplace(result, value);

  return result;
}

auto ClientSemanticTokensRequestOptions::range(
    std::optional<std::variant<std::monostate, bool, json>> range)
    -> ClientSemanticTokensRequestOptions& {
  if (!range.has_value()) {
    repr_->erase("range");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool range) { repr_->emplace("range", std::move(range)); }

    void operator()(json range) {
      lsp_runtime_error(
          "ClientSemanticTokensRequestOptions::range: not implement yet");
    }
  } v{repr_};

  std::visit(v, range.value());

  return *this;
}

auto ClientSemanticTokensRequestOptions::full(
    std::optional<std::variant<std::monostate, bool,
                               ClientSemanticTokensRequestFullDelta>>
        full) -> ClientSemanticTokensRequestOptions& {
  if (!full.has_value()) {
    repr_->erase("full");
    return *this;
  }

  // or type

  struct {
    json* repr_;

    void operator()(std::monostate) {
      lsp_runtime_error("monostate is not a valid a property value");
    }

    void operator()(bool full) { repr_->emplace("full", std::move(full)); }

    void operator()(ClientSemanticTokensRequestFullDelta full) {
      repr_->emplace("full", full);
    }
  } v{repr_};

  std::visit(v, full.value());

  return *this;
}

ClientInlayHintResolveOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("properties")) return false;
  return true;
}

auto ClientInlayHintResolveOptions::properties() const -> Vector<std::string> {
  auto& value = (*repr_)["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientInlayHintResolveOptions::properties(Vector<std::string> properties)
    -> ClientInlayHintResolveOptions& {
  lsp_runtime_error(
      "ClientInlayHintResolveOptions::properties: not implement yet");
  return *this;
}

ClientShowMessageActionItemOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientShowMessageActionItemOptions::additionalPropertiesSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("additionalPropertiesSupport")) return std::nullopt;

  auto& value = (*repr_)["additionalPropertiesSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientShowMessageActionItemOptions::additionalPropertiesSupport(
    std::optional<bool> additionalPropertiesSupport)
    -> ClientShowMessageActionItemOptions& {
  if (!additionalPropertiesSupport.has_value()) {
    repr_->erase("additionalPropertiesSupport");
    return *this;
  }
  repr_->emplace("additionalPropertiesSupport",
                 std::move(additionalPropertiesSupport.value()));
  return *this;
}

CompletionItemTagOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("valueSet")) return false;
  return true;
}

auto CompletionItemTagOptions::valueSet() const -> Vector<CompletionItemTag> {
  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<CompletionItemTag>(value);
}

auto CompletionItemTagOptions::valueSet(Vector<CompletionItemTag> valueSet)
    -> CompletionItemTagOptions& {
  lsp_runtime_error("CompletionItemTagOptions::valueSet: not implement yet");
  return *this;
}

ClientCompletionItemResolveOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("properties")) return false;
  return true;
}

auto ClientCompletionItemResolveOptions::properties() const
    -> Vector<std::string> {
  auto& value = (*repr_)["properties"];

  assert(value.is_array());
  return Vector<std::string>(value);
}

auto ClientCompletionItemResolveOptions::properties(
    Vector<std::string> properties) -> ClientCompletionItemResolveOptions& {
  lsp_runtime_error(
      "ClientCompletionItemResolveOptions::properties: not implement yet");
  return *this;
}

ClientCompletionItemInsertTextModeOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("valueSet")) return false;
  return true;
}

auto ClientCompletionItemInsertTextModeOptions::valueSet() const
    -> Vector<InsertTextMode> {
  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<InsertTextMode>(value);
}

auto ClientCompletionItemInsertTextModeOptions::valueSet(
    Vector<InsertTextMode> valueSet)
    -> ClientCompletionItemInsertTextModeOptions& {
  lsp_runtime_error(
      "ClientCompletionItemInsertTextModeOptions::valueSet: not implement yet");
  return *this;
}

ClientSignatureParameterInformationOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientSignatureParameterInformationOptions::labelOffsetSupport() const
    -> std::optional<bool> {
  if (!repr_->contains("labelOffsetSupport")) return std::nullopt;

  auto& value = (*repr_)["labelOffsetSupport"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSignatureParameterInformationOptions::labelOffsetSupport(
    std::optional<bool> labelOffsetSupport)
    -> ClientSignatureParameterInformationOptions& {
  if (!labelOffsetSupport.has_value()) {
    repr_->erase("labelOffsetSupport");
    return *this;
  }
  repr_->emplace("labelOffsetSupport", std::move(labelOffsetSupport.value()));
  return *this;
}

ClientCodeActionKindOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("valueSet")) return false;
  return true;
}

auto ClientCodeActionKindOptions::valueSet() const -> Vector<CodeActionKind> {
  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<CodeActionKind>(value);
}

auto ClientCodeActionKindOptions::valueSet(Vector<CodeActionKind> valueSet)
    -> ClientCodeActionKindOptions& {
  lsp_runtime_error("ClientCodeActionKindOptions::valueSet: not implement yet");
  return *this;
}

ClientDiagnosticsTagOptions::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  if (!repr_->contains("valueSet")) return false;
  return true;
}

auto ClientDiagnosticsTagOptions::valueSet() const -> Vector<DiagnosticTag> {
  auto& value = (*repr_)["valueSet"];

  assert(value.is_array());
  return Vector<DiagnosticTag>(value);
}

auto ClientDiagnosticsTagOptions::valueSet(Vector<DiagnosticTag> valueSet)
    -> ClientDiagnosticsTagOptions& {
  lsp_runtime_error("ClientDiagnosticsTagOptions::valueSet: not implement yet");
  return *this;
}

ClientSemanticTokensRequestFullDelta::operator bool() const {
  if (!repr_->is_object() || repr_->is_null()) return false;
  return true;
}

auto ClientSemanticTokensRequestFullDelta::delta() const
    -> std::optional<bool> {
  if (!repr_->contains("delta")) return std::nullopt;

  auto& value = (*repr_)["delta"];

  assert(value.is_boolean());
  return value.get<bool>();
}

auto ClientSemanticTokensRequestFullDelta::delta(std::optional<bool> delta)
    -> ClientSemanticTokensRequestFullDelta& {
  if (!delta.has_value()) {
    repr_->erase("delta");
    return *this;
  }
  repr_->emplace("delta", std::move(delta.value()));
  return *this;
}
}  // namespace cxx::lsp
