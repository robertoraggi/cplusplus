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

class ImplementationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> ImplementationParams&;

  auto position(Position position) -> ImplementationParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> ImplementationParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> ImplementationParams&;
};

class Location final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto range() const -> Range;

  auto uri(std::string uri) -> Location&;

  auto range(Range range) -> Location&;
};

class ImplementationRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> ImplementationRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ImplementationRegistrationOptions&;

  auto id(std::optional<std::string> id) -> ImplementationRegistrationOptions&;
};

class TypeDefinitionParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> TypeDefinitionParams&;

  auto position(Position position) -> TypeDefinitionParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> TypeDefinitionParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> TypeDefinitionParams&;
};

class TypeDefinitionRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> TypeDefinitionRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> TypeDefinitionRegistrationOptions&;

  auto id(std::optional<std::string> id) -> TypeDefinitionRegistrationOptions&;
};

class WorkspaceFolder final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto name() const -> std::string;

  auto uri(std::string uri) -> WorkspaceFolder&;

  auto name(std::string name) -> WorkspaceFolder&;
};

class DidChangeWorkspaceFoldersParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto event() const -> WorkspaceFoldersChangeEvent;

  auto event(WorkspaceFoldersChangeEvent event)
      -> DidChangeWorkspaceFoldersParams&;
};

class ConfigurationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto items() const -> Vector<ConfigurationItem>;

  auto items(Vector<ConfigurationItem> items) -> ConfigurationParams&;
};

class DocumentColorParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentColorParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentColorParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DocumentColorParams&;
};

class ColorInformation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto color() const -> Color;

  auto range(Range range) -> ColorInformation&;

  auto color(Color color) -> ColorInformation&;
};

class DocumentColorRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentColorRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentColorRegistrationOptions&;

  auto id(std::optional<std::string> id) -> DocumentColorRegistrationOptions&;
};

class ColorPresentationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto color() const -> Color;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> ColorPresentationParams&;

  auto color(Color color) -> ColorPresentationParams&;

  auto range(Range range) -> ColorPresentationParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> ColorPresentationParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> ColorPresentationParams&;
};

class ColorPresentation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::string;

  [[nodiscard]] auto textEdit() const -> std::optional<TextEdit>;

  template <typename T>
  [[nodiscard]] auto textEdit() -> T {
    auto& value = (*repr_)["textEdit"];
    return T(value);
  }

  [[nodiscard]] auto additionalTextEdits() const
      -> std::optional<Vector<TextEdit>>;

  template <typename T>
  [[nodiscard]] auto additionalTextEdits() -> T {
    auto& value = (*repr_)["additionalTextEdits"];
    return T(value);
  }

  auto label(std::string label) -> ColorPresentation&;

  auto textEdit(std::optional<TextEdit> textEdit) -> ColorPresentation&;

  auto additionalTextEdits(std::optional<Vector<TextEdit>> additionalTextEdits)
      -> ColorPresentation&;
};

class WorkDoneProgressOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> WorkDoneProgressOptions&;
};

class TextDocumentRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> TextDocumentRegistrationOptions&;
};

class FoldingRangeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> FoldingRangeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> FoldingRangeParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> FoldingRangeParams&;
};

class FoldingRange final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto startLine() const -> long;

  [[nodiscard]] auto startCharacter() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto startCharacter() -> T {
    auto& value = (*repr_)["startCharacter"];
    return T(value);
  }

  [[nodiscard]] auto endLine() const -> long;

  [[nodiscard]] auto endCharacter() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto endCharacter() -> T {
    auto& value = (*repr_)["endCharacter"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::optional<FoldingRangeKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  [[nodiscard]] auto collapsedText() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto collapsedText() -> T {
    auto& value = (*repr_)["collapsedText"];
    return T(value);
  }

  auto startLine(long startLine) -> FoldingRange&;

  auto startCharacter(std::optional<long> startCharacter) -> FoldingRange&;

  auto endLine(long endLine) -> FoldingRange&;

  auto endCharacter(std::optional<long> endCharacter) -> FoldingRange&;

  auto kind(std::optional<FoldingRangeKind> kind) -> FoldingRange&;

  auto collapsedText(std::optional<std::string> collapsedText) -> FoldingRange&;
};

class FoldingRangeRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> FoldingRangeRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> FoldingRangeRegistrationOptions&;

  auto id(std::optional<std::string> id) -> FoldingRangeRegistrationOptions&;
};

class DeclarationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> DeclarationParams&;

  auto position(Position position) -> DeclarationParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DeclarationParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DeclarationParams&;
};

class DeclarationRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DeclarationRegistrationOptions&;

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DeclarationRegistrationOptions&;

  auto id(std::optional<std::string> id) -> DeclarationRegistrationOptions&;
};

class SelectionRangeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto positions() const -> Vector<Position>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> SelectionRangeParams&;

  auto positions(Vector<Position> positions) -> SelectionRangeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> SelectionRangeParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> SelectionRangeParams&;
};

class SelectionRange final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto parent() const -> std::optional<SelectionRange>;

  template <typename T>
  [[nodiscard]] auto parent() -> T {
    auto& value = (*repr_)["parent"];
    return T(value);
  }

  auto range(Range range) -> SelectionRange&;

  auto parent(std::optional<SelectionRange> parent) -> SelectionRange&;
};

class SelectionRangeRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SelectionRangeRegistrationOptions&;

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> SelectionRangeRegistrationOptions&;

  auto id(std::optional<std::string> id) -> SelectionRangeRegistrationOptions&;
};

class WorkDoneProgressCreateParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto token() const -> ProgressToken;

  auto token(ProgressToken token) -> WorkDoneProgressCreateParams&;
};

class WorkDoneProgressCancelParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto token() const -> ProgressToken;

  auto token(ProgressToken token) -> WorkDoneProgressCancelParams&;
};

class CallHierarchyPrepareParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> CallHierarchyPrepareParams&;

  auto position(Position position) -> CallHierarchyPrepareParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> CallHierarchyPrepareParams&;
};

class CallHierarchyItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto detail() -> T {
    auto& value = (*repr_)["detail"];
    return T(value);
  }

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto selectionRange() const -> Range;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto name(std::string name) -> CallHierarchyItem&;

  auto kind(SymbolKind kind) -> CallHierarchyItem&;

  auto tags(std::optional<Vector<SymbolTag>> tags) -> CallHierarchyItem&;

  auto detail(std::optional<std::string> detail) -> CallHierarchyItem&;

  auto uri(std::string uri) -> CallHierarchyItem&;

  auto range(Range range) -> CallHierarchyItem&;

  auto selectionRange(Range selectionRange) -> CallHierarchyItem&;

  auto data(std::optional<LSPAny> data) -> CallHierarchyItem&;
};

class CallHierarchyRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> CallHierarchyRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CallHierarchyRegistrationOptions&;

  auto id(std::optional<std::string> id) -> CallHierarchyRegistrationOptions&;
};

class CallHierarchyIncomingCallsParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto item() const -> CallHierarchyItem;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto item(CallHierarchyItem item) -> CallHierarchyIncomingCallsParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> CallHierarchyIncomingCallsParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> CallHierarchyIncomingCallsParams&;
};

class CallHierarchyIncomingCall final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto from() const -> CallHierarchyItem;

  [[nodiscard]] auto fromRanges() const -> Vector<Range>;

  auto from(CallHierarchyItem from) -> CallHierarchyIncomingCall&;

  auto fromRanges(Vector<Range> fromRanges) -> CallHierarchyIncomingCall&;
};

class CallHierarchyOutgoingCallsParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto item() const -> CallHierarchyItem;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto item(CallHierarchyItem item) -> CallHierarchyOutgoingCallsParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> CallHierarchyOutgoingCallsParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> CallHierarchyOutgoingCallsParams&;
};

class CallHierarchyOutgoingCall final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto to() const -> CallHierarchyItem;

  [[nodiscard]] auto fromRanges() const -> Vector<Range>;

  auto to(CallHierarchyItem to) -> CallHierarchyOutgoingCall&;

  auto fromRanges(Vector<Range> fromRanges) -> CallHierarchyOutgoingCall&;
};

class SemanticTokensParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> SemanticTokensParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> SemanticTokensParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> SemanticTokensParams&;
};

class SemanticTokens final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto resultId() -> T {
    auto& value = (*repr_)["resultId"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> Vector<long>;

  auto resultId(std::optional<std::string> resultId) -> SemanticTokens&;

  auto data(Vector<long> data) -> SemanticTokens&;
};

class SemanticTokensPartialResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto data() const -> Vector<long>;

  auto data(Vector<long> data) -> SemanticTokensPartialResult&;
};

class SemanticTokensRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto legend() const -> SemanticTokensLegend;

  [[nodiscard]] auto range() const -> std::optional<std::variant<bool, json>>;

  template <typename T>
  [[nodiscard]] auto range() -> T {
    auto& value = (*repr_)["range"];
    return T(value);
  }

  [[nodiscard]] auto full() const
      -> std::optional<std::variant<bool, SemanticTokensFullDelta>>;

  template <typename T>
  [[nodiscard]] auto full() -> T {
    auto& value = (*repr_)["full"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> SemanticTokensRegistrationOptions&;

  auto legend(SemanticTokensLegend legend)
      -> SemanticTokensRegistrationOptions&;

  auto range(std::optional<std::variant<bool, json>> range)
      -> SemanticTokensRegistrationOptions&;

  auto full(std::optional<std::variant<bool, SemanticTokensFullDelta>> full)
      -> SemanticTokensRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SemanticTokensRegistrationOptions&;

  auto id(std::optional<std::string> id) -> SemanticTokensRegistrationOptions&;
};

class SemanticTokensDeltaParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto previousResultId() const -> std::string;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> SemanticTokensDeltaParams&;

  auto previousResultId(std::string previousResultId)
      -> SemanticTokensDeltaParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> SemanticTokensDeltaParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> SemanticTokensDeltaParams&;
};

class SemanticTokensDelta final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto resultId() -> T {
    auto& value = (*repr_)["resultId"];
    return T(value);
  }

  [[nodiscard]] auto edits() const -> Vector<SemanticTokensEdit>;

  auto resultId(std::optional<std::string> resultId) -> SemanticTokensDelta&;

  auto edits(Vector<SemanticTokensEdit> edits) -> SemanticTokensDelta&;
};

class SemanticTokensDeltaPartialResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto edits() const -> Vector<SemanticTokensEdit>;

  auto edits(Vector<SemanticTokensEdit> edits)
      -> SemanticTokensDeltaPartialResult&;
};

class SemanticTokensRangeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> SemanticTokensRangeParams&;

  auto range(Range range) -> SemanticTokensRangeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> SemanticTokensRangeParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> SemanticTokensRangeParams&;
};

class ShowDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto external() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto external() -> T {
    auto& value = (*repr_)["external"];
    return T(value);
  }

  [[nodiscard]] auto takeFocus() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto takeFocus() -> T {
    auto& value = (*repr_)["takeFocus"];
    return T(value);
  }

  [[nodiscard]] auto selection() const -> std::optional<Range>;

  template <typename T>
  [[nodiscard]] auto selection() -> T {
    auto& value = (*repr_)["selection"];
    return T(value);
  }

  auto uri(std::string uri) -> ShowDocumentParams&;

  auto external(std::optional<bool> external) -> ShowDocumentParams&;

  auto takeFocus(std::optional<bool> takeFocus) -> ShowDocumentParams&;

  auto selection(std::optional<Range> selection) -> ShowDocumentParams&;
};

class ShowDocumentResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto success() const -> bool;

  auto success(bool success) -> ShowDocumentResult&;
};

class LinkedEditingRangeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> LinkedEditingRangeParams&;

  auto position(Position position) -> LinkedEditingRangeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> LinkedEditingRangeParams&;
};

class LinkedEditingRanges final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto ranges() const -> Vector<Range>;

  [[nodiscard]] auto wordPattern() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto wordPattern() -> T {
    auto& value = (*repr_)["wordPattern"];
    return T(value);
  }

  auto ranges(Vector<Range> ranges) -> LinkedEditingRanges&;

  auto wordPattern(std::optional<std::string> wordPattern)
      -> LinkedEditingRanges&;
};

class LinkedEditingRangeRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> LinkedEditingRangeRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> LinkedEditingRangeRegistrationOptions&;

  auto id(std::optional<std::string> id)
      -> LinkedEditingRangeRegistrationOptions&;
};

class CreateFilesParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto files() const -> Vector<FileCreate>;

  auto files(Vector<FileCreate> files) -> CreateFilesParams&;
};

class WorkspaceEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto changes() const
      -> std::optional<Map<std::string, Vector<TextEdit>>>;

  template <typename T>
  [[nodiscard]] auto changes() -> T {
    auto& value = (*repr_)["changes"];
    return T(value);
  }

  [[nodiscard]] auto documentChanges() const -> std::optional<Vector<
      std::variant<TextDocumentEdit, CreateFile, RenameFile, DeleteFile>>>;

  template <typename T>
  [[nodiscard]] auto documentChanges() -> T {
    auto& value = (*repr_)["documentChanges"];
    return T(value);
  }

  [[nodiscard]] auto changeAnnotations() const
      -> std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>>;

  template <typename T>
  [[nodiscard]] auto changeAnnotations() -> T {
    auto& value = (*repr_)["changeAnnotations"];
    return T(value);
  }

  auto changes(std::optional<Map<std::string, Vector<TextEdit>>> changes)
      -> WorkspaceEdit&;

  auto documentChanges(
      std::optional<Vector<
          std::variant<TextDocumentEdit, CreateFile, RenameFile, DeleteFile>>>
          documentChanges) -> WorkspaceEdit&;

  auto changeAnnotations(
      std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>>
          changeAnnotations) -> WorkspaceEdit&;
};

class FileOperationRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto filters() const -> Vector<FileOperationFilter>;

  auto filters(Vector<FileOperationFilter> filters)
      -> FileOperationRegistrationOptions&;
};

class RenameFilesParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto files() const -> Vector<FileRename>;

  auto files(Vector<FileRename> files) -> RenameFilesParams&;
};

class DeleteFilesParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto files() const -> Vector<FileDelete>;

  auto files(Vector<FileDelete> files) -> DeleteFilesParams&;
};

class MonikerParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> MonikerParams&;

  auto position(Position position) -> MonikerParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> MonikerParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> MonikerParams&;
};

class Moniker final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto scheme() const -> std::string;

  [[nodiscard]] auto identifier() const -> std::string;

  [[nodiscard]] auto unique() const -> UniquenessLevel;

  [[nodiscard]] auto kind() const -> std::optional<MonikerKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  auto scheme(std::string scheme) -> Moniker&;

  auto identifier(std::string identifier) -> Moniker&;

  auto unique(UniquenessLevel unique) -> Moniker&;

  auto kind(std::optional<MonikerKind> kind) -> Moniker&;
};

class MonikerRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> MonikerRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> MonikerRegistrationOptions&;
};

class TypeHierarchyPrepareParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> TypeHierarchyPrepareParams&;

  auto position(Position position) -> TypeHierarchyPrepareParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> TypeHierarchyPrepareParams&;
};

class TypeHierarchyItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto detail() -> T {
    auto& value = (*repr_)["detail"];
    return T(value);
  }

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto selectionRange() const -> Range;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto name(std::string name) -> TypeHierarchyItem&;

  auto kind(SymbolKind kind) -> TypeHierarchyItem&;

  auto tags(std::optional<Vector<SymbolTag>> tags) -> TypeHierarchyItem&;

  auto detail(std::optional<std::string> detail) -> TypeHierarchyItem&;

  auto uri(std::string uri) -> TypeHierarchyItem&;

  auto range(Range range) -> TypeHierarchyItem&;

  auto selectionRange(Range selectionRange) -> TypeHierarchyItem&;

  auto data(std::optional<LSPAny> data) -> TypeHierarchyItem&;
};

class TypeHierarchyRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> TypeHierarchyRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> TypeHierarchyRegistrationOptions&;

  auto id(std::optional<std::string> id) -> TypeHierarchyRegistrationOptions&;
};

class TypeHierarchySupertypesParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto item() const -> TypeHierarchyItem;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto item(TypeHierarchyItem item) -> TypeHierarchySupertypesParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> TypeHierarchySupertypesParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> TypeHierarchySupertypesParams&;
};

class TypeHierarchySubtypesParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto item() const -> TypeHierarchyItem;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto item(TypeHierarchyItem item) -> TypeHierarchySubtypesParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> TypeHierarchySubtypesParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> TypeHierarchySubtypesParams&;
};

class InlineValueParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto context() const -> InlineValueContext;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> InlineValueParams&;

  auto range(Range range) -> InlineValueParams&;

  auto context(InlineValueContext context) -> InlineValueParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> InlineValueParams&;
};

class InlineValueRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineValueRegistrationOptions&;

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> InlineValueRegistrationOptions&;

  auto id(std::optional<std::string> id) -> InlineValueRegistrationOptions&;
};

class InlayHintParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> InlayHintParams&;

  auto range(Range range) -> InlayHintParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> InlayHintParams&;
};

class InlayHint final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto label() const
      -> std::variant<std::string, Vector<InlayHintLabelPart>>;

  template <typename T>
  [[nodiscard]] auto label() -> T {
    auto& value = (*repr_)["label"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::optional<InlayHintKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  [[nodiscard]] auto textEdits() const -> std::optional<Vector<TextEdit>>;

  template <typename T>
  [[nodiscard]] auto textEdits() -> T {
    auto& value = (*repr_)["textEdits"];
    return T(value);
  }

  [[nodiscard]] auto tooltip() const
      -> std::optional<std::variant<std::string, MarkupContent>>;

  template <typename T>
  [[nodiscard]] auto tooltip() -> T {
    auto& value = (*repr_)["tooltip"];
    return T(value);
  }

  [[nodiscard]] auto paddingLeft() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto paddingLeft() -> T {
    auto& value = (*repr_)["paddingLeft"];
    return T(value);
  }

  [[nodiscard]] auto paddingRight() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto paddingRight() -> T {
    auto& value = (*repr_)["paddingRight"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto position(Position position) -> InlayHint&;

  auto label(std::variant<std::string, Vector<InlayHintLabelPart>> label)
      -> InlayHint&;

  auto kind(std::optional<InlayHintKind> kind) -> InlayHint&;

  auto textEdits(std::optional<Vector<TextEdit>> textEdits) -> InlayHint&;

  auto tooltip(std::optional<std::variant<std::string, MarkupContent>> tooltip)
      -> InlayHint&;

  auto paddingLeft(std::optional<bool> paddingLeft) -> InlayHint&;

  auto paddingRight(std::optional<bool> paddingRight) -> InlayHint&;

  auto data(std::optional<LSPAny> data) -> InlayHint&;
};

class InlayHintRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> InlayHintRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlayHintRegistrationOptions&;

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> InlayHintRegistrationOptions&;

  auto id(std::optional<std::string> id) -> InlayHintRegistrationOptions&;
};

class DocumentDiagnosticParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto identifier() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto identifier() -> T {
    auto& value = (*repr_)["identifier"];
    return T(value);
  }

  [[nodiscard]] auto previousResultId() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto previousResultId() -> T {
    auto& value = (*repr_)["previousResultId"];
    return T(value);
  }

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentDiagnosticParams&;

  auto identifier(std::optional<std::string> identifier)
      -> DocumentDiagnosticParams&;

  auto previousResultId(std::optional<std::string> previousResultId)
      -> DocumentDiagnosticParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentDiagnosticParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DocumentDiagnosticParams&;
};

class DocumentDiagnosticReportPartialResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relatedDocuments() const
      -> Map<std::string, std::variant<FullDocumentDiagnosticReport,
                                       UnchangedDocumentDiagnosticReport>>;

  auto relatedDocuments(
      Map<std::string, std::variant<FullDocumentDiagnosticReport,
                                    UnchangedDocumentDiagnosticReport>>
          relatedDocuments) -> DocumentDiagnosticReportPartialResult&;
};

class DiagnosticServerCancellationData final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto retriggerRequest() const -> bool;

  auto retriggerRequest(bool retriggerRequest)
      -> DiagnosticServerCancellationData&;
};

class DiagnosticRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto identifier() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto identifier() -> T {
    auto& value = (*repr_)["identifier"];
    return T(value);
  }

  [[nodiscard]] auto interFileDependencies() const -> bool;

  [[nodiscard]] auto workspaceDiagnostics() const -> bool;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DiagnosticRegistrationOptions&;

  auto identifier(std::optional<std::string> identifier)
      -> DiagnosticRegistrationOptions&;

  auto interFileDependencies(bool interFileDependencies)
      -> DiagnosticRegistrationOptions&;

  auto workspaceDiagnostics(bool workspaceDiagnostics)
      -> DiagnosticRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DiagnosticRegistrationOptions&;

  auto id(std::optional<std::string> id) -> DiagnosticRegistrationOptions&;
};

class WorkspaceDiagnosticParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto identifier() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto identifier() -> T {
    auto& value = (*repr_)["identifier"];
    return T(value);
  }

  [[nodiscard]] auto previousResultIds() const -> Vector<PreviousResultId>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto identifier(std::optional<std::string> identifier)
      -> WorkspaceDiagnosticParams&;

  auto previousResultIds(Vector<PreviousResultId> previousResultIds)
      -> WorkspaceDiagnosticParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> WorkspaceDiagnosticParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> WorkspaceDiagnosticParams&;
};

class WorkspaceDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto items() const -> Vector<WorkspaceDocumentDiagnosticReport>;

  auto items(Vector<WorkspaceDocumentDiagnosticReport> items)
      -> WorkspaceDiagnosticReport&;
};

class WorkspaceDiagnosticReportPartialResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto items() const -> Vector<WorkspaceDocumentDiagnosticReport>;

  auto items(Vector<WorkspaceDocumentDiagnosticReport> items)
      -> WorkspaceDiagnosticReportPartialResult&;
};

class DidOpenNotebookDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookDocument() const -> NotebookDocument;

  [[nodiscard]] auto cellTextDocuments() const -> Vector<TextDocumentItem>;

  auto notebookDocument(NotebookDocument notebookDocument)
      -> DidOpenNotebookDocumentParams&;

  auto cellTextDocuments(Vector<TextDocumentItem> cellTextDocuments)
      -> DidOpenNotebookDocumentParams&;
};

class NotebookDocumentSyncRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookSelector() const
      -> Vector<std::variant<NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>;

  [[nodiscard]] auto save() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto save() -> T {
    auto& value = (*repr_)["save"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto notebookSelector(Vector<std::variant<NotebookDocumentFilterWithNotebook,
                                            NotebookDocumentFilterWithCells>>
                            notebookSelector)
      -> NotebookDocumentSyncRegistrationOptions&;

  auto save(std::optional<bool> save)
      -> NotebookDocumentSyncRegistrationOptions&;

  auto id(std::optional<std::string> id)
      -> NotebookDocumentSyncRegistrationOptions&;
};

class DidChangeNotebookDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookDocument() const
      -> VersionedNotebookDocumentIdentifier;

  [[nodiscard]] auto change() const -> NotebookDocumentChangeEvent;

  auto notebookDocument(VersionedNotebookDocumentIdentifier notebookDocument)
      -> DidChangeNotebookDocumentParams&;

  auto change(NotebookDocumentChangeEvent change)
      -> DidChangeNotebookDocumentParams&;
};

class DidSaveNotebookDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookDocument() const -> NotebookDocumentIdentifier;

  auto notebookDocument(NotebookDocumentIdentifier notebookDocument)
      -> DidSaveNotebookDocumentParams&;
};

class DidCloseNotebookDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookDocument() const -> NotebookDocumentIdentifier;

  [[nodiscard]] auto cellTextDocuments() const
      -> Vector<TextDocumentIdentifier>;

  auto notebookDocument(NotebookDocumentIdentifier notebookDocument)
      -> DidCloseNotebookDocumentParams&;

  auto cellTextDocuments(Vector<TextDocumentIdentifier> cellTextDocuments)
      -> DidCloseNotebookDocumentParams&;
};

class InlineCompletionParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto context() const -> InlineCompletionContext;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto context(InlineCompletionContext context) -> InlineCompletionParams&;

  auto textDocument(TextDocumentIdentifier textDocument)
      -> InlineCompletionParams&;

  auto position(Position position) -> InlineCompletionParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> InlineCompletionParams&;
};

class InlineCompletionList final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto items() const -> Vector<InlineCompletionItem>;

  auto items(Vector<InlineCompletionItem> items) -> InlineCompletionList&;
};

class InlineCompletionItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto insertText() const
      -> std::variant<std::string, StringValue>;

  template <typename T>
  [[nodiscard]] auto insertText() -> T {
    auto& value = (*repr_)["insertText"];
    return T(value);
  }

  [[nodiscard]] auto filterText() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto filterText() -> T {
    auto& value = (*repr_)["filterText"];
    return T(value);
  }

  [[nodiscard]] auto range() const -> std::optional<Range>;

  template <typename T>
  [[nodiscard]] auto range() -> T {
    auto& value = (*repr_)["range"];
    return T(value);
  }

  [[nodiscard]] auto command() const -> std::optional<Command>;

  template <typename T>
  [[nodiscard]] auto command() -> T {
    auto& value = (*repr_)["command"];
    return T(value);
  }

  auto insertText(std::variant<std::string, StringValue> insertText)
      -> InlineCompletionItem&;

  auto filterText(std::optional<std::string> filterText)
      -> InlineCompletionItem&;

  auto range(std::optional<Range> range) -> InlineCompletionItem&;

  auto command(std::optional<Command> command) -> InlineCompletionItem&;
};

class InlineCompletionRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineCompletionRegistrationOptions&;

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> InlineCompletionRegistrationOptions&;

  auto id(std::optional<std::string> id)
      -> InlineCompletionRegistrationOptions&;
};

class TextDocumentContentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> TextDocumentContentParams&;
};

class TextDocumentContentResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto text() const -> std::string;

  auto text(std::string text) -> TextDocumentContentResult&;
};

class TextDocumentContentRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto schemes() const -> Vector<std::string>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto schemes(Vector<std::string> schemes)
      -> TextDocumentContentRegistrationOptions&;

  auto schemes(std::vector<std::string> schemes)
      -> TextDocumentContentRegistrationOptions& {
    auto& value = (*repr_)["schemes"];
    value = std::move(schemes);
    return *this;
  }

  auto id(std::optional<std::string> id)
      -> TextDocumentContentRegistrationOptions&;
};

class TextDocumentContentRefreshParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> TextDocumentContentRefreshParams&;
};

class RegistrationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto registrations() const -> Vector<Registration>;

  auto registrations(Vector<Registration> registrations) -> RegistrationParams&;
};

class UnregistrationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto unregisterations() const -> Vector<Unregistration>;

  auto unregisterations(Vector<Unregistration> unregisterations)
      -> UnregistrationParams&;
};

class InitializeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto processId() const -> std::variant<int, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto processId() -> T {
    auto& value = (*repr_)["processId"];
    return T(value);
  }

  [[nodiscard]] auto clientInfo() const -> std::optional<ClientInfo>;

  template <typename T>
  [[nodiscard]] auto clientInfo() -> T {
    auto& value = (*repr_)["clientInfo"];
    return T(value);
  }

  [[nodiscard]] auto locale() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto locale() -> T {
    auto& value = (*repr_)["locale"];
    return T(value);
  }

  [[nodiscard]] auto rootPath() const
      -> std::optional<std::variant<std::string, std::nullptr_t>>;

  template <typename T>
  [[nodiscard]] auto rootPath() -> T {
    auto& value = (*repr_)["rootPath"];
    return T(value);
  }

  [[nodiscard]] auto rootUri() const
      -> std::variant<std::string, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto rootUri() -> T {
    auto& value = (*repr_)["rootUri"];
    return T(value);
  }

  [[nodiscard]] auto capabilities() const -> ClientCapabilities;

  [[nodiscard]] auto initializationOptions() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto initializationOptions() -> T {
    auto& value = (*repr_)["initializationOptions"];
    return T(value);
  }

  [[nodiscard]] auto trace() const -> std::optional<TraceValue>;

  template <typename T>
  [[nodiscard]] auto trace() -> T {
    auto& value = (*repr_)["trace"];
    return T(value);
  }

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto workspaceFolders() const
      -> std::optional<std::variant<Vector<WorkspaceFolder>, std::nullptr_t>>;

  template <typename T>
  [[nodiscard]] auto workspaceFolders() -> T {
    auto& value = (*repr_)["workspaceFolders"];
    return T(value);
  }

  auto processId(std::variant<int, std::nullptr_t> processId)
      -> InitializeParams&;

  auto clientInfo(std::optional<ClientInfo> clientInfo) -> InitializeParams&;

  auto locale(std::optional<std::string> locale) -> InitializeParams&;

  auto rootPath(
      std::optional<std::variant<std::string, std::nullptr_t>> rootPath)
      -> InitializeParams&;

  auto rootUri(std::variant<std::string, std::nullptr_t> rootUri)
      -> InitializeParams&;

  auto capabilities(ClientCapabilities capabilities) -> InitializeParams&;

  auto initializationOptions(std::optional<LSPAny> initializationOptions)
      -> InitializeParams&;

  auto trace(std::optional<TraceValue> trace) -> InitializeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> InitializeParams&;

  auto workspaceFolders(
      std::optional<std::variant<Vector<WorkspaceFolder>, std::nullptr_t>>
          workspaceFolders) -> InitializeParams&;
};

class InitializeResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto capabilities() const -> ServerCapabilities;

  [[nodiscard]] auto serverInfo() const -> std::optional<ServerInfo>;

  template <typename T>
  [[nodiscard]] auto serverInfo() -> T {
    auto& value = (*repr_)["serverInfo"];
    return T(value);
  }

  auto capabilities(ServerCapabilities capabilities) -> InitializeResult&;

  auto serverInfo(std::optional<ServerInfo> serverInfo) -> InitializeResult&;
};

class InitializeError final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto retry() const -> bool;

  auto retry(bool retry) -> InitializeError&;
};

class InitializedParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;
};

class DidChangeConfigurationParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto settings() const -> LSPAny;

  auto settings(LSPAny settings) -> DidChangeConfigurationParams&;
};

class DidChangeConfigurationRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto section() const
      -> std::optional<std::variant<std::string, Vector<std::string>>>;

  template <typename T>
  [[nodiscard]] auto section() -> T {
    auto& value = (*repr_)["section"];
    return T(value);
  }

  auto section(
      std::optional<std::variant<std::string, Vector<std::string>>> section)
      -> DidChangeConfigurationRegistrationOptions&;
};

class ShowMessageParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto type() const -> MessageType;

  [[nodiscard]] auto message() const -> std::string;

  auto type(MessageType type) -> ShowMessageParams&;

  auto message(std::string message) -> ShowMessageParams&;
};

class ShowMessageRequestParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto type() const -> MessageType;

  [[nodiscard]] auto message() const -> std::string;

  [[nodiscard]] auto actions() const
      -> std::optional<Vector<MessageActionItem>>;

  template <typename T>
  [[nodiscard]] auto actions() -> T {
    auto& value = (*repr_)["actions"];
    return T(value);
  }

  auto type(MessageType type) -> ShowMessageRequestParams&;

  auto message(std::string message) -> ShowMessageRequestParams&;

  auto actions(std::optional<Vector<MessageActionItem>> actions)
      -> ShowMessageRequestParams&;
};

class MessageActionItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto title() const -> std::string;

  auto title(std::string title) -> MessageActionItem&;
};

class LogMessageParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto type() const -> MessageType;

  [[nodiscard]] auto message() const -> std::string;

  auto type(MessageType type) -> LogMessageParams&;

  auto message(std::string message) -> LogMessageParams&;
};

class DidOpenTextDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentItem;

  auto textDocument(TextDocumentItem textDocument)
      -> DidOpenTextDocumentParams&;
};

class DidChangeTextDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> VersionedTextDocumentIdentifier;

  [[nodiscard]] auto contentChanges() const
      -> Vector<TextDocumentContentChangeEvent>;

  auto textDocument(VersionedTextDocumentIdentifier textDocument)
      -> DidChangeTextDocumentParams&;

  auto contentChanges(Vector<TextDocumentContentChangeEvent> contentChanges)
      -> DidChangeTextDocumentParams&;
};

class TextDocumentChangeRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto syncKind() const -> TextDocumentSyncKind;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  auto syncKind(TextDocumentSyncKind syncKind)
      -> TextDocumentChangeRegistrationOptions&;

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> TextDocumentChangeRegistrationOptions&;
};

class DidCloseTextDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DidCloseTextDocumentParams&;
};

class DidSaveTextDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto text() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto text() -> T {
    auto& value = (*repr_)["text"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DidSaveTextDocumentParams&;

  auto text(std::optional<std::string> text) -> DidSaveTextDocumentParams&;
};

class TextDocumentSaveRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto includeText() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto includeText() -> T {
    auto& value = (*repr_)["includeText"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> TextDocumentSaveRegistrationOptions&;

  auto includeText(std::optional<bool> includeText)
      -> TextDocumentSaveRegistrationOptions&;
};

class WillSaveTextDocumentParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto reason() const -> TextDocumentSaveReason;

  auto textDocument(TextDocumentIdentifier textDocument)
      -> WillSaveTextDocumentParams&;

  auto reason(TextDocumentSaveReason reason) -> WillSaveTextDocumentParams&;
};

class TextEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto newText() const -> std::string;

  auto range(Range range) -> TextEdit&;

  auto newText(std::string newText) -> TextEdit&;
};

class DidChangeWatchedFilesParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto changes() const -> Vector<FileEvent>;

  auto changes(Vector<FileEvent> changes) -> DidChangeWatchedFilesParams&;
};

class DidChangeWatchedFilesRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto watchers() const -> Vector<FileSystemWatcher>;

  auto watchers(Vector<FileSystemWatcher> watchers)
      -> DidChangeWatchedFilesRegistrationOptions&;
};

class PublishDiagnosticsParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto version() const -> std::optional<int>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  [[nodiscard]] auto diagnostics() const -> Vector<Diagnostic>;

  auto uri(std::string uri) -> PublishDiagnosticsParams&;

  auto version(std::optional<int> version) -> PublishDiagnosticsParams&;

  auto diagnostics(Vector<Diagnostic> diagnostics) -> PublishDiagnosticsParams&;
};

class CompletionParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto context() const -> std::optional<CompletionContext>;

  template <typename T>
  [[nodiscard]] auto context() -> T {
    auto& value = (*repr_)["context"];
    return T(value);
  }

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto context(std::optional<CompletionContext> context) -> CompletionParams&;

  auto textDocument(TextDocumentIdentifier textDocument) -> CompletionParams&;

  auto position(Position position) -> CompletionParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> CompletionParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> CompletionParams&;
};

class CompletionItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::string;

  [[nodiscard]] auto labelDetails() const
      -> std::optional<CompletionItemLabelDetails>;

  template <typename T>
  [[nodiscard]] auto labelDetails() -> T {
    auto& value = (*repr_)["labelDetails"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::optional<CompletionItemKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  [[nodiscard]] auto tags() const -> std::optional<Vector<CompletionItemTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto detail() -> T {
    auto& value = (*repr_)["detail"];
    return T(value);
  }

  [[nodiscard]] auto documentation() const
      -> std::optional<std::variant<std::string, MarkupContent>>;

  template <typename T>
  [[nodiscard]] auto documentation() -> T {
    auto& value = (*repr_)["documentation"];
    return T(value);
  }

  [[nodiscard]] auto deprecated() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto deprecated() -> T {
    auto& value = (*repr_)["deprecated"];
    return T(value);
  }

  [[nodiscard]] auto preselect() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto preselect() -> T {
    auto& value = (*repr_)["preselect"];
    return T(value);
  }

  [[nodiscard]] auto sortText() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto sortText() -> T {
    auto& value = (*repr_)["sortText"];
    return T(value);
  }

  [[nodiscard]] auto filterText() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto filterText() -> T {
    auto& value = (*repr_)["filterText"];
    return T(value);
  }

  [[nodiscard]] auto insertText() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto insertText() -> T {
    auto& value = (*repr_)["insertText"];
    return T(value);
  }

  [[nodiscard]] auto insertTextFormat() const
      -> std::optional<InsertTextFormat>;

  template <typename T>
  [[nodiscard]] auto insertTextFormat() -> T {
    auto& value = (*repr_)["insertTextFormat"];
    return T(value);
  }

  [[nodiscard]] auto insertTextMode() const -> std::optional<InsertTextMode>;

  template <typename T>
  [[nodiscard]] auto insertTextMode() -> T {
    auto& value = (*repr_)["insertTextMode"];
    return T(value);
  }

  [[nodiscard]] auto textEdit() const
      -> std::optional<std::variant<TextEdit, InsertReplaceEdit>>;

  template <typename T>
  [[nodiscard]] auto textEdit() -> T {
    auto& value = (*repr_)["textEdit"];
    return T(value);
  }

  [[nodiscard]] auto textEditText() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto textEditText() -> T {
    auto& value = (*repr_)["textEditText"];
    return T(value);
  }

  [[nodiscard]] auto additionalTextEdits() const
      -> std::optional<Vector<TextEdit>>;

  template <typename T>
  [[nodiscard]] auto additionalTextEdits() -> T {
    auto& value = (*repr_)["additionalTextEdits"];
    return T(value);
  }

  [[nodiscard]] auto commitCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto commitCharacters() -> T {
    auto& value = (*repr_)["commitCharacters"];
    return T(value);
  }

  [[nodiscard]] auto command() const -> std::optional<Command>;

  template <typename T>
  [[nodiscard]] auto command() -> T {
    auto& value = (*repr_)["command"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto label(std::string label) -> CompletionItem&;

  auto labelDetails(std::optional<CompletionItemLabelDetails> labelDetails)
      -> CompletionItem&;

  auto kind(std::optional<CompletionItemKind> kind) -> CompletionItem&;

  auto tags(std::optional<Vector<CompletionItemTag>> tags) -> CompletionItem&;

  auto detail(std::optional<std::string> detail) -> CompletionItem&;

  auto documentation(
      std::optional<std::variant<std::string, MarkupContent>> documentation)
      -> CompletionItem&;

  auto deprecated(std::optional<bool> deprecated) -> CompletionItem&;

  auto preselect(std::optional<bool> preselect) -> CompletionItem&;

  auto sortText(std::optional<std::string> sortText) -> CompletionItem&;

  auto filterText(std::optional<std::string> filterText) -> CompletionItem&;

  auto insertText(std::optional<std::string> insertText) -> CompletionItem&;

  auto insertTextFormat(std::optional<InsertTextFormat> insertTextFormat)
      -> CompletionItem&;

  auto insertTextMode(std::optional<InsertTextMode> insertTextMode)
      -> CompletionItem&;

  auto textEdit(
      std::optional<std::variant<TextEdit, InsertReplaceEdit>> textEdit)
      -> CompletionItem&;

  auto textEditText(std::optional<std::string> textEditText) -> CompletionItem&;

  auto additionalTextEdits(std::optional<Vector<TextEdit>> additionalTextEdits)
      -> CompletionItem&;

  auto commitCharacters(std::optional<Vector<std::string>> commitCharacters)
      -> CompletionItem&;

  auto commitCharacters(std::vector<std::string> commitCharacters)
      -> CompletionItem& {
    auto& value = (*repr_)["commitCharacters"];
    value = std::move(commitCharacters);
    return *this;
  }

  auto command(std::optional<Command> command) -> CompletionItem&;

  auto data(std::optional<LSPAny> data) -> CompletionItem&;
};

class CompletionList final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto isIncomplete() const -> bool;

  [[nodiscard]] auto itemDefaults() const
      -> std::optional<CompletionItemDefaults>;

  template <typename T>
  [[nodiscard]] auto itemDefaults() -> T {
    auto& value = (*repr_)["itemDefaults"];
    return T(value);
  }

  [[nodiscard]] auto applyKind() const
      -> std::optional<CompletionItemApplyKinds>;

  template <typename T>
  [[nodiscard]] auto applyKind() -> T {
    auto& value = (*repr_)["applyKind"];
    return T(value);
  }

  [[nodiscard]] auto items() const -> Vector<CompletionItem>;

  auto isIncomplete(bool isIncomplete) -> CompletionList&;

  auto itemDefaults(std::optional<CompletionItemDefaults> itemDefaults)
      -> CompletionList&;

  auto applyKind(std::optional<CompletionItemApplyKinds> applyKind)
      -> CompletionList&;

  auto items(Vector<CompletionItem> items) -> CompletionList&;
};

class CompletionRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto triggerCharacters() -> T {
    auto& value = (*repr_)["triggerCharacters"];
    return T(value);
  }

  [[nodiscard]] auto allCommitCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto allCommitCharacters() -> T {
    auto& value = (*repr_)["allCommitCharacters"];
    return T(value);
  }

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto completionItem() const
      -> std::optional<ServerCompletionItemOptions>;

  template <typename T>
  [[nodiscard]] auto completionItem() -> T {
    auto& value = (*repr_)["completionItem"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> CompletionRegistrationOptions&;

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> CompletionRegistrationOptions&;

  auto triggerCharacters(std::vector<std::string> triggerCharacters)
      -> CompletionRegistrationOptions& {
    auto& value = (*repr_)["triggerCharacters"];
    value = std::move(triggerCharacters);
    return *this;
  }

  auto allCommitCharacters(
      std::optional<Vector<std::string>> allCommitCharacters)
      -> CompletionRegistrationOptions&;

  auto allCommitCharacters(std::vector<std::string> allCommitCharacters)
      -> CompletionRegistrationOptions& {
    auto& value = (*repr_)["allCommitCharacters"];
    value = std::move(allCommitCharacters);
    return *this;
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> CompletionRegistrationOptions&;

  auto completionItem(std::optional<ServerCompletionItemOptions> completionItem)
      -> CompletionRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CompletionRegistrationOptions&;
};

class HoverParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> HoverParams&;

  auto position(Position position) -> HoverParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> HoverParams&;
};

class Hover final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto contents() const
      -> std::variant<MarkupContent, MarkedString, Vector<MarkedString>>;

  template <typename T>
  [[nodiscard]] auto contents() -> T {
    auto& value = (*repr_)["contents"];
    return T(value);
  }

  [[nodiscard]] auto range() const -> std::optional<Range>;

  template <typename T>
  [[nodiscard]] auto range() -> T {
    auto& value = (*repr_)["range"];
    return T(value);
  }

  auto contents(
      std::variant<MarkupContent, MarkedString, Vector<MarkedString>> contents)
      -> Hover&;

  auto range(std::optional<Range> range) -> Hover&;
};

class HoverRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> HoverRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> HoverRegistrationOptions&;
};

class SignatureHelpParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto context() const -> std::optional<SignatureHelpContext>;

  template <typename T>
  [[nodiscard]] auto context() -> T {
    auto& value = (*repr_)["context"];
    return T(value);
  }

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto context(std::optional<SignatureHelpContext> context)
      -> SignatureHelpParams&;

  auto textDocument(TextDocumentIdentifier textDocument)
      -> SignatureHelpParams&;

  auto position(Position position) -> SignatureHelpParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> SignatureHelpParams&;
};

class SignatureHelp final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto signatures() const -> Vector<SignatureInformation>;

  [[nodiscard]] auto activeSignature() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto activeSignature() -> T {
    auto& value = (*repr_)["activeSignature"];
    return T(value);
  }

  [[nodiscard]] auto activeParameter() const
      -> std::optional<std::variant<long, std::nullptr_t>>;

  template <typename T>
  [[nodiscard]] auto activeParameter() -> T {
    auto& value = (*repr_)["activeParameter"];
    return T(value);
  }

  auto signatures(Vector<SignatureInformation> signatures) -> SignatureHelp&;

  auto activeSignature(std::optional<long> activeSignature) -> SignatureHelp&;

  auto activeParameter(
      std::optional<std::variant<long, std::nullptr_t>> activeParameter)
      -> SignatureHelp&;
};

class SignatureHelpRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto triggerCharacters() -> T {
    auto& value = (*repr_)["triggerCharacters"];
    return T(value);
  }

  [[nodiscard]] auto retriggerCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto retriggerCharacters() -> T {
    auto& value = (*repr_)["retriggerCharacters"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> SignatureHelpRegistrationOptions&;

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> SignatureHelpRegistrationOptions&;

  auto triggerCharacters(std::vector<std::string> triggerCharacters)
      -> SignatureHelpRegistrationOptions& {
    auto& value = (*repr_)["triggerCharacters"];
    value = std::move(triggerCharacters);
    return *this;
  }

  auto retriggerCharacters(
      std::optional<Vector<std::string>> retriggerCharacters)
      -> SignatureHelpRegistrationOptions&;

  auto retriggerCharacters(std::vector<std::string> retriggerCharacters)
      -> SignatureHelpRegistrationOptions& {
    auto& value = (*repr_)["retriggerCharacters"];
    value = std::move(retriggerCharacters);
    return *this;
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SignatureHelpRegistrationOptions&;
};

class DefinitionParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> DefinitionParams&;

  auto position(Position position) -> DefinitionParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DefinitionParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DefinitionParams&;
};

class DefinitionRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DefinitionRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DefinitionRegistrationOptions&;
};

class ReferenceParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto context() const -> ReferenceContext;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto context(ReferenceContext context) -> ReferenceParams&;

  auto textDocument(TextDocumentIdentifier textDocument) -> ReferenceParams&;

  auto position(Position position) -> ReferenceParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> ReferenceParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> ReferenceParams&;
};

class ReferenceRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> ReferenceRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ReferenceRegistrationOptions&;
};

class DocumentHighlightParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentHighlightParams&;

  auto position(Position position) -> DocumentHighlightParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentHighlightParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DocumentHighlightParams&;
};

class DocumentHighlight final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto kind() const -> std::optional<DocumentHighlightKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  auto range(Range range) -> DocumentHighlight&;

  auto kind(std::optional<DocumentHighlightKind> kind) -> DocumentHighlight&;
};

class DocumentHighlightRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentHighlightRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentHighlightRegistrationOptions&;
};

class DocumentSymbolParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentSymbolParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentSymbolParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DocumentSymbolParams&;
};

class SymbolInformation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto deprecated() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto deprecated() -> T {
    auto& value = (*repr_)["deprecated"];
    return T(value);
  }

  [[nodiscard]] auto location() const -> Location;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto containerName() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto containerName() -> T {
    auto& value = (*repr_)["containerName"];
    return T(value);
  }

  auto deprecated(std::optional<bool> deprecated) -> SymbolInformation&;

  auto location(Location location) -> SymbolInformation&;

  auto name(std::string name) -> SymbolInformation&;

  auto kind(SymbolKind kind) -> SymbolInformation&;

  auto tags(std::optional<Vector<SymbolTag>> tags) -> SymbolInformation&;

  auto containerName(std::optional<std::string> containerName)
      -> SymbolInformation&;
};

class DocumentSymbol final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto detail() -> T {
    auto& value = (*repr_)["detail"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto deprecated() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto deprecated() -> T {
    auto& value = (*repr_)["deprecated"];
    return T(value);
  }

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto selectionRange() const -> Range;

  [[nodiscard]] auto children() const -> std::optional<Vector<DocumentSymbol>>;

  template <typename T>
  [[nodiscard]] auto children() -> T {
    auto& value = (*repr_)["children"];
    return T(value);
  }

  auto name(std::string name) -> DocumentSymbol&;

  auto detail(std::optional<std::string> detail) -> DocumentSymbol&;

  auto kind(SymbolKind kind) -> DocumentSymbol&;

  auto tags(std::optional<Vector<SymbolTag>> tags) -> DocumentSymbol&;

  auto deprecated(std::optional<bool> deprecated) -> DocumentSymbol&;

  auto range(Range range) -> DocumentSymbol&;

  auto selectionRange(Range selectionRange) -> DocumentSymbol&;

  auto children(std::optional<Vector<DocumentSymbol>> children)
      -> DocumentSymbol&;
};

class DocumentSymbolRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto label() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto label() -> T {
    auto& value = (*repr_)["label"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentSymbolRegistrationOptions&;

  auto label(std::optional<std::string> label)
      -> DocumentSymbolRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentSymbolRegistrationOptions&;
};

class CodeActionParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto context() const -> CodeActionContext;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> CodeActionParams&;

  auto range(Range range) -> CodeActionParams&;

  auto context(CodeActionContext context) -> CodeActionParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> CodeActionParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> CodeActionParams&;
};

class Command final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto title() const -> std::string;

  [[nodiscard]] auto tooltip() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto tooltip() -> T {
    auto& value = (*repr_)["tooltip"];
    return T(value);
  }

  [[nodiscard]] auto command() const -> std::string;

  [[nodiscard]] auto arguments() const -> std::optional<Vector<LSPAny>>;

  template <typename T>
  [[nodiscard]] auto arguments() -> T {
    auto& value = (*repr_)["arguments"];
    return T(value);
  }

  auto title(std::string title) -> Command&;

  auto tooltip(std::optional<std::string> tooltip) -> Command&;

  auto command(std::string command) -> Command&;

  auto arguments(std::optional<Vector<LSPAny>> arguments) -> Command&;
};

class CodeAction final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto title() const -> std::string;

  [[nodiscard]] auto kind() const -> std::optional<CodeActionKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  [[nodiscard]] auto diagnostics() const -> std::optional<Vector<Diagnostic>>;

  template <typename T>
  [[nodiscard]] auto diagnostics() -> T {
    auto& value = (*repr_)["diagnostics"];
    return T(value);
  }

  [[nodiscard]] auto isPreferred() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto isPreferred() -> T {
    auto& value = (*repr_)["isPreferred"];
    return T(value);
  }

  [[nodiscard]] auto disabled() const -> std::optional<CodeActionDisabled>;

  template <typename T>
  [[nodiscard]] auto disabled() -> T {
    auto& value = (*repr_)["disabled"];
    return T(value);
  }

  [[nodiscard]] auto edit() const -> std::optional<WorkspaceEdit>;

  template <typename T>
  [[nodiscard]] auto edit() -> T {
    auto& value = (*repr_)["edit"];
    return T(value);
  }

  [[nodiscard]] auto command() const -> std::optional<Command>;

  template <typename T>
  [[nodiscard]] auto command() -> T {
    auto& value = (*repr_)["command"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  [[nodiscard]] auto tags() const -> std::optional<Vector<CodeActionTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  auto title(std::string title) -> CodeAction&;

  auto kind(std::optional<CodeActionKind> kind) -> CodeAction&;

  auto diagnostics(std::optional<Vector<Diagnostic>> diagnostics)
      -> CodeAction&;

  auto isPreferred(std::optional<bool> isPreferred) -> CodeAction&;

  auto disabled(std::optional<CodeActionDisabled> disabled) -> CodeAction&;

  auto edit(std::optional<WorkspaceEdit> edit) -> CodeAction&;

  auto command(std::optional<Command> command) -> CodeAction&;

  auto data(std::optional<LSPAny> data) -> CodeAction&;

  auto tags(std::optional<Vector<CodeActionTag>> tags) -> CodeAction&;
};

class CodeActionRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto codeActionKinds() const
      -> std::optional<Vector<CodeActionKind>>;

  template <typename T>
  [[nodiscard]] auto codeActionKinds() -> T {
    auto& value = (*repr_)["codeActionKinds"];
    return T(value);
  }

  [[nodiscard]] auto documentation() const
      -> std::optional<Vector<CodeActionKindDocumentation>>;

  template <typename T>
  [[nodiscard]] auto documentation() -> T {
    auto& value = (*repr_)["documentation"];
    return T(value);
  }

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> CodeActionRegistrationOptions&;

  auto codeActionKinds(std::optional<Vector<CodeActionKind>> codeActionKinds)
      -> CodeActionRegistrationOptions&;

  auto documentation(
      std::optional<Vector<CodeActionKindDocumentation>> documentation)
      -> CodeActionRegistrationOptions&;

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> CodeActionRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CodeActionRegistrationOptions&;
};

class WorkspaceSymbolParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto query() const -> std::string;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto query(std::string query) -> WorkspaceSymbolParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> WorkspaceSymbolParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> WorkspaceSymbolParams&;
};

class WorkspaceSymbol final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto location() const
      -> std::variant<Location, LocationUriOnly>;

  template <typename T>
  [[nodiscard]] auto location() -> T {
    auto& value = (*repr_)["location"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto containerName() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto containerName() -> T {
    auto& value = (*repr_)["containerName"];
    return T(value);
  }

  auto location(std::variant<Location, LocationUriOnly> location)
      -> WorkspaceSymbol&;

  auto data(std::optional<LSPAny> data) -> WorkspaceSymbol&;

  auto name(std::string name) -> WorkspaceSymbol&;

  auto kind(SymbolKind kind) -> WorkspaceSymbol&;

  auto tags(std::optional<Vector<SymbolTag>> tags) -> WorkspaceSymbol&;

  auto containerName(std::optional<std::string> containerName)
      -> WorkspaceSymbol&;
};

class WorkspaceSymbolRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> WorkspaceSymbolRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> WorkspaceSymbolRegistrationOptions&;
};

class CodeLensParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> CodeLensParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> CodeLensParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> CodeLensParams&;
};

class CodeLens final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto command() const -> std::optional<Command>;

  template <typename T>
  [[nodiscard]] auto command() -> T {
    auto& value = (*repr_)["command"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto range(Range range) -> CodeLens&;

  auto command(std::optional<Command> command) -> CodeLens&;

  auto data(std::optional<LSPAny> data) -> CodeLens&;
};

class CodeLensRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> CodeLensRegistrationOptions&;

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> CodeLensRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CodeLensRegistrationOptions&;
};

class DocumentLinkParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> DocumentLinkParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentLinkParams&;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> DocumentLinkParams&;
};

class DocumentLink final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto target() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto target() -> T {
    auto& value = (*repr_)["target"];
    return T(value);
  }

  [[nodiscard]] auto tooltip() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto tooltip() -> T {
    auto& value = (*repr_)["tooltip"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto range(Range range) -> DocumentLink&;

  auto target(std::optional<std::string> target) -> DocumentLink&;

  auto tooltip(std::optional<std::string> tooltip) -> DocumentLink&;

  auto data(std::optional<LSPAny> data) -> DocumentLink&;
};

class DocumentLinkRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentLinkRegistrationOptions&;

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> DocumentLinkRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentLinkRegistrationOptions&;
};

class DocumentFormattingParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto options() const -> FormattingOptions;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentFormattingParams&;

  auto options(FormattingOptions options) -> DocumentFormattingParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentFormattingParams&;
};

class DocumentFormattingRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentFormattingRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentFormattingRegistrationOptions&;
};

class DocumentRangeFormattingParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto options() const -> FormattingOptions;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentRangeFormattingParams&;

  auto range(Range range) -> DocumentRangeFormattingParams&;

  auto options(FormattingOptions options) -> DocumentRangeFormattingParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentRangeFormattingParams&;
};

class DocumentRangeFormattingRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto rangesSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto rangesSupport() -> T {
    auto& value = (*repr_)["rangesSupport"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentRangeFormattingRegistrationOptions&;

  auto rangesSupport(std::optional<bool> rangesSupport)
      -> DocumentRangeFormattingRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentRangeFormattingRegistrationOptions&;
};

class DocumentRangesFormattingParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto ranges() const -> Vector<Range>;

  [[nodiscard]] auto options() const -> FormattingOptions;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentRangesFormattingParams&;

  auto ranges(Vector<Range> ranges) -> DocumentRangesFormattingParams&;

  auto options(FormattingOptions options) -> DocumentRangesFormattingParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> DocumentRangesFormattingParams&;
};

class DocumentOnTypeFormattingParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto ch() const -> std::string;

  [[nodiscard]] auto options() const -> FormattingOptions;

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DocumentOnTypeFormattingParams&;

  auto position(Position position) -> DocumentOnTypeFormattingParams&;

  auto ch(std::string ch) -> DocumentOnTypeFormattingParams&;

  auto options(FormattingOptions options) -> DocumentOnTypeFormattingParams&;
};

class DocumentOnTypeFormattingRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto firstTriggerCharacter() const -> std::string;

  [[nodiscard]] auto moreTriggerCharacter() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto moreTriggerCharacter() -> T {
    auto& value = (*repr_)["moreTriggerCharacter"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> DocumentOnTypeFormattingRegistrationOptions&;

  auto firstTriggerCharacter(std::string firstTriggerCharacter)
      -> DocumentOnTypeFormattingRegistrationOptions&;

  auto moreTriggerCharacter(
      std::optional<Vector<std::string>> moreTriggerCharacter)
      -> DocumentOnTypeFormattingRegistrationOptions&;

  auto moreTriggerCharacter(std::vector<std::string> moreTriggerCharacter)
      -> DocumentOnTypeFormattingRegistrationOptions& {
    auto& value = (*repr_)["moreTriggerCharacter"];
    value = std::move(moreTriggerCharacter);
    return *this;
  }
};

class RenameParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto newName() const -> std::string;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument) -> RenameParams&;

  auto position(Position position) -> RenameParams&;

  auto newName(std::string newName) -> RenameParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> RenameParams&;
};

class RenameRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<DocumentSelector, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto documentSelector() -> T {
    auto& value = (*repr_)["documentSelector"];
    return T(value);
  }

  [[nodiscard]] auto prepareProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto prepareProvider() -> T {
    auto& value = (*repr_)["prepareProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto documentSelector(
      std::variant<DocumentSelector, std::nullptr_t> documentSelector)
      -> RenameRegistrationOptions&;

  auto prepareProvider(std::optional<bool> prepareProvider)
      -> RenameRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> RenameRegistrationOptions&;
};

class PrepareRenameParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto textDocument(TextDocumentIdentifier textDocument)
      -> PrepareRenameParams&;

  auto position(Position position) -> PrepareRenameParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> PrepareRenameParams&;
};

class ExecuteCommandParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto command() const -> std::string;

  [[nodiscard]] auto arguments() const -> std::optional<Vector<LSPAny>>;

  template <typename T>
  [[nodiscard]] auto arguments() -> T {
    auto& value = (*repr_)["arguments"];
    return T(value);
  }

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto command(std::string command) -> ExecuteCommandParams&;

  auto arguments(std::optional<Vector<LSPAny>> arguments)
      -> ExecuteCommandParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> ExecuteCommandParams&;
};

class ExecuteCommandRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto commands() const -> Vector<std::string>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto commands(Vector<std::string> commands)
      -> ExecuteCommandRegistrationOptions&;

  auto commands(std::vector<std::string> commands)
      -> ExecuteCommandRegistrationOptions& {
    auto& value = (*repr_)["commands"];
    value = std::move(commands);
    return *this;
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ExecuteCommandRegistrationOptions&;
};

class ApplyWorkspaceEditParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto label() -> T {
    auto& value = (*repr_)["label"];
    return T(value);
  }

  [[nodiscard]] auto edit() const -> WorkspaceEdit;

  [[nodiscard]] auto metadata() const -> std::optional<WorkspaceEditMetadata>;

  template <typename T>
  [[nodiscard]] auto metadata() -> T {
    auto& value = (*repr_)["metadata"];
    return T(value);
  }

  auto label(std::optional<std::string> label) -> ApplyWorkspaceEditParams&;

  auto edit(WorkspaceEdit edit) -> ApplyWorkspaceEditParams&;

  auto metadata(std::optional<WorkspaceEditMetadata> metadata)
      -> ApplyWorkspaceEditParams&;
};

class ApplyWorkspaceEditResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto applied() const -> bool;

  [[nodiscard]] auto failureReason() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto failureReason() -> T {
    auto& value = (*repr_)["failureReason"];
    return T(value);
  }

  [[nodiscard]] auto failedChange() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto failedChange() -> T {
    auto& value = (*repr_)["failedChange"];
    return T(value);
  }

  auto applied(bool applied) -> ApplyWorkspaceEditResult&;

  auto failureReason(std::optional<std::string> failureReason)
      -> ApplyWorkspaceEditResult&;

  auto failedChange(std::optional<long> failedChange)
      -> ApplyWorkspaceEditResult&;
};

class WorkDoneProgressBegin final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto title() const -> std::string;

  [[nodiscard]] auto cancellable() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto cancellable() -> T {
    auto& value = (*repr_)["cancellable"];
    return T(value);
  }

  [[nodiscard]] auto message() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto message() -> T {
    auto& value = (*repr_)["message"];
    return T(value);
  }

  [[nodiscard]] auto percentage() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto percentage() -> T {
    auto& value = (*repr_)["percentage"];
    return T(value);
  }

  auto kind(std::string kind) -> WorkDoneProgressBegin&;

  auto title(std::string title) -> WorkDoneProgressBegin&;

  auto cancellable(std::optional<bool> cancellable) -> WorkDoneProgressBegin&;

  auto message(std::optional<std::string> message) -> WorkDoneProgressBegin&;

  auto percentage(std::optional<long> percentage) -> WorkDoneProgressBegin&;
};

class WorkDoneProgressReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto cancellable() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto cancellable() -> T {
    auto& value = (*repr_)["cancellable"];
    return T(value);
  }

  [[nodiscard]] auto message() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto message() -> T {
    auto& value = (*repr_)["message"];
    return T(value);
  }

  [[nodiscard]] auto percentage() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto percentage() -> T {
    auto& value = (*repr_)["percentage"];
    return T(value);
  }

  auto kind(std::string kind) -> WorkDoneProgressReport&;

  auto cancellable(std::optional<bool> cancellable) -> WorkDoneProgressReport&;

  auto message(std::optional<std::string> message) -> WorkDoneProgressReport&;

  auto percentage(std::optional<long> percentage) -> WorkDoneProgressReport&;
};

class WorkDoneProgressEnd final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto message() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto message() -> T {
    auto& value = (*repr_)["message"];
    return T(value);
  }

  auto kind(std::string kind) -> WorkDoneProgressEnd&;

  auto message(std::optional<std::string> message) -> WorkDoneProgressEnd&;
};

class SetTraceParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto value() const -> TraceValue;

  auto value(TraceValue value) -> SetTraceParams&;
};

class LogTraceParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto message() const -> std::string;

  [[nodiscard]] auto verbose() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto verbose() -> T {
    auto& value = (*repr_)["verbose"];
    return T(value);
  }

  auto message(std::string message) -> LogTraceParams&;

  auto verbose(std::optional<std::string> verbose) -> LogTraceParams&;
};

class CancelParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const -> std::variant<int, std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto id(std::variant<int, std::string> id) -> CancelParams&;
};

class ProgressParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto token() const -> ProgressToken;

  [[nodiscard]] auto value() const -> LSPAny;

  auto token(ProgressToken token) -> ProgressParams&;

  auto value(LSPAny value) -> ProgressParams&;
};

class TextDocumentPositionParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  auto textDocument(TextDocumentIdentifier textDocument)
      -> TextDocumentPositionParams&;

  auto position(Position position) -> TextDocumentPositionParams&;
};

class WorkDoneProgressParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> WorkDoneProgressParams&;
};

class PartialResultParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto partialResultToken() -> T {
    auto& value = (*repr_)["partialResultToken"];
    return T(value);
  }

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> PartialResultParams&;
};

class LocationLink final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto originSelectionRange() const -> std::optional<Range>;

  template <typename T>
  [[nodiscard]] auto originSelectionRange() -> T {
    auto& value = (*repr_)["originSelectionRange"];
    return T(value);
  }

  [[nodiscard]] auto targetUri() const -> std::string;

  [[nodiscard]] auto targetRange() const -> Range;

  [[nodiscard]] auto targetSelectionRange() const -> Range;

  auto originSelectionRange(std::optional<Range> originSelectionRange)
      -> LocationLink&;

  auto targetUri(std::string targetUri) -> LocationLink&;

  auto targetRange(Range targetRange) -> LocationLink&;

  auto targetSelectionRange(Range targetSelectionRange) -> LocationLink&;
};

class Range final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto start() const -> Position;

  [[nodiscard]] auto end() const -> Position;

  auto start(Position start) -> Range&;

  auto end(Position end) -> Range&;
};

class ImplementationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ImplementationOptions&;
};

class StaticRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto id() -> T {
    auto& value = (*repr_)["id"];
    return T(value);
  }

  auto id(std::optional<std::string> id) -> StaticRegistrationOptions&;
};

class TypeDefinitionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> TypeDefinitionOptions&;
};

class WorkspaceFoldersChangeEvent final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto added() const -> Vector<WorkspaceFolder>;

  [[nodiscard]] auto removed() const -> Vector<WorkspaceFolder>;

  auto added(Vector<WorkspaceFolder> added) -> WorkspaceFoldersChangeEvent&;

  auto removed(Vector<WorkspaceFolder> removed) -> WorkspaceFoldersChangeEvent&;
};

class ConfigurationItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto scopeUri() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto scopeUri() -> T {
    auto& value = (*repr_)["scopeUri"];
    return T(value);
  }

  [[nodiscard]] auto section() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto section() -> T {
    auto& value = (*repr_)["section"];
    return T(value);
  }

  auto scopeUri(std::optional<std::string> scopeUri) -> ConfigurationItem&;

  auto section(std::optional<std::string> section) -> ConfigurationItem&;
};

class TextDocumentIdentifier final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> TextDocumentIdentifier&;
};

class Color final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto red() const -> double;

  [[nodiscard]] auto green() const -> double;

  [[nodiscard]] auto blue() const -> double;

  [[nodiscard]] auto alpha() const -> double;

  auto red(double red) -> Color&;

  auto green(double green) -> Color&;

  auto blue(double blue) -> Color&;

  auto alpha(double alpha) -> Color&;
};

class DocumentColorOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentColorOptions&;
};

class FoldingRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> FoldingRangeOptions&;
};

class DeclarationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DeclarationOptions&;
};

class Position final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto line() const -> long;

  [[nodiscard]] auto character() const -> long;

  auto line(long line) -> Position&;

  auto character(long character) -> Position&;
};

class SelectionRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SelectionRangeOptions&;
};

class CallHierarchyOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CallHierarchyOptions&;
};

class SemanticTokensOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto legend() const -> SemanticTokensLegend;

  [[nodiscard]] auto range() const -> std::optional<std::variant<bool, json>>;

  template <typename T>
  [[nodiscard]] auto range() -> T {
    auto& value = (*repr_)["range"];
    return T(value);
  }

  [[nodiscard]] auto full() const
      -> std::optional<std::variant<bool, SemanticTokensFullDelta>>;

  template <typename T>
  [[nodiscard]] auto full() -> T {
    auto& value = (*repr_)["full"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto legend(SemanticTokensLegend legend) -> SemanticTokensOptions&;

  auto range(std::optional<std::variant<bool, json>> range)
      -> SemanticTokensOptions&;

  auto full(std::optional<std::variant<bool, SemanticTokensFullDelta>> full)
      -> SemanticTokensOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SemanticTokensOptions&;
};

class SemanticTokensEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto start() const -> long;

  [[nodiscard]] auto deleteCount() const -> long;

  [[nodiscard]] auto data() const -> std::optional<Vector<long>>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto start(long start) -> SemanticTokensEdit&;

  auto deleteCount(long deleteCount) -> SemanticTokensEdit&;

  auto data(std::optional<Vector<long>> data) -> SemanticTokensEdit&;
};

class LinkedEditingRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> LinkedEditingRangeOptions&;
};

class FileCreate final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> FileCreate&;
};

class TextDocumentEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const
      -> OptionalVersionedTextDocumentIdentifier;

  [[nodiscard]] auto edits() const
      -> Vector<std::variant<TextEdit, AnnotatedTextEdit, SnippetTextEdit>>;

  auto textDocument(OptionalVersionedTextDocumentIdentifier textDocument)
      -> TextDocumentEdit&;

  auto edits(
      Vector<std::variant<TextEdit, AnnotatedTextEdit, SnippetTextEdit>> edits)
      -> TextDocumentEdit&;
};

class CreateFile final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto options() const -> std::optional<CreateFileOptions>;

  template <typename T>
  [[nodiscard]] auto options() -> T {
    auto& value = (*repr_)["options"];
    return T(value);
  }

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

  template <typename T>
  [[nodiscard]] auto annotationId() -> T {
    auto& value = (*repr_)["annotationId"];
    return T(value);
  }

  auto kind(std::string kind) -> CreateFile&;

  auto uri(std::string uri) -> CreateFile&;

  auto options(std::optional<CreateFileOptions> options) -> CreateFile&;

  auto annotationId(std::optional<ChangeAnnotationIdentifier> annotationId)
      -> CreateFile&;
};

class RenameFile final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto oldUri() const -> std::string;

  [[nodiscard]] auto newUri() const -> std::string;

  [[nodiscard]] auto options() const -> std::optional<RenameFileOptions>;

  template <typename T>
  [[nodiscard]] auto options() -> T {
    auto& value = (*repr_)["options"];
    return T(value);
  }

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

  template <typename T>
  [[nodiscard]] auto annotationId() -> T {
    auto& value = (*repr_)["annotationId"];
    return T(value);
  }

  auto kind(std::string kind) -> RenameFile&;

  auto oldUri(std::string oldUri) -> RenameFile&;

  auto newUri(std::string newUri) -> RenameFile&;

  auto options(std::optional<RenameFileOptions> options) -> RenameFile&;

  auto annotationId(std::optional<ChangeAnnotationIdentifier> annotationId)
      -> RenameFile&;
};

class DeleteFile final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto options() const -> std::optional<DeleteFileOptions>;

  template <typename T>
  [[nodiscard]] auto options() -> T {
    auto& value = (*repr_)["options"];
    return T(value);
  }

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

  template <typename T>
  [[nodiscard]] auto annotationId() -> T {
    auto& value = (*repr_)["annotationId"];
    return T(value);
  }

  auto kind(std::string kind) -> DeleteFile&;

  auto uri(std::string uri) -> DeleteFile&;

  auto options(std::optional<DeleteFileOptions> options) -> DeleteFile&;

  auto annotationId(std::optional<ChangeAnnotationIdentifier> annotationId)
      -> DeleteFile&;
};

class ChangeAnnotation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::string;

  [[nodiscard]] auto needsConfirmation() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto needsConfirmation() -> T {
    auto& value = (*repr_)["needsConfirmation"];
    return T(value);
  }

  [[nodiscard]] auto description() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto description() -> T {
    auto& value = (*repr_)["description"];
    return T(value);
  }

  auto label(std::string label) -> ChangeAnnotation&;

  auto needsConfirmation(std::optional<bool> needsConfirmation)
      -> ChangeAnnotation&;

  auto description(std::optional<std::string> description) -> ChangeAnnotation&;
};

class FileOperationFilter final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto scheme() -> T {
    auto& value = (*repr_)["scheme"];
    return T(value);
  }

  [[nodiscard]] auto pattern() const -> FileOperationPattern;

  auto scheme(std::optional<std::string> scheme) -> FileOperationFilter&;

  auto pattern(FileOperationPattern pattern) -> FileOperationFilter&;
};

class FileRename final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto oldUri() const -> std::string;

  [[nodiscard]] auto newUri() const -> std::string;

  auto oldUri(std::string oldUri) -> FileRename&;

  auto newUri(std::string newUri) -> FileRename&;
};

class FileDelete final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> FileDelete&;
};

class MonikerOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> MonikerOptions&;
};

class TypeHierarchyOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> TypeHierarchyOptions&;
};

class InlineValueContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto frameId() const -> int;

  [[nodiscard]] auto stoppedLocation() const -> Range;

  auto frameId(int frameId) -> InlineValueContext&;

  auto stoppedLocation(Range stoppedLocation) -> InlineValueContext&;
};

class InlineValueText final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto text() const -> std::string;

  auto range(Range range) -> InlineValueText&;

  auto text(std::string text) -> InlineValueText&;
};

class InlineValueVariableLookup final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto variableName() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto variableName() -> T {
    auto& value = (*repr_)["variableName"];
    return T(value);
  }

  [[nodiscard]] auto caseSensitiveLookup() const -> bool;

  auto range(Range range) -> InlineValueVariableLookup&;

  auto variableName(std::optional<std::string> variableName)
      -> InlineValueVariableLookup&;

  auto caseSensitiveLookup(bool caseSensitiveLookup)
      -> InlineValueVariableLookup&;
};

class InlineValueEvaluatableExpression final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto expression() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto expression() -> T {
    auto& value = (*repr_)["expression"];
    return T(value);
  }

  auto range(Range range) -> InlineValueEvaluatableExpression&;

  auto expression(std::optional<std::string> expression)
      -> InlineValueEvaluatableExpression&;
};

class InlineValueOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineValueOptions&;
};

class InlayHintLabelPart final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto value() const -> std::string;

  [[nodiscard]] auto tooltip() const
      -> std::optional<std::variant<std::string, MarkupContent>>;

  template <typename T>
  [[nodiscard]] auto tooltip() -> T {
    auto& value = (*repr_)["tooltip"];
    return T(value);
  }

  [[nodiscard]] auto location() const -> std::optional<Location>;

  template <typename T>
  [[nodiscard]] auto location() -> T {
    auto& value = (*repr_)["location"];
    return T(value);
  }

  [[nodiscard]] auto command() const -> std::optional<Command>;

  template <typename T>
  [[nodiscard]] auto command() -> T {
    auto& value = (*repr_)["command"];
    return T(value);
  }

  auto value(std::string value) -> InlayHintLabelPart&;

  auto tooltip(std::optional<std::variant<std::string, MarkupContent>> tooltip)
      -> InlayHintLabelPart&;

  auto location(std::optional<Location> location) -> InlayHintLabelPart&;

  auto command(std::optional<Command> command) -> InlayHintLabelPart&;
};

class MarkupContent final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> MarkupKind;

  [[nodiscard]] auto value() const -> std::string;

  auto kind(MarkupKind kind) -> MarkupContent&;

  auto value(std::string value) -> MarkupContent&;
};

class InlayHintOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> InlayHintOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlayHintOptions&;
};

class RelatedFullDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relatedDocuments() const -> std::optional<
      Map<std::string, std::variant<FullDocumentDiagnosticReport,
                                    UnchangedDocumentDiagnosticReport>>>;

  template <typename T>
  [[nodiscard]] auto relatedDocuments() -> T {
    auto& value = (*repr_)["relatedDocuments"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto resultId() -> T {
    auto& value = (*repr_)["resultId"];
    return T(value);
  }

  [[nodiscard]] auto items() const -> Vector<Diagnostic>;

  auto relatedDocuments(
      std::optional<
          Map<std::string, std::variant<FullDocumentDiagnosticReport,
                                        UnchangedDocumentDiagnosticReport>>>
          relatedDocuments) -> RelatedFullDocumentDiagnosticReport&;

  auto kind(std::string kind) -> RelatedFullDocumentDiagnosticReport&;

  auto resultId(std::optional<std::string> resultId)
      -> RelatedFullDocumentDiagnosticReport&;

  auto items(Vector<Diagnostic> items) -> RelatedFullDocumentDiagnosticReport&;
};

class RelatedUnchangedDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relatedDocuments() const -> std::optional<
      Map<std::string, std::variant<FullDocumentDiagnosticReport,
                                    UnchangedDocumentDiagnosticReport>>>;

  template <typename T>
  [[nodiscard]] auto relatedDocuments() -> T {
    auto& value = (*repr_)["relatedDocuments"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::string;

  auto relatedDocuments(
      std::optional<
          Map<std::string, std::variant<FullDocumentDiagnosticReport,
                                        UnchangedDocumentDiagnosticReport>>>
          relatedDocuments) -> RelatedUnchangedDocumentDiagnosticReport&;

  auto kind(std::string kind) -> RelatedUnchangedDocumentDiagnosticReport&;

  auto resultId(std::string resultId)
      -> RelatedUnchangedDocumentDiagnosticReport&;
};

class FullDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto resultId() -> T {
    auto& value = (*repr_)["resultId"];
    return T(value);
  }

  [[nodiscard]] auto items() const -> Vector<Diagnostic>;

  auto kind(std::string kind) -> FullDocumentDiagnosticReport&;

  auto resultId(std::optional<std::string> resultId)
      -> FullDocumentDiagnosticReport&;

  auto items(Vector<Diagnostic> items) -> FullDocumentDiagnosticReport&;
};

class UnchangedDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::string;

  auto kind(std::string kind) -> UnchangedDocumentDiagnosticReport&;

  auto resultId(std::string resultId) -> UnchangedDocumentDiagnosticReport&;
};

class DiagnosticOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto identifier() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto identifier() -> T {
    auto& value = (*repr_)["identifier"];
    return T(value);
  }

  [[nodiscard]] auto interFileDependencies() const -> bool;

  [[nodiscard]] auto workspaceDiagnostics() const -> bool;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto identifier(std::optional<std::string> identifier) -> DiagnosticOptions&;

  auto interFileDependencies(bool interFileDependencies) -> DiagnosticOptions&;

  auto workspaceDiagnostics(bool workspaceDiagnostics) -> DiagnosticOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DiagnosticOptions&;
};

class PreviousResultId final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto value() const -> std::string;

  auto uri(std::string uri) -> PreviousResultId&;

  auto value(std::string value) -> PreviousResultId&;
};

class NotebookDocument final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto notebookType() const -> std::string;

  [[nodiscard]] auto version() const -> int;

  [[nodiscard]] auto metadata() const -> std::optional<LSPObject>;

  template <typename T>
  [[nodiscard]] auto metadata() -> T {
    auto& value = (*repr_)["metadata"];
    return T(value);
  }

  [[nodiscard]] auto cells() const -> Vector<NotebookCell>;

  auto uri(std::string uri) -> NotebookDocument&;

  auto notebookType(std::string notebookType) -> NotebookDocument&;

  auto version(int version) -> NotebookDocument&;

  auto metadata(std::optional<LSPObject> metadata) -> NotebookDocument&;

  auto cells(Vector<NotebookCell> cells) -> NotebookDocument&;
};

class TextDocumentItem final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto languageId() const -> LanguageKind;

  [[nodiscard]] auto version() const -> int;

  [[nodiscard]] auto text() const -> std::string;

  auto uri(std::string uri) -> TextDocumentItem&;

  auto languageId(LanguageKind languageId) -> TextDocumentItem&;

  auto version(int version) -> TextDocumentItem&;

  auto text(std::string text) -> TextDocumentItem&;
};

class NotebookDocumentSyncOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookSelector() const
      -> Vector<std::variant<NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>;

  [[nodiscard]] auto save() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto save() -> T {
    auto& value = (*repr_)["save"];
    return T(value);
  }

  auto notebookSelector(Vector<std::variant<NotebookDocumentFilterWithNotebook,
                                            NotebookDocumentFilterWithCells>>
                            notebookSelector) -> NotebookDocumentSyncOptions&;

  auto save(std::optional<bool> save) -> NotebookDocumentSyncOptions&;
};

class VersionedNotebookDocumentIdentifier final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto version() const -> int;

  [[nodiscard]] auto uri() const -> std::string;

  auto version(int version) -> VersionedNotebookDocumentIdentifier&;

  auto uri(std::string uri) -> VersionedNotebookDocumentIdentifier&;
};

class NotebookDocumentChangeEvent final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto metadata() const -> std::optional<LSPObject>;

  template <typename T>
  [[nodiscard]] auto metadata() -> T {
    auto& value = (*repr_)["metadata"];
    return T(value);
  }

  [[nodiscard]] auto cells() const
      -> std::optional<NotebookDocumentCellChanges>;

  template <typename T>
  [[nodiscard]] auto cells() -> T {
    auto& value = (*repr_)["cells"];
    return T(value);
  }

  auto metadata(std::optional<LSPObject> metadata)
      -> NotebookDocumentChangeEvent&;

  auto cells(std::optional<NotebookDocumentCellChanges> cells)
      -> NotebookDocumentChangeEvent&;
};

class NotebookDocumentIdentifier final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> NotebookDocumentIdentifier&;
};

class InlineCompletionContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerKind() const -> InlineCompletionTriggerKind;

  [[nodiscard]] auto selectedCompletionInfo() const
      -> std::optional<SelectedCompletionInfo>;

  template <typename T>
  [[nodiscard]] auto selectedCompletionInfo() -> T {
    auto& value = (*repr_)["selectedCompletionInfo"];
    return T(value);
  }

  auto triggerKind(InlineCompletionTriggerKind triggerKind)
      -> InlineCompletionContext&;

  auto selectedCompletionInfo(
      std::optional<SelectedCompletionInfo> selectedCompletionInfo)
      -> InlineCompletionContext&;
};

class StringValue final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto value() const -> std::string;

  auto kind(std::string kind) -> StringValue&;

  auto value(std::string value) -> StringValue&;
};

class InlineCompletionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineCompletionOptions&;
};

class TextDocumentContentOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto schemes() const -> Vector<std::string>;

  auto schemes(Vector<std::string> schemes) -> TextDocumentContentOptions&;

  auto schemes(std::vector<std::string> schemes)
      -> TextDocumentContentOptions& {
    auto& value = (*repr_)["schemes"];
    value = std::move(schemes);
    return *this;
  }
};

class Registration final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const -> std::string;

  [[nodiscard]] auto method() const -> std::string;

  [[nodiscard]] auto registerOptions() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto registerOptions() -> T {
    auto& value = (*repr_)["registerOptions"];
    return T(value);
  }

  auto id(std::string id) -> Registration&;

  auto method(std::string method) -> Registration&;

  auto registerOptions(std::optional<LSPAny> registerOptions) -> Registration&;
};

class Unregistration final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const -> std::string;

  [[nodiscard]] auto method() const -> std::string;

  auto id(std::string id) -> Unregistration&;

  auto method(std::string method) -> Unregistration&;
};

class _InitializeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto processId() const -> std::variant<int, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto processId() -> T {
    auto& value = (*repr_)["processId"];
    return T(value);
  }

  [[nodiscard]] auto clientInfo() const -> std::optional<ClientInfo>;

  template <typename T>
  [[nodiscard]] auto clientInfo() -> T {
    auto& value = (*repr_)["clientInfo"];
    return T(value);
  }

  [[nodiscard]] auto locale() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto locale() -> T {
    auto& value = (*repr_)["locale"];
    return T(value);
  }

  [[nodiscard]] auto rootPath() const
      -> std::optional<std::variant<std::string, std::nullptr_t>>;

  template <typename T>
  [[nodiscard]] auto rootPath() -> T {
    auto& value = (*repr_)["rootPath"];
    return T(value);
  }

  [[nodiscard]] auto rootUri() const
      -> std::variant<std::string, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto rootUri() -> T {
    auto& value = (*repr_)["rootUri"];
    return T(value);
  }

  [[nodiscard]] auto capabilities() const -> ClientCapabilities;

  [[nodiscard]] auto initializationOptions() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto initializationOptions() -> T {
    auto& value = (*repr_)["initializationOptions"];
    return T(value);
  }

  [[nodiscard]] auto trace() const -> std::optional<TraceValue>;

  template <typename T>
  [[nodiscard]] auto trace() -> T {
    auto& value = (*repr_)["trace"];
    return T(value);
  }

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  template <typename T>
  [[nodiscard]] auto workDoneToken() -> T {
    auto& value = (*repr_)["workDoneToken"];
    return T(value);
  }

  auto processId(std::variant<int, std::nullptr_t> processId)
      -> _InitializeParams&;

  auto clientInfo(std::optional<ClientInfo> clientInfo) -> _InitializeParams&;

  auto locale(std::optional<std::string> locale) -> _InitializeParams&;

  auto rootPath(
      std::optional<std::variant<std::string, std::nullptr_t>> rootPath)
      -> _InitializeParams&;

  auto rootUri(std::variant<std::string, std::nullptr_t> rootUri)
      -> _InitializeParams&;

  auto capabilities(ClientCapabilities capabilities) -> _InitializeParams&;

  auto initializationOptions(std::optional<LSPAny> initializationOptions)
      -> _InitializeParams&;

  auto trace(std::optional<TraceValue> trace) -> _InitializeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> _InitializeParams&;
};

class WorkspaceFoldersInitializeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workspaceFolders() const
      -> std::optional<std::variant<Vector<WorkspaceFolder>, std::nullptr_t>>;

  template <typename T>
  [[nodiscard]] auto workspaceFolders() -> T {
    auto& value = (*repr_)["workspaceFolders"];
    return T(value);
  }

  auto workspaceFolders(
      std::optional<std::variant<Vector<WorkspaceFolder>, std::nullptr_t>>
          workspaceFolders) -> WorkspaceFoldersInitializeParams&;
};

class ServerCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto positionEncoding() const
      -> std::optional<PositionEncodingKind>;

  template <typename T>
  [[nodiscard]] auto positionEncoding() -> T {
    auto& value = (*repr_)["positionEncoding"];
    return T(value);
  }

  [[nodiscard]] auto textDocumentSync() const -> std::optional<
      std::variant<TextDocumentSyncOptions, TextDocumentSyncKind>>;

  template <typename T>
  [[nodiscard]] auto textDocumentSync() -> T {
    auto& value = (*repr_)["textDocumentSync"];
    return T(value);
  }

  [[nodiscard]] auto notebookDocumentSync() const
      -> std::optional<std::variant<NotebookDocumentSyncOptions,
                                    NotebookDocumentSyncRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto notebookDocumentSync() -> T {
    auto& value = (*repr_)["notebookDocumentSync"];
    return T(value);
  }

  [[nodiscard]] auto completionProvider() const
      -> std::optional<CompletionOptions>;

  template <typename T>
  [[nodiscard]] auto completionProvider() -> T {
    auto& value = (*repr_)["completionProvider"];
    return T(value);
  }

  [[nodiscard]] auto hoverProvider() const
      -> std::optional<std::variant<bool, HoverOptions>>;

  template <typename T>
  [[nodiscard]] auto hoverProvider() -> T {
    auto& value = (*repr_)["hoverProvider"];
    return T(value);
  }

  [[nodiscard]] auto signatureHelpProvider() const
      -> std::optional<SignatureHelpOptions>;

  template <typename T>
  [[nodiscard]] auto signatureHelpProvider() -> T {
    auto& value = (*repr_)["signatureHelpProvider"];
    return T(value);
  }

  [[nodiscard]] auto declarationProvider() const -> std::optional<
      std::variant<bool, DeclarationOptions, DeclarationRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto declarationProvider() -> T {
    auto& value = (*repr_)["declarationProvider"];
    return T(value);
  }

  [[nodiscard]] auto definitionProvider() const
      -> std::optional<std::variant<bool, DefinitionOptions>>;

  template <typename T>
  [[nodiscard]] auto definitionProvider() -> T {
    auto& value = (*repr_)["definitionProvider"];
    return T(value);
  }

  [[nodiscard]] auto typeDefinitionProvider() const
      -> std::optional<std::variant<bool, TypeDefinitionOptions,
                                    TypeDefinitionRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto typeDefinitionProvider() -> T {
    auto& value = (*repr_)["typeDefinitionProvider"];
    return T(value);
  }

  [[nodiscard]] auto implementationProvider() const
      -> std::optional<std::variant<bool, ImplementationOptions,
                                    ImplementationRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto implementationProvider() -> T {
    auto& value = (*repr_)["implementationProvider"];
    return T(value);
  }

  [[nodiscard]] auto referencesProvider() const
      -> std::optional<std::variant<bool, ReferenceOptions>>;

  template <typename T>
  [[nodiscard]] auto referencesProvider() -> T {
    auto& value = (*repr_)["referencesProvider"];
    return T(value);
  }

  [[nodiscard]] auto documentHighlightProvider() const
      -> std::optional<std::variant<bool, DocumentHighlightOptions>>;

  template <typename T>
  [[nodiscard]] auto documentHighlightProvider() -> T {
    auto& value = (*repr_)["documentHighlightProvider"];
    return T(value);
  }

  [[nodiscard]] auto documentSymbolProvider() const
      -> std::optional<std::variant<bool, DocumentSymbolOptions>>;

  template <typename T>
  [[nodiscard]] auto documentSymbolProvider() -> T {
    auto& value = (*repr_)["documentSymbolProvider"];
    return T(value);
  }

  [[nodiscard]] auto codeActionProvider() const
      -> std::optional<std::variant<bool, CodeActionOptions>>;

  template <typename T>
  [[nodiscard]] auto codeActionProvider() -> T {
    auto& value = (*repr_)["codeActionProvider"];
    return T(value);
  }

  [[nodiscard]] auto codeLensProvider() const -> std::optional<CodeLensOptions>;

  template <typename T>
  [[nodiscard]] auto codeLensProvider() -> T {
    auto& value = (*repr_)["codeLensProvider"];
    return T(value);
  }

  [[nodiscard]] auto documentLinkProvider() const
      -> std::optional<DocumentLinkOptions>;

  template <typename T>
  [[nodiscard]] auto documentLinkProvider() -> T {
    auto& value = (*repr_)["documentLinkProvider"];
    return T(value);
  }

  [[nodiscard]] auto colorProvider() const
      -> std::optional<std::variant<bool, DocumentColorOptions,
                                    DocumentColorRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto colorProvider() -> T {
    auto& value = (*repr_)["colorProvider"];
    return T(value);
  }

  [[nodiscard]] auto workspaceSymbolProvider() const
      -> std::optional<std::variant<bool, WorkspaceSymbolOptions>>;

  template <typename T>
  [[nodiscard]] auto workspaceSymbolProvider() -> T {
    auto& value = (*repr_)["workspaceSymbolProvider"];
    return T(value);
  }

  [[nodiscard]] auto documentFormattingProvider() const
      -> std::optional<std::variant<bool, DocumentFormattingOptions>>;

  template <typename T>
  [[nodiscard]] auto documentFormattingProvider() -> T {
    auto& value = (*repr_)["documentFormattingProvider"];
    return T(value);
  }

  [[nodiscard]] auto documentRangeFormattingProvider() const
      -> std::optional<std::variant<bool, DocumentRangeFormattingOptions>>;

  template <typename T>
  [[nodiscard]] auto documentRangeFormattingProvider() -> T {
    auto& value = (*repr_)["documentRangeFormattingProvider"];
    return T(value);
  }

  [[nodiscard]] auto documentOnTypeFormattingProvider() const
      -> std::optional<DocumentOnTypeFormattingOptions>;

  template <typename T>
  [[nodiscard]] auto documentOnTypeFormattingProvider() -> T {
    auto& value = (*repr_)["documentOnTypeFormattingProvider"];
    return T(value);
  }

  [[nodiscard]] auto renameProvider() const
      -> std::optional<std::variant<bool, RenameOptions>>;

  template <typename T>
  [[nodiscard]] auto renameProvider() -> T {
    auto& value = (*repr_)["renameProvider"];
    return T(value);
  }

  [[nodiscard]] auto foldingRangeProvider() const -> std::optional<
      std::variant<bool, FoldingRangeOptions, FoldingRangeRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto foldingRangeProvider() -> T {
    auto& value = (*repr_)["foldingRangeProvider"];
    return T(value);
  }

  [[nodiscard]] auto selectionRangeProvider() const
      -> std::optional<std::variant<bool, SelectionRangeOptions,
                                    SelectionRangeRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto selectionRangeProvider() -> T {
    auto& value = (*repr_)["selectionRangeProvider"];
    return T(value);
  }

  [[nodiscard]] auto executeCommandProvider() const
      -> std::optional<ExecuteCommandOptions>;

  template <typename T>
  [[nodiscard]] auto executeCommandProvider() -> T {
    auto& value = (*repr_)["executeCommandProvider"];
    return T(value);
  }

  [[nodiscard]] auto callHierarchyProvider() const
      -> std::optional<std::variant<bool, CallHierarchyOptions,
                                    CallHierarchyRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto callHierarchyProvider() -> T {
    auto& value = (*repr_)["callHierarchyProvider"];
    return T(value);
  }

  [[nodiscard]] auto linkedEditingRangeProvider() const
      -> std::optional<std::variant<bool, LinkedEditingRangeOptions,
                                    LinkedEditingRangeRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto linkedEditingRangeProvider() -> T {
    auto& value = (*repr_)["linkedEditingRangeProvider"];
    return T(value);
  }

  [[nodiscard]] auto semanticTokensProvider() const -> std::optional<
      std::variant<SemanticTokensOptions, SemanticTokensRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto semanticTokensProvider() -> T {
    auto& value = (*repr_)["semanticTokensProvider"];
    return T(value);
  }

  [[nodiscard]] auto monikerProvider() const -> std::optional<
      std::variant<bool, MonikerOptions, MonikerRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto monikerProvider() -> T {
    auto& value = (*repr_)["monikerProvider"];
    return T(value);
  }

  [[nodiscard]] auto typeHierarchyProvider() const
      -> std::optional<std::variant<bool, TypeHierarchyOptions,
                                    TypeHierarchyRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto typeHierarchyProvider() -> T {
    auto& value = (*repr_)["typeHierarchyProvider"];
    return T(value);
  }

  [[nodiscard]] auto inlineValueProvider() const -> std::optional<
      std::variant<bool, InlineValueOptions, InlineValueRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto inlineValueProvider() -> T {
    auto& value = (*repr_)["inlineValueProvider"];
    return T(value);
  }

  [[nodiscard]] auto inlayHintProvider() const -> std::optional<
      std::variant<bool, InlayHintOptions, InlayHintRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto inlayHintProvider() -> T {
    auto& value = (*repr_)["inlayHintProvider"];
    return T(value);
  }

  [[nodiscard]] auto diagnosticProvider() const -> std::optional<
      std::variant<DiagnosticOptions, DiagnosticRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto diagnosticProvider() -> T {
    auto& value = (*repr_)["diagnosticProvider"];
    return T(value);
  }

  [[nodiscard]] auto inlineCompletionProvider() const
      -> std::optional<std::variant<bool, InlineCompletionOptions>>;

  template <typename T>
  [[nodiscard]] auto inlineCompletionProvider() -> T {
    auto& value = (*repr_)["inlineCompletionProvider"];
    return T(value);
  }

  [[nodiscard]] auto workspace() const -> std::optional<WorkspaceOptions>;

  template <typename T>
  [[nodiscard]] auto workspace() -> T {
    auto& value = (*repr_)["workspace"];
    return T(value);
  }

  [[nodiscard]] auto experimental() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto experimental() -> T {
    auto& value = (*repr_)["experimental"];
    return T(value);
  }

  auto positionEncoding(std::optional<PositionEncodingKind> positionEncoding)
      -> ServerCapabilities&;

  auto textDocumentSync(
      std::optional<std::variant<TextDocumentSyncOptions, TextDocumentSyncKind>>
          textDocumentSync) -> ServerCapabilities&;

  auto notebookDocumentSync(
      std::optional<std::variant<NotebookDocumentSyncOptions,
                                 NotebookDocumentSyncRegistrationOptions>>
          notebookDocumentSync) -> ServerCapabilities&;

  auto completionProvider(std::optional<CompletionOptions> completionProvider)
      -> ServerCapabilities&;

  auto hoverProvider(
      std::optional<std::variant<bool, HoverOptions>> hoverProvider)
      -> ServerCapabilities&;

  auto signatureHelpProvider(
      std::optional<SignatureHelpOptions> signatureHelpProvider)
      -> ServerCapabilities&;

  auto declarationProvider(
      std::optional<std::variant<bool, DeclarationOptions,
                                 DeclarationRegistrationOptions>>
          declarationProvider) -> ServerCapabilities&;

  auto definitionProvider(
      std::optional<std::variant<bool, DefinitionOptions>> definitionProvider)
      -> ServerCapabilities&;

  auto typeDefinitionProvider(
      std::optional<std::variant<bool, TypeDefinitionOptions,
                                 TypeDefinitionRegistrationOptions>>
          typeDefinitionProvider) -> ServerCapabilities&;

  auto implementationProvider(
      std::optional<std::variant<bool, ImplementationOptions,
                                 ImplementationRegistrationOptions>>
          implementationProvider) -> ServerCapabilities&;

  auto referencesProvider(
      std::optional<std::variant<bool, ReferenceOptions>> referencesProvider)
      -> ServerCapabilities&;

  auto documentHighlightProvider(
      std::optional<std::variant<bool, DocumentHighlightOptions>>
          documentHighlightProvider) -> ServerCapabilities&;

  auto documentSymbolProvider(
      std::optional<std::variant<bool, DocumentSymbolOptions>>
          documentSymbolProvider) -> ServerCapabilities&;

  auto codeActionProvider(
      std::optional<std::variant<bool, CodeActionOptions>> codeActionProvider)
      -> ServerCapabilities&;

  auto codeLensProvider(std::optional<CodeLensOptions> codeLensProvider)
      -> ServerCapabilities&;

  auto documentLinkProvider(
      std::optional<DocumentLinkOptions> documentLinkProvider)
      -> ServerCapabilities&;

  auto colorProvider(
      std::optional<std::variant<bool, DocumentColorOptions,
                                 DocumentColorRegistrationOptions>>
          colorProvider) -> ServerCapabilities&;

  auto workspaceSymbolProvider(
      std::optional<std::variant<bool, WorkspaceSymbolOptions>>
          workspaceSymbolProvider) -> ServerCapabilities&;

  auto documentFormattingProvider(
      std::optional<std::variant<bool, DocumentFormattingOptions>>
          documentFormattingProvider) -> ServerCapabilities&;

  auto documentRangeFormattingProvider(
      std::optional<std::variant<bool, DocumentRangeFormattingOptions>>
          documentRangeFormattingProvider) -> ServerCapabilities&;

  auto documentOnTypeFormattingProvider(
      std::optional<DocumentOnTypeFormattingOptions>
          documentOnTypeFormattingProvider) -> ServerCapabilities&;

  auto renameProvider(
      std::optional<std::variant<bool, RenameOptions>> renameProvider)
      -> ServerCapabilities&;

  auto foldingRangeProvider(
      std::optional<std::variant<bool, FoldingRangeOptions,
                                 FoldingRangeRegistrationOptions>>
          foldingRangeProvider) -> ServerCapabilities&;

  auto selectionRangeProvider(
      std::optional<std::variant<bool, SelectionRangeOptions,
                                 SelectionRangeRegistrationOptions>>
          selectionRangeProvider) -> ServerCapabilities&;

  auto executeCommandProvider(
      std::optional<ExecuteCommandOptions> executeCommandProvider)
      -> ServerCapabilities&;

  auto callHierarchyProvider(
      std::optional<std::variant<bool, CallHierarchyOptions,
                                 CallHierarchyRegistrationOptions>>
          callHierarchyProvider) -> ServerCapabilities&;

  auto linkedEditingRangeProvider(
      std::optional<std::variant<bool, LinkedEditingRangeOptions,
                                 LinkedEditingRangeRegistrationOptions>>
          linkedEditingRangeProvider) -> ServerCapabilities&;

  auto semanticTokensProvider(
      std::optional<std::variant<SemanticTokensOptions,
                                 SemanticTokensRegistrationOptions>>
          semanticTokensProvider) -> ServerCapabilities&;

  auto monikerProvider(
      std::optional<
          std::variant<bool, MonikerOptions, MonikerRegistrationOptions>>
          monikerProvider) -> ServerCapabilities&;

  auto typeHierarchyProvider(
      std::optional<std::variant<bool, TypeHierarchyOptions,
                                 TypeHierarchyRegistrationOptions>>
          typeHierarchyProvider) -> ServerCapabilities&;

  auto inlineValueProvider(
      std::optional<std::variant<bool, InlineValueOptions,
                                 InlineValueRegistrationOptions>>
          inlineValueProvider) -> ServerCapabilities&;

  auto inlayHintProvider(
      std::optional<
          std::variant<bool, InlayHintOptions, InlayHintRegistrationOptions>>
          inlayHintProvider) -> ServerCapabilities&;

  auto diagnosticProvider(
      std::optional<
          std::variant<DiagnosticOptions, DiagnosticRegistrationOptions>>
          diagnosticProvider) -> ServerCapabilities&;

  auto inlineCompletionProvider(
      std::optional<std::variant<bool, InlineCompletionOptions>>
          inlineCompletionProvider) -> ServerCapabilities&;

  auto workspace(std::optional<WorkspaceOptions> workspace)
      -> ServerCapabilities&;

  auto experimental(std::optional<LSPAny> experimental) -> ServerCapabilities&;
};

class ServerInfo final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto version() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  auto name(std::string name) -> ServerInfo&;

  auto version(std::optional<std::string> version) -> ServerInfo&;
};

class VersionedTextDocumentIdentifier final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto version() const -> int;

  [[nodiscard]] auto uri() const -> std::string;

  auto version(int version) -> VersionedTextDocumentIdentifier&;

  auto uri(std::string uri) -> VersionedTextDocumentIdentifier&;
};

class SaveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto includeText() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto includeText() -> T {
    auto& value = (*repr_)["includeText"];
    return T(value);
  }

  auto includeText(std::optional<bool> includeText) -> SaveOptions&;
};

class FileEvent final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto type() const -> FileChangeType;

  auto uri(std::string uri) -> FileEvent&;

  auto type(FileChangeType type) -> FileEvent&;
};

class FileSystemWatcher final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto globPattern() const -> GlobPattern;

  [[nodiscard]] auto kind() const -> std::optional<WatchKind>;

  template <typename T>
  [[nodiscard]] auto kind() -> T {
    auto& value = (*repr_)["kind"];
    return T(value);
  }

  auto globPattern(GlobPattern globPattern) -> FileSystemWatcher&;

  auto kind(std::optional<WatchKind> kind) -> FileSystemWatcher&;
};

class Diagnostic final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto severity() const -> std::optional<DiagnosticSeverity>;

  template <typename T>
  [[nodiscard]] auto severity() -> T {
    auto& value = (*repr_)["severity"];
    return T(value);
  }

  [[nodiscard]] auto code() const
      -> std::optional<std::variant<int, std::string>>;

  template <typename T>
  [[nodiscard]] auto code() -> T {
    auto& value = (*repr_)["code"];
    return T(value);
  }

  [[nodiscard]] auto codeDescription() const -> std::optional<CodeDescription>;

  template <typename T>
  [[nodiscard]] auto codeDescription() -> T {
    auto& value = (*repr_)["codeDescription"];
    return T(value);
  }

  [[nodiscard]] auto source() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto source() -> T {
    auto& value = (*repr_)["source"];
    return T(value);
  }

  [[nodiscard]] auto message() const -> std::string;

  [[nodiscard]] auto tags() const -> std::optional<Vector<DiagnosticTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto relatedInformation() const
      -> std::optional<Vector<DiagnosticRelatedInformation>>;

  template <typename T>
  [[nodiscard]] auto relatedInformation() -> T {
    auto& value = (*repr_)["relatedInformation"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto range(Range range) -> Diagnostic&;

  auto severity(std::optional<DiagnosticSeverity> severity) -> Diagnostic&;

  auto code(std::optional<std::variant<int, std::string>> code) -> Diagnostic&;

  auto codeDescription(std::optional<CodeDescription> codeDescription)
      -> Diagnostic&;

  auto source(std::optional<std::string> source) -> Diagnostic&;

  auto message(std::string message) -> Diagnostic&;

  auto tags(std::optional<Vector<DiagnosticTag>> tags) -> Diagnostic&;

  auto relatedInformation(
      std::optional<Vector<DiagnosticRelatedInformation>> relatedInformation)
      -> Diagnostic&;

  auto data(std::optional<LSPAny> data) -> Diagnostic&;
};

class CompletionContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerKind() const -> CompletionTriggerKind;

  [[nodiscard]] auto triggerCharacter() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto triggerCharacter() -> T {
    auto& value = (*repr_)["triggerCharacter"];
    return T(value);
  }

  auto triggerKind(CompletionTriggerKind triggerKind) -> CompletionContext&;

  auto triggerCharacter(std::optional<std::string> triggerCharacter)
      -> CompletionContext&;
};

class CompletionItemLabelDetails final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto detail() -> T {
    auto& value = (*repr_)["detail"];
    return T(value);
  }

  [[nodiscard]] auto description() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto description() -> T {
    auto& value = (*repr_)["description"];
    return T(value);
  }

  auto detail(std::optional<std::string> detail) -> CompletionItemLabelDetails&;

  auto description(std::optional<std::string> description)
      -> CompletionItemLabelDetails&;
};

class InsertReplaceEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto newText() const -> std::string;

  [[nodiscard]] auto insert() const -> Range;

  [[nodiscard]] auto replace() const -> Range;

  auto newText(std::string newText) -> InsertReplaceEdit&;

  auto insert(Range insert) -> InsertReplaceEdit&;

  auto replace(Range replace) -> InsertReplaceEdit&;
};

class CompletionItemDefaults final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto commitCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto commitCharacters() -> T {
    auto& value = (*repr_)["commitCharacters"];
    return T(value);
  }

  [[nodiscard]] auto editRange() const
      -> std::optional<std::variant<Range, EditRangeWithInsertReplace>>;

  template <typename T>
  [[nodiscard]] auto editRange() -> T {
    auto& value = (*repr_)["editRange"];
    return T(value);
  }

  [[nodiscard]] auto insertTextFormat() const
      -> std::optional<InsertTextFormat>;

  template <typename T>
  [[nodiscard]] auto insertTextFormat() -> T {
    auto& value = (*repr_)["insertTextFormat"];
    return T(value);
  }

  [[nodiscard]] auto insertTextMode() const -> std::optional<InsertTextMode>;

  template <typename T>
  [[nodiscard]] auto insertTextMode() -> T {
    auto& value = (*repr_)["insertTextMode"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto commitCharacters(std::optional<Vector<std::string>> commitCharacters)
      -> CompletionItemDefaults&;

  auto commitCharacters(std::vector<std::string> commitCharacters)
      -> CompletionItemDefaults& {
    auto& value = (*repr_)["commitCharacters"];
    value = std::move(commitCharacters);
    return *this;
  }

  auto editRange(
      std::optional<std::variant<Range, EditRangeWithInsertReplace>> editRange)
      -> CompletionItemDefaults&;

  auto insertTextFormat(std::optional<InsertTextFormat> insertTextFormat)
      -> CompletionItemDefaults&;

  auto insertTextMode(std::optional<InsertTextMode> insertTextMode)
      -> CompletionItemDefaults&;

  auto data(std::optional<LSPAny> data) -> CompletionItemDefaults&;
};

class CompletionItemApplyKinds final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto commitCharacters() const -> std::optional<ApplyKind>;

  template <typename T>
  [[nodiscard]] auto commitCharacters() -> T {
    auto& value = (*repr_)["commitCharacters"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<ApplyKind>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  auto commitCharacters(std::optional<ApplyKind> commitCharacters)
      -> CompletionItemApplyKinds&;

  auto data(std::optional<ApplyKind> data) -> CompletionItemApplyKinds&;
};

class CompletionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto triggerCharacters() -> T {
    auto& value = (*repr_)["triggerCharacters"];
    return T(value);
  }

  [[nodiscard]] auto allCommitCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto allCommitCharacters() -> T {
    auto& value = (*repr_)["allCommitCharacters"];
    return T(value);
  }

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto completionItem() const
      -> std::optional<ServerCompletionItemOptions>;

  template <typename T>
  [[nodiscard]] auto completionItem() -> T {
    auto& value = (*repr_)["completionItem"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> CompletionOptions&;

  auto triggerCharacters(std::vector<std::string> triggerCharacters)
      -> CompletionOptions& {
    auto& value = (*repr_)["triggerCharacters"];
    value = std::move(triggerCharacters);
    return *this;
  }

  auto allCommitCharacters(
      std::optional<Vector<std::string>> allCommitCharacters)
      -> CompletionOptions&;

  auto allCommitCharacters(std::vector<std::string> allCommitCharacters)
      -> CompletionOptions& {
    auto& value = (*repr_)["allCommitCharacters"];
    value = std::move(allCommitCharacters);
    return *this;
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> CompletionOptions&;

  auto completionItem(std::optional<ServerCompletionItemOptions> completionItem)
      -> CompletionOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CompletionOptions&;
};

class HoverOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress) -> HoverOptions&;
};

class SignatureHelpContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerKind() const -> SignatureHelpTriggerKind;

  [[nodiscard]] auto triggerCharacter() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto triggerCharacter() -> T {
    auto& value = (*repr_)["triggerCharacter"];
    return T(value);
  }

  [[nodiscard]] auto isRetrigger() const -> bool;

  [[nodiscard]] auto activeSignatureHelp() const
      -> std::optional<SignatureHelp>;

  template <typename T>
  [[nodiscard]] auto activeSignatureHelp() -> T {
    auto& value = (*repr_)["activeSignatureHelp"];
    return T(value);
  }

  auto triggerKind(SignatureHelpTriggerKind triggerKind)
      -> SignatureHelpContext&;

  auto triggerCharacter(std::optional<std::string> triggerCharacter)
      -> SignatureHelpContext&;

  auto isRetrigger(bool isRetrigger) -> SignatureHelpContext&;

  auto activeSignatureHelp(std::optional<SignatureHelp> activeSignatureHelp)
      -> SignatureHelpContext&;
};

class SignatureInformation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::string;

  [[nodiscard]] auto documentation() const
      -> std::optional<std::variant<std::string, MarkupContent>>;

  template <typename T>
  [[nodiscard]] auto documentation() -> T {
    auto& value = (*repr_)["documentation"];
    return T(value);
  }

  [[nodiscard]] auto parameters() const
      -> std::optional<Vector<ParameterInformation>>;

  template <typename T>
  [[nodiscard]] auto parameters() -> T {
    auto& value = (*repr_)["parameters"];
    return T(value);
  }

  [[nodiscard]] auto activeParameter() const
      -> std::optional<std::variant<long, std::nullptr_t>>;

  template <typename T>
  [[nodiscard]] auto activeParameter() -> T {
    auto& value = (*repr_)["activeParameter"];
    return T(value);
  }

  auto label(std::string label) -> SignatureInformation&;

  auto documentation(
      std::optional<std::variant<std::string, MarkupContent>> documentation)
      -> SignatureInformation&;

  auto parameters(std::optional<Vector<ParameterInformation>> parameters)
      -> SignatureInformation&;

  auto activeParameter(
      std::optional<std::variant<long, std::nullptr_t>> activeParameter)
      -> SignatureInformation&;
};

class SignatureHelpOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto triggerCharacters() -> T {
    auto& value = (*repr_)["triggerCharacters"];
    return T(value);
  }

  [[nodiscard]] auto retriggerCharacters() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto retriggerCharacters() -> T {
    auto& value = (*repr_)["retriggerCharacters"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> SignatureHelpOptions&;

  auto triggerCharacters(std::vector<std::string> triggerCharacters)
      -> SignatureHelpOptions& {
    auto& value = (*repr_)["triggerCharacters"];
    value = std::move(triggerCharacters);
    return *this;
  }

  auto retriggerCharacters(
      std::optional<Vector<std::string>> retriggerCharacters)
      -> SignatureHelpOptions&;

  auto retriggerCharacters(std::vector<std::string> retriggerCharacters)
      -> SignatureHelpOptions& {
    auto& value = (*repr_)["retriggerCharacters"];
    value = std::move(retriggerCharacters);
    return *this;
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SignatureHelpOptions&;
};

class DefinitionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DefinitionOptions&;
};

class ReferenceContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto includeDeclaration() const -> bool;

  auto includeDeclaration(bool includeDeclaration) -> ReferenceContext&;
};

class ReferenceOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ReferenceOptions&;
};

class DocumentHighlightOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentHighlightOptions&;
};

class BaseSymbolInformation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  template <typename T>
  [[nodiscard]] auto tags() -> T {
    auto& value = (*repr_)["tags"];
    return T(value);
  }

  [[nodiscard]] auto containerName() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto containerName() -> T {
    auto& value = (*repr_)["containerName"];
    return T(value);
  }

  auto name(std::string name) -> BaseSymbolInformation&;

  auto kind(SymbolKind kind) -> BaseSymbolInformation&;

  auto tags(std::optional<Vector<SymbolTag>> tags) -> BaseSymbolInformation&;

  auto containerName(std::optional<std::string> containerName)
      -> BaseSymbolInformation&;
};

class DocumentSymbolOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto label() -> T {
    auto& value = (*repr_)["label"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto label(std::optional<std::string> label) -> DocumentSymbolOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentSymbolOptions&;
};

class CodeActionContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto diagnostics() const -> Vector<Diagnostic>;

  [[nodiscard]] auto only() const -> std::optional<Vector<CodeActionKind>>;

  template <typename T>
  [[nodiscard]] auto only() -> T {
    auto& value = (*repr_)["only"];
    return T(value);
  }

  [[nodiscard]] auto triggerKind() const
      -> std::optional<CodeActionTriggerKind>;

  template <typename T>
  [[nodiscard]] auto triggerKind() -> T {
    auto& value = (*repr_)["triggerKind"];
    return T(value);
  }

  auto diagnostics(Vector<Diagnostic> diagnostics) -> CodeActionContext&;

  auto only(std::optional<Vector<CodeActionKind>> only) -> CodeActionContext&;

  auto triggerKind(std::optional<CodeActionTriggerKind> triggerKind)
      -> CodeActionContext&;
};

class CodeActionDisabled final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto reason() const -> std::string;

  auto reason(std::string reason) -> CodeActionDisabled&;
};

class CodeActionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto codeActionKinds() const
      -> std::optional<Vector<CodeActionKind>>;

  template <typename T>
  [[nodiscard]] auto codeActionKinds() -> T {
    auto& value = (*repr_)["codeActionKinds"];
    return T(value);
  }

  [[nodiscard]] auto documentation() const
      -> std::optional<Vector<CodeActionKindDocumentation>>;

  template <typename T>
  [[nodiscard]] auto documentation() -> T {
    auto& value = (*repr_)["documentation"];
    return T(value);
  }

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto codeActionKinds(std::optional<Vector<CodeActionKind>> codeActionKinds)
      -> CodeActionOptions&;

  auto documentation(
      std::optional<Vector<CodeActionKindDocumentation>> documentation)
      -> CodeActionOptions&;

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> CodeActionOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CodeActionOptions&;
};

class LocationUriOnly final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  auto uri(std::string uri) -> LocationUriOnly&;
};

class WorkspaceSymbolOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> WorkspaceSymbolOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> WorkspaceSymbolOptions&;
};

class CodeLensOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto resolveProvider(std::optional<bool> resolveProvider) -> CodeLensOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CodeLensOptions&;
};

class DocumentLinkOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto resolveProvider() -> T {
    auto& value = (*repr_)["resolveProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> DocumentLinkOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentLinkOptions&;
};

class FormattingOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto tabSize() const -> long;

  [[nodiscard]] auto insertSpaces() const -> bool;

  [[nodiscard]] auto trimTrailingWhitespace() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto trimTrailingWhitespace() -> T {
    auto& value = (*repr_)["trimTrailingWhitespace"];
    return T(value);
  }

  [[nodiscard]] auto insertFinalNewline() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto insertFinalNewline() -> T {
    auto& value = (*repr_)["insertFinalNewline"];
    return T(value);
  }

  [[nodiscard]] auto trimFinalNewlines() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto trimFinalNewlines() -> T {
    auto& value = (*repr_)["trimFinalNewlines"];
    return T(value);
  }

  auto tabSize(long tabSize) -> FormattingOptions&;

  auto insertSpaces(bool insertSpaces) -> FormattingOptions&;

  auto trimTrailingWhitespace(std::optional<bool> trimTrailingWhitespace)
      -> FormattingOptions&;

  auto insertFinalNewline(std::optional<bool> insertFinalNewline)
      -> FormattingOptions&;

  auto trimFinalNewlines(std::optional<bool> trimFinalNewlines)
      -> FormattingOptions&;
};

class DocumentFormattingOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentFormattingOptions&;
};

class DocumentRangeFormattingOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto rangesSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto rangesSupport() -> T {
    auto& value = (*repr_)["rangesSupport"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto rangesSupport(std::optional<bool> rangesSupport)
      -> DocumentRangeFormattingOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentRangeFormattingOptions&;
};

class DocumentOnTypeFormattingOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto firstTriggerCharacter() const -> std::string;

  [[nodiscard]] auto moreTriggerCharacter() const
      -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto moreTriggerCharacter() -> T {
    auto& value = (*repr_)["moreTriggerCharacter"];
    return T(value);
  }

  auto firstTriggerCharacter(std::string firstTriggerCharacter)
      -> DocumentOnTypeFormattingOptions&;

  auto moreTriggerCharacter(
      std::optional<Vector<std::string>> moreTriggerCharacter)
      -> DocumentOnTypeFormattingOptions&;

  auto moreTriggerCharacter(std::vector<std::string> moreTriggerCharacter)
      -> DocumentOnTypeFormattingOptions& {
    auto& value = (*repr_)["moreTriggerCharacter"];
    value = std::move(moreTriggerCharacter);
    return *this;
  }
};

class RenameOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto prepareProvider() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto prepareProvider() -> T {
    auto& value = (*repr_)["prepareProvider"];
    return T(value);
  }

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto prepareProvider(std::optional<bool> prepareProvider) -> RenameOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress) -> RenameOptions&;
};

class PrepareRenamePlaceholder final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto placeholder() const -> std::string;

  auto range(Range range) -> PrepareRenamePlaceholder&;

  auto placeholder(std::string placeholder) -> PrepareRenamePlaceholder&;
};

class PrepareRenameDefaultBehavior final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto defaultBehavior() const -> bool;

  auto defaultBehavior(bool defaultBehavior) -> PrepareRenameDefaultBehavior&;
};

class ExecuteCommandOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto commands() const -> Vector<std::string>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  auto commands(Vector<std::string> commands) -> ExecuteCommandOptions&;

  auto commands(std::vector<std::string> commands) -> ExecuteCommandOptions& {
    auto& value = (*repr_)["commands"];
    value = std::move(commands);
    return *this;
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ExecuteCommandOptions&;
};

class WorkspaceEditMetadata final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto isRefactoring() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto isRefactoring() -> T {
    auto& value = (*repr_)["isRefactoring"];
    return T(value);
  }

  auto isRefactoring(std::optional<bool> isRefactoring)
      -> WorkspaceEditMetadata&;
};

class SemanticTokensLegend final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto tokenTypes() const -> Vector<std::string>;

  [[nodiscard]] auto tokenModifiers() const -> Vector<std::string>;

  auto tokenTypes(Vector<std::string> tokenTypes) -> SemanticTokensLegend&;

  auto tokenTypes(std::vector<std::string> tokenTypes)
      -> SemanticTokensLegend& {
    auto& value = (*repr_)["tokenTypes"];
    value = std::move(tokenTypes);
    return *this;
  }

  auto tokenModifiers(Vector<std::string> tokenModifiers)
      -> SemanticTokensLegend&;

  auto tokenModifiers(std::vector<std::string> tokenModifiers)
      -> SemanticTokensLegend& {
    auto& value = (*repr_)["tokenModifiers"];
    value = std::move(tokenModifiers);
    return *this;
  }
};

class SemanticTokensFullDelta final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto delta() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto delta() -> T {
    auto& value = (*repr_)["delta"];
    return T(value);
  }

  auto delta(std::optional<bool> delta) -> SemanticTokensFullDelta&;
};

class OptionalVersionedTextDocumentIdentifier final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto version() const -> std::variant<int, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  [[nodiscard]] auto uri() const -> std::string;

  auto version(std::variant<int, std::nullptr_t> version)
      -> OptionalVersionedTextDocumentIdentifier&;

  auto uri(std::string uri) -> OptionalVersionedTextDocumentIdentifier&;
};

class AnnotatedTextEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto annotationId() const -> ChangeAnnotationIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto newText() const -> std::string;

  auto annotationId(ChangeAnnotationIdentifier annotationId)
      -> AnnotatedTextEdit&;

  auto range(Range range) -> AnnotatedTextEdit&;

  auto newText(std::string newText) -> AnnotatedTextEdit&;
};

class SnippetTextEdit final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto snippet() const -> StringValue;

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

  template <typename T>
  [[nodiscard]] auto annotationId() -> T {
    auto& value = (*repr_)["annotationId"];
    return T(value);
  }

  auto range(Range range) -> SnippetTextEdit&;

  auto snippet(StringValue snippet) -> SnippetTextEdit&;

  auto annotationId(std::optional<ChangeAnnotationIdentifier> annotationId)
      -> SnippetTextEdit&;
};

class ResourceOperation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

  template <typename T>
  [[nodiscard]] auto annotationId() -> T {
    auto& value = (*repr_)["annotationId"];
    return T(value);
  }

  auto kind(std::string kind) -> ResourceOperation&;

  auto annotationId(std::optional<ChangeAnnotationIdentifier> annotationId)
      -> ResourceOperation&;
};

class CreateFileOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto overwrite() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto overwrite() -> T {
    auto& value = (*repr_)["overwrite"];
    return T(value);
  }

  [[nodiscard]] auto ignoreIfExists() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto ignoreIfExists() -> T {
    auto& value = (*repr_)["ignoreIfExists"];
    return T(value);
  }

  auto overwrite(std::optional<bool> overwrite) -> CreateFileOptions&;

  auto ignoreIfExists(std::optional<bool> ignoreIfExists) -> CreateFileOptions&;
};

class RenameFileOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto overwrite() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto overwrite() -> T {
    auto& value = (*repr_)["overwrite"];
    return T(value);
  }

  [[nodiscard]] auto ignoreIfExists() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto ignoreIfExists() -> T {
    auto& value = (*repr_)["ignoreIfExists"];
    return T(value);
  }

  auto overwrite(std::optional<bool> overwrite) -> RenameFileOptions&;

  auto ignoreIfExists(std::optional<bool> ignoreIfExists) -> RenameFileOptions&;
};

class DeleteFileOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto recursive() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto recursive() -> T {
    auto& value = (*repr_)["recursive"];
    return T(value);
  }

  [[nodiscard]] auto ignoreIfNotExists() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto ignoreIfNotExists() -> T {
    auto& value = (*repr_)["ignoreIfNotExists"];
    return T(value);
  }

  auto recursive(std::optional<bool> recursive) -> DeleteFileOptions&;

  auto ignoreIfNotExists(std::optional<bool> ignoreIfNotExists)
      -> DeleteFileOptions&;
};

class FileOperationPattern final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto glob() const -> std::string;

  [[nodiscard]] auto matches() const -> std::optional<FileOperationPatternKind>;

  template <typename T>
  [[nodiscard]] auto matches() -> T {
    auto& value = (*repr_)["matches"];
    return T(value);
  }

  [[nodiscard]] auto options() const
      -> std::optional<FileOperationPatternOptions>;

  template <typename T>
  [[nodiscard]] auto options() -> T {
    auto& value = (*repr_)["options"];
    return T(value);
  }

  auto glob(std::string glob) -> FileOperationPattern&;

  auto matches(std::optional<FileOperationPatternKind> matches)
      -> FileOperationPattern&;

  auto options(std::optional<FileOperationPatternOptions> options)
      -> FileOperationPattern&;
};

class WorkspaceFullDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto version() const -> std::variant<int, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto resultId() -> T {
    auto& value = (*repr_)["resultId"];
    return T(value);
  }

  [[nodiscard]] auto items() const -> Vector<Diagnostic>;

  auto uri(std::string uri) -> WorkspaceFullDocumentDiagnosticReport&;

  auto version(std::variant<int, std::nullptr_t> version)
      -> WorkspaceFullDocumentDiagnosticReport&;

  auto kind(std::string kind) -> WorkspaceFullDocumentDiagnosticReport&;

  auto resultId(std::optional<std::string> resultId)
      -> WorkspaceFullDocumentDiagnosticReport&;

  auto items(Vector<Diagnostic> items)
      -> WorkspaceFullDocumentDiagnosticReport&;
};

class WorkspaceUnchangedDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto version() const -> std::variant<int, std::nullptr_t>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::string;

  auto uri(std::string uri) -> WorkspaceUnchangedDocumentDiagnosticReport&;

  auto version(std::variant<int, std::nullptr_t> version)
      -> WorkspaceUnchangedDocumentDiagnosticReport&;

  auto kind(std::string kind) -> WorkspaceUnchangedDocumentDiagnosticReport&;

  auto resultId(std::string resultId)
      -> WorkspaceUnchangedDocumentDiagnosticReport&;
};

class NotebookCell final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> NotebookCellKind;

  [[nodiscard]] auto document() const -> std::string;

  [[nodiscard]] auto metadata() const -> std::optional<LSPObject>;

  template <typename T>
  [[nodiscard]] auto metadata() -> T {
    auto& value = (*repr_)["metadata"];
    return T(value);
  }

  [[nodiscard]] auto executionSummary() const
      -> std::optional<ExecutionSummary>;

  template <typename T>
  [[nodiscard]] auto executionSummary() -> T {
    auto& value = (*repr_)["executionSummary"];
    return T(value);
  }

  auto kind(NotebookCellKind kind) -> NotebookCell&;

  auto document(std::string document) -> NotebookCell&;

  auto metadata(std::optional<LSPObject> metadata) -> NotebookCell&;

  auto executionSummary(std::optional<ExecutionSummary> executionSummary)
      -> NotebookCell&;
};

class NotebookDocumentFilterWithNotebook final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebook() const
      -> std::variant<std::string, NotebookDocumentFilter>;

  template <typename T>
  [[nodiscard]] auto notebook() -> T {
    auto& value = (*repr_)["notebook"];
    return T(value);
  }

  [[nodiscard]] auto cells() const
      -> std::optional<Vector<NotebookCellLanguage>>;

  template <typename T>
  [[nodiscard]] auto cells() -> T {
    auto& value = (*repr_)["cells"];
    return T(value);
  }

  auto notebook(std::variant<std::string, NotebookDocumentFilter> notebook)
      -> NotebookDocumentFilterWithNotebook&;

  auto cells(std::optional<Vector<NotebookCellLanguage>> cells)
      -> NotebookDocumentFilterWithNotebook&;
};

class NotebookDocumentFilterWithCells final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebook() const
      -> std::optional<std::variant<std::string, NotebookDocumentFilter>>;

  template <typename T>
  [[nodiscard]] auto notebook() -> T {
    auto& value = (*repr_)["notebook"];
    return T(value);
  }

  [[nodiscard]] auto cells() const -> Vector<NotebookCellLanguage>;

  auto notebook(
      std::optional<std::variant<std::string, NotebookDocumentFilter>> notebook)
      -> NotebookDocumentFilterWithCells&;

  auto cells(Vector<NotebookCellLanguage> cells)
      -> NotebookDocumentFilterWithCells&;
};

class NotebookDocumentCellChanges final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto structure() const
      -> std::optional<NotebookDocumentCellChangeStructure>;

  template <typename T>
  [[nodiscard]] auto structure() -> T {
    auto& value = (*repr_)["structure"];
    return T(value);
  }

  [[nodiscard]] auto data() const -> std::optional<Vector<NotebookCell>>;

  template <typename T>
  [[nodiscard]] auto data() -> T {
    auto& value = (*repr_)["data"];
    return T(value);
  }

  [[nodiscard]] auto textContent() const
      -> std::optional<Vector<NotebookDocumentCellContentChanges>>;

  template <typename T>
  [[nodiscard]] auto textContent() -> T {
    auto& value = (*repr_)["textContent"];
    return T(value);
  }

  auto structure(std::optional<NotebookDocumentCellChangeStructure> structure)
      -> NotebookDocumentCellChanges&;

  auto data(std::optional<Vector<NotebookCell>> data)
      -> NotebookDocumentCellChanges&;

  auto textContent(
      std::optional<Vector<NotebookDocumentCellContentChanges>> textContent)
      -> NotebookDocumentCellChanges&;
};

class SelectedCompletionInfo final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto text() const -> std::string;

  auto range(Range range) -> SelectedCompletionInfo&;

  auto text(std::string text) -> SelectedCompletionInfo&;
};

class ClientInfo final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto version() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  auto name(std::string name) -> ClientInfo&;

  auto version(std::optional<std::string> version) -> ClientInfo&;
};

class ClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workspace() const
      -> std::optional<WorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto workspace() -> T {
    auto& value = (*repr_)["workspace"];
    return T(value);
  }

  [[nodiscard]] auto textDocument() const
      -> std::optional<TextDocumentClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto textDocument() -> T {
    auto& value = (*repr_)["textDocument"];
    return T(value);
  }

  [[nodiscard]] auto notebookDocument() const
      -> std::optional<NotebookDocumentClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto notebookDocument() -> T {
    auto& value = (*repr_)["notebookDocument"];
    return T(value);
  }

  [[nodiscard]] auto window() const -> std::optional<WindowClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto window() -> T {
    auto& value = (*repr_)["window"];
    return T(value);
  }

  [[nodiscard]] auto general() const
      -> std::optional<GeneralClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto general() -> T {
    auto& value = (*repr_)["general"];
    return T(value);
  }

  [[nodiscard]] auto experimental() const -> std::optional<LSPAny>;

  template <typename T>
  [[nodiscard]] auto experimental() -> T {
    auto& value = (*repr_)["experimental"];
    return T(value);
  }

  auto workspace(std::optional<WorkspaceClientCapabilities> workspace)
      -> ClientCapabilities&;

  auto textDocument(std::optional<TextDocumentClientCapabilities> textDocument)
      -> ClientCapabilities&;

  auto notebookDocument(
      std::optional<NotebookDocumentClientCapabilities> notebookDocument)
      -> ClientCapabilities&;

  auto window(std::optional<WindowClientCapabilities> window)
      -> ClientCapabilities&;

  auto general(std::optional<GeneralClientCapabilities> general)
      -> ClientCapabilities&;

  auto experimental(std::optional<LSPAny> experimental) -> ClientCapabilities&;
};

class TextDocumentSyncOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto openClose() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto openClose() -> T {
    auto& value = (*repr_)["openClose"];
    return T(value);
  }

  [[nodiscard]] auto change() const -> std::optional<TextDocumentSyncKind>;

  template <typename T>
  [[nodiscard]] auto change() -> T {
    auto& value = (*repr_)["change"];
    return T(value);
  }

  [[nodiscard]] auto willSave() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willSave() -> T {
    auto& value = (*repr_)["willSave"];
    return T(value);
  }

  [[nodiscard]] auto willSaveWaitUntil() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willSaveWaitUntil() -> T {
    auto& value = (*repr_)["willSaveWaitUntil"];
    return T(value);
  }

  [[nodiscard]] auto save() const
      -> std::optional<std::variant<bool, SaveOptions>>;

  template <typename T>
  [[nodiscard]] auto save() -> T {
    auto& value = (*repr_)["save"];
    return T(value);
  }

  auto openClose(std::optional<bool> openClose) -> TextDocumentSyncOptions&;

  auto change(std::optional<TextDocumentSyncKind> change)
      -> TextDocumentSyncOptions&;

  auto willSave(std::optional<bool> willSave) -> TextDocumentSyncOptions&;

  auto willSaveWaitUntil(std::optional<bool> willSaveWaitUntil)
      -> TextDocumentSyncOptions&;

  auto save(std::optional<std::variant<bool, SaveOptions>> save)
      -> TextDocumentSyncOptions&;
};

class WorkspaceOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workspaceFolders() const
      -> std::optional<WorkspaceFoldersServerCapabilities>;

  template <typename T>
  [[nodiscard]] auto workspaceFolders() -> T {
    auto& value = (*repr_)["workspaceFolders"];
    return T(value);
  }

  [[nodiscard]] auto fileOperations() const
      -> std::optional<FileOperationOptions>;

  template <typename T>
  [[nodiscard]] auto fileOperations() -> T {
    auto& value = (*repr_)["fileOperations"];
    return T(value);
  }

  [[nodiscard]] auto textDocumentContent() const
      -> std::optional<std::variant<TextDocumentContentOptions,
                                    TextDocumentContentRegistrationOptions>>;

  template <typename T>
  [[nodiscard]] auto textDocumentContent() -> T {
    auto& value = (*repr_)["textDocumentContent"];
    return T(value);
  }

  auto workspaceFolders(
      std::optional<WorkspaceFoldersServerCapabilities> workspaceFolders)
      -> WorkspaceOptions&;

  auto fileOperations(std::optional<FileOperationOptions> fileOperations)
      -> WorkspaceOptions&;

  auto textDocumentContent(
      std::optional<std::variant<TextDocumentContentOptions,
                                 TextDocumentContentRegistrationOptions>>
          textDocumentContent) -> WorkspaceOptions&;
};

class TextDocumentContentChangePartial final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto rangeLength() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto rangeLength() -> T {
    auto& value = (*repr_)["rangeLength"];
    return T(value);
  }

  [[nodiscard]] auto text() const -> std::string;

  auto range(Range range) -> TextDocumentContentChangePartial&;

  auto rangeLength(std::optional<long> rangeLength)
      -> TextDocumentContentChangePartial&;

  auto text(std::string text) -> TextDocumentContentChangePartial&;
};

class TextDocumentContentChangeWholeDocument final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto text() const -> std::string;

  auto text(std::string text) -> TextDocumentContentChangeWholeDocument&;
};

class CodeDescription final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto href() const -> std::string;

  auto href(std::string href) -> CodeDescription&;
};

class DiagnosticRelatedInformation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto location() const -> Location;

  [[nodiscard]] auto message() const -> std::string;

  auto location(Location location) -> DiagnosticRelatedInformation&;

  auto message(std::string message) -> DiagnosticRelatedInformation&;
};

class EditRangeWithInsertReplace final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto insert() const -> Range;

  [[nodiscard]] auto replace() const -> Range;

  auto insert(Range insert) -> EditRangeWithInsertReplace&;

  auto replace(Range replace) -> EditRangeWithInsertReplace&;
};

class ServerCompletionItemOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto labelDetailsSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto labelDetailsSupport() -> T {
    auto& value = (*repr_)["labelDetailsSupport"];
    return T(value);
  }

  auto labelDetailsSupport(std::optional<bool> labelDetailsSupport)
      -> ServerCompletionItemOptions&;
};

class MarkedStringWithLanguage final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto language() const -> std::string;

  [[nodiscard]] auto value() const -> std::string;

  auto language(std::string language) -> MarkedStringWithLanguage&;

  auto value(std::string value) -> MarkedStringWithLanguage&;
};

class ParameterInformation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const
      -> std::variant<std::string, std::tuple<long, long>>;

  template <typename T>
  [[nodiscard]] auto label() -> T {
    auto& value = (*repr_)["label"];
    return T(value);
  }

  [[nodiscard]] auto documentation() const
      -> std::optional<std::variant<std::string, MarkupContent>>;

  template <typename T>
  [[nodiscard]] auto documentation() -> T {
    auto& value = (*repr_)["documentation"];
    return T(value);
  }

  auto label(std::variant<std::string, std::tuple<long, long>> label)
      -> ParameterInformation&;

  auto documentation(
      std::optional<std::variant<std::string, MarkupContent>> documentation)
      -> ParameterInformation&;
};

class CodeActionKindDocumentation final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> CodeActionKind;

  [[nodiscard]] auto command() const -> Command;

  auto kind(CodeActionKind kind) -> CodeActionKindDocumentation&;

  auto command(Command command) -> CodeActionKindDocumentation&;
};

class NotebookCellTextDocumentFilter final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebook() const
      -> std::variant<std::string, NotebookDocumentFilter>;

  template <typename T>
  [[nodiscard]] auto notebook() -> T {
    auto& value = (*repr_)["notebook"];
    return T(value);
  }

  [[nodiscard]] auto language() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto language() -> T {
    auto& value = (*repr_)["language"];
    return T(value);
  }

  auto notebook(std::variant<std::string, NotebookDocumentFilter> notebook)
      -> NotebookCellTextDocumentFilter&;

  auto language(std::optional<std::string> language)
      -> NotebookCellTextDocumentFilter&;
};

class FileOperationPatternOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto ignoreCase() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto ignoreCase() -> T {
    auto& value = (*repr_)["ignoreCase"];
    return T(value);
  }

  auto ignoreCase(std::optional<bool> ignoreCase)
      -> FileOperationPatternOptions&;
};

class ExecutionSummary final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto executionOrder() const -> long;

  [[nodiscard]] auto success() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto success() -> T {
    auto& value = (*repr_)["success"];
    return T(value);
  }

  auto executionOrder(long executionOrder) -> ExecutionSummary&;

  auto success(std::optional<bool> success) -> ExecutionSummary&;
};

class NotebookCellLanguage final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto language() const -> std::string;

  auto language(std::string language) -> NotebookCellLanguage&;
};

class NotebookDocumentCellChangeStructure final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto array() const -> NotebookCellArrayChange;

  [[nodiscard]] auto didOpen() const -> std::optional<Vector<TextDocumentItem>>;

  template <typename T>
  [[nodiscard]] auto didOpen() -> T {
    auto& value = (*repr_)["didOpen"];
    return T(value);
  }

  [[nodiscard]] auto didClose() const
      -> std::optional<Vector<TextDocumentIdentifier>>;

  template <typename T>
  [[nodiscard]] auto didClose() -> T {
    auto& value = (*repr_)["didClose"];
    return T(value);
  }

  auto array(NotebookCellArrayChange array)
      -> NotebookDocumentCellChangeStructure&;

  auto didOpen(std::optional<Vector<TextDocumentItem>> didOpen)
      -> NotebookDocumentCellChangeStructure&;

  auto didClose(std::optional<Vector<TextDocumentIdentifier>> didClose)
      -> NotebookDocumentCellChangeStructure&;
};

class NotebookDocumentCellContentChanges final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto document() const -> VersionedTextDocumentIdentifier;

  [[nodiscard]] auto changes() const -> Vector<TextDocumentContentChangeEvent>;

  auto document(VersionedTextDocumentIdentifier document)
      -> NotebookDocumentCellContentChanges&;

  auto changes(Vector<TextDocumentContentChangeEvent> changes)
      -> NotebookDocumentCellContentChanges&;
};

class WorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto applyEdit() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto applyEdit() -> T {
    auto& value = (*repr_)["applyEdit"];
    return T(value);
  }

  [[nodiscard]] auto workspaceEdit() const
      -> std::optional<WorkspaceEditClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto workspaceEdit() -> T {
    auto& value = (*repr_)["workspaceEdit"];
    return T(value);
  }

  [[nodiscard]] auto didChangeConfiguration() const
      -> std::optional<DidChangeConfigurationClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto didChangeConfiguration() -> T {
    auto& value = (*repr_)["didChangeConfiguration"];
    return T(value);
  }

  [[nodiscard]] auto didChangeWatchedFiles() const
      -> std::optional<DidChangeWatchedFilesClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto didChangeWatchedFiles() -> T {
    auto& value = (*repr_)["didChangeWatchedFiles"];
    return T(value);
  }

  [[nodiscard]] auto symbol() const
      -> std::optional<WorkspaceSymbolClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto symbol() -> T {
    auto& value = (*repr_)["symbol"];
    return T(value);
  }

  [[nodiscard]] auto executeCommand() const
      -> std::optional<ExecuteCommandClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto executeCommand() -> T {
    auto& value = (*repr_)["executeCommand"];
    return T(value);
  }

  [[nodiscard]] auto workspaceFolders() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workspaceFolders() -> T {
    auto& value = (*repr_)["workspaceFolders"];
    return T(value);
  }

  [[nodiscard]] auto configuration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto configuration() -> T {
    auto& value = (*repr_)["configuration"];
    return T(value);
  }

  [[nodiscard]] auto semanticTokens() const
      -> std::optional<SemanticTokensWorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto semanticTokens() -> T {
    auto& value = (*repr_)["semanticTokens"];
    return T(value);
  }

  [[nodiscard]] auto codeLens() const
      -> std::optional<CodeLensWorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto codeLens() -> T {
    auto& value = (*repr_)["codeLens"];
    return T(value);
  }

  [[nodiscard]] auto fileOperations() const
      -> std::optional<FileOperationClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto fileOperations() -> T {
    auto& value = (*repr_)["fileOperations"];
    return T(value);
  }

  [[nodiscard]] auto inlineValue() const
      -> std::optional<InlineValueWorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto inlineValue() -> T {
    auto& value = (*repr_)["inlineValue"];
    return T(value);
  }

  [[nodiscard]] auto inlayHint() const
      -> std::optional<InlayHintWorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto inlayHint() -> T {
    auto& value = (*repr_)["inlayHint"];
    return T(value);
  }

  [[nodiscard]] auto diagnostics() const
      -> std::optional<DiagnosticWorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto diagnostics() -> T {
    auto& value = (*repr_)["diagnostics"];
    return T(value);
  }

  [[nodiscard]] auto foldingRange() const
      -> std::optional<FoldingRangeWorkspaceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto foldingRange() -> T {
    auto& value = (*repr_)["foldingRange"];
    return T(value);
  }

  [[nodiscard]] auto textDocumentContent() const
      -> std::optional<TextDocumentContentClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto textDocumentContent() -> T {
    auto& value = (*repr_)["textDocumentContent"];
    return T(value);
  }

  auto applyEdit(std::optional<bool> applyEdit) -> WorkspaceClientCapabilities&;

  auto workspaceEdit(
      std::optional<WorkspaceEditClientCapabilities> workspaceEdit)
      -> WorkspaceClientCapabilities&;

  auto didChangeConfiguration(
      std::optional<DidChangeConfigurationClientCapabilities>
          didChangeConfiguration) -> WorkspaceClientCapabilities&;

  auto didChangeWatchedFiles(
      std::optional<DidChangeWatchedFilesClientCapabilities>
          didChangeWatchedFiles) -> WorkspaceClientCapabilities&;

  auto symbol(std::optional<WorkspaceSymbolClientCapabilities> symbol)
      -> WorkspaceClientCapabilities&;

  auto executeCommand(
      std::optional<ExecuteCommandClientCapabilities> executeCommand)
      -> WorkspaceClientCapabilities&;

  auto workspaceFolders(std::optional<bool> workspaceFolders)
      -> WorkspaceClientCapabilities&;

  auto configuration(std::optional<bool> configuration)
      -> WorkspaceClientCapabilities&;

  auto semanticTokens(
      std::optional<SemanticTokensWorkspaceClientCapabilities> semanticTokens)
      -> WorkspaceClientCapabilities&;

  auto codeLens(std::optional<CodeLensWorkspaceClientCapabilities> codeLens)
      -> WorkspaceClientCapabilities&;

  auto fileOperations(
      std::optional<FileOperationClientCapabilities> fileOperations)
      -> WorkspaceClientCapabilities&;

  auto inlineValue(
      std::optional<InlineValueWorkspaceClientCapabilities> inlineValue)
      -> WorkspaceClientCapabilities&;

  auto inlayHint(std::optional<InlayHintWorkspaceClientCapabilities> inlayHint)
      -> WorkspaceClientCapabilities&;

  auto diagnostics(
      std::optional<DiagnosticWorkspaceClientCapabilities> diagnostics)
      -> WorkspaceClientCapabilities&;

  auto foldingRange(
      std::optional<FoldingRangeWorkspaceClientCapabilities> foldingRange)
      -> WorkspaceClientCapabilities&;

  auto textDocumentContent(
      std::optional<TextDocumentContentClientCapabilities> textDocumentContent)
      -> WorkspaceClientCapabilities&;
};

class TextDocumentClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto synchronization() const
      -> std::optional<TextDocumentSyncClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto synchronization() -> T {
    auto& value = (*repr_)["synchronization"];
    return T(value);
  }

  [[nodiscard]] auto filters() const
      -> std::optional<TextDocumentFilterClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto filters() -> T {
    auto& value = (*repr_)["filters"];
    return T(value);
  }

  [[nodiscard]] auto completion() const
      -> std::optional<CompletionClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto completion() -> T {
    auto& value = (*repr_)["completion"];
    return T(value);
  }

  [[nodiscard]] auto hover() const -> std::optional<HoverClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto hover() -> T {
    auto& value = (*repr_)["hover"];
    return T(value);
  }

  [[nodiscard]] auto signatureHelp() const
      -> std::optional<SignatureHelpClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto signatureHelp() -> T {
    auto& value = (*repr_)["signatureHelp"];
    return T(value);
  }

  [[nodiscard]] auto declaration() const
      -> std::optional<DeclarationClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto declaration() -> T {
    auto& value = (*repr_)["declaration"];
    return T(value);
  }

  [[nodiscard]] auto definition() const
      -> std::optional<DefinitionClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto definition() -> T {
    auto& value = (*repr_)["definition"];
    return T(value);
  }

  [[nodiscard]] auto typeDefinition() const
      -> std::optional<TypeDefinitionClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto typeDefinition() -> T {
    auto& value = (*repr_)["typeDefinition"];
    return T(value);
  }

  [[nodiscard]] auto implementation() const
      -> std::optional<ImplementationClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto implementation() -> T {
    auto& value = (*repr_)["implementation"];
    return T(value);
  }

  [[nodiscard]] auto references() const
      -> std::optional<ReferenceClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto references() -> T {
    auto& value = (*repr_)["references"];
    return T(value);
  }

  [[nodiscard]] auto documentHighlight() const
      -> std::optional<DocumentHighlightClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto documentHighlight() -> T {
    auto& value = (*repr_)["documentHighlight"];
    return T(value);
  }

  [[nodiscard]] auto documentSymbol() const
      -> std::optional<DocumentSymbolClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto documentSymbol() -> T {
    auto& value = (*repr_)["documentSymbol"];
    return T(value);
  }

  [[nodiscard]] auto codeAction() const
      -> std::optional<CodeActionClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto codeAction() -> T {
    auto& value = (*repr_)["codeAction"];
    return T(value);
  }

  [[nodiscard]] auto codeLens() const
      -> std::optional<CodeLensClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto codeLens() -> T {
    auto& value = (*repr_)["codeLens"];
    return T(value);
  }

  [[nodiscard]] auto documentLink() const
      -> std::optional<DocumentLinkClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto documentLink() -> T {
    auto& value = (*repr_)["documentLink"];
    return T(value);
  }

  [[nodiscard]] auto colorProvider() const
      -> std::optional<DocumentColorClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto colorProvider() -> T {
    auto& value = (*repr_)["colorProvider"];
    return T(value);
  }

  [[nodiscard]] auto formatting() const
      -> std::optional<DocumentFormattingClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto formatting() -> T {
    auto& value = (*repr_)["formatting"];
    return T(value);
  }

  [[nodiscard]] auto rangeFormatting() const
      -> std::optional<DocumentRangeFormattingClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto rangeFormatting() -> T {
    auto& value = (*repr_)["rangeFormatting"];
    return T(value);
  }

  [[nodiscard]] auto onTypeFormatting() const
      -> std::optional<DocumentOnTypeFormattingClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto onTypeFormatting() -> T {
    auto& value = (*repr_)["onTypeFormatting"];
    return T(value);
  }

  [[nodiscard]] auto rename() const -> std::optional<RenameClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto rename() -> T {
    auto& value = (*repr_)["rename"];
    return T(value);
  }

  [[nodiscard]] auto foldingRange() const
      -> std::optional<FoldingRangeClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto foldingRange() -> T {
    auto& value = (*repr_)["foldingRange"];
    return T(value);
  }

  [[nodiscard]] auto selectionRange() const
      -> std::optional<SelectionRangeClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto selectionRange() -> T {
    auto& value = (*repr_)["selectionRange"];
    return T(value);
  }

  [[nodiscard]] auto publishDiagnostics() const
      -> std::optional<PublishDiagnosticsClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto publishDiagnostics() -> T {
    auto& value = (*repr_)["publishDiagnostics"];
    return T(value);
  }

  [[nodiscard]] auto callHierarchy() const
      -> std::optional<CallHierarchyClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto callHierarchy() -> T {
    auto& value = (*repr_)["callHierarchy"];
    return T(value);
  }

  [[nodiscard]] auto semanticTokens() const
      -> std::optional<SemanticTokensClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto semanticTokens() -> T {
    auto& value = (*repr_)["semanticTokens"];
    return T(value);
  }

  [[nodiscard]] auto linkedEditingRange() const
      -> std::optional<LinkedEditingRangeClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto linkedEditingRange() -> T {
    auto& value = (*repr_)["linkedEditingRange"];
    return T(value);
  }

  [[nodiscard]] auto moniker() const
      -> std::optional<MonikerClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto moniker() -> T {
    auto& value = (*repr_)["moniker"];
    return T(value);
  }

  [[nodiscard]] auto typeHierarchy() const
      -> std::optional<TypeHierarchyClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto typeHierarchy() -> T {
    auto& value = (*repr_)["typeHierarchy"];
    return T(value);
  }

  [[nodiscard]] auto inlineValue() const
      -> std::optional<InlineValueClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto inlineValue() -> T {
    auto& value = (*repr_)["inlineValue"];
    return T(value);
  }

  [[nodiscard]] auto inlayHint() const
      -> std::optional<InlayHintClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto inlayHint() -> T {
    auto& value = (*repr_)["inlayHint"];
    return T(value);
  }

  [[nodiscard]] auto diagnostic() const
      -> std::optional<DiagnosticClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto diagnostic() -> T {
    auto& value = (*repr_)["diagnostic"];
    return T(value);
  }

  [[nodiscard]] auto inlineCompletion() const
      -> std::optional<InlineCompletionClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto inlineCompletion() -> T {
    auto& value = (*repr_)["inlineCompletion"];
    return T(value);
  }

  auto synchronization(
      std::optional<TextDocumentSyncClientCapabilities> synchronization)
      -> TextDocumentClientCapabilities&;

  auto filters(std::optional<TextDocumentFilterClientCapabilities> filters)
      -> TextDocumentClientCapabilities&;

  auto completion(std::optional<CompletionClientCapabilities> completion)
      -> TextDocumentClientCapabilities&;

  auto hover(std::optional<HoverClientCapabilities> hover)
      -> TextDocumentClientCapabilities&;

  auto signatureHelp(
      std::optional<SignatureHelpClientCapabilities> signatureHelp)
      -> TextDocumentClientCapabilities&;

  auto declaration(std::optional<DeclarationClientCapabilities> declaration)
      -> TextDocumentClientCapabilities&;

  auto definition(std::optional<DefinitionClientCapabilities> definition)
      -> TextDocumentClientCapabilities&;

  auto typeDefinition(
      std::optional<TypeDefinitionClientCapabilities> typeDefinition)
      -> TextDocumentClientCapabilities&;

  auto implementation(
      std::optional<ImplementationClientCapabilities> implementation)
      -> TextDocumentClientCapabilities&;

  auto references(std::optional<ReferenceClientCapabilities> references)
      -> TextDocumentClientCapabilities&;

  auto documentHighlight(
      std::optional<DocumentHighlightClientCapabilities> documentHighlight)
      -> TextDocumentClientCapabilities&;

  auto documentSymbol(
      std::optional<DocumentSymbolClientCapabilities> documentSymbol)
      -> TextDocumentClientCapabilities&;

  auto codeAction(std::optional<CodeActionClientCapabilities> codeAction)
      -> TextDocumentClientCapabilities&;

  auto codeLens(std::optional<CodeLensClientCapabilities> codeLens)
      -> TextDocumentClientCapabilities&;

  auto documentLink(std::optional<DocumentLinkClientCapabilities> documentLink)
      -> TextDocumentClientCapabilities&;

  auto colorProvider(
      std::optional<DocumentColorClientCapabilities> colorProvider)
      -> TextDocumentClientCapabilities&;

  auto formatting(
      std::optional<DocumentFormattingClientCapabilities> formatting)
      -> TextDocumentClientCapabilities&;

  auto rangeFormatting(
      std::optional<DocumentRangeFormattingClientCapabilities> rangeFormatting)
      -> TextDocumentClientCapabilities&;

  auto onTypeFormatting(
      std::optional<DocumentOnTypeFormattingClientCapabilities>
          onTypeFormatting) -> TextDocumentClientCapabilities&;

  auto rename(std::optional<RenameClientCapabilities> rename)
      -> TextDocumentClientCapabilities&;

  auto foldingRange(std::optional<FoldingRangeClientCapabilities> foldingRange)
      -> TextDocumentClientCapabilities&;

  auto selectionRange(
      std::optional<SelectionRangeClientCapabilities> selectionRange)
      -> TextDocumentClientCapabilities&;

  auto publishDiagnostics(
      std::optional<PublishDiagnosticsClientCapabilities> publishDiagnostics)
      -> TextDocumentClientCapabilities&;

  auto callHierarchy(
      std::optional<CallHierarchyClientCapabilities> callHierarchy)
      -> TextDocumentClientCapabilities&;

  auto semanticTokens(
      std::optional<SemanticTokensClientCapabilities> semanticTokens)
      -> TextDocumentClientCapabilities&;

  auto linkedEditingRange(
      std::optional<LinkedEditingRangeClientCapabilities> linkedEditingRange)
      -> TextDocumentClientCapabilities&;

  auto moniker(std::optional<MonikerClientCapabilities> moniker)
      -> TextDocumentClientCapabilities&;

  auto typeHierarchy(
      std::optional<TypeHierarchyClientCapabilities> typeHierarchy)
      -> TextDocumentClientCapabilities&;

  auto inlineValue(std::optional<InlineValueClientCapabilities> inlineValue)
      -> TextDocumentClientCapabilities&;

  auto inlayHint(std::optional<InlayHintClientCapabilities> inlayHint)
      -> TextDocumentClientCapabilities&;

  auto diagnostic(std::optional<DiagnosticClientCapabilities> diagnostic)
      -> TextDocumentClientCapabilities&;

  auto inlineCompletion(
      std::optional<InlineCompletionClientCapabilities> inlineCompletion)
      -> TextDocumentClientCapabilities&;
};

class NotebookDocumentClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto synchronization() const
      -> NotebookDocumentSyncClientCapabilities;

  auto synchronization(NotebookDocumentSyncClientCapabilities synchronization)
      -> NotebookDocumentClientCapabilities&;
};

class WindowClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto workDoneProgress() -> T {
    auto& value = (*repr_)["workDoneProgress"];
    return T(value);
  }

  [[nodiscard]] auto showMessage() const
      -> std::optional<ShowMessageRequestClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto showMessage() -> T {
    auto& value = (*repr_)["showMessage"];
    return T(value);
  }

  [[nodiscard]] auto showDocument() const
      -> std::optional<ShowDocumentClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto showDocument() -> T {
    auto& value = (*repr_)["showDocument"];
    return T(value);
  }

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> WindowClientCapabilities&;

  auto showMessage(
      std::optional<ShowMessageRequestClientCapabilities> showMessage)
      -> WindowClientCapabilities&;

  auto showDocument(std::optional<ShowDocumentClientCapabilities> showDocument)
      -> WindowClientCapabilities&;
};

class GeneralClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto staleRequestSupport() const
      -> std::optional<StaleRequestSupportOptions>;

  template <typename T>
  [[nodiscard]] auto staleRequestSupport() -> T {
    auto& value = (*repr_)["staleRequestSupport"];
    return T(value);
  }

  [[nodiscard]] auto regularExpressions() const
      -> std::optional<RegularExpressionsClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto regularExpressions() -> T {
    auto& value = (*repr_)["regularExpressions"];
    return T(value);
  }

  [[nodiscard]] auto markdown() const
      -> std::optional<MarkdownClientCapabilities>;

  template <typename T>
  [[nodiscard]] auto markdown() -> T {
    auto& value = (*repr_)["markdown"];
    return T(value);
  }

  [[nodiscard]] auto positionEncodings() const
      -> std::optional<Vector<PositionEncodingKind>>;

  template <typename T>
  [[nodiscard]] auto positionEncodings() -> T {
    auto& value = (*repr_)["positionEncodings"];
    return T(value);
  }

  auto staleRequestSupport(
      std::optional<StaleRequestSupportOptions> staleRequestSupport)
      -> GeneralClientCapabilities&;

  auto regularExpressions(
      std::optional<RegularExpressionsClientCapabilities> regularExpressions)
      -> GeneralClientCapabilities&;

  auto markdown(std::optional<MarkdownClientCapabilities> markdown)
      -> GeneralClientCapabilities&;

  auto positionEncodings(
      std::optional<Vector<PositionEncodingKind>> positionEncodings)
      -> GeneralClientCapabilities&;
};

class WorkspaceFoldersServerCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto supported() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto supported() -> T {
    auto& value = (*repr_)["supported"];
    return T(value);
  }

  [[nodiscard]] auto changeNotifications() const
      -> std::optional<std::variant<std::string, bool>>;

  template <typename T>
  [[nodiscard]] auto changeNotifications() -> T {
    auto& value = (*repr_)["changeNotifications"];
    return T(value);
  }

  auto supported(std::optional<bool> supported)
      -> WorkspaceFoldersServerCapabilities&;

  auto changeNotifications(
      std::optional<std::variant<std::string, bool>> changeNotifications)
      -> WorkspaceFoldersServerCapabilities&;
};

class FileOperationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto didCreate() const
      -> std::optional<FileOperationRegistrationOptions>;

  template <typename T>
  [[nodiscard]] auto didCreate() -> T {
    auto& value = (*repr_)["didCreate"];
    return T(value);
  }

  [[nodiscard]] auto willCreate() const
      -> std::optional<FileOperationRegistrationOptions>;

  template <typename T>
  [[nodiscard]] auto willCreate() -> T {
    auto& value = (*repr_)["willCreate"];
    return T(value);
  }

  [[nodiscard]] auto didRename() const
      -> std::optional<FileOperationRegistrationOptions>;

  template <typename T>
  [[nodiscard]] auto didRename() -> T {
    auto& value = (*repr_)["didRename"];
    return T(value);
  }

  [[nodiscard]] auto willRename() const
      -> std::optional<FileOperationRegistrationOptions>;

  template <typename T>
  [[nodiscard]] auto willRename() -> T {
    auto& value = (*repr_)["willRename"];
    return T(value);
  }

  [[nodiscard]] auto didDelete() const
      -> std::optional<FileOperationRegistrationOptions>;

  template <typename T>
  [[nodiscard]] auto didDelete() -> T {
    auto& value = (*repr_)["didDelete"];
    return T(value);
  }

  [[nodiscard]] auto willDelete() const
      -> std::optional<FileOperationRegistrationOptions>;

  template <typename T>
  [[nodiscard]] auto willDelete() -> T {
    auto& value = (*repr_)["willDelete"];
    return T(value);
  }

  auto didCreate(std::optional<FileOperationRegistrationOptions> didCreate)
      -> FileOperationOptions&;

  auto willCreate(std::optional<FileOperationRegistrationOptions> willCreate)
      -> FileOperationOptions&;

  auto didRename(std::optional<FileOperationRegistrationOptions> didRename)
      -> FileOperationOptions&;

  auto willRename(std::optional<FileOperationRegistrationOptions> willRename)
      -> FileOperationOptions&;

  auto didDelete(std::optional<FileOperationRegistrationOptions> didDelete)
      -> FileOperationOptions&;

  auto willDelete(std::optional<FileOperationRegistrationOptions> willDelete)
      -> FileOperationOptions&;
};

class RelativePattern final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto baseUri() const
      -> std::variant<WorkspaceFolder, std::string>;

  template <typename T>
  [[nodiscard]] auto baseUri() -> T {
    auto& value = (*repr_)["baseUri"];
    return T(value);
  }

  [[nodiscard]] auto pattern() const -> Pattern;

  auto baseUri(std::variant<WorkspaceFolder, std::string> baseUri)
      -> RelativePattern&;

  auto pattern(Pattern pattern) -> RelativePattern&;
};

class TextDocumentFilterLanguage final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto language() const -> std::string;

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto scheme() -> T {
    auto& value = (*repr_)["scheme"];
    return T(value);
  }

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

  template <typename T>
  [[nodiscard]] auto pattern() -> T {
    auto& value = (*repr_)["pattern"];
    return T(value);
  }

  auto language(std::string language) -> TextDocumentFilterLanguage&;

  auto scheme(std::optional<std::string> scheme) -> TextDocumentFilterLanguage&;

  auto pattern(std::optional<GlobPattern> pattern)
      -> TextDocumentFilterLanguage&;
};

class TextDocumentFilterScheme final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto language() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto language() -> T {
    auto& value = (*repr_)["language"];
    return T(value);
  }

  [[nodiscard]] auto scheme() const -> std::string;

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

  template <typename T>
  [[nodiscard]] auto pattern() -> T {
    auto& value = (*repr_)["pattern"];
    return T(value);
  }

  auto language(std::optional<std::string> language)
      -> TextDocumentFilterScheme&;

  auto scheme(std::string scheme) -> TextDocumentFilterScheme&;

  auto pattern(std::optional<GlobPattern> pattern) -> TextDocumentFilterScheme&;
};

class TextDocumentFilterPattern final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto language() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto language() -> T {
    auto& value = (*repr_)["language"];
    return T(value);
  }

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto scheme() -> T {
    auto& value = (*repr_)["scheme"];
    return T(value);
  }

  [[nodiscard]] auto pattern() const -> GlobPattern;

  auto language(std::optional<std::string> language)
      -> TextDocumentFilterPattern&;

  auto scheme(std::optional<std::string> scheme) -> TextDocumentFilterPattern&;

  auto pattern(GlobPattern pattern) -> TextDocumentFilterPattern&;
};

class NotebookDocumentFilterNotebookType final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookType() const -> std::string;

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto scheme() -> T {
    auto& value = (*repr_)["scheme"];
    return T(value);
  }

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

  template <typename T>
  [[nodiscard]] auto pattern() -> T {
    auto& value = (*repr_)["pattern"];
    return T(value);
  }

  auto notebookType(std::string notebookType)
      -> NotebookDocumentFilterNotebookType&;

  auto scheme(std::optional<std::string> scheme)
      -> NotebookDocumentFilterNotebookType&;

  auto pattern(std::optional<GlobPattern> pattern)
      -> NotebookDocumentFilterNotebookType&;
};

class NotebookDocumentFilterScheme final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookType() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto notebookType() -> T {
    auto& value = (*repr_)["notebookType"];
    return T(value);
  }

  [[nodiscard]] auto scheme() const -> std::string;

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

  template <typename T>
  [[nodiscard]] auto pattern() -> T {
    auto& value = (*repr_)["pattern"];
    return T(value);
  }

  auto notebookType(std::optional<std::string> notebookType)
      -> NotebookDocumentFilterScheme&;

  auto scheme(std::string scheme) -> NotebookDocumentFilterScheme&;

  auto pattern(std::optional<GlobPattern> pattern)
      -> NotebookDocumentFilterScheme&;
};

class NotebookDocumentFilterPattern final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebookType() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto notebookType() -> T {
    auto& value = (*repr_)["notebookType"];
    return T(value);
  }

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto scheme() -> T {
    auto& value = (*repr_)["scheme"];
    return T(value);
  }

  [[nodiscard]] auto pattern() const -> GlobPattern;

  auto notebookType(std::optional<std::string> notebookType)
      -> NotebookDocumentFilterPattern&;

  auto scheme(std::optional<std::string> scheme)
      -> NotebookDocumentFilterPattern&;

  auto pattern(GlobPattern pattern) -> NotebookDocumentFilterPattern&;
};

class NotebookCellArrayChange final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto start() const -> long;

  [[nodiscard]] auto deleteCount() const -> long;

  [[nodiscard]] auto cells() const -> std::optional<Vector<NotebookCell>>;

  template <typename T>
  [[nodiscard]] auto cells() -> T {
    auto& value = (*repr_)["cells"];
    return T(value);
  }

  auto start(long start) -> NotebookCellArrayChange&;

  auto deleteCount(long deleteCount) -> NotebookCellArrayChange&;

  auto cells(std::optional<Vector<NotebookCell>> cells)
      -> NotebookCellArrayChange&;
};

class WorkspaceEditClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentChanges() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto documentChanges() -> T {
    auto& value = (*repr_)["documentChanges"];
    return T(value);
  }

  [[nodiscard]] auto resourceOperations() const
      -> std::optional<Vector<ResourceOperationKind>>;

  template <typename T>
  [[nodiscard]] auto resourceOperations() -> T {
    auto& value = (*repr_)["resourceOperations"];
    return T(value);
  }

  [[nodiscard]] auto failureHandling() const
      -> std::optional<FailureHandlingKind>;

  template <typename T>
  [[nodiscard]] auto failureHandling() -> T {
    auto& value = (*repr_)["failureHandling"];
    return T(value);
  }

  [[nodiscard]] auto normalizesLineEndings() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto normalizesLineEndings() -> T {
    auto& value = (*repr_)["normalizesLineEndings"];
    return T(value);
  }

  [[nodiscard]] auto changeAnnotationSupport() const
      -> std::optional<ChangeAnnotationsSupportOptions>;

  template <typename T>
  [[nodiscard]] auto changeAnnotationSupport() -> T {
    auto& value = (*repr_)["changeAnnotationSupport"];
    return T(value);
  }

  [[nodiscard]] auto metadataSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto metadataSupport() -> T {
    auto& value = (*repr_)["metadataSupport"];
    return T(value);
  }

  [[nodiscard]] auto snippetEditSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto snippetEditSupport() -> T {
    auto& value = (*repr_)["snippetEditSupport"];
    return T(value);
  }

  auto documentChanges(std::optional<bool> documentChanges)
      -> WorkspaceEditClientCapabilities&;

  auto resourceOperations(
      std::optional<Vector<ResourceOperationKind>> resourceOperations)
      -> WorkspaceEditClientCapabilities&;

  auto failureHandling(std::optional<FailureHandlingKind> failureHandling)
      -> WorkspaceEditClientCapabilities&;

  auto normalizesLineEndings(std::optional<bool> normalizesLineEndings)
      -> WorkspaceEditClientCapabilities&;

  auto changeAnnotationSupport(
      std::optional<ChangeAnnotationsSupportOptions> changeAnnotationSupport)
      -> WorkspaceEditClientCapabilities&;

  auto metadataSupport(std::optional<bool> metadataSupport)
      -> WorkspaceEditClientCapabilities&;

  auto snippetEditSupport(std::optional<bool> snippetEditSupport)
      -> WorkspaceEditClientCapabilities&;
};

class DidChangeConfigurationClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DidChangeConfigurationClientCapabilities&;
};

class DidChangeWatchedFilesClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto relativePatternSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto relativePatternSupport() -> T {
    auto& value = (*repr_)["relativePatternSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DidChangeWatchedFilesClientCapabilities&;

  auto relativePatternSupport(std::optional<bool> relativePatternSupport)
      -> DidChangeWatchedFilesClientCapabilities&;
};

class WorkspaceSymbolClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto symbolKind() const
      -> std::optional<ClientSymbolKindOptions>;

  template <typename T>
  [[nodiscard]] auto symbolKind() -> T {
    auto& value = (*repr_)["symbolKind"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientSymbolTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientSymbolResolveOptions>;

  template <typename T>
  [[nodiscard]] auto resolveSupport() -> T {
    auto& value = (*repr_)["resolveSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> WorkspaceSymbolClientCapabilities&;

  auto symbolKind(std::optional<ClientSymbolKindOptions> symbolKind)
      -> WorkspaceSymbolClientCapabilities&;

  auto tagSupport(std::optional<ClientSymbolTagOptions> tagSupport)
      -> WorkspaceSymbolClientCapabilities&;

  auto resolveSupport(std::optional<ClientSymbolResolveOptions> resolveSupport)
      -> WorkspaceSymbolClientCapabilities&;
};

class ExecuteCommandClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> ExecuteCommandClientCapabilities&;
};

class SemanticTokensWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto refreshSupport() -> T {
    auto& value = (*repr_)["refreshSupport"];
    return T(value);
  }

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> SemanticTokensWorkspaceClientCapabilities&;
};

class CodeLensWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto refreshSupport() -> T {
    auto& value = (*repr_)["refreshSupport"];
    return T(value);
  }

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> CodeLensWorkspaceClientCapabilities&;
};

class FileOperationClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto didCreate() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto didCreate() -> T {
    auto& value = (*repr_)["didCreate"];
    return T(value);
  }

  [[nodiscard]] auto willCreate() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willCreate() -> T {
    auto& value = (*repr_)["willCreate"];
    return T(value);
  }

  [[nodiscard]] auto didRename() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto didRename() -> T {
    auto& value = (*repr_)["didRename"];
    return T(value);
  }

  [[nodiscard]] auto willRename() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willRename() -> T {
    auto& value = (*repr_)["willRename"];
    return T(value);
  }

  [[nodiscard]] auto didDelete() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto didDelete() -> T {
    auto& value = (*repr_)["didDelete"];
    return T(value);
  }

  [[nodiscard]] auto willDelete() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willDelete() -> T {
    auto& value = (*repr_)["willDelete"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> FileOperationClientCapabilities&;

  auto didCreate(std::optional<bool> didCreate)
      -> FileOperationClientCapabilities&;

  auto willCreate(std::optional<bool> willCreate)
      -> FileOperationClientCapabilities&;

  auto didRename(std::optional<bool> didRename)
      -> FileOperationClientCapabilities&;

  auto willRename(std::optional<bool> willRename)
      -> FileOperationClientCapabilities&;

  auto didDelete(std::optional<bool> didDelete)
      -> FileOperationClientCapabilities&;

  auto willDelete(std::optional<bool> willDelete)
      -> FileOperationClientCapabilities&;
};

class InlineValueWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto refreshSupport() -> T {
    auto& value = (*repr_)["refreshSupport"];
    return T(value);
  }

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> InlineValueWorkspaceClientCapabilities&;
};

class InlayHintWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto refreshSupport() -> T {
    auto& value = (*repr_)["refreshSupport"];
    return T(value);
  }

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> InlayHintWorkspaceClientCapabilities&;
};

class DiagnosticWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto refreshSupport() -> T {
    auto& value = (*repr_)["refreshSupport"];
    return T(value);
  }

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> DiagnosticWorkspaceClientCapabilities&;
};

class FoldingRangeWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto refreshSupport() -> T {
    auto& value = (*repr_)["refreshSupport"];
    return T(value);
  }

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> FoldingRangeWorkspaceClientCapabilities&;
};

class TextDocumentContentClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> TextDocumentContentClientCapabilities&;
};

class TextDocumentSyncClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto willSave() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willSave() -> T {
    auto& value = (*repr_)["willSave"];
    return T(value);
  }

  [[nodiscard]] auto willSaveWaitUntil() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto willSaveWaitUntil() -> T {
    auto& value = (*repr_)["willSaveWaitUntil"];
    return T(value);
  }

  [[nodiscard]] auto didSave() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto didSave() -> T {
    auto& value = (*repr_)["didSave"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> TextDocumentSyncClientCapabilities&;

  auto willSave(std::optional<bool> willSave)
      -> TextDocumentSyncClientCapabilities&;

  auto willSaveWaitUntil(std::optional<bool> willSaveWaitUntil)
      -> TextDocumentSyncClientCapabilities&;

  auto didSave(std::optional<bool> didSave)
      -> TextDocumentSyncClientCapabilities&;
};

class TextDocumentFilterClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relativePatternSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto relativePatternSupport() -> T {
    auto& value = (*repr_)["relativePatternSupport"];
    return T(value);
  }

  auto relativePatternSupport(std::optional<bool> relativePatternSupport)
      -> TextDocumentFilterClientCapabilities&;
};

class CompletionClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto completionItem() const
      -> std::optional<ClientCompletionItemOptions>;

  template <typename T>
  [[nodiscard]] auto completionItem() -> T {
    auto& value = (*repr_)["completionItem"];
    return T(value);
  }

  [[nodiscard]] auto completionItemKind() const
      -> std::optional<ClientCompletionItemOptionsKind>;

  template <typename T>
  [[nodiscard]] auto completionItemKind() -> T {
    auto& value = (*repr_)["completionItemKind"];
    return T(value);
  }

  [[nodiscard]] auto insertTextMode() const -> std::optional<InsertTextMode>;

  template <typename T>
  [[nodiscard]] auto insertTextMode() -> T {
    auto& value = (*repr_)["insertTextMode"];
    return T(value);
  }

  [[nodiscard]] auto contextSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto contextSupport() -> T {
    auto& value = (*repr_)["contextSupport"];
    return T(value);
  }

  [[nodiscard]] auto completionList() const
      -> std::optional<CompletionListCapabilities>;

  template <typename T>
  [[nodiscard]] auto completionList() -> T {
    auto& value = (*repr_)["completionList"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> CompletionClientCapabilities&;

  auto completionItem(std::optional<ClientCompletionItemOptions> completionItem)
      -> CompletionClientCapabilities&;

  auto completionItemKind(
      std::optional<ClientCompletionItemOptionsKind> completionItemKind)
      -> CompletionClientCapabilities&;

  auto insertTextMode(std::optional<InsertTextMode> insertTextMode)
      -> CompletionClientCapabilities&;

  auto contextSupport(std::optional<bool> contextSupport)
      -> CompletionClientCapabilities&;

  auto completionList(std::optional<CompletionListCapabilities> completionList)
      -> CompletionClientCapabilities&;
};

class HoverClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto contentFormat() const -> std::optional<Vector<MarkupKind>>;

  template <typename T>
  [[nodiscard]] auto contentFormat() -> T {
    auto& value = (*repr_)["contentFormat"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> HoverClientCapabilities&;

  auto contentFormat(std::optional<Vector<MarkupKind>> contentFormat)
      -> HoverClientCapabilities&;
};

class SignatureHelpClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto signatureInformation() const
      -> std::optional<ClientSignatureInformationOptions>;

  template <typename T>
  [[nodiscard]] auto signatureInformation() -> T {
    auto& value = (*repr_)["signatureInformation"];
    return T(value);
  }

  [[nodiscard]] auto contextSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto contextSupport() -> T {
    auto& value = (*repr_)["contextSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> SignatureHelpClientCapabilities&;

  auto signatureInformation(
      std::optional<ClientSignatureInformationOptions> signatureInformation)
      -> SignatureHelpClientCapabilities&;

  auto contextSupport(std::optional<bool> contextSupport)
      -> SignatureHelpClientCapabilities&;
};

class DeclarationClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto linkSupport() -> T {
    auto& value = (*repr_)["linkSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DeclarationClientCapabilities&;

  auto linkSupport(std::optional<bool> linkSupport)
      -> DeclarationClientCapabilities&;
};

class DefinitionClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto linkSupport() -> T {
    auto& value = (*repr_)["linkSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DefinitionClientCapabilities&;

  auto linkSupport(std::optional<bool> linkSupport)
      -> DefinitionClientCapabilities&;
};

class TypeDefinitionClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto linkSupport() -> T {
    auto& value = (*repr_)["linkSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> TypeDefinitionClientCapabilities&;

  auto linkSupport(std::optional<bool> linkSupport)
      -> TypeDefinitionClientCapabilities&;
};

class ImplementationClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto linkSupport() -> T {
    auto& value = (*repr_)["linkSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> ImplementationClientCapabilities&;

  auto linkSupport(std::optional<bool> linkSupport)
      -> ImplementationClientCapabilities&;
};

class ReferenceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> ReferenceClientCapabilities&;
};

class DocumentHighlightClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentHighlightClientCapabilities&;
};

class DocumentSymbolClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto symbolKind() const
      -> std::optional<ClientSymbolKindOptions>;

  template <typename T>
  [[nodiscard]] auto symbolKind() -> T {
    auto& value = (*repr_)["symbolKind"];
    return T(value);
  }

  [[nodiscard]] auto hierarchicalDocumentSymbolSupport() const
      -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto hierarchicalDocumentSymbolSupport() -> T {
    auto& value = (*repr_)["hierarchicalDocumentSymbolSupport"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientSymbolTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  [[nodiscard]] auto labelSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto labelSupport() -> T {
    auto& value = (*repr_)["labelSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentSymbolClientCapabilities&;

  auto symbolKind(std::optional<ClientSymbolKindOptions> symbolKind)
      -> DocumentSymbolClientCapabilities&;

  auto hierarchicalDocumentSymbolSupport(
      std::optional<bool> hierarchicalDocumentSymbolSupport)
      -> DocumentSymbolClientCapabilities&;

  auto tagSupport(std::optional<ClientSymbolTagOptions> tagSupport)
      -> DocumentSymbolClientCapabilities&;

  auto labelSupport(std::optional<bool> labelSupport)
      -> DocumentSymbolClientCapabilities&;
};

class CodeActionClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto codeActionLiteralSupport() const
      -> std::optional<ClientCodeActionLiteralOptions>;

  template <typename T>
  [[nodiscard]] auto codeActionLiteralSupport() -> T {
    auto& value = (*repr_)["codeActionLiteralSupport"];
    return T(value);
  }

  [[nodiscard]] auto isPreferredSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto isPreferredSupport() -> T {
    auto& value = (*repr_)["isPreferredSupport"];
    return T(value);
  }

  [[nodiscard]] auto disabledSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto disabledSupport() -> T {
    auto& value = (*repr_)["disabledSupport"];
    return T(value);
  }

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dataSupport() -> T {
    auto& value = (*repr_)["dataSupport"];
    return T(value);
  }

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientCodeActionResolveOptions>;

  template <typename T>
  [[nodiscard]] auto resolveSupport() -> T {
    auto& value = (*repr_)["resolveSupport"];
    return T(value);
  }

  [[nodiscard]] auto honorsChangeAnnotations() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto honorsChangeAnnotations() -> T {
    auto& value = (*repr_)["honorsChangeAnnotations"];
    return T(value);
  }

  [[nodiscard]] auto documentationSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto documentationSupport() -> T {
    auto& value = (*repr_)["documentationSupport"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const -> std::optional<CodeActionTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> CodeActionClientCapabilities&;

  auto codeActionLiteralSupport(
      std::optional<ClientCodeActionLiteralOptions> codeActionLiteralSupport)
      -> CodeActionClientCapabilities&;

  auto isPreferredSupport(std::optional<bool> isPreferredSupport)
      -> CodeActionClientCapabilities&;

  auto disabledSupport(std::optional<bool> disabledSupport)
      -> CodeActionClientCapabilities&;

  auto dataSupport(std::optional<bool> dataSupport)
      -> CodeActionClientCapabilities&;

  auto resolveSupport(
      std::optional<ClientCodeActionResolveOptions> resolveSupport)
      -> CodeActionClientCapabilities&;

  auto honorsChangeAnnotations(std::optional<bool> honorsChangeAnnotations)
      -> CodeActionClientCapabilities&;

  auto documentationSupport(std::optional<bool> documentationSupport)
      -> CodeActionClientCapabilities&;

  auto tagSupport(std::optional<CodeActionTagOptions> tagSupport)
      -> CodeActionClientCapabilities&;
};

class CodeLensClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientCodeLensResolveOptions>;

  template <typename T>
  [[nodiscard]] auto resolveSupport() -> T {
    auto& value = (*repr_)["resolveSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> CodeLensClientCapabilities&;

  auto resolveSupport(
      std::optional<ClientCodeLensResolveOptions> resolveSupport)
      -> CodeLensClientCapabilities&;
};

class DocumentLinkClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto tooltipSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto tooltipSupport() -> T {
    auto& value = (*repr_)["tooltipSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentLinkClientCapabilities&;

  auto tooltipSupport(std::optional<bool> tooltipSupport)
      -> DocumentLinkClientCapabilities&;
};

class DocumentColorClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentColorClientCapabilities&;
};

class DocumentFormattingClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentFormattingClientCapabilities&;
};

class DocumentRangeFormattingClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto rangesSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto rangesSupport() -> T {
    auto& value = (*repr_)["rangesSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentRangeFormattingClientCapabilities&;

  auto rangesSupport(std::optional<bool> rangesSupport)
      -> DocumentRangeFormattingClientCapabilities&;
};

class DocumentOnTypeFormattingClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentOnTypeFormattingClientCapabilities&;
};

class RenameClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto prepareSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto prepareSupport() -> T {
    auto& value = (*repr_)["prepareSupport"];
    return T(value);
  }

  [[nodiscard]] auto prepareSupportDefaultBehavior() const
      -> std::optional<PrepareSupportDefaultBehavior>;

  template <typename T>
  [[nodiscard]] auto prepareSupportDefaultBehavior() -> T {
    auto& value = (*repr_)["prepareSupportDefaultBehavior"];
    return T(value);
  }

  [[nodiscard]] auto honorsChangeAnnotations() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto honorsChangeAnnotations() -> T {
    auto& value = (*repr_)["honorsChangeAnnotations"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> RenameClientCapabilities&;

  auto prepareSupport(std::optional<bool> prepareSupport)
      -> RenameClientCapabilities&;

  auto prepareSupportDefaultBehavior(
      std::optional<PrepareSupportDefaultBehavior>
          prepareSupportDefaultBehavior) -> RenameClientCapabilities&;

  auto honorsChangeAnnotations(std::optional<bool> honorsChangeAnnotations)
      -> RenameClientCapabilities&;
};

class FoldingRangeClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto rangeLimit() const -> std::optional<long>;

  template <typename T>
  [[nodiscard]] auto rangeLimit() -> T {
    auto& value = (*repr_)["rangeLimit"];
    return T(value);
  }

  [[nodiscard]] auto lineFoldingOnly() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto lineFoldingOnly() -> T {
    auto& value = (*repr_)["lineFoldingOnly"];
    return T(value);
  }

  [[nodiscard]] auto foldingRangeKind() const
      -> std::optional<ClientFoldingRangeKindOptions>;

  template <typename T>
  [[nodiscard]] auto foldingRangeKind() -> T {
    auto& value = (*repr_)["foldingRangeKind"];
    return T(value);
  }

  [[nodiscard]] auto foldingRange() const
      -> std::optional<ClientFoldingRangeOptions>;

  template <typename T>
  [[nodiscard]] auto foldingRange() -> T {
    auto& value = (*repr_)["foldingRange"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> FoldingRangeClientCapabilities&;

  auto rangeLimit(std::optional<long> rangeLimit)
      -> FoldingRangeClientCapabilities&;

  auto lineFoldingOnly(std::optional<bool> lineFoldingOnly)
      -> FoldingRangeClientCapabilities&;

  auto foldingRangeKind(
      std::optional<ClientFoldingRangeKindOptions> foldingRangeKind)
      -> FoldingRangeClientCapabilities&;

  auto foldingRange(std::optional<ClientFoldingRangeOptions> foldingRange)
      -> FoldingRangeClientCapabilities&;
};

class SelectionRangeClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> SelectionRangeClientCapabilities&;
};

class PublishDiagnosticsClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto versionSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto versionSupport() -> T {
    auto& value = (*repr_)["versionSupport"];
    return T(value);
  }

  [[nodiscard]] auto relatedInformation() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto relatedInformation() -> T {
    auto& value = (*repr_)["relatedInformation"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientDiagnosticsTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  [[nodiscard]] auto codeDescriptionSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto codeDescriptionSupport() -> T {
    auto& value = (*repr_)["codeDescriptionSupport"];
    return T(value);
  }

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dataSupport() -> T {
    auto& value = (*repr_)["dataSupport"];
    return T(value);
  }

  auto versionSupport(std::optional<bool> versionSupport)
      -> PublishDiagnosticsClientCapabilities&;

  auto relatedInformation(std::optional<bool> relatedInformation)
      -> PublishDiagnosticsClientCapabilities&;

  auto tagSupport(std::optional<ClientDiagnosticsTagOptions> tagSupport)
      -> PublishDiagnosticsClientCapabilities&;

  auto codeDescriptionSupport(std::optional<bool> codeDescriptionSupport)
      -> PublishDiagnosticsClientCapabilities&;

  auto dataSupport(std::optional<bool> dataSupport)
      -> PublishDiagnosticsClientCapabilities&;
};

class CallHierarchyClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> CallHierarchyClientCapabilities&;
};

class SemanticTokensClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto requests() const -> ClientSemanticTokensRequestOptions;

  [[nodiscard]] auto tokenTypes() const -> Vector<std::string>;

  [[nodiscard]] auto tokenModifiers() const -> Vector<std::string>;

  [[nodiscard]] auto formats() const -> Vector<TokenFormat>;

  [[nodiscard]] auto overlappingTokenSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto overlappingTokenSupport() -> T {
    auto& value = (*repr_)["overlappingTokenSupport"];
    return T(value);
  }

  [[nodiscard]] auto multilineTokenSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto multilineTokenSupport() -> T {
    auto& value = (*repr_)["multilineTokenSupport"];
    return T(value);
  }

  [[nodiscard]] auto serverCancelSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto serverCancelSupport() -> T {
    auto& value = (*repr_)["serverCancelSupport"];
    return T(value);
  }

  [[nodiscard]] auto augmentsSyntaxTokens() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto augmentsSyntaxTokens() -> T {
    auto& value = (*repr_)["augmentsSyntaxTokens"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> SemanticTokensClientCapabilities&;

  auto requests(ClientSemanticTokensRequestOptions requests)
      -> SemanticTokensClientCapabilities&;

  auto tokenTypes(Vector<std::string> tokenTypes)
      -> SemanticTokensClientCapabilities&;

  auto tokenTypes(std::vector<std::string> tokenTypes)
      -> SemanticTokensClientCapabilities& {
    auto& value = (*repr_)["tokenTypes"];
    value = std::move(tokenTypes);
    return *this;
  }

  auto tokenModifiers(Vector<std::string> tokenModifiers)
      -> SemanticTokensClientCapabilities&;

  auto tokenModifiers(std::vector<std::string> tokenModifiers)
      -> SemanticTokensClientCapabilities& {
    auto& value = (*repr_)["tokenModifiers"];
    value = std::move(tokenModifiers);
    return *this;
  }

  auto formats(Vector<TokenFormat> formats)
      -> SemanticTokensClientCapabilities&;

  auto overlappingTokenSupport(std::optional<bool> overlappingTokenSupport)
      -> SemanticTokensClientCapabilities&;

  auto multilineTokenSupport(std::optional<bool> multilineTokenSupport)
      -> SemanticTokensClientCapabilities&;

  auto serverCancelSupport(std::optional<bool> serverCancelSupport)
      -> SemanticTokensClientCapabilities&;

  auto augmentsSyntaxTokens(std::optional<bool> augmentsSyntaxTokens)
      -> SemanticTokensClientCapabilities&;
};

class LinkedEditingRangeClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> LinkedEditingRangeClientCapabilities&;
};

class MonikerClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> MonikerClientCapabilities&;
};

class TypeHierarchyClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> TypeHierarchyClientCapabilities&;
};

class InlineValueClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> InlineValueClientCapabilities&;
};

class InlayHintClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientInlayHintResolveOptions>;

  template <typename T>
  [[nodiscard]] auto resolveSupport() -> T {
    auto& value = (*repr_)["resolveSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> InlayHintClientCapabilities&;

  auto resolveSupport(
      std::optional<ClientInlayHintResolveOptions> resolveSupport)
      -> InlayHintClientCapabilities&;
};

class DiagnosticClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto relatedDocumentSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto relatedDocumentSupport() -> T {
    auto& value = (*repr_)["relatedDocumentSupport"];
    return T(value);
  }

  [[nodiscard]] auto relatedInformation() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto relatedInformation() -> T {
    auto& value = (*repr_)["relatedInformation"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientDiagnosticsTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  [[nodiscard]] auto codeDescriptionSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto codeDescriptionSupport() -> T {
    auto& value = (*repr_)["codeDescriptionSupport"];
    return T(value);
  }

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dataSupport() -> T {
    auto& value = (*repr_)["dataSupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DiagnosticClientCapabilities&;

  auto relatedDocumentSupport(std::optional<bool> relatedDocumentSupport)
      -> DiagnosticClientCapabilities&;

  auto relatedInformation(std::optional<bool> relatedInformation)
      -> DiagnosticClientCapabilities&;

  auto tagSupport(std::optional<ClientDiagnosticsTagOptions> tagSupport)
      -> DiagnosticClientCapabilities&;

  auto codeDescriptionSupport(std::optional<bool> codeDescriptionSupport)
      -> DiagnosticClientCapabilities&;

  auto dataSupport(std::optional<bool> dataSupport)
      -> DiagnosticClientCapabilities&;
};

class InlineCompletionClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> InlineCompletionClientCapabilities&;
};

class NotebookDocumentSyncClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dynamicRegistration() -> T {
    auto& value = (*repr_)["dynamicRegistration"];
    return T(value);
  }

  [[nodiscard]] auto executionSummarySupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto executionSummarySupport() -> T {
    auto& value = (*repr_)["executionSummarySupport"];
    return T(value);
  }

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> NotebookDocumentSyncClientCapabilities&;

  auto executionSummarySupport(std::optional<bool> executionSummarySupport)
      -> NotebookDocumentSyncClientCapabilities&;
};

class ShowMessageRequestClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto messageActionItem() const
      -> std::optional<ClientShowMessageActionItemOptions>;

  template <typename T>
  [[nodiscard]] auto messageActionItem() -> T {
    auto& value = (*repr_)["messageActionItem"];
    return T(value);
  }

  auto messageActionItem(
      std::optional<ClientShowMessageActionItemOptions> messageActionItem)
      -> ShowMessageRequestClientCapabilities&;
};

class ShowDocumentClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto support() const -> bool;

  auto support(bool support) -> ShowDocumentClientCapabilities&;
};

class StaleRequestSupportOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto cancel() const -> bool;

  [[nodiscard]] auto retryOnContentModified() const -> Vector<std::string>;

  auto cancel(bool cancel) -> StaleRequestSupportOptions&;

  auto retryOnContentModified(Vector<std::string> retryOnContentModified)
      -> StaleRequestSupportOptions&;

  auto retryOnContentModified(std::vector<std::string> retryOnContentModified)
      -> StaleRequestSupportOptions& {
    auto& value = (*repr_)["retryOnContentModified"];
    value = std::move(retryOnContentModified);
    return *this;
  }
};

class RegularExpressionsClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto engine() const -> RegularExpressionEngineKind;

  [[nodiscard]] auto version() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  auto engine(RegularExpressionEngineKind engine)
      -> RegularExpressionsClientCapabilities&;

  auto version(std::optional<std::string> version)
      -> RegularExpressionsClientCapabilities&;
};

class MarkdownClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto parser() const -> std::string;

  [[nodiscard]] auto version() const -> std::optional<std::string>;

  template <typename T>
  [[nodiscard]] auto version() -> T {
    auto& value = (*repr_)["version"];
    return T(value);
  }

  [[nodiscard]] auto allowedTags() const -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto allowedTags() -> T {
    auto& value = (*repr_)["allowedTags"];
    return T(value);
  }

  auto parser(std::string parser) -> MarkdownClientCapabilities&;

  auto version(std::optional<std::string> version)
      -> MarkdownClientCapabilities&;

  auto allowedTags(std::optional<Vector<std::string>> allowedTags)
      -> MarkdownClientCapabilities&;

  auto allowedTags(std::vector<std::string> allowedTags)
      -> MarkdownClientCapabilities& {
    auto& value = (*repr_)["allowedTags"];
    value = std::move(allowedTags);
    return *this;
  }
};

class ChangeAnnotationsSupportOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto groupsOnLabel() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto groupsOnLabel() -> T {
    auto& value = (*repr_)["groupsOnLabel"];
    return T(value);
  }

  auto groupsOnLabel(std::optional<bool> groupsOnLabel)
      -> ChangeAnnotationsSupportOptions&;
};

class ClientSymbolKindOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> std::optional<Vector<SymbolKind>>;

  template <typename T>
  [[nodiscard]] auto valueSet() -> T {
    auto& value = (*repr_)["valueSet"];
    return T(value);
  }

  auto valueSet(std::optional<Vector<SymbolKind>> valueSet)
      -> ClientSymbolKindOptions&;
};

class ClientSymbolTagOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> Vector<SymbolTag>;

  auto valueSet(Vector<SymbolTag> valueSet) -> ClientSymbolTagOptions&;
};

class ClientSymbolResolveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto properties() const -> Vector<std::string>;

  auto properties(Vector<std::string> properties)
      -> ClientSymbolResolveOptions&;

  auto properties(std::vector<std::string> properties)
      -> ClientSymbolResolveOptions& {
    auto& value = (*repr_)["properties"];
    value = std::move(properties);
    return *this;
  }
};

class ClientCompletionItemOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto snippetSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto snippetSupport() -> T {
    auto& value = (*repr_)["snippetSupport"];
    return T(value);
  }

  [[nodiscard]] auto commitCharactersSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto commitCharactersSupport() -> T {
    auto& value = (*repr_)["commitCharactersSupport"];
    return T(value);
  }

  [[nodiscard]] auto documentationFormat() const
      -> std::optional<Vector<MarkupKind>>;

  template <typename T>
  [[nodiscard]] auto documentationFormat() -> T {
    auto& value = (*repr_)["documentationFormat"];
    return T(value);
  }

  [[nodiscard]] auto deprecatedSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto deprecatedSupport() -> T {
    auto& value = (*repr_)["deprecatedSupport"];
    return T(value);
  }

  [[nodiscard]] auto preselectSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto preselectSupport() -> T {
    auto& value = (*repr_)["preselectSupport"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const
      -> std::optional<CompletionItemTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  [[nodiscard]] auto insertReplaceSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto insertReplaceSupport() -> T {
    auto& value = (*repr_)["insertReplaceSupport"];
    return T(value);
  }

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientCompletionItemResolveOptions>;

  template <typename T>
  [[nodiscard]] auto resolveSupport() -> T {
    auto& value = (*repr_)["resolveSupport"];
    return T(value);
  }

  [[nodiscard]] auto insertTextModeSupport() const
      -> std::optional<ClientCompletionItemInsertTextModeOptions>;

  template <typename T>
  [[nodiscard]] auto insertTextModeSupport() -> T {
    auto& value = (*repr_)["insertTextModeSupport"];
    return T(value);
  }

  [[nodiscard]] auto labelDetailsSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto labelDetailsSupport() -> T {
    auto& value = (*repr_)["labelDetailsSupport"];
    return T(value);
  }

  auto snippetSupport(std::optional<bool> snippetSupport)
      -> ClientCompletionItemOptions&;

  auto commitCharactersSupport(std::optional<bool> commitCharactersSupport)
      -> ClientCompletionItemOptions&;

  auto documentationFormat(
      std::optional<Vector<MarkupKind>> documentationFormat)
      -> ClientCompletionItemOptions&;

  auto deprecatedSupport(std::optional<bool> deprecatedSupport)
      -> ClientCompletionItemOptions&;

  auto preselectSupport(std::optional<bool> preselectSupport)
      -> ClientCompletionItemOptions&;

  auto tagSupport(std::optional<CompletionItemTagOptions> tagSupport)
      -> ClientCompletionItemOptions&;

  auto insertReplaceSupport(std::optional<bool> insertReplaceSupport)
      -> ClientCompletionItemOptions&;

  auto resolveSupport(
      std::optional<ClientCompletionItemResolveOptions> resolveSupport)
      -> ClientCompletionItemOptions&;

  auto insertTextModeSupport(
      std::optional<ClientCompletionItemInsertTextModeOptions>
          insertTextModeSupport) -> ClientCompletionItemOptions&;

  auto labelDetailsSupport(std::optional<bool> labelDetailsSupport)
      -> ClientCompletionItemOptions&;
};

class ClientCompletionItemOptionsKind final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const
      -> std::optional<Vector<CompletionItemKind>>;

  template <typename T>
  [[nodiscard]] auto valueSet() -> T {
    auto& value = (*repr_)["valueSet"];
    return T(value);
  }

  auto valueSet(std::optional<Vector<CompletionItemKind>> valueSet)
      -> ClientCompletionItemOptionsKind&;
};

class CompletionListCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto itemDefaults() const -> std::optional<Vector<std::string>>;

  template <typename T>
  [[nodiscard]] auto itemDefaults() -> T {
    auto& value = (*repr_)["itemDefaults"];
    return T(value);
  }

  [[nodiscard]] auto applyKindSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto applyKindSupport() -> T {
    auto& value = (*repr_)["applyKindSupport"];
    return T(value);
  }

  auto itemDefaults(std::optional<Vector<std::string>> itemDefaults)
      -> CompletionListCapabilities&;

  auto itemDefaults(std::vector<std::string> itemDefaults)
      -> CompletionListCapabilities& {
    auto& value = (*repr_)["itemDefaults"];
    value = std::move(itemDefaults);
    return *this;
  }

  auto applyKindSupport(std::optional<bool> applyKindSupport)
      -> CompletionListCapabilities&;
};

class ClientSignatureInformationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentationFormat() const
      -> std::optional<Vector<MarkupKind>>;

  template <typename T>
  [[nodiscard]] auto documentationFormat() -> T {
    auto& value = (*repr_)["documentationFormat"];
    return T(value);
  }

  [[nodiscard]] auto parameterInformation() const
      -> std::optional<ClientSignatureParameterInformationOptions>;

  template <typename T>
  [[nodiscard]] auto parameterInformation() -> T {
    auto& value = (*repr_)["parameterInformation"];
    return T(value);
  }

  [[nodiscard]] auto activeParameterSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto activeParameterSupport() -> T {
    auto& value = (*repr_)["activeParameterSupport"];
    return T(value);
  }

  [[nodiscard]] auto noActiveParameterSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto noActiveParameterSupport() -> T {
    auto& value = (*repr_)["noActiveParameterSupport"];
    return T(value);
  }

  auto documentationFormat(
      std::optional<Vector<MarkupKind>> documentationFormat)
      -> ClientSignatureInformationOptions&;

  auto parameterInformation(
      std::optional<ClientSignatureParameterInformationOptions>
          parameterInformation) -> ClientSignatureInformationOptions&;

  auto activeParameterSupport(std::optional<bool> activeParameterSupport)
      -> ClientSignatureInformationOptions&;

  auto noActiveParameterSupport(std::optional<bool> noActiveParameterSupport)
      -> ClientSignatureInformationOptions&;
};

class ClientCodeActionLiteralOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto codeActionKind() const -> ClientCodeActionKindOptions;

  auto codeActionKind(ClientCodeActionKindOptions codeActionKind)
      -> ClientCodeActionLiteralOptions&;
};

class ClientCodeActionResolveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto properties() const -> Vector<std::string>;

  auto properties(Vector<std::string> properties)
      -> ClientCodeActionResolveOptions&;

  auto properties(std::vector<std::string> properties)
      -> ClientCodeActionResolveOptions& {
    auto& value = (*repr_)["properties"];
    value = std::move(properties);
    return *this;
  }
};

class CodeActionTagOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> Vector<CodeActionTag>;

  auto valueSet(Vector<CodeActionTag> valueSet) -> CodeActionTagOptions&;
};

class ClientCodeLensResolveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto properties() const -> Vector<std::string>;

  auto properties(Vector<std::string> properties)
      -> ClientCodeLensResolveOptions&;

  auto properties(std::vector<std::string> properties)
      -> ClientCodeLensResolveOptions& {
    auto& value = (*repr_)["properties"];
    value = std::move(properties);
    return *this;
  }
};

class ClientFoldingRangeKindOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const
      -> std::optional<Vector<FoldingRangeKind>>;

  template <typename T>
  [[nodiscard]] auto valueSet() -> T {
    auto& value = (*repr_)["valueSet"];
    return T(value);
  }

  auto valueSet(std::optional<Vector<FoldingRangeKind>> valueSet)
      -> ClientFoldingRangeKindOptions&;
};

class ClientFoldingRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto collapsedText() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto collapsedText() -> T {
    auto& value = (*repr_)["collapsedText"];
    return T(value);
  }

  auto collapsedText(std::optional<bool> collapsedText)
      -> ClientFoldingRangeOptions&;
};

class DiagnosticsCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relatedInformation() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto relatedInformation() -> T {
    auto& value = (*repr_)["relatedInformation"];
    return T(value);
  }

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientDiagnosticsTagOptions>;

  template <typename T>
  [[nodiscard]] auto tagSupport() -> T {
    auto& value = (*repr_)["tagSupport"];
    return T(value);
  }

  [[nodiscard]] auto codeDescriptionSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto codeDescriptionSupport() -> T {
    auto& value = (*repr_)["codeDescriptionSupport"];
    return T(value);
  }

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto dataSupport() -> T {
    auto& value = (*repr_)["dataSupport"];
    return T(value);
  }

  auto relatedInformation(std::optional<bool> relatedInformation)
      -> DiagnosticsCapabilities&;

  auto tagSupport(std::optional<ClientDiagnosticsTagOptions> tagSupport)
      -> DiagnosticsCapabilities&;

  auto codeDescriptionSupport(std::optional<bool> codeDescriptionSupport)
      -> DiagnosticsCapabilities&;

  auto dataSupport(std::optional<bool> dataSupport) -> DiagnosticsCapabilities&;
};

class ClientSemanticTokensRequestOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> std::optional<std::variant<bool, json>>;

  template <typename T>
  [[nodiscard]] auto range() -> T {
    auto& value = (*repr_)["range"];
    return T(value);
  }

  [[nodiscard]] auto full() const -> std::optional<
      std::variant<bool, ClientSemanticTokensRequestFullDelta>>;

  template <typename T>
  [[nodiscard]] auto full() -> T {
    auto& value = (*repr_)["full"];
    return T(value);
  }

  auto range(std::optional<std::variant<bool, json>> range)
      -> ClientSemanticTokensRequestOptions&;

  auto full(
      std::optional<std::variant<bool, ClientSemanticTokensRequestFullDelta>>
          full) -> ClientSemanticTokensRequestOptions&;
};

class ClientInlayHintResolveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto properties() const -> Vector<std::string>;

  auto properties(Vector<std::string> properties)
      -> ClientInlayHintResolveOptions&;

  auto properties(std::vector<std::string> properties)
      -> ClientInlayHintResolveOptions& {
    auto& value = (*repr_)["properties"];
    value = std::move(properties);
    return *this;
  }
};

class ClientShowMessageActionItemOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto additionalPropertiesSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto additionalPropertiesSupport() -> T {
    auto& value = (*repr_)["additionalPropertiesSupport"];
    return T(value);
  }

  auto additionalPropertiesSupport(
      std::optional<bool> additionalPropertiesSupport)
      -> ClientShowMessageActionItemOptions&;
};

class CompletionItemTagOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> Vector<CompletionItemTag>;

  auto valueSet(Vector<CompletionItemTag> valueSet)
      -> CompletionItemTagOptions&;
};

class ClientCompletionItemResolveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto properties() const -> Vector<std::string>;

  auto properties(Vector<std::string> properties)
      -> ClientCompletionItemResolveOptions&;

  auto properties(std::vector<std::string> properties)
      -> ClientCompletionItemResolveOptions& {
    auto& value = (*repr_)["properties"];
    value = std::move(properties);
    return *this;
  }
};

class ClientCompletionItemInsertTextModeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> Vector<InsertTextMode>;

  auto valueSet(Vector<InsertTextMode> valueSet)
      -> ClientCompletionItemInsertTextModeOptions&;
};

class ClientSignatureParameterInformationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto labelOffsetSupport() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto labelOffsetSupport() -> T {
    auto& value = (*repr_)["labelOffsetSupport"];
    return T(value);
  }

  auto labelOffsetSupport(std::optional<bool> labelOffsetSupport)
      -> ClientSignatureParameterInformationOptions&;
};

class ClientCodeActionKindOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> Vector<CodeActionKind>;

  auto valueSet(Vector<CodeActionKind> valueSet)
      -> ClientCodeActionKindOptions&;
};

class ClientDiagnosticsTagOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> Vector<DiagnosticTag>;

  auto valueSet(Vector<DiagnosticTag> valueSet) -> ClientDiagnosticsTagOptions&;
};

class ClientSemanticTokensRequestFullDelta final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto delta() const -> std::optional<bool>;

  template <typename T>
  [[nodiscard]] auto delta() -> T {
    auto& value = (*repr_)["delta"];
    return T(value);
  }

  auto delta(std::optional<bool> delta)
      -> ClientSemanticTokensRequestFullDelta&;
};
}  // namespace cxx::lsp
