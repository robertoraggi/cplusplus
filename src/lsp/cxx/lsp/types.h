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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> ImplementationRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> TypeDefinitionRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentColorRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto additionalTextEdits() const
      -> std::optional<Vector<TextEdit>>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> WorkDoneProgressOptions&;
};

class TextDocumentRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> TextDocumentRegistrationOptions&;
};

class FoldingRangeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto endLine() const -> long;

  [[nodiscard]] auto endCharacter() const -> std::optional<long>;

  [[nodiscard]] auto kind() const -> std::optional<FoldingRangeKind>;

  [[nodiscard]] auto collapsedText() const -> std::optional<std::string>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> FoldingRangeRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DeclarationRegistrationOptions&;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DeclarationRegistrationOptions&;

  auto id(std::optional<std::string> id) -> DeclarationRegistrationOptions&;
};

class SelectionRangeParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto positions() const -> Vector<Position>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  auto range(Range range) -> SelectionRange&;

  auto parent(std::optional<SelectionRange> parent) -> SelectionRange&;
};

class SelectionRangeRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SelectionRangeRegistrationOptions&;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> SelectionRangeRegistrationOptions&;

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

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto selectionRange() const -> Range;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> CallHierarchyRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto legend() const -> SemanticTokensLegend;

  [[nodiscard]] auto range() const
      -> std::optional<std::variant<std::monostate, bool, json>>;

  [[nodiscard]] auto full() const -> std::optional<
      std::variant<std::monostate, bool, SemanticTokensFullDelta>>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> SemanticTokensRegistrationOptions&;

  auto legend(SemanticTokensLegend legend)
      -> SemanticTokensRegistrationOptions&;

  auto range(std::optional<std::variant<std::monostate, bool, json>> range)
      -> SemanticTokensRegistrationOptions&;

  auto full(
      std::optional<std::variant<std::monostate, bool, SemanticTokensFullDelta>>
          full) -> SemanticTokensRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto takeFocus() const -> std::optional<bool>;

  [[nodiscard]] auto selection() const -> std::optional<Range>;

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

  auto ranges(Vector<Range> ranges) -> LinkedEditingRanges&;

  auto wordPattern(std::optional<std::string> wordPattern)
      -> LinkedEditingRanges&;
};

class LinkedEditingRangeRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> LinkedEditingRangeRegistrationOptions&;

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

  [[nodiscard]] auto
  documentChanges() const -> std::optional<Vector<std::variant<
      std::monostate, TextDocumentEdit, CreateFile, RenameFile, DeleteFile>>>;

  [[nodiscard]] auto changeAnnotations() const
      -> std::optional<Map<ChangeAnnotationIdentifier, ChangeAnnotation>>;

  auto changes(std::optional<Map<std::string, Vector<TextEdit>>> changes)
      -> WorkspaceEdit&;

  auto documentChanges(
      std::optional<Vector<std::variant<std::monostate, TextDocumentEdit,
                                        CreateFile, RenameFile, DeleteFile>>>
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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> MonikerRegistrationOptions&;

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

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto selectionRange() const -> Range;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> TypeHierarchyRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineValueRegistrationOptions&;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> InlineValueRegistrationOptions&;

  auto id(std::optional<std::string> id) -> InlineValueRegistrationOptions&;
};

class InlayHintParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>>;

  [[nodiscard]] auto kind() const -> std::optional<InlayHintKind>;

  [[nodiscard]] auto textEdits() const -> std::optional<Vector<TextEdit>>;

  [[nodiscard]] auto tooltip() const -> std::optional<
      std::variant<std::monostate, std::string, MarkupContent>>;

  [[nodiscard]] auto paddingLeft() const -> std::optional<bool>;

  [[nodiscard]] auto paddingRight() const -> std::optional<bool>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  auto position(Position position) -> InlayHint&;

  auto label(
      std::variant<std::monostate, std::string, Vector<InlayHintLabelPart>>
          label) -> InlayHint&;

  auto kind(std::optional<InlayHintKind> kind) -> InlayHint&;

  auto textEdits(std::optional<Vector<TextEdit>> textEdits) -> InlayHint&;

  auto tooltip(
      std::optional<std::variant<std::monostate, std::string, MarkupContent>>
          tooltip) -> InlayHint&;

  auto paddingLeft(std::optional<bool> paddingLeft) -> InlayHint&;

  auto paddingRight(std::optional<bool> paddingRight) -> InlayHint&;

  auto data(std::optional<LSPAny> data) -> InlayHint&;
};

class InlayHintRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> InlayHintRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlayHintRegistrationOptions&;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> InlayHintRegistrationOptions&;

  auto id(std::optional<std::string> id) -> InlayHintRegistrationOptions&;
};

class DocumentDiagnosticParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto identifier() const -> std::optional<std::string>;

  [[nodiscard]] auto previousResultId() const -> std::optional<std::string>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> Map<std::string,
             std::variant<std::monostate, FullDocumentDiagnosticReport,
                          UnchangedDocumentDiagnosticReport>>;

  auto relatedDocuments(
      Map<std::string,
          std::variant<std::monostate, FullDocumentDiagnosticReport,
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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto identifier() const -> std::optional<std::string>;

  [[nodiscard]] auto interFileDependencies() const -> bool;

  [[nodiscard]] auto workspaceDiagnostics() const -> bool;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DiagnosticRegistrationOptions&;

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

  [[nodiscard]] auto previousResultIds() const -> Vector<PreviousResultId>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>;

  [[nodiscard]] auto save() const -> std::optional<bool>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto notebookSelector(
      Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                          NotebookDocumentFilterWithCells>>
          notebookSelector) -> NotebookDocumentSyncRegistrationOptions&;

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
      -> std::variant<std::monostate, std::string, StringValue>;

  [[nodiscard]] auto filterText() const -> std::optional<std::string>;

  [[nodiscard]] auto range() const -> std::optional<Range>;

  [[nodiscard]] auto command() const -> std::optional<Command>;

  auto insertText(
      std::variant<std::monostate, std::string, StringValue> insertText)
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

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineCompletionRegistrationOptions&;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> InlineCompletionRegistrationOptions&;

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

  auto schemes(Vector<std::string> schemes)
      -> TextDocumentContentRegistrationOptions&;

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

  [[nodiscard]] auto processId() const
      -> std::variant<std::monostate, int, std::nullptr_t>;

  [[nodiscard]] auto clientInfo() const -> std::optional<ClientInfo>;

  [[nodiscard]] auto locale() const -> std::optional<std::string>;

  [[nodiscard]] auto rootPath() const -> std::optional<
      std::variant<std::monostate, std::string, std::nullptr_t>>;

  [[nodiscard]] auto rootUri() const
      -> std::variant<std::monostate, std::string, std::nullptr_t>;

  [[nodiscard]] auto capabilities() const -> ClientCapabilities;

  [[nodiscard]] auto initializationOptions() const -> std::optional<LSPAny>;

  [[nodiscard]] auto trace() const -> std::optional<TraceValue>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto workspaceFolders() const -> std::optional<
      std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>;

  auto processId(std::variant<std::monostate, int, std::nullptr_t> processId)
      -> InitializeParams&;

  auto clientInfo(std::optional<ClientInfo> clientInfo) -> InitializeParams&;

  auto locale(std::optional<std::string> locale) -> InitializeParams&;

  auto rootPath(
      std::optional<std::variant<std::monostate, std::string, std::nullptr_t>>
          rootPath) -> InitializeParams&;

  auto rootUri(
      std::variant<std::monostate, std::string, std::nullptr_t> rootUri)
      -> InitializeParams&;

  auto capabilities(ClientCapabilities capabilities) -> InitializeParams&;

  auto initializationOptions(std::optional<LSPAny> initializationOptions)
      -> InitializeParams&;

  auto trace(std::optional<TraceValue> trace) -> InitializeParams&;

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> InitializeParams&;

  auto workspaceFolders(
      std::optional<
          std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>
          workspaceFolders) -> InitializeParams&;
};

class InitializeResult final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto capabilities() const -> ServerCapabilities;

  [[nodiscard]] auto serverInfo() const -> std::optional<ServerInfo>;

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

  [[nodiscard]] auto section() const -> std::optional<
      std::variant<std::monostate, std::string, Vector<std::string>>>;

  auto section(std::optional<
               std::variant<std::monostate, std::string, Vector<std::string>>>
                   section) -> DidChangeConfigurationRegistrationOptions&;
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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  auto syncKind(TextDocumentSyncKind syncKind)
      -> TextDocumentChangeRegistrationOptions&;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> TextDocumentChangeRegistrationOptions&;
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

  auto textDocument(TextDocumentIdentifier textDocument)
      -> DidSaveTextDocumentParams&;

  auto text(std::optional<std::string> text) -> DidSaveTextDocumentParams&;
};

class TextDocumentSaveRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto includeText() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> TextDocumentSaveRegistrationOptions&;

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

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto kind() const -> std::optional<CompletionItemKind>;

  [[nodiscard]] auto tags() const -> std::optional<Vector<CompletionItemTag>>;

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  [[nodiscard]] auto documentation() const -> std::optional<
      std::variant<std::monostate, std::string, MarkupContent>>;

  [[nodiscard]] auto deprecated() const -> std::optional<bool>;

  [[nodiscard]] auto preselect() const -> std::optional<bool>;

  [[nodiscard]] auto sortText() const -> std::optional<std::string>;

  [[nodiscard]] auto filterText() const -> std::optional<std::string>;

  [[nodiscard]] auto insertText() const -> std::optional<std::string>;

  [[nodiscard]] auto insertTextFormat() const
      -> std::optional<InsertTextFormat>;

  [[nodiscard]] auto insertTextMode() const -> std::optional<InsertTextMode>;

  [[nodiscard]] auto textEdit() const -> std::optional<
      std::variant<std::monostate, TextEdit, InsertReplaceEdit>>;

  [[nodiscard]] auto textEditText() const -> std::optional<std::string>;

  [[nodiscard]] auto additionalTextEdits() const
      -> std::optional<Vector<TextEdit>>;

  [[nodiscard]] auto commitCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto command() const -> std::optional<Command>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  auto label(std::string label) -> CompletionItem&;

  auto labelDetails(std::optional<CompletionItemLabelDetails> labelDetails)
      -> CompletionItem&;

  auto kind(std::optional<CompletionItemKind> kind) -> CompletionItem&;

  auto tags(std::optional<Vector<CompletionItemTag>> tags) -> CompletionItem&;

  auto detail(std::optional<std::string> detail) -> CompletionItem&;

  auto documentation(
      std::optional<std::variant<std::monostate, std::string, MarkupContent>>
          documentation) -> CompletionItem&;

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
      std::optional<std::variant<std::monostate, TextEdit, InsertReplaceEdit>>
          textEdit) -> CompletionItem&;

  auto textEditText(std::optional<std::string> textEditText) -> CompletionItem&;

  auto additionalTextEdits(std::optional<Vector<TextEdit>> additionalTextEdits)
      -> CompletionItem&;

  auto commitCharacters(std::optional<Vector<std::string>> commitCharacters)
      -> CompletionItem&;

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

  [[nodiscard]] auto applyKind() const
      -> std::optional<CompletionItemApplyKinds>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto allCommitCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto completionItem() const
      -> std::optional<ServerCompletionItemOptions>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> CompletionRegistrationOptions&;

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> CompletionRegistrationOptions&;

  auto allCommitCharacters(
      std::optional<Vector<std::string>> allCommitCharacters)
      -> CompletionRegistrationOptions&;

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
      -> std::variant<std::monostate, MarkupContent, MarkedString,
                      Vector<MarkedString>>;

  [[nodiscard]] auto range() const -> std::optional<Range>;

  auto contents(std::variant<std::monostate, MarkupContent, MarkedString,
                             Vector<MarkedString>>
                    contents) -> Hover&;

  auto range(std::optional<Range> range) -> Hover&;
};

class HoverRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> HoverRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> HoverRegistrationOptions&;
};

class SignatureHelpParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto context() const -> std::optional<SignatureHelpContext>;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto activeParameter() const
      -> std::optional<std::variant<std::monostate, long, std::nullptr_t>>;

  auto signatures(Vector<SignatureInformation> signatures) -> SignatureHelp&;

  auto activeSignature(std::optional<long> activeSignature) -> SignatureHelp&;

  auto activeParameter(
      std::optional<std::variant<std::monostate, long, std::nullptr_t>>
          activeParameter) -> SignatureHelp&;
};

class SignatureHelpRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto retriggerCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> SignatureHelpRegistrationOptions&;

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> SignatureHelpRegistrationOptions&;

  auto retriggerCharacters(
      std::optional<Vector<std::string>> retriggerCharacters)
      -> SignatureHelpRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DefinitionRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> ReferenceRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  auto range(Range range) -> DocumentHighlight&;

  auto kind(std::optional<DocumentHighlightKind> kind) -> DocumentHighlight&;
};

class DocumentHighlightRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentHighlightRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentHighlightRegistrationOptions&;
};

class DocumentSymbolParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto location() const -> Location;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  [[nodiscard]] auto containerName() const -> std::optional<std::string>;

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

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  [[nodiscard]] auto deprecated() const -> std::optional<bool>;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto selectionRange() const -> Range;

  [[nodiscard]] auto children() const -> std::optional<Vector<DocumentSymbol>>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto label() const -> std::optional<std::string>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentSymbolRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto command() const -> std::string;

  [[nodiscard]] auto arguments() const -> std::optional<Vector<LSPAny>>;

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

  [[nodiscard]] auto diagnostics() const -> std::optional<Vector<Diagnostic>>;

  [[nodiscard]] auto isPreferred() const -> std::optional<bool>;

  [[nodiscard]] auto disabled() const -> std::optional<CodeActionDisabled>;

  [[nodiscard]] auto edit() const -> std::optional<WorkspaceEdit>;

  [[nodiscard]] auto command() const -> std::optional<Command>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  [[nodiscard]] auto tags() const -> std::optional<Vector<CodeActionTag>>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto codeActionKinds() const
      -> std::optional<Vector<CodeActionKind>>;

  [[nodiscard]] auto documentation() const
      -> std::optional<Vector<CodeActionKindDocumentation>>;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> CodeActionRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, Location, LocationUriOnly>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  [[nodiscard]] auto name() const -> std::string;

  [[nodiscard]] auto kind() const -> SymbolKind;

  [[nodiscard]] auto tags() const -> std::optional<Vector<SymbolTag>>;

  [[nodiscard]] auto containerName() const -> std::optional<std::string>;

  auto location(
      std::variant<std::monostate, Location, LocationUriOnly> location)
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

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  auto range(Range range) -> CodeLens&;

  auto command(std::optional<Command> command) -> CodeLens&;

  auto data(std::optional<LSPAny> data) -> CodeLens&;
};

class CodeLensRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentSelector() const
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> CodeLensRegistrationOptions&;

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

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

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

  [[nodiscard]] auto tooltip() const -> std::optional<std::string>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentLinkRegistrationOptions&;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentFormattingRegistrationOptions&;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto rangesSupport() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentRangeFormattingRegistrationOptions&;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto firstTriggerCharacter() const -> std::string;

  [[nodiscard]] auto moreTriggerCharacter() const
      -> std::optional<Vector<std::string>>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> DocumentOnTypeFormattingRegistrationOptions&;

  auto firstTriggerCharacter(std::string firstTriggerCharacter)
      -> DocumentOnTypeFormattingRegistrationOptions&;

  auto moreTriggerCharacter(
      std::optional<Vector<std::string>> moreTriggerCharacter)
      -> DocumentOnTypeFormattingRegistrationOptions&;
};

class RenameParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto textDocument() const -> TextDocumentIdentifier;

  [[nodiscard]] auto position() const -> Position;

  [[nodiscard]] auto newName() const -> std::string;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

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
      -> std::variant<std::monostate, DocumentSelector, std::nullptr_t>;

  [[nodiscard]] auto prepareProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto documentSelector(
      std::variant<std::monostate, DocumentSelector, std::nullptr_t>
          documentSelector) -> RenameRegistrationOptions&;

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

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

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

  auto commands(Vector<std::string> commands)
      -> ExecuteCommandRegistrationOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ExecuteCommandRegistrationOptions&;
};

class ApplyWorkspaceEditParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto label() const -> std::optional<std::string>;

  [[nodiscard]] auto edit() const -> WorkspaceEdit;

  [[nodiscard]] auto metadata() const -> std::optional<WorkspaceEditMetadata>;

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

  [[nodiscard]] auto failedChange() const -> std::optional<long>;

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

  [[nodiscard]] auto message() const -> std::optional<std::string>;

  [[nodiscard]] auto percentage() const -> std::optional<long>;

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

  [[nodiscard]] auto message() const -> std::optional<std::string>;

  [[nodiscard]] auto percentage() const -> std::optional<long>;

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

  auto message(std::string message) -> LogTraceParams&;

  auto verbose(std::optional<std::string> verbose) -> LogTraceParams&;
};

class CancelParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const
      -> std::variant<std::monostate, int, std::string>;

  auto id(std::variant<std::monostate, int, std::string> id) -> CancelParams&;
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

  auto workDoneToken(std::optional<ProgressToken> workDoneToken)
      -> WorkDoneProgressParams&;
};

class PartialResultParams final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto partialResultToken() const -> std::optional<ProgressToken>;

  auto partialResultToken(std::optional<ProgressToken> partialResultToken)
      -> PartialResultParams&;
};

class LocationLink final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto originSelectionRange() const -> std::optional<Range>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ImplementationOptions&;
};

class StaticRegistrationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const -> std::optional<std::string>;

  auto id(std::optional<std::string> id) -> StaticRegistrationOptions&;
};

class TypeDefinitionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto section() const -> std::optional<std::string>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentColorOptions&;
};

class FoldingRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> FoldingRangeOptions&;
};

class DeclarationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SelectionRangeOptions&;
};

class CallHierarchyOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CallHierarchyOptions&;
};

class SemanticTokensOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto legend() const -> SemanticTokensLegend;

  [[nodiscard]] auto range() const
      -> std::optional<std::variant<std::monostate, bool, json>>;

  [[nodiscard]] auto full() const -> std::optional<
      std::variant<std::monostate, bool, SemanticTokensFullDelta>>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto legend(SemanticTokensLegend legend) -> SemanticTokensOptions&;

  auto range(std::optional<std::variant<std::monostate, bool, json>> range)
      -> SemanticTokensOptions&;

  auto full(
      std::optional<std::variant<std::monostate, bool, SemanticTokensFullDelta>>
          full) -> SemanticTokensOptions&;

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

  auto start(long start) -> SemanticTokensEdit&;

  auto deleteCount(long deleteCount) -> SemanticTokensEdit&;

  auto data(std::optional<Vector<long>> data) -> SemanticTokensEdit&;
};

class LinkedEditingRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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
      -> Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                             SnippetTextEdit>>;

  auto textDocument(OptionalVersionedTextDocumentIdentifier textDocument)
      -> TextDocumentEdit&;

  auto edits(Vector<std::variant<std::monostate, TextEdit, AnnotatedTextEdit,
                                 SnippetTextEdit>>
                 edits) -> TextDocumentEdit&;
};

class CreateFile final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto uri() const -> std::string;

  [[nodiscard]] auto options() const -> std::optional<CreateFileOptions>;

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

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

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

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

  [[nodiscard]] auto annotationId() const
      -> std::optional<ChangeAnnotationIdentifier>;

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

  [[nodiscard]] auto description() const -> std::optional<std::string>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> MonikerOptions&;
};

class TypeHierarchyOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  auto range(Range range) -> InlineValueEvaluatableExpression&;

  auto expression(std::optional<std::string> expression)
      -> InlineValueEvaluatableExpression&;
};

class InlineValueOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineValueOptions&;
};

class InlayHintLabelPart final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto value() const -> std::string;

  [[nodiscard]] auto tooltip() const -> std::optional<
      std::variant<std::monostate, std::string, MarkupContent>>;

  [[nodiscard]] auto location() const -> std::optional<Location>;

  [[nodiscard]] auto command() const -> std::optional<Command>;

  auto value(std::string value) -> InlayHintLabelPart&;

  auto tooltip(
      std::optional<std::variant<std::monostate, std::string, MarkupContent>>
          tooltip) -> InlayHintLabelPart&;

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

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto resolveProvider(std::optional<bool> resolveProvider)
      -> InlayHintOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlayHintOptions&;
};

class RelatedFullDocumentDiagnosticReport final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relatedDocuments() const -> std::optional<Map<
      std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                UnchangedDocumentDiagnosticReport>>>;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  [[nodiscard]] auto items() const -> Vector<Diagnostic>;

  auto relatedDocuments(
      std::optional<
          Map<std::string,
              std::variant<std::monostate, FullDocumentDiagnosticReport,
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

  [[nodiscard]] auto relatedDocuments() const -> std::optional<Map<
      std::string, std::variant<std::monostate, FullDocumentDiagnosticReport,
                                UnchangedDocumentDiagnosticReport>>>;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::string;

  auto relatedDocuments(
      std::optional<
          Map<std::string,
              std::variant<std::monostate, FullDocumentDiagnosticReport,
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

  [[nodiscard]] auto interFileDependencies() const -> bool;

  [[nodiscard]] auto workspaceDiagnostics() const -> bool;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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
      -> Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
                             NotebookDocumentFilterWithCells>>;

  [[nodiscard]] auto save() const -> std::optional<bool>;

  auto notebookSelector(
      Vector<std::variant<std::monostate, NotebookDocumentFilterWithNotebook,
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

  [[nodiscard]] auto cells() const
      -> std::optional<NotebookDocumentCellChanges>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> InlineCompletionOptions&;
};

class TextDocumentContentOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto schemes() const -> Vector<std::string>;

  auto schemes(Vector<std::string> schemes) -> TextDocumentContentOptions&;
};

class Registration final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto id() const -> std::string;

  [[nodiscard]] auto method() const -> std::string;

  [[nodiscard]] auto registerOptions() const -> std::optional<LSPAny>;

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

  [[nodiscard]] auto processId() const
      -> std::variant<std::monostate, int, std::nullptr_t>;

  [[nodiscard]] auto clientInfo() const -> std::optional<ClientInfo>;

  [[nodiscard]] auto locale() const -> std::optional<std::string>;

  [[nodiscard]] auto rootPath() const -> std::optional<
      std::variant<std::monostate, std::string, std::nullptr_t>>;

  [[nodiscard]] auto rootUri() const
      -> std::variant<std::monostate, std::string, std::nullptr_t>;

  [[nodiscard]] auto capabilities() const -> ClientCapabilities;

  [[nodiscard]] auto initializationOptions() const -> std::optional<LSPAny>;

  [[nodiscard]] auto trace() const -> std::optional<TraceValue>;

  [[nodiscard]] auto workDoneToken() const -> std::optional<ProgressToken>;

  auto processId(std::variant<std::monostate, int, std::nullptr_t> processId)
      -> _InitializeParams&;

  auto clientInfo(std::optional<ClientInfo> clientInfo) -> _InitializeParams&;

  auto locale(std::optional<std::string> locale) -> _InitializeParams&;

  auto rootPath(
      std::optional<std::variant<std::monostate, std::string, std::nullptr_t>>
          rootPath) -> _InitializeParams&;

  auto rootUri(
      std::variant<std::monostate, std::string, std::nullptr_t> rootUri)
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

  [[nodiscard]] auto workspaceFolders() const -> std::optional<
      std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>;

  auto workspaceFolders(
      std::optional<
          std::variant<std::monostate, Vector<WorkspaceFolder>, std::nullptr_t>>
          workspaceFolders) -> WorkspaceFoldersInitializeParams&;
};

class ServerCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto positionEncoding() const
      -> std::optional<PositionEncodingKind>;

  [[nodiscard]] auto textDocumentSync() const
      -> std::optional<std::variant<std::monostate, TextDocumentSyncOptions,
                                    TextDocumentSyncKind>>;

  [[nodiscard]] auto notebookDocumentSync() const
      -> std::optional<std::variant<std::monostate, NotebookDocumentSyncOptions,
                                    NotebookDocumentSyncRegistrationOptions>>;

  [[nodiscard]] auto completionProvider() const
      -> std::optional<CompletionOptions>;

  [[nodiscard]] auto hoverProvider() const
      -> std::optional<std::variant<std::monostate, bool, HoverOptions>>;

  [[nodiscard]] auto signatureHelpProvider() const
      -> std::optional<SignatureHelpOptions>;

  [[nodiscard]] auto declarationProvider() const
      -> std::optional<std::variant<std::monostate, bool, DeclarationOptions,
                                    DeclarationRegistrationOptions>>;

  [[nodiscard]] auto definitionProvider() const
      -> std::optional<std::variant<std::monostate, bool, DefinitionOptions>>;

  [[nodiscard]] auto typeDefinitionProvider() const
      -> std::optional<std::variant<std::monostate, bool, TypeDefinitionOptions,
                                    TypeDefinitionRegistrationOptions>>;

  [[nodiscard]] auto implementationProvider() const
      -> std::optional<std::variant<std::monostate, bool, ImplementationOptions,
                                    ImplementationRegistrationOptions>>;

  [[nodiscard]] auto referencesProvider() const
      -> std::optional<std::variant<std::monostate, bool, ReferenceOptions>>;

  [[nodiscard]] auto documentHighlightProvider() const -> std::optional<
      std::variant<std::monostate, bool, DocumentHighlightOptions>>;

  [[nodiscard]] auto documentSymbolProvider() const -> std::optional<
      std::variant<std::monostate, bool, DocumentSymbolOptions>>;

  [[nodiscard]] auto codeActionProvider() const
      -> std::optional<std::variant<std::monostate, bool, CodeActionOptions>>;

  [[nodiscard]] auto codeLensProvider() const -> std::optional<CodeLensOptions>;

  [[nodiscard]] auto documentLinkProvider() const
      -> std::optional<DocumentLinkOptions>;

  [[nodiscard]] auto colorProvider() const
      -> std::optional<std::variant<std::monostate, bool, DocumentColorOptions,
                                    DocumentColorRegistrationOptions>>;

  [[nodiscard]] auto workspaceSymbolProvider() const -> std::optional<
      std::variant<std::monostate, bool, WorkspaceSymbolOptions>>;

  [[nodiscard]] auto documentFormattingProvider() const -> std::optional<
      std::variant<std::monostate, bool, DocumentFormattingOptions>>;

  [[nodiscard]] auto documentRangeFormattingProvider() const -> std::optional<
      std::variant<std::monostate, bool, DocumentRangeFormattingOptions>>;

  [[nodiscard]] auto documentOnTypeFormattingProvider() const
      -> std::optional<DocumentOnTypeFormattingOptions>;

  [[nodiscard]] auto renameProvider() const
      -> std::optional<std::variant<std::monostate, bool, RenameOptions>>;

  [[nodiscard]] auto foldingRangeProvider() const
      -> std::optional<std::variant<std::monostate, bool, FoldingRangeOptions,
                                    FoldingRangeRegistrationOptions>>;

  [[nodiscard]] auto selectionRangeProvider() const
      -> std::optional<std::variant<std::monostate, bool, SelectionRangeOptions,
                                    SelectionRangeRegistrationOptions>>;

  [[nodiscard]] auto executeCommandProvider() const
      -> std::optional<ExecuteCommandOptions>;

  [[nodiscard]] auto callHierarchyProvider() const
      -> std::optional<std::variant<std::monostate, bool, CallHierarchyOptions,
                                    CallHierarchyRegistrationOptions>>;

  [[nodiscard]] auto linkedEditingRangeProvider() const -> std::optional<
      std::variant<std::monostate, bool, LinkedEditingRangeOptions,
                   LinkedEditingRangeRegistrationOptions>>;

  [[nodiscard]] auto semanticTokensProvider() const
      -> std::optional<std::variant<std::monostate, SemanticTokensOptions,
                                    SemanticTokensRegistrationOptions>>;

  [[nodiscard]] auto monikerProvider() const
      -> std::optional<std::variant<std::monostate, bool, MonikerOptions,
                                    MonikerRegistrationOptions>>;

  [[nodiscard]] auto typeHierarchyProvider() const
      -> std::optional<std::variant<std::monostate, bool, TypeHierarchyOptions,
                                    TypeHierarchyRegistrationOptions>>;

  [[nodiscard]] auto inlineValueProvider() const
      -> std::optional<std::variant<std::monostate, bool, InlineValueOptions,
                                    InlineValueRegistrationOptions>>;

  [[nodiscard]] auto inlayHintProvider() const
      -> std::optional<std::variant<std::monostate, bool, InlayHintOptions,
                                    InlayHintRegistrationOptions>>;

  [[nodiscard]] auto diagnosticProvider() const
      -> std::optional<std::variant<std::monostate, DiagnosticOptions,
                                    DiagnosticRegistrationOptions>>;

  [[nodiscard]] auto inlineCompletionProvider() const -> std::optional<
      std::variant<std::monostate, bool, InlineCompletionOptions>>;

  [[nodiscard]] auto workspace() const -> std::optional<WorkspaceOptions>;

  [[nodiscard]] auto experimental() const -> std::optional<LSPAny>;

  auto positionEncoding(std::optional<PositionEncodingKind> positionEncoding)
      -> ServerCapabilities&;

  auto textDocumentSync(
      std::optional<std::variant<std::monostate, TextDocumentSyncOptions,
                                 TextDocumentSyncKind>>
          textDocumentSync) -> ServerCapabilities&;

  auto notebookDocumentSync(
      std::optional<std::variant<std::monostate, NotebookDocumentSyncOptions,
                                 NotebookDocumentSyncRegistrationOptions>>
          notebookDocumentSync) -> ServerCapabilities&;

  auto completionProvider(std::optional<CompletionOptions> completionProvider)
      -> ServerCapabilities&;

  auto hoverProvider(
      std::optional<std::variant<std::monostate, bool, HoverOptions>>
          hoverProvider) -> ServerCapabilities&;

  auto signatureHelpProvider(
      std::optional<SignatureHelpOptions> signatureHelpProvider)
      -> ServerCapabilities&;

  auto declarationProvider(
      std::optional<std::variant<std::monostate, bool, DeclarationOptions,
                                 DeclarationRegistrationOptions>>
          declarationProvider) -> ServerCapabilities&;

  auto definitionProvider(
      std::optional<std::variant<std::monostate, bool, DefinitionOptions>>
          definitionProvider) -> ServerCapabilities&;

  auto typeDefinitionProvider(
      std::optional<std::variant<std::monostate, bool, TypeDefinitionOptions,
                                 TypeDefinitionRegistrationOptions>>
          typeDefinitionProvider) -> ServerCapabilities&;

  auto implementationProvider(
      std::optional<std::variant<std::monostate, bool, ImplementationOptions,
                                 ImplementationRegistrationOptions>>
          implementationProvider) -> ServerCapabilities&;

  auto referencesProvider(
      std::optional<std::variant<std::monostate, bool, ReferenceOptions>>
          referencesProvider) -> ServerCapabilities&;

  auto documentHighlightProvider(
      std::optional<
          std::variant<std::monostate, bool, DocumentHighlightOptions>>
          documentHighlightProvider) -> ServerCapabilities&;

  auto documentSymbolProvider(
      std::optional<std::variant<std::monostate, bool, DocumentSymbolOptions>>
          documentSymbolProvider) -> ServerCapabilities&;

  auto codeActionProvider(
      std::optional<std::variant<std::monostate, bool, CodeActionOptions>>
          codeActionProvider) -> ServerCapabilities&;

  auto codeLensProvider(std::optional<CodeLensOptions> codeLensProvider)
      -> ServerCapabilities&;

  auto documentLinkProvider(
      std::optional<DocumentLinkOptions> documentLinkProvider)
      -> ServerCapabilities&;

  auto colorProvider(
      std::optional<std::variant<std::monostate, bool, DocumentColorOptions,
                                 DocumentColorRegistrationOptions>>
          colorProvider) -> ServerCapabilities&;

  auto workspaceSymbolProvider(
      std::optional<std::variant<std::monostate, bool, WorkspaceSymbolOptions>>
          workspaceSymbolProvider) -> ServerCapabilities&;

  auto documentFormattingProvider(
      std::optional<
          std::variant<std::monostate, bool, DocumentFormattingOptions>>
          documentFormattingProvider) -> ServerCapabilities&;

  auto documentRangeFormattingProvider(
      std::optional<
          std::variant<std::monostate, bool, DocumentRangeFormattingOptions>>
          documentRangeFormattingProvider) -> ServerCapabilities&;

  auto documentOnTypeFormattingProvider(
      std::optional<DocumentOnTypeFormattingOptions>
          documentOnTypeFormattingProvider) -> ServerCapabilities&;

  auto renameProvider(
      std::optional<std::variant<std::monostate, bool, RenameOptions>>
          renameProvider) -> ServerCapabilities&;

  auto foldingRangeProvider(
      std::optional<std::variant<std::monostate, bool, FoldingRangeOptions,
                                 FoldingRangeRegistrationOptions>>
          foldingRangeProvider) -> ServerCapabilities&;

  auto selectionRangeProvider(
      std::optional<std::variant<std::monostate, bool, SelectionRangeOptions,
                                 SelectionRangeRegistrationOptions>>
          selectionRangeProvider) -> ServerCapabilities&;

  auto executeCommandProvider(
      std::optional<ExecuteCommandOptions> executeCommandProvider)
      -> ServerCapabilities&;

  auto callHierarchyProvider(
      std::optional<std::variant<std::monostate, bool, CallHierarchyOptions,
                                 CallHierarchyRegistrationOptions>>
          callHierarchyProvider) -> ServerCapabilities&;

  auto linkedEditingRangeProvider(
      std::optional<
          std::variant<std::monostate, bool, LinkedEditingRangeOptions,
                       LinkedEditingRangeRegistrationOptions>>
          linkedEditingRangeProvider) -> ServerCapabilities&;

  auto semanticTokensProvider(
      std::optional<std::variant<std::monostate, SemanticTokensOptions,
                                 SemanticTokensRegistrationOptions>>
          semanticTokensProvider) -> ServerCapabilities&;

  auto monikerProvider(
      std::optional<std::variant<std::monostate, bool, MonikerOptions,
                                 MonikerRegistrationOptions>>
          monikerProvider) -> ServerCapabilities&;

  auto typeHierarchyProvider(
      std::optional<std::variant<std::monostate, bool, TypeHierarchyOptions,
                                 TypeHierarchyRegistrationOptions>>
          typeHierarchyProvider) -> ServerCapabilities&;

  auto inlineValueProvider(
      std::optional<std::variant<std::monostate, bool, InlineValueOptions,
                                 InlineValueRegistrationOptions>>
          inlineValueProvider) -> ServerCapabilities&;

  auto inlayHintProvider(
      std::optional<std::variant<std::monostate, bool, InlayHintOptions,
                                 InlayHintRegistrationOptions>>
          inlayHintProvider) -> ServerCapabilities&;

  auto diagnosticProvider(
      std::optional<std::variant<std::monostate, DiagnosticOptions,
                                 DiagnosticRegistrationOptions>>
          diagnosticProvider) -> ServerCapabilities&;

  auto inlineCompletionProvider(
      std::optional<std::variant<std::monostate, bool, InlineCompletionOptions>>
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

  auto globPattern(GlobPattern globPattern) -> FileSystemWatcher&;

  auto kind(std::optional<WatchKind> kind) -> FileSystemWatcher&;
};

class Diagnostic final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto severity() const -> std::optional<DiagnosticSeverity>;

  [[nodiscard]] auto code() const
      -> std::optional<std::variant<std::monostate, int, std::string>>;

  [[nodiscard]] auto codeDescription() const -> std::optional<CodeDescription>;

  [[nodiscard]] auto source() const -> std::optional<std::string>;

  [[nodiscard]] auto message() const -> std::string;

  [[nodiscard]] auto tags() const -> std::optional<Vector<DiagnosticTag>>;

  [[nodiscard]] auto relatedInformation() const
      -> std::optional<Vector<DiagnosticRelatedInformation>>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  auto range(Range range) -> Diagnostic&;

  auto severity(std::optional<DiagnosticSeverity> severity) -> Diagnostic&;

  auto code(std::optional<std::variant<std::monostate, int, std::string>> code)
      -> Diagnostic&;

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

  auto triggerKind(CompletionTriggerKind triggerKind) -> CompletionContext&;

  auto triggerCharacter(std::optional<std::string> triggerCharacter)
      -> CompletionContext&;
};

class CompletionItemLabelDetails final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto detail() const -> std::optional<std::string>;

  [[nodiscard]] auto description() const -> std::optional<std::string>;

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

  [[nodiscard]] auto editRange() const -> std::optional<
      std::variant<std::monostate, Range, EditRangeWithInsertReplace>>;

  [[nodiscard]] auto insertTextFormat() const
      -> std::optional<InsertTextFormat>;

  [[nodiscard]] auto insertTextMode() const -> std::optional<InsertTextMode>;

  [[nodiscard]] auto data() const -> std::optional<LSPAny>;

  auto commitCharacters(std::optional<Vector<std::string>> commitCharacters)
      -> CompletionItemDefaults&;

  auto editRange(
      std::optional<
          std::variant<std::monostate, Range, EditRangeWithInsertReplace>>
          editRange) -> CompletionItemDefaults&;

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

  [[nodiscard]] auto data() const -> std::optional<ApplyKind>;

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

  [[nodiscard]] auto allCommitCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto completionItem() const
      -> std::optional<ServerCompletionItemOptions>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> CompletionOptions&;

  auto allCommitCharacters(
      std::optional<Vector<std::string>> allCommitCharacters)
      -> CompletionOptions&;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress) -> HoverOptions&;
};

class SignatureHelpContext final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerKind() const -> SignatureHelpTriggerKind;

  [[nodiscard]] auto triggerCharacter() const -> std::optional<std::string>;

  [[nodiscard]] auto isRetrigger() const -> bool;

  [[nodiscard]] auto activeSignatureHelp() const
      -> std::optional<SignatureHelp>;

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

  [[nodiscard]] auto documentation() const -> std::optional<
      std::variant<std::monostate, std::string, MarkupContent>>;

  [[nodiscard]] auto parameters() const
      -> std::optional<Vector<ParameterInformation>>;

  [[nodiscard]] auto activeParameter() const
      -> std::optional<std::variant<std::monostate, long, std::nullptr_t>>;

  auto label(std::string label) -> SignatureInformation&;

  auto documentation(
      std::optional<std::variant<std::monostate, std::string, MarkupContent>>
          documentation) -> SignatureInformation&;

  auto parameters(std::optional<Vector<ParameterInformation>> parameters)
      -> SignatureInformation&;

  auto activeParameter(
      std::optional<std::variant<std::monostate, long, std::nullptr_t>>
          activeParameter) -> SignatureInformation&;
};

class SignatureHelpOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto triggerCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto retriggerCharacters() const
      -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto triggerCharacters(std::optional<Vector<std::string>> triggerCharacters)
      -> SignatureHelpOptions&;

  auto retriggerCharacters(
      std::optional<Vector<std::string>> retriggerCharacters)
      -> SignatureHelpOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> SignatureHelpOptions&;
};

class DefinitionOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ReferenceOptions&;
};

class DocumentHighlightOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto containerName() const -> std::optional<std::string>;

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

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto triggerKind() const
      -> std::optional<CodeActionTriggerKind>;

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

  [[nodiscard]] auto documentation() const
      -> std::optional<Vector<CodeActionKindDocumentation>>;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

  auto resolveProvider(std::optional<bool> resolveProvider) -> CodeLensOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> CodeLensOptions&;
};

class DocumentLinkOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto resolveProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  [[nodiscard]] auto insertFinalNewline() const -> std::optional<bool>;

  [[nodiscard]] auto trimFinalNewlines() const -> std::optional<bool>;

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

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> DocumentFormattingOptions&;
};

class DocumentRangeFormattingOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto rangesSupport() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  auto firstTriggerCharacter(std::string firstTriggerCharacter)
      -> DocumentOnTypeFormattingOptions&;

  auto moreTriggerCharacter(
      std::optional<Vector<std::string>> moreTriggerCharacter)
      -> DocumentOnTypeFormattingOptions&;
};

class RenameOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto prepareProvider() const -> std::optional<bool>;

  [[nodiscard]] auto workDoneProgress() const -> std::optional<bool>;

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

  auto commands(Vector<std::string> commands) -> ExecuteCommandOptions&;

  auto workDoneProgress(std::optional<bool> workDoneProgress)
      -> ExecuteCommandOptions&;
};

class WorkspaceEditMetadata final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto isRefactoring() const -> std::optional<bool>;

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

  auto tokenModifiers(Vector<std::string> tokenModifiers)
      -> SemanticTokensLegend&;
};

class SemanticTokensFullDelta final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto delta() const -> std::optional<bool>;

  auto delta(std::optional<bool> delta) -> SemanticTokensFullDelta&;
};

class OptionalVersionedTextDocumentIdentifier final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto version() const
      -> std::variant<std::monostate, int, std::nullptr_t>;

  [[nodiscard]] auto uri() const -> std::string;

  auto version(std::variant<std::monostate, int, std::nullptr_t> version)
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

  auto kind(std::string kind) -> ResourceOperation&;

  auto annotationId(std::optional<ChangeAnnotationIdentifier> annotationId)
      -> ResourceOperation&;
};

class CreateFileOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto overwrite() const -> std::optional<bool>;

  [[nodiscard]] auto ignoreIfExists() const -> std::optional<bool>;

  auto overwrite(std::optional<bool> overwrite) -> CreateFileOptions&;

  auto ignoreIfExists(std::optional<bool> ignoreIfExists) -> CreateFileOptions&;
};

class RenameFileOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto overwrite() const -> std::optional<bool>;

  [[nodiscard]] auto ignoreIfExists() const -> std::optional<bool>;

  auto overwrite(std::optional<bool> overwrite) -> RenameFileOptions&;

  auto ignoreIfExists(std::optional<bool> ignoreIfExists) -> RenameFileOptions&;
};

class DeleteFileOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto recursive() const -> std::optional<bool>;

  [[nodiscard]] auto ignoreIfNotExists() const -> std::optional<bool>;

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

  [[nodiscard]] auto options() const
      -> std::optional<FileOperationPatternOptions>;

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

  [[nodiscard]] auto version() const
      -> std::variant<std::monostate, int, std::nullptr_t>;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::optional<std::string>;

  [[nodiscard]] auto items() const -> Vector<Diagnostic>;

  auto uri(std::string uri) -> WorkspaceFullDocumentDiagnosticReport&;

  auto version(std::variant<std::monostate, int, std::nullptr_t> version)
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

  [[nodiscard]] auto version() const
      -> std::variant<std::monostate, int, std::nullptr_t>;

  [[nodiscard]] auto kind() const -> std::string;

  [[nodiscard]] auto resultId() const -> std::string;

  auto uri(std::string uri) -> WorkspaceUnchangedDocumentDiagnosticReport&;

  auto version(std::variant<std::monostate, int, std::nullptr_t> version)
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

  [[nodiscard]] auto executionSummary() const
      -> std::optional<ExecutionSummary>;

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
      -> std::variant<std::monostate, std::string, NotebookDocumentFilter>;

  [[nodiscard]] auto cells() const
      -> std::optional<Vector<NotebookCellLanguage>>;

  auto notebook(
      std::variant<std::monostate, std::string, NotebookDocumentFilter>
          notebook) -> NotebookDocumentFilterWithNotebook&;

  auto cells(std::optional<Vector<NotebookCellLanguage>> cells)
      -> NotebookDocumentFilterWithNotebook&;
};

class NotebookDocumentFilterWithCells final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto notebook() const -> std::optional<
      std::variant<std::monostate, std::string, NotebookDocumentFilter>>;

  [[nodiscard]] auto cells() const -> Vector<NotebookCellLanguage>;

  auto notebook(
      std::optional<
          std::variant<std::monostate, std::string, NotebookDocumentFilter>>
          notebook) -> NotebookDocumentFilterWithCells&;

  auto cells(Vector<NotebookCellLanguage> cells)
      -> NotebookDocumentFilterWithCells&;
};

class NotebookDocumentCellChanges final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto structure() const
      -> std::optional<NotebookDocumentCellChangeStructure>;

  [[nodiscard]] auto data() const -> std::optional<Vector<NotebookCell>>;

  [[nodiscard]] auto textContent() const
      -> std::optional<Vector<NotebookDocumentCellContentChanges>>;

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

  auto name(std::string name) -> ClientInfo&;

  auto version(std::optional<std::string> version) -> ClientInfo&;
};

class ClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workspace() const
      -> std::optional<WorkspaceClientCapabilities>;

  [[nodiscard]] auto textDocument() const
      -> std::optional<TextDocumentClientCapabilities>;

  [[nodiscard]] auto notebookDocument() const
      -> std::optional<NotebookDocumentClientCapabilities>;

  [[nodiscard]] auto window() const -> std::optional<WindowClientCapabilities>;

  [[nodiscard]] auto general() const
      -> std::optional<GeneralClientCapabilities>;

  [[nodiscard]] auto experimental() const -> std::optional<LSPAny>;

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

  [[nodiscard]] auto change() const -> std::optional<TextDocumentSyncKind>;

  [[nodiscard]] auto willSave() const -> std::optional<bool>;

  [[nodiscard]] auto willSaveWaitUntil() const -> std::optional<bool>;

  [[nodiscard]] auto save() const
      -> std::optional<std::variant<std::monostate, bool, SaveOptions>>;

  auto openClose(std::optional<bool> openClose) -> TextDocumentSyncOptions&;

  auto change(std::optional<TextDocumentSyncKind> change)
      -> TextDocumentSyncOptions&;

  auto willSave(std::optional<bool> willSave) -> TextDocumentSyncOptions&;

  auto willSaveWaitUntil(std::optional<bool> willSaveWaitUntil)
      -> TextDocumentSyncOptions&;

  auto save(std::optional<std::variant<std::monostate, bool, SaveOptions>> save)
      -> TextDocumentSyncOptions&;
};

class WorkspaceOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto workspaceFolders() const
      -> std::optional<WorkspaceFoldersServerCapabilities>;

  [[nodiscard]] auto fileOperations() const
      -> std::optional<FileOperationOptions>;

  [[nodiscard]] auto textDocumentContent() const
      -> std::optional<std::variant<std::monostate, TextDocumentContentOptions,
                                    TextDocumentContentRegistrationOptions>>;

  auto workspaceFolders(
      std::optional<WorkspaceFoldersServerCapabilities> workspaceFolders)
      -> WorkspaceOptions&;

  auto fileOperations(std::optional<FileOperationOptions> fileOperations)
      -> WorkspaceOptions&;

  auto textDocumentContent(
      std::optional<std::variant<std::monostate, TextDocumentContentOptions,
                                 TextDocumentContentRegistrationOptions>>
          textDocumentContent) -> WorkspaceOptions&;
};

class TextDocumentContentChangePartial final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto range() const -> Range;

  [[nodiscard]] auto rangeLength() const -> std::optional<long>;

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
      -> std::variant<std::monostate, std::string, std::tuple<long, long>>;

  [[nodiscard]] auto documentation() const -> std::optional<
      std::variant<std::monostate, std::string, MarkupContent>>;

  auto label(
      std::variant<std::monostate, std::string, std::tuple<long, long>> label)
      -> ParameterInformation&;

  auto documentation(
      std::optional<std::variant<std::monostate, std::string, MarkupContent>>
          documentation) -> ParameterInformation&;
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
      -> std::variant<std::monostate, std::string, NotebookDocumentFilter>;

  [[nodiscard]] auto language() const -> std::optional<std::string>;

  auto notebook(
      std::variant<std::monostate, std::string, NotebookDocumentFilter>
          notebook) -> NotebookCellTextDocumentFilter&;

  auto language(std::optional<std::string> language)
      -> NotebookCellTextDocumentFilter&;
};

class FileOperationPatternOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto ignoreCase() const -> std::optional<bool>;

  auto ignoreCase(std::optional<bool> ignoreCase)
      -> FileOperationPatternOptions&;
};

class ExecutionSummary final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto executionOrder() const -> long;

  [[nodiscard]] auto success() const -> std::optional<bool>;

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

  [[nodiscard]] auto didClose() const
      -> std::optional<Vector<TextDocumentIdentifier>>;

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

  [[nodiscard]] auto workspaceEdit() const
      -> std::optional<WorkspaceEditClientCapabilities>;

  [[nodiscard]] auto didChangeConfiguration() const
      -> std::optional<DidChangeConfigurationClientCapabilities>;

  [[nodiscard]] auto didChangeWatchedFiles() const
      -> std::optional<DidChangeWatchedFilesClientCapabilities>;

  [[nodiscard]] auto symbol() const
      -> std::optional<WorkspaceSymbolClientCapabilities>;

  [[nodiscard]] auto executeCommand() const
      -> std::optional<ExecuteCommandClientCapabilities>;

  [[nodiscard]] auto workspaceFolders() const -> std::optional<bool>;

  [[nodiscard]] auto configuration() const -> std::optional<bool>;

  [[nodiscard]] auto semanticTokens() const
      -> std::optional<SemanticTokensWorkspaceClientCapabilities>;

  [[nodiscard]] auto codeLens() const
      -> std::optional<CodeLensWorkspaceClientCapabilities>;

  [[nodiscard]] auto fileOperations() const
      -> std::optional<FileOperationClientCapabilities>;

  [[nodiscard]] auto inlineValue() const
      -> std::optional<InlineValueWorkspaceClientCapabilities>;

  [[nodiscard]] auto inlayHint() const
      -> std::optional<InlayHintWorkspaceClientCapabilities>;

  [[nodiscard]] auto diagnostics() const
      -> std::optional<DiagnosticWorkspaceClientCapabilities>;

  [[nodiscard]] auto foldingRange() const
      -> std::optional<FoldingRangeWorkspaceClientCapabilities>;

  [[nodiscard]] auto textDocumentContent() const
      -> std::optional<TextDocumentContentClientCapabilities>;

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

  [[nodiscard]] auto filters() const
      -> std::optional<TextDocumentFilterClientCapabilities>;

  [[nodiscard]] auto completion() const
      -> std::optional<CompletionClientCapabilities>;

  [[nodiscard]] auto hover() const -> std::optional<HoverClientCapabilities>;

  [[nodiscard]] auto signatureHelp() const
      -> std::optional<SignatureHelpClientCapabilities>;

  [[nodiscard]] auto declaration() const
      -> std::optional<DeclarationClientCapabilities>;

  [[nodiscard]] auto definition() const
      -> std::optional<DefinitionClientCapabilities>;

  [[nodiscard]] auto typeDefinition() const
      -> std::optional<TypeDefinitionClientCapabilities>;

  [[nodiscard]] auto implementation() const
      -> std::optional<ImplementationClientCapabilities>;

  [[nodiscard]] auto references() const
      -> std::optional<ReferenceClientCapabilities>;

  [[nodiscard]] auto documentHighlight() const
      -> std::optional<DocumentHighlightClientCapabilities>;

  [[nodiscard]] auto documentSymbol() const
      -> std::optional<DocumentSymbolClientCapabilities>;

  [[nodiscard]] auto codeAction() const
      -> std::optional<CodeActionClientCapabilities>;

  [[nodiscard]] auto codeLens() const
      -> std::optional<CodeLensClientCapabilities>;

  [[nodiscard]] auto documentLink() const
      -> std::optional<DocumentLinkClientCapabilities>;

  [[nodiscard]] auto colorProvider() const
      -> std::optional<DocumentColorClientCapabilities>;

  [[nodiscard]] auto formatting() const
      -> std::optional<DocumentFormattingClientCapabilities>;

  [[nodiscard]] auto rangeFormatting() const
      -> std::optional<DocumentRangeFormattingClientCapabilities>;

  [[nodiscard]] auto onTypeFormatting() const
      -> std::optional<DocumentOnTypeFormattingClientCapabilities>;

  [[nodiscard]] auto rename() const -> std::optional<RenameClientCapabilities>;

  [[nodiscard]] auto foldingRange() const
      -> std::optional<FoldingRangeClientCapabilities>;

  [[nodiscard]] auto selectionRange() const
      -> std::optional<SelectionRangeClientCapabilities>;

  [[nodiscard]] auto publishDiagnostics() const
      -> std::optional<PublishDiagnosticsClientCapabilities>;

  [[nodiscard]] auto callHierarchy() const
      -> std::optional<CallHierarchyClientCapabilities>;

  [[nodiscard]] auto semanticTokens() const
      -> std::optional<SemanticTokensClientCapabilities>;

  [[nodiscard]] auto linkedEditingRange() const
      -> std::optional<LinkedEditingRangeClientCapabilities>;

  [[nodiscard]] auto moniker() const
      -> std::optional<MonikerClientCapabilities>;

  [[nodiscard]] auto typeHierarchy() const
      -> std::optional<TypeHierarchyClientCapabilities>;

  [[nodiscard]] auto inlineValue() const
      -> std::optional<InlineValueClientCapabilities>;

  [[nodiscard]] auto inlayHint() const
      -> std::optional<InlayHintClientCapabilities>;

  [[nodiscard]] auto diagnostic() const
      -> std::optional<DiagnosticClientCapabilities>;

  [[nodiscard]] auto inlineCompletion() const
      -> std::optional<InlineCompletionClientCapabilities>;

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

  [[nodiscard]] auto showMessage() const
      -> std::optional<ShowMessageRequestClientCapabilities>;

  [[nodiscard]] auto showDocument() const
      -> std::optional<ShowDocumentClientCapabilities>;

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

  [[nodiscard]] auto regularExpressions() const
      -> std::optional<RegularExpressionsClientCapabilities>;

  [[nodiscard]] auto markdown() const
      -> std::optional<MarkdownClientCapabilities>;

  [[nodiscard]] auto positionEncodings() const
      -> std::optional<Vector<PositionEncodingKind>>;

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

  [[nodiscard]] auto changeNotifications() const
      -> std::optional<std::variant<std::monostate, std::string, bool>>;

  auto supported(std::optional<bool> supported)
      -> WorkspaceFoldersServerCapabilities&;

  auto changeNotifications(
      std::optional<std::variant<std::monostate, std::string, bool>>
          changeNotifications) -> WorkspaceFoldersServerCapabilities&;
};

class FileOperationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto didCreate() const
      -> std::optional<FileOperationRegistrationOptions>;

  [[nodiscard]] auto willCreate() const
      -> std::optional<FileOperationRegistrationOptions>;

  [[nodiscard]] auto didRename() const
      -> std::optional<FileOperationRegistrationOptions>;

  [[nodiscard]] auto willRename() const
      -> std::optional<FileOperationRegistrationOptions>;

  [[nodiscard]] auto didDelete() const
      -> std::optional<FileOperationRegistrationOptions>;

  [[nodiscard]] auto willDelete() const
      -> std::optional<FileOperationRegistrationOptions>;

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
      -> std::variant<std::monostate, WorkspaceFolder, std::string>;

  [[nodiscard]] auto pattern() const -> Pattern;

  auto baseUri(
      std::variant<std::monostate, WorkspaceFolder, std::string> baseUri)
      -> RelativePattern&;

  auto pattern(Pattern pattern) -> RelativePattern&;
};

class TextDocumentFilterLanguage final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto language() const -> std::string;

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

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

  [[nodiscard]] auto scheme() const -> std::string;

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

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

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

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

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

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

  [[nodiscard]] auto scheme() const -> std::string;

  [[nodiscard]] auto pattern() const -> std::optional<GlobPattern>;

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

  [[nodiscard]] auto scheme() const -> std::optional<std::string>;

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

  [[nodiscard]] auto resourceOperations() const
      -> std::optional<Vector<ResourceOperationKind>>;

  [[nodiscard]] auto failureHandling() const
      -> std::optional<FailureHandlingKind>;

  [[nodiscard]] auto normalizesLineEndings() const -> std::optional<bool>;

  [[nodiscard]] auto changeAnnotationSupport() const
      -> std::optional<ChangeAnnotationsSupportOptions>;

  [[nodiscard]] auto metadataSupport() const -> std::optional<bool>;

  [[nodiscard]] auto snippetEditSupport() const -> std::optional<bool>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DidChangeConfigurationClientCapabilities&;
};

class DidChangeWatchedFilesClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto relativePatternSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto symbolKind() const
      -> std::optional<ClientSymbolKindOptions>;

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientSymbolTagOptions>;

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientSymbolResolveOptions>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> ExecuteCommandClientCapabilities&;
};

class SemanticTokensWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> SemanticTokensWorkspaceClientCapabilities&;
};

class CodeLensWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> CodeLensWorkspaceClientCapabilities&;
};

class FileOperationClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto didCreate() const -> std::optional<bool>;

  [[nodiscard]] auto willCreate() const -> std::optional<bool>;

  [[nodiscard]] auto didRename() const -> std::optional<bool>;

  [[nodiscard]] auto willRename() const -> std::optional<bool>;

  [[nodiscard]] auto didDelete() const -> std::optional<bool>;

  [[nodiscard]] auto willDelete() const -> std::optional<bool>;

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

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> InlineValueWorkspaceClientCapabilities&;
};

class InlayHintWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> InlayHintWorkspaceClientCapabilities&;
};

class DiagnosticWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> DiagnosticWorkspaceClientCapabilities&;
};

class FoldingRangeWorkspaceClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto refreshSupport() const -> std::optional<bool>;

  auto refreshSupport(std::optional<bool> refreshSupport)
      -> FoldingRangeWorkspaceClientCapabilities&;
};

class TextDocumentContentClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> TextDocumentContentClientCapabilities&;
};

class TextDocumentSyncClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto willSave() const -> std::optional<bool>;

  [[nodiscard]] auto willSaveWaitUntil() const -> std::optional<bool>;

  [[nodiscard]] auto didSave() const -> std::optional<bool>;

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

  auto relativePatternSupport(std::optional<bool> relativePatternSupport)
      -> TextDocumentFilterClientCapabilities&;
};

class CompletionClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto completionItem() const
      -> std::optional<ClientCompletionItemOptions>;

  [[nodiscard]] auto completionItemKind() const
      -> std::optional<ClientCompletionItemOptionsKind>;

  [[nodiscard]] auto insertTextMode() const -> std::optional<InsertTextMode>;

  [[nodiscard]] auto contextSupport() const -> std::optional<bool>;

  [[nodiscard]] auto completionList() const
      -> std::optional<CompletionListCapabilities>;

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

  [[nodiscard]] auto contentFormat() const -> std::optional<Vector<MarkupKind>>;

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

  [[nodiscard]] auto signatureInformation() const
      -> std::optional<ClientSignatureInformationOptions>;

  [[nodiscard]] auto contextSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto linkSupport() const -> std::optional<bool>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> ReferenceClientCapabilities&;
};

class DocumentHighlightClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentHighlightClientCapabilities&;
};

class DocumentSymbolClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto symbolKind() const
      -> std::optional<ClientSymbolKindOptions>;

  [[nodiscard]] auto hierarchicalDocumentSymbolSupport() const
      -> std::optional<bool>;

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientSymbolTagOptions>;

  [[nodiscard]] auto labelSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto codeActionLiteralSupport() const
      -> std::optional<ClientCodeActionLiteralOptions>;

  [[nodiscard]] auto isPreferredSupport() const -> std::optional<bool>;

  [[nodiscard]] auto disabledSupport() const -> std::optional<bool>;

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientCodeActionResolveOptions>;

  [[nodiscard]] auto honorsChangeAnnotations() const -> std::optional<bool>;

  [[nodiscard]] auto documentationSupport() const -> std::optional<bool>;

  [[nodiscard]] auto tagSupport() const -> std::optional<CodeActionTagOptions>;

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

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientCodeLensResolveOptions>;

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

  [[nodiscard]] auto tooltipSupport() const -> std::optional<bool>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentColorClientCapabilities&;
};

class DocumentFormattingClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentFormattingClientCapabilities&;
};

class DocumentRangeFormattingClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto rangesSupport() const -> std::optional<bool>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> DocumentOnTypeFormattingClientCapabilities&;
};

class RenameClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto prepareSupport() const -> std::optional<bool>;

  [[nodiscard]] auto prepareSupportDefaultBehavior() const
      -> std::optional<PrepareSupportDefaultBehavior>;

  [[nodiscard]] auto honorsChangeAnnotations() const -> std::optional<bool>;

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

  [[nodiscard]] auto rangeLimit() const -> std::optional<long>;

  [[nodiscard]] auto lineFoldingOnly() const -> std::optional<bool>;

  [[nodiscard]] auto foldingRangeKind() const
      -> std::optional<ClientFoldingRangeKindOptions>;

  [[nodiscard]] auto foldingRange() const
      -> std::optional<ClientFoldingRangeOptions>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> SelectionRangeClientCapabilities&;
};

class PublishDiagnosticsClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto versionSupport() const -> std::optional<bool>;

  [[nodiscard]] auto relatedInformation() const -> std::optional<bool>;

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientDiagnosticsTagOptions>;

  [[nodiscard]] auto codeDescriptionSupport() const -> std::optional<bool>;

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> CallHierarchyClientCapabilities&;
};

class SemanticTokensClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto requests() const -> ClientSemanticTokensRequestOptions;

  [[nodiscard]] auto tokenTypes() const -> Vector<std::string>;

  [[nodiscard]] auto tokenModifiers() const -> Vector<std::string>;

  [[nodiscard]] auto formats() const -> Vector<TokenFormat>;

  [[nodiscard]] auto overlappingTokenSupport() const -> std::optional<bool>;

  [[nodiscard]] auto multilineTokenSupport() const -> std::optional<bool>;

  [[nodiscard]] auto serverCancelSupport() const -> std::optional<bool>;

  [[nodiscard]] auto augmentsSyntaxTokens() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> SemanticTokensClientCapabilities&;

  auto requests(ClientSemanticTokensRequestOptions requests)
      -> SemanticTokensClientCapabilities&;

  auto tokenTypes(Vector<std::string> tokenTypes)
      -> SemanticTokensClientCapabilities&;

  auto tokenModifiers(Vector<std::string> tokenModifiers)
      -> SemanticTokensClientCapabilities&;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> LinkedEditingRangeClientCapabilities&;
};

class MonikerClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> MonikerClientCapabilities&;
};

class TypeHierarchyClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> TypeHierarchyClientCapabilities&;
};

class InlineValueClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> InlineValueClientCapabilities&;
};

class InlayHintClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientInlayHintResolveOptions>;

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

  [[nodiscard]] auto relatedDocumentSupport() const -> std::optional<bool>;

  [[nodiscard]] auto relatedInformation() const -> std::optional<bool>;

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientDiagnosticsTagOptions>;

  [[nodiscard]] auto codeDescriptionSupport() const -> std::optional<bool>;

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

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

  auto dynamicRegistration(std::optional<bool> dynamicRegistration)
      -> InlineCompletionClientCapabilities&;
};

class NotebookDocumentSyncClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto dynamicRegistration() const -> std::optional<bool>;

  [[nodiscard]] auto executionSummarySupport() const -> std::optional<bool>;

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
};

class RegularExpressionsClientCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto engine() const -> RegularExpressionEngineKind;

  [[nodiscard]] auto version() const -> std::optional<std::string>;

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

  [[nodiscard]] auto allowedTags() const -> std::optional<Vector<std::string>>;

  auto parser(std::string parser) -> MarkdownClientCapabilities&;

  auto version(std::optional<std::string> version)
      -> MarkdownClientCapabilities&;

  auto allowedTags(std::optional<Vector<std::string>> allowedTags)
      -> MarkdownClientCapabilities&;
};

class ChangeAnnotationsSupportOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto groupsOnLabel() const -> std::optional<bool>;

  auto groupsOnLabel(std::optional<bool> groupsOnLabel)
      -> ChangeAnnotationsSupportOptions&;
};

class ClientSymbolKindOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const -> std::optional<Vector<SymbolKind>>;

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
};

class ClientCompletionItemOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto snippetSupport() const -> std::optional<bool>;

  [[nodiscard]] auto commitCharactersSupport() const -> std::optional<bool>;

  [[nodiscard]] auto documentationFormat() const
      -> std::optional<Vector<MarkupKind>>;

  [[nodiscard]] auto deprecatedSupport() const -> std::optional<bool>;

  [[nodiscard]] auto preselectSupport() const -> std::optional<bool>;

  [[nodiscard]] auto tagSupport() const
      -> std::optional<CompletionItemTagOptions>;

  [[nodiscard]] auto insertReplaceSupport() const -> std::optional<bool>;

  [[nodiscard]] auto resolveSupport() const
      -> std::optional<ClientCompletionItemResolveOptions>;

  [[nodiscard]] auto insertTextModeSupport() const
      -> std::optional<ClientCompletionItemInsertTextModeOptions>;

  [[nodiscard]] auto labelDetailsSupport() const -> std::optional<bool>;

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

  auto valueSet(std::optional<Vector<CompletionItemKind>> valueSet)
      -> ClientCompletionItemOptionsKind&;
};

class CompletionListCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto itemDefaults() const -> std::optional<Vector<std::string>>;

  [[nodiscard]] auto applyKindSupport() const -> std::optional<bool>;

  auto itemDefaults(std::optional<Vector<std::string>> itemDefaults)
      -> CompletionListCapabilities&;

  auto applyKindSupport(std::optional<bool> applyKindSupport)
      -> CompletionListCapabilities&;
};

class ClientSignatureInformationOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto documentationFormat() const
      -> std::optional<Vector<MarkupKind>>;

  [[nodiscard]] auto parameterInformation() const
      -> std::optional<ClientSignatureParameterInformationOptions>;

  [[nodiscard]] auto activeParameterSupport() const -> std::optional<bool>;

  [[nodiscard]] auto noActiveParameterSupport() const -> std::optional<bool>;

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
};

class ClientFoldingRangeKindOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto valueSet() const
      -> std::optional<Vector<FoldingRangeKind>>;

  auto valueSet(std::optional<Vector<FoldingRangeKind>> valueSet)
      -> ClientFoldingRangeKindOptions&;
};

class ClientFoldingRangeOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto collapsedText() const -> std::optional<bool>;

  auto collapsedText(std::optional<bool> collapsedText)
      -> ClientFoldingRangeOptions&;
};

class DiagnosticsCapabilities final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto relatedInformation() const -> std::optional<bool>;

  [[nodiscard]] auto tagSupport() const
      -> std::optional<ClientDiagnosticsTagOptions>;

  [[nodiscard]] auto codeDescriptionSupport() const -> std::optional<bool>;

  [[nodiscard]] auto dataSupport() const -> std::optional<bool>;

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

  [[nodiscard]] auto range() const
      -> std::optional<std::variant<std::monostate, bool, json>>;

  [[nodiscard]] auto full() const -> std::optional<
      std::variant<std::monostate, bool, ClientSemanticTokensRequestFullDelta>>;

  auto range(std::optional<std::variant<std::monostate, bool, json>> range)
      -> ClientSemanticTokensRequestOptions&;

  auto full(std::optional<std::variant<std::monostate, bool,
                                       ClientSemanticTokensRequestFullDelta>>
                full) -> ClientSemanticTokensRequestOptions&;
};

class ClientInlayHintResolveOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto properties() const -> Vector<std::string>;

  auto properties(Vector<std::string> properties)
      -> ClientInlayHintResolveOptions&;
};

class ClientShowMessageActionItemOptions final : public LSPObject {
 public:
  using LSPObject::LSPObject;

  explicit operator bool() const;

  [[nodiscard]] auto additionalPropertiesSupport() const -> std::optional<bool>;

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

  auto delta(std::optional<bool> delta)
      -> ClientSemanticTokensRequestFullDelta&;
};
}  // namespace cxx::lsp
