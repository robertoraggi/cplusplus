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

#include "cxx_document.h"

#include <cxx/access_control.h>
#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/lexer.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/types.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/private/utf8.h>
#include <cxx/symbols.h>
#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#ifndef CXX_NO_THREADS
#include <atomic>
#include <mutex>
#endif

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace cxx::lsp {

namespace {

constexpr int kMaxDiagnostics = 100;

constexpr std::array kSemanticTokenTypes{
    SemanticTokenTypes::kNamespace,     SemanticTokenTypes::kType,
    SemanticTokenTypes::kClass,         SemanticTokenTypes::kEnum,
    SemanticTokenTypes::kInterface,     SemanticTokenTypes::kStruct,
    SemanticTokenTypes::kTypeParameter, SemanticTokenTypes::kParameter,
    SemanticTokenTypes::kVariable,      SemanticTokenTypes::kProperty,
    SemanticTokenTypes::kEnumMember,    SemanticTokenTypes::kEvent,
    SemanticTokenTypes::kFunction,      SemanticTokenTypes::kMethod,
    SemanticTokenTypes::kMacro,         SemanticTokenTypes::kKeyword,
    SemanticTokenTypes::kModifier,      SemanticTokenTypes::kComment,
    SemanticTokenTypes::kString,        SemanticTokenTypes::kNumber,
    SemanticTokenTypes::kRegexp,        SemanticTokenTypes::kOperator,
    SemanticTokenTypes::kDecorator,     SemanticTokenTypes::kLabel,
};

constexpr std::array kSemanticTokenModifiers{
    SemanticTokenModifiers::kDeclaration,
};

template <typename Enum, std::size_t Size>
auto enumNames(const std::array<Enum, Size>& values)
    -> std::vector<std::string> {
  std::vector<std::string> names;
  names.reserve(values.size());
  for (auto value : values) names.push_back(to_string(value));
  return names;
}

template <typename Enum, std::size_t Size>
auto enumIndex(const std::array<Enum, Size>& values, Enum value) -> long {
  auto position = std::ranges::find(values, value);
  return long(position - values.begin());
}

auto semanticTokenModifierMask(SemanticTokenModifiers modifier) -> long {
  return 1L << enumIndex(kSemanticTokenModifiers, modifier);
}

auto templateParametersOf(Symbol* symbol) -> TemplateParametersSymbol*;
auto signatureLabelOf(FunctionSymbol* function) -> std::string;

struct TextPosition {
  std::uint32_t line = 0;
  std::uint32_t character = 0;
};

class SourceText {
 public:
  explicit SourceText(std::string_view source) : source_(source) {
    lineStarts_.push_back(0);
    for (std::size_t index = 0; index < source_.size(); ++index) {
      if (source_[index] == '\n') lineStarts_.push_back(index + 1);
    }
  }

  [[nodiscard]] auto positionAt(std::size_t offset) const -> TextPosition {
    if (offset > source_.size()) offset = source_.size();

    auto nextLine = std::ranges::upper_bound(lineStarts_, offset);
    auto line = std::size_t(nextLine - lineStarts_.begin());
    if (line != 0) --line;

    const auto lineStart = lineStarts_.at(line);
    return TextPosition{.line = std::uint32_t(line),
                        .character = utf16Length(lineStart, offset)};
  }

  [[nodiscard]] auto utf16Length(std::size_t start, std::size_t end) const
      -> std::uint32_t {
    auto first = source_.begin() + std::ptrdiff_t(start);
    const auto last = source_.begin() + std::ptrdiff_t(end);
    std::uint32_t length = 0;

    while (first != last) {
      const auto codepoint = utf8::next(first, last);
      ++length;
      if (codepoint > 0xFFFF) ++length;
    }

    return length;
  }

 private:
  std::string_view source_;
  std::vector<std::size_t> lineStarts_;
};

struct SymbolOccurrence {
  SourceLocation location;
  Symbol* symbol = nullptr;
  bool isDeclaration = false;
};

auto soleFunctionOf(Symbol* overloadSet) -> FunctionSymbol* {
  FunctionSymbol* result = nullptr;

  for (auto function : views::each_function(overloadSet)) {
    auto canonical = function->canonical();
    if (!result) {
      result = canonical;
      continue;
    }
    if (result != canonical) return nullptr;
  }

  return result;
}

auto symbolIdentity(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;
  symbol = resolve_using_declaration(symbol);
  if (!symbol) return nullptr;

  if (auto injected = symbol_cast<InjectedClassNameSymbol>(symbol)) {
    symbol = injected->classSymbol();
  } else if (auto baseClass = symbol_cast<BaseClassSymbol>(symbol)) {
    symbol = baseClass->symbol();
  }

  if (!symbol) return nullptr;

  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    if (auto principal = function->structorPrincipal()) symbol = principal;
  }

  if (symbol->isOverloadSet()) {
    if (auto function = soleFunctionOf(symbol)) symbol = function;
  }

  return symbol->canonical();
}

auto isSameDeclaration(Symbol* lhs, Symbol* rhs) -> bool {
  if (lhs == rhs) return true;
  if (!lhs->location()) return false;
  if (lhs->location() != rhs->location()) return false;
  return lhs->name() == rhs->name();
}

auto overloadSetContains(Symbol* overloadSet, Symbol* symbol) -> bool {
  for (auto function : views::each_function(overloadSet)) {
    if (isSameDeclaration(function->canonical(), symbol)) return true;
  }
  return false;
}

auto isSameEntity(Symbol* lhs, Symbol* rhs) -> bool {
  lhs = symbolIdentity(lhs);
  rhs = symbolIdentity(rhs);

  if (!lhs || !rhs) return false;
  if (lhs->isOverloadSet()) return overloadSetContains(lhs, rhs);
  if (rhs->isOverloadSet()) return overloadSetContains(rhs, lhs);
  return isSameDeclaration(lhs, rhs);
}

auto redeclarationsOf(Symbol* symbol) -> std::vector<Symbol*> {
  return cxx::visit(
      []<typename S>(S* symbol) -> std::vector<Symbol*> {
        if constexpr (requires { symbol->redeclarations(); }) {
          const auto& redeclarations = symbol->redeclarations();
          return std::vector<Symbol*>(redeclarations.begin(),
                                      redeclarations.end());
        } else {
          return {};
        }
      },
      symbol);
}

auto isImplicitlyDeclared(TranslationUnit* unit, Symbol* symbol) -> bool {
  auto identifier = name_cast<Identifier>(symbol->name());

  if (unit->ownsLocation(symbol->location())) {
    if (unit->tokenKind(symbol->location()) != TokenKind::T_IDENTIFIER) {
      return identifier != nullptr;
    }

    if (unit->identifier(symbol->location()) != identifier) return true;
  }

  auto function = symbol_cast<FunctionSymbol>(symbol);
  if (!function) return false;
  if (function->isStructorVariant()) return true;
  if (function->inheritedConstructor()) return true;

  auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
  if (!classSymbol) return false;

  return function->location() == classSymbol->location();
}

auto symbolOccurrencePriority(Symbol* symbol) -> int {
  if (!symbol) return 0;

  switch (symbol->kind()) {
    case cxx::SymbolKind::kNamespace:
    case cxx::SymbolKind::kNamespaceAlias:
    case cxx::SymbolKind::kConcept:
    case cxx::SymbolKind::kClass:
    case cxx::SymbolKind::kEnum:
    case cxx::SymbolKind::kScopedEnum:
    case cxx::SymbolKind::kTypeAlias:
    case cxx::SymbolKind::kInjectedClassName:
      return 3;
    case cxx::SymbolKind::kVariable:
    case cxx::SymbolKind::kField:
    case cxx::SymbolKind::kParameter:
    case cxx::SymbolKind::kParameterPack:
    case cxx::SymbolKind::kEnumerator:
    case cxx::SymbolKind::kTypeParameter:
    case cxx::SymbolKind::kNonTypeParameter:
    case cxx::SymbolKind::kTemplateTypeParameter:
    case cxx::SymbolKind::kConstraintTypeParameter:
      return 2;
    default:
      return 1;
  }
}

class SymbolOccurrences final : public ASTVisitor {
 public:
  explicit SymbolOccurrences(TranslationUnit* unit) : unit_(unit) {
    collectScope(unit_->globalScope());
    accept(unit_->ast());

    std::ranges::sort(occurrences_, [](const SymbolOccurrence& lhs,
                                       const SymbolOccurrence& rhs) {
      if (lhs.location != rhs.location) return lhs.location < rhs.location;
      if (lhs.isDeclaration != rhs.isDeclaration)
        return lhs.isDeclaration && !rhs.isDeclaration;
      return symbolOccurrencePriority(lhs.symbol) >
             symbolOccurrencePriority(rhs.symbol);
    });

    auto sameOccurrence = [](const SymbolOccurrence& lhs,
                             const SymbolOccurrence& rhs) {
      if (lhs.location != rhs.location) return false;
      return isSameEntity(lhs.symbol, rhs.symbol);
    };

    auto end = std::ranges::unique(occurrences_, sameOccurrence).begin();
    occurrences_.erase(end, occurrences_.end());
  }

  [[nodiscard]] auto all() const -> const std::vector<SymbolOccurrence>& {
    return occurrences_;
  }

  [[nodiscard]] auto atOffset(std::size_t offset) const
      -> const SymbolOccurrence* {
    for (const auto& occurrence : occurrences_) {
      const auto& token = unit_->tokenAt(occurrence.location);
      if (offset < token.offset()) continue;
      if (offset >= token.offset() + token.length()) continue;
      return &occurrence;
    }

    return nullptr;
  }

  [[nodiscard]] auto rangeOf(SourceLocation location) const
      -> std::pair<std::size_t, std::size_t> {
    const auto& token = unit_->tokenAt(location);
    return {token.offset(), token.offset() + token.length()};
  }

  void visit(IdExpressionAST* ast) override {
    add(get_name_location(ast), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(MemberExpressionAST* ast) override {
    add(get_name_location(ast), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(DotDesignatorAST* ast) override {
    add(get_name_location(ast), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(NamedTypeSpecifierAST* ast) override {
    add(firstSourceLocation(ast->unqualifiedId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(ElaboratedTypeSpecifierAST* ast) override {
    add(firstSourceLocation(ast->unqualifiedId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(TypenameSpecifierAST* ast) override {
    add(firstSourceLocation(ast->unqualifiedId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(SimpleTemplateIdAST* ast) override {
    add(ast->identifierLoc, ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(SimpleNestedNameSpecifierAST* ast) override {
    add(ast->identifierLoc, ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(TemplateNestedNameSpecifierAST* ast) override {
    add(firstSourceLocation(ast->templateId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(TypeConstraintAST* ast) override {
    add(ast->identifierLoc, ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(SizeofPackExpressionAST* ast) override {
    add(ast->identifierLoc, ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(BuiltinOffsetofExpressionAST* ast) override {
    add(ast->identifierLoc, ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(NamespaceReflectExpressionAST* ast) override {
    add(ast->identifierLoc, ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(FunctionDefinitionAST* ast) override {
    addDeclarator(ast->declarator, ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(InitDeclaratorAST* ast) override {
    addDeclarator(ast->declarator, ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(BaseSpecifierAST* ast) override {
    add(firstSourceLocation(ast->unqualifiedId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(ParenMemInitializerAST* ast) override {
    add(firstSourceLocation(ast->unqualifiedId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

  void visit(BracedMemInitializerAST* ast) override {
    add(firstSourceLocation(ast->unqualifiedId), ast->symbol, false);
    ASTVisitor::visit(ast);
  }

 private:
  void addDeclarator(DeclaratorAST* declarator, Symbol* symbol) {
    auto declaratorId = getDeclaratorId(declarator);
    if (!declaratorId) return;
    add(firstSourceLocation(declaratorId->unqualifiedId), symbol, true);
  }

  void add(SourceLocation location, Symbol* symbol, bool isDeclaration) {
    if (!location || !symbol) return;
    if (!symbol->name()) return;
    if (!unit_->isMainFileLocation(location)) return;

    occurrences_.push_back(SymbolOccurrence{.location = location,
                                            .symbol = symbol,
                                            .isDeclaration = isDeclaration});
  }

  void collectScope(ScopeSymbol* scope) {
    if (!scope) return;
    for (auto member : scope->members()) collectMember(member);
  }

  void collectMember(Symbol* symbol) {
    if (!symbol) return;
    if (!visitedSymbols_.insert(symbol).second) return;

    if (symbol->isOverloadSet()) {
      auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol);
      for (auto usingDeclaration : overloadSet->usingDeclarations())
        collectMember(usingDeclaration);
      for (auto function : overloadSet->declaredFunctions())
        collectMember(function);
      return;
    }

    if (!symbol->isBaseClass() && !isImplicitlyDeclared(unit_, symbol)) {
      add(symbol->location(), symbol, true);
    }

    if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
      for (auto constructor : classSymbol->declaredConstructors())
        collectMember(constructor);
      for (auto deductionGuide : classSymbol->deductionGuides())
        collectMember(deductionGuide);
    }

    collectScope(templateParametersOf(symbol));
    collectScope(symbol->asScopeSymbol());

    for (auto redeclaration : redeclarationsOf(symbol))
      collectMember(redeclaration);
  }

  TranslationUnit* unit_;
  std::vector<SymbolOccurrence> occurrences_;
  std::unordered_set<Symbol*> visitedSymbols_;
};

auto semanticTokenTypeOf(Symbol* symbol) -> std::optional<SemanticTokenTypes> {
  if (!symbol) return std::nullopt;

  return cxx::visit(
      []<typename S>(S* symbol) -> std::optional<SemanticTokenTypes> {
        using SymbolType = std::remove_cvref_t<decltype(*symbol)>;

        if constexpr (std::is_same_v<SymbolType, NamespaceSymbol> ||
                      std::is_same_v<SymbolType, NamespaceAliasSymbol>) {
          return SemanticTokenTypes::kNamespace;
        } else if constexpr (std::is_same_v<SymbolType, ClassSymbol> ||
                             std::is_same_v<SymbolType,
                                            InjectedClassNameSymbol>) {
          return SemanticTokenTypes::kClass;
        } else if constexpr (std::is_same_v<SymbolType, EnumSymbol> ||
                             std::is_same_v<SymbolType, ScopedEnumSymbol>) {
          return SemanticTokenTypes::kEnum;
        } else if constexpr (std::is_same_v<SymbolType, TypeAliasSymbol> ||
                             std::is_same_v<SymbolType, ConceptSymbol>) {
          return SemanticTokenTypes::kType;
        } else if constexpr (std::is_same_v<SymbolType, TypeParameterSymbol> ||
                             std::is_same_v<SymbolType,
                                            TemplateTypeParameterSymbol> ||
                             std::is_same_v<SymbolType,
                                            ConstraintTypeParameterSymbol>) {
          return SemanticTokenTypes::kTypeParameter;
        } else if constexpr (std::is_same_v<SymbolType, ParameterSymbol> ||
                             std::is_same_v<SymbolType, ParameterPackSymbol> ||
                             std::is_same_v<SymbolType,
                                            NonTypeParameterSymbol>) {
          return SemanticTokenTypes::kParameter;
        } else if constexpr (std::is_same_v<SymbolType, VariableSymbol>) {
          return SemanticTokenTypes::kVariable;
        } else if constexpr (std::is_same_v<SymbolType, FieldSymbol>) {
          return SemanticTokenTypes::kProperty;
        } else if constexpr (std::is_same_v<SymbolType, EnumeratorSymbol>) {
          return SemanticTokenTypes::kEnumMember;
        } else if constexpr (std::is_same_v<SymbolType, FunctionSymbol> ||
                             std::is_same_v<SymbolType, DeductionGuideSymbol> ||
                             std::is_same_v<SymbolType, OverloadSetSymbol> ||
                             std::is_same_v<SymbolType, LambdaSymbol>) {
          if (symbol->parent() && symbol->parent()->isClass()) {
            return SemanticTokenTypes::kMethod;
          }
          return SemanticTokenTypes::kFunction;
        } else if constexpr (std::is_same_v<SymbolType,
                                            UsingDeclarationSymbol>) {
          return semanticTokenTypeOf(symbol->target());
        } else {
          return std::nullopt;
        }
      },
      symbol);
}

auto lexicalSemanticTokenType(TokenKind kind)
    -> std::optional<SemanticTokenTypes> {
  switch (kind) {
#define CXX_LSP_KEYWORD_CASE(name, spelling) \
  case TokenKind::T_##name:                  \
    return SemanticTokenTypes::kKeyword;
    FOR_EACH_KEYWORD(CXX_LSP_KEYWORD_CASE)
#undef CXX_LSP_KEYWORD_CASE

    case TokenKind::T_COMMENT:
      return SemanticTokenTypes::kComment;
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
    case TokenKind::T_UTF16_STRING_LITERAL:
    case TokenKind::T_UTF32_STRING_LITERAL:
    case TokenKind::T_UTF8_STRING_LITERAL:
    case TokenKind::T_WIDE_STRING_LITERAL:
      return SemanticTokenTypes::kString;
    case TokenKind::T_FLOATING_POINT_LITERAL:
    case TokenKind::T_INTEGER_LITERAL:
      return SemanticTokenTypes::kNumber;
    default:
      break;
  }

  switch (kind) {
#define CXX_LSP_OPERATOR_CASE(name, spelling) \
  case TokenKind::T_##name:                   \
    return SemanticTokenTypes::kOperator;
    FOR_EACH_OPERATOR(CXX_LSP_OPERATOR_CASE)
#undef CXX_LSP_OPERATOR_CASE

    default:
      return std::nullopt;
  }
}

auto semanticTokenPriority(SemanticTokenTypes type) -> int {
  switch (type) {
    case SemanticTokenTypes::kNamespace:
    case SemanticTokenTypes::kType:
    case SemanticTokenTypes::kClass:
    case SemanticTokenTypes::kEnum:
    case SemanticTokenTypes::kInterface:
    case SemanticTokenTypes::kStruct:
    case SemanticTokenTypes::kTypeParameter:
      return 3;
    case SemanticTokenTypes::kFunction:
    case SemanticTokenTypes::kMethod:
      return 2;
    default:
      return 1;
  }
}

auto hoverTextOf(Symbol* symbol) -> std::string {
  if (!symbol) return {};
  symbol = resolve_using_declaration(symbol);
  if (!symbol) return {};

  return cxx::visit(
      []<typename S>(S* symbol) -> std::string {
        using SymbolType = std::remove_cvref_t<decltype(*symbol)>;
        auto name = to_string(symbol->name());

        if constexpr (std::is_same_v<SymbolType, NamespaceSymbol>) {
          return std::format("namespace {}", name);
        } else if constexpr (std::is_same_v<SymbolType, NamespaceAliasSymbol>) {
          return std::format("namespace {}", name);
        } else if constexpr (std::is_same_v<SymbolType, ClassSymbol>) {
          if (symbol->isUnion()) return std::format("union {}", name);
          return std::format("class {}", name);
        } else if constexpr (std::is_same_v<SymbolType,
                                            InjectedClassNameSymbol>) {
          return std::format("class {}", name);
        } else if constexpr (std::is_same_v<SymbolType, EnumSymbol> ||
                             std::is_same_v<SymbolType, ScopedEnumSymbol>) {
          return std::format("enum {}", name);
        } else if constexpr (std::is_same_v<SymbolType, TypeAliasSymbol>) {
          return std::format("using {} = {}", name, to_string(symbol->type()));
        } else if constexpr (std::is_same_v<SymbolType, ConceptSymbol>) {
          return std::format("concept {}", name);
        } else if constexpr (std::is_same_v<SymbolType, OverloadSetSymbol>) {
          std::string label;
          for (auto function : views::each_function(symbol)) {
            if (function->canonical() != function) continue;
            if (!label.empty()) label += "\n";
            label += signatureLabelOf(function);
          }
          if (label.empty()) return name;
          return label;
        } else {
          if (!symbol->type()) return name;
          return to_string(symbol->type(), symbol->name());
        }
      },
      symbol);
}

auto isBefore(TextPosition lhs, TextPosition rhs) -> bool {
  if (lhs.line < rhs.line) return true;
  if (lhs.line > rhs.line) return false;
  return lhs.character < rhs.character;
}

auto overlaps(TextPosition tokenStart, TextPosition tokenEnd,
              const Range& range) -> bool {
  auto start = range.start();
  auto end = range.end();
  TextPosition rangeStart{.line = std::uint32_t(start.line()),
                          .character = std::uint32_t(start.character())};
  TextPosition rangeEnd{.line = std::uint32_t(end.line()),
                        .character = std::uint32_t(end.character())};
  if (!isBefore(rangeStart, rangeEnd)) return false;
  if (!isBefore(tokenStart, rangeEnd)) return false;
  return isBefore(rangeStart, tokenEnd);
}

auto diagnosticSeverityOf(cxx::Severity severity) -> DiagnosticSeverity {
  switch (severity) {
    case cxx::Severity::Message:
      return DiagnosticSeverity::kInformation;
    case cxx::Severity::Note:
      return DiagnosticSeverity::kHint;
    case cxx::Severity::Warning:
      return DiagnosticSeverity::kWarning;
    case cxx::Severity::Error:
    case cxx::Severity::Fatal:
      return DiagnosticSeverity::kError;
  }

  return DiagnosticSeverity::kError;
}

struct Diagnostics final : cxx::DiagnosticsClient {
  json messages = json::array();
  Vector<lsp::Diagnostic> diagnostics{messages};
  bool hasErrors = false;

  void report(const cxx::Diagnostic& diag) override {
    if (diag.severity() == cxx::Severity::Error ||
        diag.severity() == cxx::Severity::Fatal) {
      hasErrors = true;
    }

    auto start = sourceResolver()->tokenStartPosition(diag.token());
    auto end = sourceResolver()->tokenEndPosition(diag.token());

    auto tmp = json::object();

    auto d = diagnostics.emplace_back();

    int s = std::max(int(start.line) - 1, 0);
    int sc = std::max(int(start.column) - 1, 0);
    int e = std::max(int(end.line) - 1, 0);
    int ec = std::max(int(end.column) - 1, 0);

    d.message(diag.message());
    d.severity(diagnosticSeverityOf(diag.severity()));
    d.range().start(lsp::Position(tmp).line(s).character(sc));
    d.range().end(lsp::Position(tmp).line(e).character(ec));
  }
};

auto classSymbolOf(const TypeTraits& traits, const Type* objectType)
    -> ClassSymbol* {
  auto unwrapped = traits.remove_cvref(objectType);

  if (auto pointerType = type_cast<PointerType>(unwrapped)) {
    unwrapped = traits.remove_cvref(pointerType->elementType());
  }

  auto classType = type_cast<ClassType>(unwrapped);
  if (!classType) return nullptr;

  return classType->symbol();
}

auto templateParametersOf(Symbol* symbol) -> TemplateParametersSymbol* {
  return cxx::visit(
      []<typename S>(S* symbol) -> TemplateParametersSymbol* {
        if constexpr (requires { symbol->templateParameters(); })
          return symbol->templateParameters();
        else
          return nullptr;
      },
      symbol);
}

auto functionCompletionItemKind(FunctionSymbol* function, bool memberOfClass)
    -> CompletionItemKind {
  if (function->isConstructor()) return CompletionItemKind::kConstructor;
  if (memberOfClass) return CompletionItemKind::kMethod;
  return CompletionItemKind::kFunction;
}

struct CompletionItemKindOf {
  bool memberOfClass = false;

  auto operator()(NamespaceSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kModule;
  }

  auto operator()(ConceptSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kInterface;
  }

  auto operator()(ClassSymbol* symbol) const -> CompletionItemKind {
    if (symbol->isUnion()) return CompletionItemKind::kStruct;
    return CompletionItemKind::kClass;
  }

  auto operator()(InjectedClassNameSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kClass;
  }

  auto operator()(TypeAliasSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kClass;
  }

  auto operator()(EnumSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kEnum;
  }

  auto operator()(ScopedEnumSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kEnum;
  }

  auto operator()(EnumeratorSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kEnumMember;
  }

  auto operator()(FunctionSymbol* symbol) const -> CompletionItemKind {
    return functionCompletionItemKind(symbol, memberOfClass);
  }

  auto operator()(OverloadSetSymbol* symbol) const -> CompletionItemKind {
    auto functions = symbol->declaredFunctions();
    if (functions.empty()) return CompletionItemKind::kFunction;
    return functionCompletionItemKind(functions.front(), memberOfClass);
  }

  auto operator()(DeductionGuideSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kFunction;
  }

  auto operator()(LambdaSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kFunction;
  }

  auto operator()(FieldSymbol* symbol) const -> CompletionItemKind {
    if (symbol->isStatic()) return CompletionItemKind::kVariable;
    return CompletionItemKind::kField;
  }

  auto operator()(VariableSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kVariable;
  }

  auto operator()(ParameterSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kVariable;
  }

  auto operator()(ParameterPackSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kVariable;
  }

  auto operator()(NonTypeParameterSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kVariable;
  }

  auto operator()(TypeParameterSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kTypeParameter;
  }

  auto operator()(TemplateTypeParameterSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kTypeParameter;
  }

  auto operator()(ConstraintTypeParameterSymbol*) const -> CompletionItemKind {
    return CompletionItemKind::kTypeParameter;
  }

  auto operator()(UsingDeclarationSymbol* symbol) const -> CompletionItemKind {
    auto target = symbol->target();
    if (!target) return CompletionItemKind::kReference;
    return cxx::visit(*this, target);
  }

  auto operator()(Symbol*) const -> CompletionItemKind {
    return CompletionItemKind::kText;
  }
};

[[nodiscard]] auto isUnnameable(Symbol* member) -> bool {
  if (auto classSymbol = symbol_cast<ClassSymbol>(member))
    if (classSymbol->isClosureType()) return true;

  if (auto identifier = name_cast<Identifier>(member->name()))
    return identifier->isAnonymous();

  return false;
}

class CompletionItemCollector {
 public:
  CompletionItemCollector(Vector<CompletionItem>& completionItems,
                          std::vector<std::string>& labels,
                          const AccessContext& accessContext,
                          CompletionEditRange editRange)
      : completionItems_(completionItems),
        labels_(labels),
        accessContext_(accessContext),
        editRange_(editRange) {}

  void addScope(ScopeSymbol* scope, ClassSymbol* objectClass) {
    if (!scope) return;
    if (std::ranges::contains(visitedScopes_, scope)) return;
    visitedScopes_.push_back(scope);

    auto designatingClass = symbol_cast<ClassSymbol>(scope);

    for (auto member : views::members(scope)) {
      if (!member->name()) continue;
      if (member->isHidden()) continue;
      if (isUnnameable(member)) continue;
      if (!accessContext_.isAccessible(member, designatingClass, objectClass))
        continue;
      addItem(member, designatingClass != nullptr);
    }

    for (auto directive : scope->usingDirectives()) {
      auto namespaceSymbol = symbol_cast<NamespaceSymbol>(directive);
      if (!namespaceSymbol) continue;
      if (!namespaceSymbol->isInline() && namespaceSymbol->name()) continue;
      addScope(namespaceSymbol, objectClass);
    }

    if (!designatingClass) return;

    for (auto baseClass :
         designatingClass->resolvedDefinition()->baseClasses()) {
      auto base = symbol_cast<ClassSymbol>(baseClass->symbol());
      if (!base) continue;
      if (!accessContext_.isAccessibleBaseClass(designatingClass, base))
        continue;
      addScope(base->resolvedDefinition(), objectClass);
    }
  }

  void addDesignators(ClassSymbol* classSymbol) {
    if (!classSymbol) return;

    for (auto member : views::members(classSymbol->resolvedDefinition())) {
      auto field = symbol_cast<FieldSymbol>(member);
      if (!field) continue;
      if (field->isStatic()) continue;
      if (!field->name()) continue;
      if (!accessContext_.isAccessible(field, classSymbol, classSymbol))
        continue;
      addItem(field, true);
    }
  }

  void addEnclosingScopes(ScopeSymbol* scope) {
    for (auto current = scope; current; current = current->parent()) {
      auto objectClass = symbol_cast<ClassSymbol>(current);
      addScope(current, objectClass);
      addScope(templateParametersOf(current), nullptr);
    }
  }

 private:
  void addItem(Symbol* symbol, bool memberOfClass) {
    auto label = to_string(symbol->name());
    if (std::ranges::contains(labels_, label)) return;

    auto item = completionItems_.emplace_back();
    item.label(label);
    item.kind(cxx::visit(CompletionItemKindOf{memberOfClass}, symbol));

    json startStorage;
    Position start{startStorage};
    start.line(editRange_.line).character(editRange_.startColumn);

    json endStorage;
    Position end{endStorage};
    end.line(editRange_.line).character(editRange_.endColumn);

    json rangeStorage;
    Range range{rangeStorage};
    range.start(start).end(end);

    json textEditStorage;
    TextEdit textEdit{textEditStorage};
    textEdit.range(range).newText(label);
    item.textEdit(std::variant<TextEdit, InsertReplaceEdit>{textEdit});

    labels_.push_back(std::move(label));
  }

  Vector<CompletionItem>& completionItems_;
  std::vector<std::string>& labels_;
  const AccessContext& accessContext_;
  CompletionEditRange editRange_;
  std::vector<ScopeSymbol*> visitedScopes_;
};

struct CompletionSink {
  TranslationUnit* unit;
  Vector<CompletionItem> completionItems;
  CompletionEditRange editRange;
  std::vector<std::string> labels;

  void operator()(const MemberCompletionContext& context) {
    auto objectClass = classSymbolOf(unit->typeTraits(), context.objectType);
    if (!objectClass) return;
    AccessContext accessContext{unit, context.accessingScope};
    auto collector = collectorFor(accessContext);
    collector.addScope(objectClass->resolvedDefinition(), objectClass);
  }

  void operator()(const ScopeCompletionContext& context) {
    AccessContext accessContext{unit, context.accessingScope};
    auto collector = collectorFor(accessContext);
    collector.addScope(context.scope, symbol_cast<ClassSymbol>(context.scope));
  }

  void operator()(const UnqualifiedCompletionContext& context) {
    AccessContext accessContext{unit, context.scope};
    auto collector = collectorFor(accessContext);
    collector.addEnclosingScopes(context.scope);
  }

  void operator()(const DesignatorCompletionContext& context) {
    AccessContext accessContext{unit, context.accessingScope};
    auto collector = collectorFor(accessContext);
    collector.addDesignators(
        classSymbolOf(unit->typeTraits(), context.objectType));
  }

  void operator()(const ArgumentHintsContext&) const {}
  void operator()(const TemplateArgumentHintsContext&) const {}

 private:
  auto collectorFor(const AccessContext& accessContext)
      -> CompletionItemCollector {
    return CompletionItemCollector{completionItems, labels, accessContext,
                                   editRange};
  }
};

auto signatureLabelOf(FunctionSymbol* function) -> std::string {
  TypePrintOptions options;
  options.omitFunctionReturnType = function->isConstructor();
  return to_string(function->type(), function->name(), options);
}

struct TemplateParameterLabel {
  auto operator()(TypeParameterSymbol* symbol) const -> std::string {
    return named("class", symbol);
  }

  auto operator()(ConstraintTypeParameterSymbol* symbol) const -> std::string {
    return named(constraintName(symbol), symbol);
  }

  auto operator()(TemplateTypeParameterSymbol* symbol) const -> std::string {
    return named(templateParameterTypeLabel(symbol->type()), symbol);
  }

  auto operator()(NonTypeParameterSymbol* symbol) const -> std::string {
    return named(to_string(symbol->objectType()), symbol);
  }

  auto operator()(Symbol* symbol) const -> std::string {
    return named(to_string(symbol->type()), symbol);
  }

 private:
  auto named(std::string kind, Symbol* symbol) const -> std::string {
    if (isPack(symbol)) kind += "...";
    if (!symbol->name()) return kind;
    kind += " ";
    kind += to_string(symbol->name());
    return kind;
  }

  auto isPack(Symbol* symbol) const -> bool {
    auto info = template_parameter_info(symbol);
    if (!info) return false;
    return info->isPack;
  }

  auto constraintName(ConstraintTypeParameterSymbol* symbol) const
      -> std::string {
    auto typeConstraint = symbol->typeConstraint();
    if (!typeConstraint) return "class";
    if (!typeConstraint->identifier) return "class";
    return typeConstraint->identifier->name();
  }

  auto templateParameterTypeLabel(const Type* type) const -> std::string {
    if (type_cast<TypeParameterType>(type)) return "class";

    auto templateType = type_cast<TemplateTypeParameterType>(type);
    if (!templateType) return to_string(type);

    std::string clause = "template <";
    std::string_view separator;
    for (auto parameterType : templateType->templateParameters()) {
      clause += separator;
      clause += templateParameterTypeLabel(parameterType);
      separator = ", ";
    }
    clause += "> class";
    return clause;
  }
};

void addTemplateSignature(SignatureHelp& result, Symbol* templateSymbol,
                          int activeParameter) {
  auto templateParameters = templateParametersOf(templateSymbol);
  if (!templateParameters) return;

  std::string label = "template <";
  std::vector<std::string> parameterLabels;

  for (auto parameter : views::members(templateParameters)) {
    if (!parameterLabels.empty()) label += ", ";
    auto parameterLabel = cxx::visit(TemplateParameterLabel{}, parameter);
    label += parameterLabel;
    parameterLabels.push_back(std::move(parameterLabel));
  }

  if (parameterLabels.empty()) return;

  label += ">";

  auto signatures = result.signatures();
  auto signature = signatures.emplace_back();
  signature.label(std::move(label));

  auto parameterList = signature.parameters<Vector<ParameterInformation>>();
  for (auto& parameterLabel : parameterLabels) {
    auto parameterInfo = parameterList.emplace_back();
    parameterInfo.label(std::move(parameterLabel));
  }

  result.activeSignature(0);
  result.activeParameter(long(activeParameter));
}

struct SignatureHelpSink {
  SignatureHelp result;

  void operator()(const ArgumentHintsContext& context) {
    clearResult();

    if (context.candidates.empty()) return;

    auto signatures = result.signatures();
    int activeSignature = 0;
    bool foundActiveSignature = false;

    for (auto function : context.candidates) {
      auto signature = signatures.emplace_back();

      signature.label(signatureLabelOf(function));

      auto parameterList = signature.parameters<Vector<ParameterInformation>>();

      int parameterCount = 0;

      if (auto functionParameters = function->functionParameters()) {
        for (auto member : views::members(functionParameters)) {
          auto parameterSymbol = symbol_cast<ParameterSymbol>(member);
          if (!parameterSymbol) continue;

          auto parameterInfo = parameterList.emplace_back();
          parameterInfo.label(
              to_string(parameterSymbol->type(), parameterSymbol->name()));

          ++parameterCount;
        }
      }

      if (foundActiveSignature) continue;

      if (parameterCount > context.activeParameter) {
        foundActiveSignature = true;
        continue;
      }

      ++activeSignature;
    }

    if (!foundActiveSignature) activeSignature = 0;

    result.activeSignature(activeSignature);
    result.activeParameter(long(context.activeParameter));
  }

  void operator()(const TemplateArgumentHintsContext& context) {
    clearResult();

    addTemplateSignature(result, context.templateSymbol,
                         context.activeParameter);
  }

  void operator()(const MemberCompletionContext&) const {}
  void operator()(const ScopeCompletionContext&) const {}
  void operator()(const UnqualifiedCompletionContext&) const {}
  void operator()(const DesignatorCompletionContext&) const {}

 private:
  void clearResult() { result.get() = json::object(); }
};

}  // namespace

struct CxxDocument::Private {
  std::string fileName;
  long version;
  Diagnostics diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  std::shared_ptr<Toolchain> toolchain;
  std::function<void(const CodeCompletionContext&)> complete;

#ifndef CXX_NO_THREADS
  std::atomic<bool> cancelled{false};
#else
  bool cancelled{false};
#endif

  Private(std::string fileName, long version)
      : fileName(std::move(fileName)), version(version) {
    diagnosticsClient.setErrorLimit(kMaxDiagnostics);
  }

  auto symbolOccurrences() -> const SymbolOccurrences& {
    auto lock = lockCaches();
    if (!occurrences) occurrences = std::make_unique<SymbolOccurrences>(&unit);
    return *occurrences;
  }

  auto sourceText() -> const SourceText& {
    auto lock = lockCaches();
    if (!source) {
      source = std::make_unique<SourceText>(
          unit.preprocessor()->source(unit.preprocessor()->mainSourceFileId()));
    }
    return *source;
  }

 private:
#ifndef CXX_NO_THREADS
  auto lockCaches() -> std::unique_lock<std::mutex> {
    return std::unique_lock(cachesMutex);
  }

  std::mutex cachesMutex;
#else
  struct NoLock {};

  auto lockCaches() -> NoLock { return {}; }
#endif

  std::unique_ptr<SymbolOccurrences> occurrences;
  std::unique_ptr<SourceText> source;
};

auto CxxDocument::semanticTokenTypeLegend() -> const std::vector<std::string>& {
  static const auto tokenTypes = enumNames(kSemanticTokenTypes);
  return tokenTypes;
}

auto CxxDocument::semanticTokenModifierLegend()
    -> const std::vector<std::string>& {
  static const auto tokenModifiers = enumNames(kSemanticTokenModifiers);
  return tokenModifiers;
}

CxxDocument::CxxDocument(std::string fileName, long version)
    : d(std::make_unique<Private>(std::move(fileName), version)) {}

CxxDocument::~CxxDocument() {}

auto CxxDocument::isCancelled() const -> bool {
#ifndef CXX_NO_THREADS
  return d->cancelled.load();
#else
  return d->cancelled;
#endif
}

void CxxDocument::cancel() {
#ifndef CXX_NO_THREADS
  d->cancelled.store(true);
#else
  d->cancelled = true;
#endif
}

auto CxxDocument::fileName() const -> const std::string& { return d->fileName; }

auto CxxDocument::version() const -> long { return d->version; }

auto CxxDocument::translationUnit() const -> TranslationUnit* {
  return &d->unit;
}

auto CxxDocument::parserConfiguration() const -> ParserConfiguration {
  return ParserConfiguration{
      .checkTypes = true,
      .stopParsingPredicate = [this] { return isCancelled(); },
      .complete = d->complete,
  };
}

void CxxDocument::setToolchain(std::shared_ptr<Toolchain> toolchain) {
  d->toolchain = std::move(toolchain);
}

void CxxDocument::requestCodeCompletionAt(std::uint32_t line,
                                          std::uint32_t column,
                                          CompletionEditRange editRange,
                                          Vector<CompletionItem> result) {
  auto& unit = d->unit;

  (void)unit.blockErrors(true);

  unit.preprocessor()->requestCodeCompletionAt(line, column);

  d->complete = [sink = CompletionSink{&unit, result, editRange}](
                    const CodeCompletionContext& context) mutable {
    std::visit(sink, context);
  };
}

void CxxDocument::requestSignatureHelpAt(std::uint32_t line,
                                         std::uint32_t column,
                                         SignatureHelp result) {
  auto& unit = d->unit;

  (void)unit.blockErrors(true);

  unit.preprocessor()->requestCodeCompletionAt(line, column);

  d->complete = [sink = SignatureHelpSink{result}](
                    const CodeCompletionContext& context) mutable {
    std::visit(sink, context);
  };
}

void CxxDocument::semanticTokens(std::optional<Range> range,
                                 Vector<long> result) const {
  auto& unit = d->unit;
  auto preprocessor = unit.preprocessor();
  const std::string_view source =
      preprocessor->source(preprocessor->mainSourceFileId());
  const auto& sourceText = d->sourceText();
  const auto& occurrences = d->symbolOccurrences();

  struct SymbolToken {
    SemanticTokenTypes type;
    long modifiers = 0;
  };

  std::unordered_map<std::size_t, SymbolToken> symbolTokens;

  for (const auto& occurrence : occurrences.all()) {
    const auto& token = unit.tokenAt(occurrence.location);

    auto type = semanticTokenTypeOf(occurrence.symbol);
    if (!type.has_value()) continue;

    long modifiers = 0;
    if (occurrence.isDeclaration) {
      modifiers =
          semanticTokenModifierMask(SemanticTokenModifiers::kDeclaration);
    }
    auto current = symbolTokens.find(token.offset());
    if (current == symbolTokens.end()) {
      symbolTokens.emplace(token.offset(),
                           SymbolToken{.type = *type, .modifiers = modifiers});
      continue;
    }

    if (current->second.modifiers < modifiers) {
      current->second = SymbolToken{.type = *type, .modifiers = modifiers};
      continue;
    }

    if (current->second.modifiers > modifiers) continue;
    if (semanticTokenPriority(current->second.type) >=
        semanticTokenPriority(*type))
      continue;
    current->second = SymbolToken{.type = *type, .modifiers = modifiers};
  }

  std::uint32_t previousLine = 0;
  std::uint32_t previousCharacter = 0;
  bool hasPreviousToken = false;

  auto emit = [&](std::size_t startOffset, std::size_t endOffset,
                  SemanticTokenTypes type, long modifiers) {
    if (startOffset == endOffset) return;

    auto position = sourceText.positionAt(startOffset);
    auto endPosition = sourceText.positionAt(endOffset);
    if (range.has_value()) {
      if (!overlaps(position, endPosition, *range)) return;
    }

    auto length = sourceText.utf16Length(startOffset, endOffset);
    if (length == 0) return;

    std::uint32_t deltaLine = position.line;
    std::uint32_t deltaCharacter = position.character;

    if (hasPreviousToken) {
      deltaLine -= previousLine;
      if (deltaLine == 0) deltaCharacter -= previousCharacter;
    }

    result.emplace_back(long(deltaLine));
    result.emplace_back(long(deltaCharacter));
    result.emplace_back(long(length));
    result.emplace_back(enumIndex(kSemanticTokenTypes, type));
    result.emplace_back(modifiers);

    previousLine = position.line;
    previousCharacter = position.character;
    hasPreviousToken = true;
  };

  Lexer lexer(source, unit.language());
  lexer.setKeepComments(true);

  for (;;) {
    const auto kind = lexer.next();
    if (kind == TokenKind::T_EOF_SYMBOL) break;

    const auto tokenStart = std::size_t(lexer.tokenPos());
    const auto tokenEnd = tokenStart + lexer.tokenLength();
    auto modifiers = 0L;
    std::optional<SemanticTokenTypes> type;

    if (kind == TokenKind::T_IDENTIFIER) {
      auto symbolToken = symbolTokens.find(tokenStart);
      if (symbolToken != symbolTokens.end()) {
        type = symbolToken->second.type;
        modifiers = symbolToken->second.modifiers;
      }
    } else {
      type = lexicalSemanticTokenType(kind);
    }

    if (!type.has_value()) continue;

    auto segmentStart = tokenStart;
    while (segmentStart < tokenEnd) {
      auto newline = source.find('\n', segmentStart);
      auto segmentEnd = tokenEnd;
      if (newline != std::string_view::npos && newline < tokenEnd) {
        segmentEnd = newline;
      }

      emit(segmentStart, segmentEnd, *type, modifiers);

      if (segmentEnd == tokenEnd) break;
      segmentStart = segmentEnd + 1;
    }
  }
}

auto CxxDocument::hoverAt(std::size_t offset, Hover result) const -> bool {
  const auto& occurrences = d->symbolOccurrences();
  auto occurrence = occurrences.atOffset(offset);
  if (!occurrence) return false;

  auto text = hoverTextOf(occurrence->symbol);
  if (text.empty()) return false;

  auto contents = result.contents<MarkupContent>();
  contents.kind(MarkupKind::kMarkdown);
  contents.value(std::format("```cpp\n{}\n```", text));

  const auto& sourceText = d->sourceText();
  auto [startOffset, endOffset] = occurrences.rangeOf(occurrence->location);
  auto start = sourceText.positionAt(startOffset);
  auto end = sourceText.positionAt(endOffset);
  auto range = result.range<Range>();
  range.start().line(start.line).character(start.character);
  range.end().line(end.line).character(end.character);
  return true;
}

void CxxDocument::documentHighlightsAt(std::size_t offset,
                                       Vector<DocumentHighlight> result) const {
  const auto& occurrences = d->symbolOccurrences();
  auto selected = occurrences.atOffset(offset);
  if (!selected) return;

  const auto& sourceText = d->sourceText();

  for (const auto& occurrence : occurrences.all()) {
    if (!isSameEntity(occurrence.symbol, selected->symbol)) continue;

    auto [startOffset, endOffset] = occurrences.rangeOf(occurrence.location);
    auto start = sourceText.positionAt(startOffset);
    auto end = sourceText.positionAt(endOffset);
    auto highlight = result.emplace_back();
    highlight.kind(DocumentHighlightKind::kText);
    highlight.range().start().line(start.line).character(start.character);
    highlight.range().end().line(end.line).character(end.character);
  }
}

auto CxxDocument::diagnostics() const -> Vector<Diagnostic> {
  return Vector<Diagnostic>(d->diagnosticsClient.messages);
}

auto CxxDocument::hasErrors() const -> bool {
  return d->diagnosticsClient.hasErrors;
}

auto CxxDocument::textOf(AST* ast) -> std::optional<std::string_view> {
  return textInRange(ast->firstSourceLocation(), ast->lastSourceLocation());
}

auto CxxDocument::textInRange(SourceLocation start, SourceLocation end)
    -> std::optional<std::string_view> {
  auto& unit = d->unit;
  auto preprocessor = unit.preprocessor();

  if (!unit.ownsLocation(start) || !unit.ownsLocation(end.previous())) {
    return std::nullopt;
  }

  const auto startToken = unit.tokenAt(start);
  const auto endToken = unit.tokenAt(end.previous());

  if (startToken.fileId() != endToken.fileId()) {
    return std::nullopt;
  }

  std::string_view source = preprocessor->source(startToken.fileId());

  const auto offset = startToken.offset();
  const auto length = endToken.offset() + endToken.length() - offset;

  return source.substr(offset, length);
}

}  // namespace cxx::lsp
