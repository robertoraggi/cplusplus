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
#include <cxx/control.h>
#include <cxx/lsp/enums.h>
#include <cxx/lsp/types.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#ifndef CXX_NO_THREADS
#include <atomic>
#endif

#include <algorithm>
#include <functional>

namespace cxx::lsp {

namespace {

constexpr int kMaxDiagnostics = 100;

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

    auto start = preprocessor()->tokenStartPosition(diag.token());
    auto end = preprocessor()->tokenEndPosition(diag.token());

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
};

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
