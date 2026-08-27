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
#include <cxx/toolchain_config.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#ifndef CXX_NO_THREADS
#include <atomic>
#endif

#include <algorithm>
#include <format>
#include <iostream>

namespace cxx::lsp {

namespace {

constexpr int kMaxDiagnostics = 100;

struct Diagnostics final : cxx::DiagnosticsClient {
  json messages = json::array();
  Vector<lsp::Diagnostic> diagnostics{messages};

  void report(const cxx::Diagnostic& diag) override {
    auto start = preprocessor()->tokenStartPosition(diag.token());
    auto end = preprocessor()->tokenEndPosition(diag.token());

    auto tmp = json::object();

    auto d = diagnostics.emplace_back();

    int s = std::max(int(start.line) - 1, 0);
    int sc = std::max(int(start.column) - 1, 0);
    int e = std::max(int(end.line) - 1, 0);
    int ec = std::max(int(end.column) - 1, 0);

    d.message(diag.message());
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

class CompletionItemCollector {
 public:
  CompletionItemCollector(Vector<CompletionItem>& completionItems,
                          std::vector<std::string>& labels,
                          const AccessContext& accessContext)
      : completionItems_(completionItems),
        labels_(labels),
        accessContext_(accessContext) {}

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
      addLabel(to_string(member->name()));
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
      addLabel(to_string(field->name()));
    }
  }

  void addEnclosingScopes(ScopeSymbol* scope) {
    for (auto current = scope; current; current = current->parent()) {
      auto objectClass = symbol_cast<ClassSymbol>(current);
      addScope(current, objectClass);
    }
  }

 private:
  void addLabel(std::string label) {
    if (std::ranges::contains(labels_, label)) return;
    labels_.push_back(label);
    auto item = completionItems_.emplace_back();
    item.label(std::move(label));
  }

  Vector<CompletionItem>& completionItems_;
  std::vector<std::string>& labels_;
  const AccessContext& accessContext_;
  std::vector<ScopeSymbol*> visitedScopes_;
};

struct CompletionSink {
  TranslationUnit* unit;
  TypeTraits traits;
  Vector<CompletionItem>& completionItems;
  std::vector<std::string> labels;

  void operator()(const MemberCompletionContext& context) {
    auto objectClass = classSymbolOf(traits, context.objectType);
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
    collector.addDesignators(classSymbolOf(traits, context.objectType));
  }

  void operator()(const ArgumentHintsContext&) const {}
  void operator()(const TemplateArgumentHintsContext&) const {}

 private:
  auto collectorFor(const AccessContext& accessContext)
      -> CompletionItemCollector {
    return CompletionItemCollector{completionItems, labels, accessContext};
  }
};

struct SignatureHelpSink {
  std::vector<FunctionSymbol*> candidates;
  Symbol* templateSymbol = nullptr;
  int activeParameter = 0;

  void operator()(const ArgumentHintsContext& context) {
    candidates = context.candidates;
    templateSymbol = nullptr;
    activeParameter = context.activeParameter;
  }

  void operator()(const TemplateArgumentHintsContext& context) {
    candidates.clear();
    templateSymbol = context.templateSymbol;
    activeParameter = context.activeParameter;
  }

  void operator()(const MemberCompletionContext&) const {}
  void operator()(const ScopeCompletionContext&) const {}
  void operator()(const UnqualifiedCompletionContext&) const {}
  void operator()(const DesignatorCompletionContext&) const {}
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
  auto templateParameters = cxx::visit(
      []<typename S>(S* symbol) -> TemplateParametersSymbol* {
        if constexpr (requires { symbol->templateParameters(); })
          return symbol->templateParameters();
        else
          return nullptr;
      },
      templateSymbol);
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

}  // namespace

struct CxxDocument::Private {
  const CLI& cli;
  std::string fileName;
  long version;
  Diagnostics diagnosticsClient;
  TranslationUnit unit{&diagnosticsClient};
  std::shared_ptr<Toolchain> toolchain;

#ifndef CXX_NO_THREADS
  std::atomic<bool> cancelled{false};
#else
  bool cancelled{false};
#endif

  Private(const CLI& cli, std::string fileName, long version)
      : cli(cli), fileName(std::move(fileName)), version(version) {
    diagnosticsClient.setErrorLimit(kMaxDiagnostics);
  }

  void configure();
};

void CxxDocument::Private::configure() {
  auto preprocessor = unit.preprocessor();
  std::string error;
  toolchain = createToolchain(cli, preprocessor, error);
  if (!error.empty()) toolchain.reset();
  if (toolchain) unit.control()->setMemoryLayout(toolchain->memoryLayout());
}

CxxDocument::CxxDocument(const CLI& cli, std::string fileName, long version)
    : d(std::make_unique<Private>(cli, std::move(fileName), version)) {}

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

void CxxDocument::codeCompletionAt(std::string source, std::uint32_t line,
                                   std::uint32_t column,
                                   Vector<CompletionItem> completionItems) {
  auto& unit = d->unit;

  (void)unit.blockErrors(true);

  unit.preprocessor()->requestCodeCompletionAt(line, column);

  auto traits = unit.typeTraits();

  CompletionSink sink{&unit, traits, completionItems};

  parse(std::move(source), [&sink](const CodeCompletionContext& context) {
    std::visit(sink, context);
  });
}

void CxxDocument::signatureHelpAt(std::string source, std::uint32_t line,
                                  std::uint32_t column, SignatureHelp result) {
  auto& unit = d->unit;

  (void)unit.blockErrors(true);

  unit.preprocessor()->requestCodeCompletionAt(line, column);

  SignatureHelpSink sink;

  parse(std::move(source), [&sink](const CodeCompletionContext& context) {
    std::visit(sink, context);
  });

  if (sink.templateSymbol) {
    addTemplateSignature(result, sink.templateSymbol, sink.activeParameter);
    return;
  }

  if (sink.candidates.empty()) return;

  auto signatures = result.signatures();
  int activeSignature = 0;
  bool foundActiveSignature = false;

  for (auto function : sink.candidates) {
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

    if (parameterCount > sink.activeParameter) {
      foundActiveSignature = true;
      continue;
    }

    ++activeSignature;
  }

  if (!foundActiveSignature) activeSignature = 0;

  result.activeSignature(activeSignature);
  result.activeParameter(long(sink.activeParameter));
}

void CxxDocument::parse(std::string source) { parse(std::move(source), {}); }

void CxxDocument::parse(
    std::string source,
    std::function<void(const CodeCompletionContext&)> complete) {
  d->configure();

  auto& unit = d->unit;

  auto preprocessor = unit.preprocessor();

  DefaultPreprocessorState state{*preprocessor};

  unit.beginPreprocessing(std::move(source), d->fileName);

  while (state) {
    if (isCancelled()) break;
    std::visit(state, unit.continuePreprocessing());
  }

  unit.endPreprocessing();

  auto stopParsingPredicate = [this] { return isCancelled(); };

  unit.parse(ParserConfiguration{
      .checkTypes = true,
      .stopParsingPredicate = stopParsingPredicate,
      .complete = std::move(complete),
  });
}

CxxDocument::~CxxDocument() {}

auto CxxDocument::version() const -> long { return d->version; }

auto CxxDocument::diagnostics() const -> Vector<Diagnostic> {
  return Vector<Diagnostic>(d->diagnosticsClient.messages);
}

auto CxxDocument::translationUnit() const -> TranslationUnit* {
  return &d->unit;
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
