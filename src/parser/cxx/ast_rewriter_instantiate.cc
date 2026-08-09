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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>
#include <cxx/views/symbols.h>

#include <format>
#include <unordered_set>

namespace cxx {
namespace {
class InstantiationDepthGuard {
 public:
  explicit InstantiationDepthGuard(TranslationUnit* unit) : unit_(unit) {
    unit_->setTemplateInstantiationDepth(unit_->templateInstantiationDepth() +
                                         1);
  }

  ~InstantiationDepthGuard() {
    unit_->setTemplateInstantiationDepth(unit_->templateInstantiationDepth() -
                                         1);
  }

  [[nodiscard]] auto exceeded() const -> bool {
    return unit_->templateInstantiationDepth() >
           TranslationUnit::kMaxTemplateInstantiationDepth;
  }

 private:
  TranslationUnit* unit_;
};

struct GetSpecialization {
  const std::vector<TemplateArgument>& templateArguments;

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(TypeAliasSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    return symbol->findSpecialization(templateArguments);
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

struct Instantiate {
  ASTRewriter& rewriter;
  ScopeSymbol* parentScope = nullptr;
  bool declarationOnly = false;

  void attachDeferredBody(FunctionDefinitionAST* instance,
                          FunctionDefinitionAST* pattern) {
    if (instance->functionBody) return;
    auto fn = symbol_cast<FunctionSymbol>(instance->symbol);
    if (!fn || fn->hasPendingBody()) return;
    auto pending = std::make_unique<PendingBodyInstantiation>();
    pending->originalDefinition = pattern;
    pending->templateArguments = rewriter.templateArguments();
    pending->parentScope = parentScope;
    pending->depth = rewriter.depth();
    fn->setPendingBody(std::move(pending));
  }

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    auto classSpecifier = ast_cast<ClassSpecifierAST>(symbol->declaration());
    if (!classSpecifier) return nullptr;

    auto instance =
        ast_cast<ClassSpecifierAST>(rewriter.specifier(classSpecifier));
    if (!instance) return nullptr;

    return instance->symbol;
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto declaration = templateDecl->declaration;
    auto simpleDecl = ast_cast<SimpleDeclarationAST>(declaration);
    if (!simpleDecl) return nullptr;

    auto instance =
        ast_cast<SimpleDeclarationAST>(rewriter.declaration(simpleDecl));
    if (!instance || !instance->initDeclaratorList ||
        !instance->initDeclaratorList->value) {
      return nullptr;
    }

    auto instantiatedSymbol = instance->initDeclaratorList->value->symbol;
    if (!instantiatedSymbol) return nullptr;
    return symbol_cast<VariableSymbol>(instantiatedSymbol);
  }

  auto operator()(TypeAliasSymbol* symbol) -> Symbol* {
    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto declaration = ast_cast<AliasDeclarationAST>(templateDecl->declaration);
    if (!declaration) return nullptr;

    auto instance =
        ast_cast<AliasDeclarationAST>(rewriter.declaration(declaration));
    if (!instance) return nullptr;

    if (auto written =
            rewriter.writtenArgumentForAliasedParameter(declaration->typeId)) {
      if (auto alias = symbol_cast<TypeAliasSymbol>(instance->symbol))
        alias->setExpansionTypeId(written);
    }

    return instance->symbol;
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    rewriter.retryPendingMemberTemplateAttachment(symbol);

    if (symbol->hasPendingBody()) {
      rewriter.completePendingBody(symbol);
    }

    auto functionDef = symbol->declaration();
    auto definingSymbol = symbol;

    if (!functionDef) {
      for (auto redecl : symbol->redeclarations()) {
        if (auto def = redecl->declaration()) {
          functionDef = def;
          definingSymbol = redecl;
          break;
        }
      }
    }

    if (definingSymbol != symbol) {
      if (auto definingTemplateDecl = definingSymbol->templateDeclaration()) {
        rewriter.setDepth(definingTemplateDecl->depth);
      }
    }

    rewriter.setInstantiatingFunctionTemplateSpecialization(
        symbol->templateDeclaration() != nullptr);

    if (functionDef) {
      auto instance =
          ast_cast<FunctionDefinitionAST>(rewriter.declaration(functionDef));
      if (!instance) return nullptr;
      if (declarationOnly) attachDeferredBody(instance, functionDef);
      return instance->symbol;
    }

    auto templateDecl = symbol->templateDeclaration();
    if (!templateDecl) return nullptr;

    auto declaration =
        ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
    if (!declaration) return nullptr;

    auto instance =
        ast_cast<SimpleDeclarationAST>(rewriter.declaration(declaration));
    if (!instance || !instance->initDeclaratorList ||
        !instance->initDeclaratorList->value) {
      return nullptr;
    }

    return instance->initDeclaratorList->value->symbol;
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

[[nodiscard]] auto isPrimaryTemplate(
    const std::vector<TemplateArgument>& templateArguments) -> bool {
  if (templateArguments.empty()) return false;

  int expected = 0;
  for (const auto& arg : templateArguments) {
    if (!std::holds_alternative<Symbol*>(arg)) return false;

    auto sym = std::get<Symbol*>(arg);
    if (!sym) return false;

    if (auto pack = symbol_cast<ParameterPackSymbol>(sym)) {
      if (pack->elements().size() != 1) return false;
      auto element = pack->elements()[0];
      if (!element) return false;

      auto elementType = element->type();
      if (!elementType) return false;

      auto ty = getTypeParamInfo(elementType);
      if (!ty) return false;
      if (ty->index != expected) return false;
      if (!ty->isPack) return false;
      ++expected;
      continue;
    }

    auto symType = sym->type();
    if (!symType) return false;

    auto ty = getTypeParamInfo(symType);
    if (!ty) return false;
    if (ty->index != expected) return false;
    ++expected;
  }
  return true;
}

[[nodiscard]] auto templateParameterCount(TemplateDeclarationAST* templateDecl)
    -> int {
  if (!templateDecl) return 0;
  int count = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    (void)parameter;
    ++count;
  }
  return count;
}

[[nodiscard]] auto computeInstantiationClassName(
    TranslationUnit* unit, Symbol* primaryTemplate,
    const std::vector<TemplateArgument>& templateArguments) -> std::string {
  if (!primaryTemplate) return "template";
  return to_string(unit->control()->getTemplateId(primaryTemplate->name(),
                                                  templateArguments));
}

[[nodiscard]] auto instantiationLabel(Symbol* symbol) -> std::string_view {
  return symbol_cast<FunctionSymbol>(symbol)
             ? "function template specialization"
             : "template class";
}

[[nodiscard]] auto findMutableSpecialization(Symbol* primary, Symbol* spec)
    -> TemplateSpecialization* {
  if (!primary || !spec) return nullptr;
  auto search = [spec](auto sym) -> TemplateSpecialization* {
    for (auto& s : sym->mutableSpecializations())
      if (s.symbol == spec) return &s;
    return nullptr;
  };
  if (auto cs = symbol_cast<ClassSymbol>(primary)) return search(cs);
  if (auto as = symbol_cast<TypeAliasSymbol>(primary)) return search(as);
  if (auto vs = symbol_cast<VariableSymbol>(primary)) return search(vs);
  if (auto fs = symbol_cast<FunctionSymbol>(primary)) return search(fs);
  return nullptr;
}

[[nodiscard]] auto instantiateBuiltinMakeIntegerSeq(
    TranslationUnit* unit,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation instantiationLoc, bool sfinaeContext, bool argsComplete,
    bool declarationOnly) -> Symbol* {
  if (templateArguments.size() != 3) return nullptr;

  Symbol* seqClass = nullptr;
  if (auto symArg = std::get_if<Symbol*>(&templateArguments[0])) {
    seqClass = template_name_symbol(*symArg);
  } else if (auto typeArg = std::get_if<const Type*>(&templateArguments[0])) {
    if (auto classType = type_cast<ClassType>(*typeArg)) {
      seqClass = template_name_symbol(classType->symbol());
    }
  }
  if (!seqClass || !template_declaration_of(seqClass)) return nullptr;

  auto elementType = template_argument_type(templateArguments[1]);
  if (!elementType) return nullptr;

  std::optional<std::intmax_t> N;
  if (auto val = template_argument_value(templateArguments[2])) {
    auto interp = ASTInterpreter{unit};
    if (auto intVal = interp.toInt(*val)) {
      N = *intVal;
    }
  } else if (auto expr = std::get_if<ExpressionAST*>(&templateArguments[2])) {
    auto interp = ASTInterpreter{unit};
    if (auto val = interp.evaluate(*expr)) {
      if (auto intVal = interp.toInt(*val)) {
        N = *intVal;
      }
    }
  }
  if (!N.has_value() || *N < 0) return nullptr;

  auto ar = unit->arena();
  List<TemplateArgumentAST*>* expandedArgs = nullptr;
  List<TemplateArgumentAST*>** it = &expandedArgs;

  auto typeId = TypeIdAST::create(ar);
  typeId->type = elementType;
  auto expandedTypeArg = TypeTemplateArgumentAST::create(ar, typeId);
  *it = make_list_node(ar, static_cast<TemplateArgumentAST*>(expandedTypeArg));
  it = &(*it)->next;

  for (std::intmax_t i = 0; i < *N; ++i) {
    std::string spelling = std::format("{}", i);
    auto literal = unit->control()->integerLiteral(spelling);
    auto intExpr = IntLiteralExpressionAST::create(
        ar, literal, ValueCategory::kPrValue, elementType);
    auto exprArg = ExpressionTemplateArgumentAST::create(ar, intExpr);
    *it = make_list_node(ar, static_cast<TemplateArgumentAST*>(exprArg));
    it = &(*it)->next;
  }

  return ASTRewriter::instantiate(unit, expandedArgs, seqClass,
                                  instantiationLoc, sfinaeContext, argsComplete,
                                  declarationOnly);
}

[[nodiscard]] auto instantiateBuiltinTypePackElement(
    TranslationUnit* unit, Symbol* symbol,
    const std::vector<TemplateArgument>& templateArguments) -> Symbol* {
  if (templateArguments.size() < 2) return nullptr;

  std::optional<std::intmax_t> N;
  if (auto val = template_argument_value(templateArguments[0])) {
    auto interp = ASTInterpreter{unit};
    if (auto intVal = interp.toInt(*val)) {
      N = *intVal;
    }
  } else if (auto expr = std::get_if<ExpressionAST*>(&templateArguments[0])) {
    auto interp = ASTInterpreter{unit};
    if (auto val = interp.evaluate(*expr)) {
      if (auto intVal = interp.toInt(*val)) {
        N = *intVal;
      }
    }
  }

  auto packSize = static_cast<std::intmax_t>(templateArguments.size() - 1);
  if (!N.has_value() || *N < 0 || *N >= packSize) return nullptr;

  auto elementType = template_argument_type(templateArguments[1 + *N]);
  if (!elementType) return nullptr;

  auto alias = unit->control()->newTypeAliasSymbol(nullptr, {});
  alias->setName(symbol->name());
  alias->setType(elementType);
  return alias;
}

[[nodiscard]] auto instantiateBuiltinCommonType(
    TranslationUnit* unit,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation instantiationLoc, bool sfinaeContext, bool argsComplete,
    bool declarationOnly) -> Symbol* {
  auto traits = TypeTraits{unit};

  if (templateArguments.size() < 3) return nullptr;

  ClassSymbol* identityClass = nullptr;
  if (auto identityArgType = template_argument_type(templateArguments[1])) {
    if (auto identityType = type_cast<ClassType>(identityArgType)) {
      identityClass = symbol_cast<ClassSymbol>(identityType->symbol());
    }
  } else if (auto sym = std::get_if<Symbol*>(&templateArguments[1])) {
    identityClass = symbol_cast<ClassSymbol>(*sym);
  }
  if (!identityClass || !template_declaration_of(identityClass)) return nullptr;

  ClassSymbol* emptyClass = nullptr;
  if (auto emptyArgType = template_argument_type(templateArguments[2])) {
    if (auto ctype = type_cast<ClassType>(emptyArgType)) {
      emptyClass = symbol_cast<ClassSymbol>(ctype->symbol());
    }
  } else if (auto sym = std::get_if<Symbol*>(&templateArguments[2])) {
    emptyClass = symbol_cast<ClassSymbol>(*sym);
  }
  if (!emptyClass) return nullptr;

  std::vector<const Type*> operands;
  for (std::size_t i = 3; i < templateArguments.size(); ++i) {
    auto type = template_argument_type(templateArguments[i]);
    if (!type) return nullptr;
    operands.push_back(type);
  }

  if (operands.empty()) return emptyClass;

  const Type* result = traits.remove_cvref(operands[0]);
  for (std::size_t i = 1; i < operands.size(); ++i) {
    auto next = traits.remove_cvref(operands[i]);
    if (traits.is_same(result, next)) continue;
    auto combined = StandardConversion{unit}.commonArithmeticType(result, next);
    if (!combined) return emptyClass;
    result = combined;
  }

  auto ar = unit->arena();
  List<TemplateArgumentAST*>* expandedArgs = nullptr;
  List<TemplateArgumentAST*>** it = &expandedArgs;

  auto typeId = TypeIdAST::create(ar);
  typeId->type = result;
  auto typeArg = TypeTemplateArgumentAST::create(ar, typeId);
  *it = make_list_node(ar, static_cast<TemplateArgumentAST*>(typeArg));

  return ASTRewriter::instantiate(unit, expandedArgs, identityClass,
                                  instantiationLoc, sfinaeContext, argsComplete,
                                  declarationOnly);
}

[[nodiscard]] auto instantiateBuiltinTemplate(
    TranslationUnit* unit, Symbol* symbol, BuiltinTemplateKind builtinKind,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation instantiationLoc, bool sfinaeContext, bool argsComplete,
    bool declarationOnly) -> Symbol* {
  auto expandedArguments = expand_template_arguments(templateArguments);

  switch (builtinKind) {
    case BuiltinTemplateKind::T___MAKE_INTEGER_SEQ:
      return instantiateBuiltinMakeIntegerSeq(unit, expandedArguments,
                                              instantiationLoc, sfinaeContext,
                                              argsComplete, declarationOnly);
    case BuiltinTemplateKind::T___TYPE_PACK_ELEMENT:
      return instantiateBuiltinTypePackElement(unit, symbol, expandedArguments);
    case BuiltinTemplateKind::T___BUILTIN_COMMON_TYPE:
      return instantiateBuiltinCommonType(unit, expandedArguments,
                                          instantiationLoc, sfinaeContext,
                                          argsComplete, declarationOnly);
    default:
      return nullptr;
  }
}

}  // namespace

auto ASTRewriter::paste(TranslationUnit* unit, ScopeSymbol* scope,
                        StatementAST* ast) -> StatementAST* {
  auto rewriter = ASTRewriter{unit, scope, {}};
  auto result = rewriter.statement(ast);
  return result;
}

auto ASTRewriter::substituteDefaultTypeId(
    TranslationUnit* unit, TypeIdAST* typeId,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> TypeIdAST* {
  if (!typeId) return nullptr;
  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;
  return rewriter.typeId(typeId);
}

auto ASTRewriter::substituteDefaultExpression(
    TranslationUnit* unit, ExpressionAST* expression,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> ExpressionAST* {
  if (!expression) return nullptr;
  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;
  return rewriter.expression(expression);
}

auto ASTRewriter::substituteParameterClause(
    TranslationUnit* unit, ParameterDeclarationClauseAST* parameters,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> ParameterDeclarationClauseAST* {
  if (!parameters) return nullptr;

  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;

  return rewriter.parameterDeclarationClause(parameters);
}

auto ASTRewriter::substituteParameterTypes(
    TranslationUnit* unit, ParameterDeclarationClauseAST* parameters,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> std::optional<std::vector<const Type*>> {
  if (!parameters) return std::vector<const Type*>{};

  auto rewritten = substituteParameterClause(unit, parameters,
                                             templateArguments, depth, scope);
  if (!rewritten) return std::nullopt;

  std::vector<const Type*> parameterTypes;
  for (auto parameter : ListView{rewritten->parameterDeclarationList}) {
    if (!parameter->type) return std::nullopt;
    parameterTypes.push_back(parameter->type);
  }

  return parameterTypes;
}

void ASTRewriter::reportPendingInstantiationErrors(
    TranslationUnit* unit, Symbol* primaryTemplate, Symbol* instantiated,
    SourceLocation instantiationLoc) {
  if (!primaryTemplate || !instantiated || !instantiationLoc) return;
  if (auto spec = findMutableSpecialization(primaryTemplate, instantiated)) {
    if (!spec->instantiationErrors.empty()) {
      for (auto& diag : spec->instantiationErrors)
        unit->diagnosticsClient()->report(diag);
      spec->instantiationErrors.clear();
      auto name =
          computeInstantiationClassName(unit, primaryTemplate, spec->arguments);
      auto label = instantiationLabel(primaryTemplate);
      unit->note(instantiationLoc,
                 std::format("in instantiation of {} '{}' requested here",
                             label, name));
    }
  }
}

auto ASTRewriter::instantiateForArgs(
    TranslationUnit* unit, List<TemplateArgumentAST*>* deducedArguments,
    FunctionSymbol* function, SourceLocation instantiationLoc,
    bool argsComplete, bool declarationOnly) -> FunctionSymbol* {
  return symbol_cast<FunctionSymbol>(
      instantiate(unit, deducedArguments, function, instantiationLoc,
                  /*sfinaeContext=*/true, argsComplete, declarationOnly));
}

auto ASTRewriter::instantiate(TranslationUnit* unit,
                              List<TemplateArgumentAST*>* templateArgumentList,
                              Symbol* symbol, SourceLocation instantiationLoc,
                              bool sfinaeContext, bool argsComplete,
                              bool declarationOnly,
                              bool retainEnclosingTemplateLevels) -> Symbol* {
  if (!symbol) return nullptr;

  if (!unit->config().checkTypes) return nullptr;

  InstantiationDepthGuard depthGuard{unit};

  if (depthGuard.exceeded()) {
    auto message = std::format(
        "recursive template instantiation exceeded maximum depth "
        "of {} while instantiating '{}'",
        TranslationUnit::kMaxTemplateInstantiationDepth,
        to_string(symbol->name()));

    if (auto client = unit->reportingDiagnosticsClient();
        client && unit->diagnosticsClient() &&
        unit->diagnosticsClient()->isSfinae()) {
      client->report(unit->tokenAt(instantiationLoc), Severity::Error,
                     std::move(message));
    } else {
      unit->error(instantiationLoc, std::move(message));
    }

    return nullptr;
  }

  const auto activeClientIsSfinae =
      unit->diagnosticsClient() && unit->diagnosticsClient()->isSfinae();

  if (!sfinaeContext && activeClientIsSfinae) {
    sfinaeContext = true;
  }

  auto templateDecl = template_declaration_of(symbol);
  if (!templateDecl) return nullptr;

  auto declaration = template_declaration_ast(symbol);
  if (!declaration) return nullptr;

  const bool ownsSfinaeClient = sfinaeContext && !activeClientIsSfinae;

  std::optional<SilentDiagnosticsClient> sfinaeClient;
  DiagnosticsClient* savedDiagClient = nullptr;
  if (ownsSfinaeClient) {
    sfinaeClient.emplace();
    savedDiagClient = unit->changeDiagnosticsClient(&*sfinaeClient);
  }

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList,
                                  argsComplete);

  if (!subst) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return nullptr;
  }

  auto templateArguments = std::move(*subst).templateArguments();

  auto identifier = name_cast<Identifier>(symbol->name());
  if (identifier &&
      identifier->builtinTemplate() != BuiltinTemplateKind::T_NONE) {
    auto builtinKind = identifier->builtinTemplate();
    auto result = instantiateBuiltinTemplate(
        unit, symbol, builtinKind, templateArguments, instantiationLoc,
        sfinaeContext, argsComplete, declarationOnly);
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return result;
  }

  if (symbol_cast<FunctionSymbol>(symbol) &&
      static_cast<int>(templateArguments.size()) <
          templateParameterCount(templateDecl)) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return symbol;
  }

  if (isPrimaryTemplate(templateArguments)) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return symbol;
  }

  auto cached = retainEnclosingTemplateLevels
                    ? nullptr
                    : visit(GetSpecialization{templateArguments}, symbol);

  if (cached) {
    auto cachedClass = symbol_cast<ClassSymbol>(cached);
    if (!cachedClass) {
      if (!declarationOnly) {
        if (auto cachedFn = symbol_cast<FunctionSymbol>(cached);
            cachedFn && cachedFn->hasPendingBody()) {
          auto bodyErrors = ASTRewriter::completePendingBodyFor(
              unit, cachedFn, /*captureBodyErrors=*/true);
          if (!bodyErrors.empty()) {
            if (auto spec = findMutableSpecialization(symbol, cached)) {
              spec->instantiationErrors = std::move(bodyErrors);
            }
          }
        }
      }
      if (!sfinaeContext)
        reportPendingInstantiationErrors(unit, symbol, cached,
                                         instantiationLoc);
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return cached;
    }
    if (cachedClass->declaration()) {
      if (!sfinaeContext)
        reportPendingInstantiationErrors(unit, symbol, cached,
                                         instantiationLoc);
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return cached;
    }
  }

  if (!checkAssociatedConstraints(unit, symbol, templateArguments,
                                  templateDecl->depth)) {
    if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
    return nullptr;
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, classSymbol, templateArguments);
    if (partial.handled()) {
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return partial.symbol;
    }
  }

  if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, variableSymbol, templateArguments);
    if (partial.handled()) {
      if (savedDiagClient) (void)unit->changeDiagnosticsClient(savedDiagClient);
      return partial.symbol;
    }
  }

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;
  rewriter.inheritEnclosingTemplateArguments(parentScope);
  rewriter.writtenTemplateArgumentList_ = templateArgumentList;
  rewriter.setRetainsEnclosingTemplateLevels(retainEnclosingTemplateLevels);
  rewriter.binder().setInstantiatingSymbol(symbol);
  rewriter.binder().setInstantiationLoc(instantiationLoc);
  if (declarationOnly) rewriter.setRestrictedToDeclarations(true);

  auto registerFunctionSpecialization = [&](Symbol* result) {
    if (!result || result == symbol) return;
    auto fnTemplate = symbol_cast<FunctionSymbol>(symbol);
    if (!fnTemplate) return;
    auto instance = symbol_cast<FunctionSymbol>(result);
    if (!instance || instance->isSpecialization()) return;

    if (fnTemplate->isFriend()) instance->setFriend(true);

    if (fnTemplate->findSpecialization(templateArguments)) return;
    fnTemplate->addSpecialization(templateArguments, instance);
  };

  if (sfinaeContext) {
    auto instance =
        visit(Instantiate{rewriter, parentScope, declarationOnly}, symbol);
    if (ownsSfinaeClient) {
      (void)unit->changeDiagnosticsClient(savedDiagClient);
      if (sfinaeClient->hadError()) return nullptr;
    }
    if (rewriter.substitutionFailed()) return nullptr;

    registerFunctionSpecialization(instance);

    auto bodyErrors = rewriter.takeBodyErrors();
    if (!bodyErrors.empty() && instance) {
      if (auto spec = findMutableSpecialization(symbol, instance)) {
        spec->instantiationErrors = std::move(bodyErrors);
      }
    }

    return instance;
  }

  CapturingDiagnosticsClient capturing{unit->diagnosticsClient()};
  (void)unit->changeDiagnosticsClient(&capturing);

  auto instantiatedSymbol =
      visit(Instantiate{rewriter, parentScope, declarationOnly}, symbol);

  (void)unit->changeDiagnosticsClient(capturing.parent);

  registerFunctionSpecialization(instantiatedSymbol);

  auto bodyErrors = rewriter.takeBodyErrors();
  capturing.diagnostics.insert(capturing.diagnostics.end(),
                               std::make_move_iterator(bodyErrors.begin()),
                               std::make_move_iterator(bodyErrors.end()));

  if (!capturing.diagnostics.empty()) {
    if (auto spec = findMutableSpecialization(symbol, instantiatedSymbol)) {
      spec->instantiationErrors = std::move(capturing.diagnostics);
    }
    if (instantiationLoc) {
      auto name =
          computeInstantiationClassName(unit, symbol, templateArguments);
      auto label = instantiationLabel(symbol);
      unit->note(instantiationLoc,
                 std::format("in instantiation of {} '{}' requested here",
                             label, name));
    }
  }

  return instantiatedSymbol;
}

void ASTRewriter::markExplicitInstantiationDeclared(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    Symbol* symbol) {
  if (!symbol) return;
  if (!unit->config().checkTypes) return;

  auto classSymbol = symbol_cast<ClassSymbol>(symbol);
  if (!classSymbol) return;

  auto templateDecl = template_declaration_of(symbol);
  if (!templateDecl) return;

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList,
                                  /*argsComplete=*/true);
  if (!subst) return;

  auto templateArguments = std::move(*subst).templateArguments();

  if (isPrimaryTemplate(templateArguments)) return;

  classSymbol->addExternInstantiationDeclaration(std::move(templateArguments));
}

auto ASTRewriter::ensureCompleteClass(TranslationUnit* unit,
                                      ClassSymbol* classSymbol) -> bool {
  if (!classSymbol) return false;
  if (classSymbol->resolvedDefinition()->isComplete()) return true;
  if (!classSymbol->isSpecialization()) return false;

  auto primaryTemplate = classSymbol->primaryTemplateSymbol();
  if (!primaryTemplate) return false;

  TemplateSpecialization* spec = nullptr;
  for (auto& s : primaryTemplate->mutableSpecializations()) {
    if (s.symbol == classSymbol) {
      spec = &s;
      break;
    }
  }

  if (!spec || !spec->isPendingInstantiation) return false;

  auto pendingArgList = spec->pendingArgumentList;
  auto pendingLoc = spec->pendingInstantiationLoc;
  spec->isPendingInstantiation = false;
  spec->pendingArgumentList = nullptr;

  auto result =
      instantiate(unit, pendingArgList, primaryTemplate, pendingLoc, false);

  if (!result) return false;

  auto resultClass = symbol_cast<ClassSymbol>(result);
  if (!resultClass || !resultClass->isComplete()) return false;

  if (resultClass != classSymbol) {
    classSymbol->addRedeclaration(resultClass);
    classSymbol->setDefinition(resultClass);
    resultClass->setType(classSymbol->type());
  }

  return resultClass->isComplete();
}

void ASTRewriter::instantiateOutOfClassMemberDefinitions(ClassSymbol* pattern) {
  if (!pattern) return;
  if (templateArguments_.empty()) return;

  auto instantiateDefinition = [&](Symbol* def, DeclarationAST* declaration,
                                   TemplateDeclarationAST* templateDecl) {
    auto lexicalScope = def->enclosingNonTemplateParametersScope();
    while (lexicalScope && lexicalScope->isClass()) {
      lexicalScope = lexicalScope->enclosingNonTemplateParametersScope();
    }
    auto rewriter = ASTRewriter{
        unit_, lexicalScope, std::vector<TemplateArgument>(templateArguments_)};
    rewriter.depth_ = templateDecl->depth;
    rewriter.binder_.setInstantiatingSymbol(def);
    if (auto instance = symbol_cast<ClassSymbol>(
            pattern->findSpecialization(templateArguments_))) {
      rewriter.remapScopeMembers(pattern, instance);
    }
    (void)rewriter.declaration(declaration);
  };

  auto instanceClass =
      symbol_cast<ClassSymbol>(pattern->findSpecialization(templateArguments_));
  if (instanceClass) {
    if (auto def = symbol_cast<ClassSymbol>(instanceClass->definition());
        def && def != instanceClass) {
      instanceClass = def;
    }
  }

  if (instanceClass) {
    unit_->addPendingMemberInstantiation(instanceClass);
    remapScopeMembers(pattern, instanceClass);
  }

  auto attachPendingBody = [&](FunctionSymbol* target, FunctionSymbol* def,
                               FunctionDefinitionAST* defAst, int depth) {
    auto lexicalScope = def->enclosingNonTemplateParametersScope();
    while (lexicalScope && lexicalScope->isClass()) {
      lexicalScope = lexicalScope->enclosingNonTemplateParametersScope();
    }

    auto pending = std::make_unique<PendingBodyInstantiation>();
    pending->originalDefinition = defAst;
    pending->templateArguments = templateArguments_;
    pending->parentScope = lexicalScope;
    pending->depth = depth;
    target->setPendingBody(std::move(pending));
    if (target->isDefinitionRequired()) {
      unit_->addPendingBodyCompletion(target);
    }
  };

  auto attachPendingDefinition = [&](FunctionSymbol* member) {
    if (!instanceClass) return;
    if (member->isFriend()) return;
    auto def = symbol_cast<FunctionSymbol>(member->definition());
    if (!def || def == member) return;
    auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
    if (!defAst) return;
    auto classTemplateDecl = pattern->templateDeclaration();
    if (!classTemplateDecl) return;

    auto instanceMember = symbol_cast<FunctionSymbol>(remapSymbol(member));
    if (!instanceMember || instanceMember == member) return;
    if (!instanceMember->templateDeclaration()) return;
    if (instanceMember->declaration()) return;
    if (instanceMember->hasPendingBody()) return;

    attachPendingBody(instanceMember, def, defAst, classTemplateDecl->depth);
  };

  auto instantiateFunctionDefinition = [&](FunctionSymbol* member) {
    if (member->templateDeclaration()) return;
    auto def = symbol_cast<FunctionSymbol>(member->definition());
    if (!def || def == member) return;
    auto templateDecl = def->templateDeclaration();
    if (!templateDecl) return;
    auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
    if (!defAst) return;
    auto instanceMember = symbol_cast<FunctionSymbol>(remapSymbol(member));
    if (!instanceMember || instanceMember == member) return;
    if (instanceMember->isDefined()) return;
    if (instanceMember->hasPendingBody()) return;
    attachPendingBody(instanceMember, def, defAst, templateDecl->depth);
  };

  for (auto member : pattern->members()) {
    for (auto function : views::each_function(member)) {
      if (function->templateDeclaration()) attachPendingDefinition(function);
    }
  }

  if (instanceClass) {
    auto classTemplateDecl = pattern->templateDeclaration();
    if (classTemplateDecl) {
      for (auto instanceCtor : instanceClass->declaredConstructors()) {
        if (!instanceCtor->templateDeclaration()) continue;
        if (instanceCtor->declaration()) continue;
        if (instanceCtor->hasPendingBody()) continue;

        FunctionSymbol* patternCtor = nullptr;
        for (auto candidate : pattern->declaredConstructors()) {
          if (!candidate->templateDeclaration()) continue;
          if (candidate->isFriend()) continue;
          if (!areFunctionTemplateHeadsEquivalentForRedeclaration(
                  unit_, pattern, instanceCtor->templateDeclaration(),
                  candidate->templateDeclaration())) {
            continue;
          }
          patternCtor = candidate;
          break;
        }
        if (!patternCtor) continue;

        auto def = symbol_cast<FunctionSymbol>(patternCtor->definition());
        if (!def || def == patternCtor) continue;
        auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
        if (!defAst) continue;

        attachPendingBody(instanceCtor, def, defAst, classTemplateDecl->depth);
      }
    }
  }

  for (auto member : pattern->members()) {
    auto memberClass = symbol_cast<ClassSymbol>(member);
    if (!memberClass) continue;
    auto def = symbol_cast<ClassSymbol>(memberClass->definition());
    if (!def || def == memberClass) continue;
    auto templateDecl = def->templateDeclaration();
    if (!templateDecl) continue;
    auto declAst = ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
    if (!declAst) continue;
    if (instanceClass) {
      bool alreadyComplete = false;
      for (auto cand : instanceClass->find(memberClass->name())) {
        if (auto cls = symbol_cast<ClassSymbol>(cand);
            cls && cls->isComplete()) {
          alreadyComplete = true;
          break;
        }
      }
      if (alreadyComplete) continue;
    }
    instantiateDefinition(def, declAst, templateDecl);
  }

  for (auto member : pattern->members()) {
    for (auto function : views::each_function(member)) {
      instantiateFunctionDefinition(function);
    }
    if (auto field = symbol_cast<FieldSymbol>(member)) {
      if (!field->isStatic()) continue;
      auto def = field->definition();
      if (!def) continue;
      auto templateDecl = def->templateDeclaration();
      if (!templateDecl) continue;
      auto declAst = ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
      if (!declAst) continue;
      if (def->findSpecialization(templateArguments_)) continue;
      instantiateDefinition(def, declAst, templateDecl);
    }
  }

  if (instanceClass) {
    std::vector<FunctionSymbol*> patternCtors;
    for (auto ctor : pattern->declaredConstructors()) {
      if (ctor->canonical() == ctor && !ctor->isDefaulted())
        patternCtors.push_back(ctor);
    }
    std::vector<FunctionSymbol*> instanceCtors;
    for (auto ctor : instanceClass->declaredConstructors()) {
      if (ctor->canonical() == ctor && !ctor->isDefaulted())
        instanceCtors.push_back(ctor);
    }
    auto classTemplateDecl = pattern->templateDeclaration();
    for (std::size_t i = 0; classTemplateDecl && i < patternCtors.size() &&
                            i < instanceCtors.size();
         ++i) {
      auto ctor = patternCtors[i];
      if (ctor->templateDeclaration()) continue;
      auto def = symbol_cast<FunctionSymbol>(ctor->definition());
      if (!def || def == ctor) continue;
      auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
      if (!defAst) continue;

      auto instanceCtor = instanceCtors[i];
      const bool alreadyHandled = instanceCtor->hasPendingBody() ||
                                  (instanceCtor->declaration() &&
                                   instanceCtor->declaration()->functionBody);
      if (alreadyHandled) continue;

      auto ctorType = type_cast<FunctionType>(ctor->type());
      auto instanceCtorType = type_cast<FunctionType>(instanceCtor->type());
      if (!ctorType || !instanceCtorType ||
          ctorType->parameterTypes().size() !=
              instanceCtorType->parameterTypes().size()) {
        continue;
      }

      attachPendingBody(instanceCtor, def, defAst, classTemplateDecl->depth);
      unit_->addPendingBodyCompletion(instanceCtor);
    }
  }
}

void ASTRewriter::retryPendingMemberTemplateAttachment(FunctionSymbol* member) {
  if (!member || member->hasPendingBody() || member->declaration()) return;
  if (!member->templateDeclaration() || member->isFriend()) return;

  auto instanceClass =
      symbol_cast<ClassSymbol>(member->enclosingNonTemplateParametersScope());
  if (!instanceClass) return;

  auto pattern = instanceClass->primaryTemplateSymbol();
  if (!pattern) return;

  auto classTemplateDecl = pattern->templateDeclaration();
  if (!classTemplateDecl) return;

  FunctionSymbol* patternDef = nullptr;
  FunctionDefinitionAST* patternDefAst = nullptr;
  for (auto candidate : pattern->find(member->name())) {
    for (auto function : views::each_function(candidate)) {
      if (!function->templateDeclaration()) continue;
      if (function->isFriend()) continue;
      auto def = symbol_cast<FunctionSymbol>(function->definition());
      if (!def || def == function) continue;
      auto defAst = ast_cast<FunctionDefinitionAST>(def->declaration());
      if (!defAst) continue;
      if (!areFunctionTemplateHeadsEquivalentForRedeclaration(
              unit_, pattern, member->templateDeclaration(),
              function->templateDeclaration())) {
        continue;
      }
      patternDef = def;
      patternDefAst = defAst;
      break;
    }
    if (patternDef) break;
  }
  if (!patternDef) return;

  auto lexicalScope = patternDef->enclosingNonTemplateParametersScope();
  while (lexicalScope && lexicalScope->isClass()) {
    lexicalScope = lexicalScope->enclosingNonTemplateParametersScope();
  }

  auto classArgs = instanceClass->templateArguments();
  auto pending = std::make_unique<PendingBodyInstantiation>();
  pending->originalDefinition = patternDefAst;
  pending->templateArguments =
      std::vector<TemplateArgument>(classArgs.begin(), classArgs.end());
  pending->parentScope = lexicalScope;
  pending->depth = classTemplateDecl->depth;
  member->setPendingBody(std::move(pending));
  if (member->isDefinitionRequired()) {
    unit_->addPendingBodyCompletion(member);
  }
}

void ASTRewriter::completePendingMemberInstantiations(TranslationUnit* unit) {
  if (!unit || !unit->config().checkTypes) return;

  std::unordered_set<ClassSymbol*> processed;
  for (int round = 0; round < 32; ++round) {
    bool any = false;

    auto pendingBodies = unit->takePendingBodyCompletions();
    for (auto function : pendingBodies) {
      if (!function || !function->hasPendingBody()) continue;
      any = true;
      CapturingDiagnosticsClient capture;
      auto saved = unit->changeDiagnosticsClient(&capture);
      auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
      rewriter.completePendingBody(function);
      (void)unit->changeDiagnosticsClient(saved);
      unit->deferBodyDiagnostics(function, std::move(capture.diagnostics));
    }

    auto pending = unit->takePendingMemberInstantiations();
    for (auto instance : pending) {
      if (!instance) continue;
      if (!processed.insert(instance).second) continue;
      auto pattern = instance->primaryTemplateSymbol();
      if (!pattern) continue;
      auto args = instance->templateArguments();
      if (args.empty()) continue;
      any = true;
      auto rewriter =
          ASTRewriter{unit, unit->globalScope(),
                      std::vector<TemplateArgument>(args.begin(), args.end())};
      rewriter.instantiateOutOfClassMemberDefinitions(pattern);
    }
    if (!any) break;
  }
}
}  // namespace cxx
