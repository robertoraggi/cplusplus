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
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <ranges>
#include <unordered_set>

namespace cxx {
namespace {
class InstantiationDepthGuard {
 public:
  explicit InstantiationDepthGuard(TranslationUnit* unit)
      : unit_(unit), scope_(unit) {}

  [[nodiscard]] auto exceeded() const -> bool {
    return unit_->templateInstantiationDepth() >
           TranslationUnit::kMaxTemplateInstantiationDepth;
  }

 private:
  TranslationUnit* unit_;
  TranslationUnit::TemplateInstantiationScope scope_;
};

[[nodiscard]] auto outOfClassDefinitionOf(FunctionSymbol* member)
    -> FunctionSymbol* {
  auto definition = symbol_cast<FunctionSymbol>(member->definition());
  if (!definition || definition == member) return nullptr;
  if (!ast_cast<FunctionDefinitionAST>(definition->declaration()))
    return nullptr;
  return definition;
}

[[nodiscard]] auto isPrimaryTemplate(
    const std::vector<TemplateArgument>& templateArguments, int depth) -> bool {
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
      if (ty->depth != depth) return false;
      if (ty->index != expected) return false;
      if (!ty->isPack) return false;
      ++expected;
      continue;
    }

    auto symType = sym->type();
    if (!symType) return false;

    auto ty = getTypeParamInfo(symType);
    if (!ty) return false;
    if (ty->depth != depth) return false;
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
  if (symbol_cast<FunctionSymbol>(symbol))
    return "function template specialization";
  return "template class";
}

void noteInstantiationRequestedHere(
    TranslationUnit* unit, Symbol* primaryTemplate,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation instantiationLoc) {
  if (!instantiationLoc) return;
  auto client = unit->reportingDiagnosticsClient();
  if (!client) return;
  auto name =
      computeInstantiationClassName(unit, primaryTemplate, templateArguments);
  auto label = instantiationLabel(primaryTemplate);
  unit->report(
      client, instantiationLoc, Severity::Note,
      std::format("in instantiation of {} '{}' requested here", label, name));
}

[[nodiscard]] auto hasPlaceholderReturnType(FunctionSymbol* function) -> bool {
  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return false;
  return isPlaceholderType(functionType->returnType());
}

[[nodiscard]] auto definitionNeededAtPointOfReference(TranslationUnit* unit,
                                                      FunctionSymbol* function)
    -> bool {
  if (unit->isInstantiatingTemplate()) return true;
  if (function->isConstexpr()) return true;
  return hasPlaceholderReturnType(function);
}

void recordFunctionInstantiationRequest(Symbol* primary, Symbol* specialization,
                                        SourceLocation instantiationLoc) {
  if (!instantiationLoc) return;
  if (!symbol_cast<FunctionSymbol>(specialization)) return;
  auto spec = specialization_entry_of(primary, specialization);
  if (!spec) return;
  spec->pendingInstantiationLoc = instantiationLoc;
}

[[nodiscard]] auto templateTemplateArgument(const TemplateArgument& argument)
    -> Symbol* {
  Symbol* symbol = nullptr;
  if (auto name = std::get_if<Symbol*>(&argument)) {
    symbol = template_name_symbol(*name);
  } else if (auto classType =
                 type_cast<ClassType>(template_argument_type(argument))) {
    symbol = template_name_symbol(classType->symbol());
  }
  if (!template_declaration_of(symbol)) return nullptr;
  return symbol;
}

[[nodiscard]] auto instantiateBuiltinMakeIntegerSeq(
    TranslationUnit* unit,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation instantiationLoc, bool sfinaeContext, bool argsComplete,
    bool declarationOnly) -> Symbol* {
  if (templateArguments.size() != 3) return nullptr;

  auto seqClass = templateTemplateArgument(templateArguments[0]);
  if (!seqClass) return nullptr;

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
        ar, literal, /*literalOperatorCall=*/nullptr, ValueCategory::kPrValue,
        elementType);
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

[[nodiscard]] auto typeTemplateArgumentList(TranslationUnit* unit,
                                            std::span<const Type* const> types)
    -> List<TemplateArgumentAST*>* {
  auto arena = unit->arena();
  List<TemplateArgumentAST*>* arguments = nullptr;
  auto it = &arguments;
  for (auto type : types) {
    auto typeId = TypeIdAST::create(arena);
    typeId->type = type;
    auto argument = TypeTemplateArgumentAST::create(arena, typeId);
    *it = make_list_node(arena, static_cast<TemplateArgumentAST*>(argument));
    it = &(*it)->next;
  }
  return arguments;
}

class BuiltinCommonType {
 public:
  BuiltinCommonType(TranslationUnit* unit, Symbol* commonTypeTemplate,
                    SourceLocation instantiationLoc)
      : unit_(unit),
        traits_(unit),
        commonTypeTemplate_(commonTypeTemplate),
        instantiationLoc_(instantiationLoc) {}

  [[nodiscard]] auto operator()(std::span<const Type* const> types)
      -> const Type* {
    if (types.empty()) return nullptr;
    if (types.size() == 1) return commonTypeOf({types[0], types[0]});
    if (types.size() == 2) return commonTypeOfPair(types[0], types[1]);
    auto first = commonTypeOf({types[0], types[1]});
    if (!first) return nullptr;
    std::vector<const Type*> remaining{first};
    remaining.insert(remaining.end(), types.begin() + 2, types.end());
    return commonTypeOf(remaining);
  }

 private:
  [[nodiscard]] auto commonTypeOfPair(const Type* first, const Type* second)
      -> const Type* {
    if (!isDecayed(first) || !isDecayed(second))
      return commonTypeOf({traits_.decay(first), traits_.decay(second)});
    if (auto type = traits_.conditional_operator_type(first, second))
      return traits_.decay(type);
    auto type = traits_.conditional_operator_type(
        traits_.add_const_ref(first), traits_.add_const_ref(second));
    if (!type) return nullptr;
    return traits_.decay(type);
  }

  [[nodiscard]] auto isDecayed(const Type* type) const -> bool {
    return traits_.is_same(type, traits_.decay(type));
  }

  [[nodiscard]] auto commonTypeOf(const std::vector<const Type*>& types)
      -> const Type* {
    const auto sfinaeContext = true;
    const auto argsComplete = true;
    const auto declarationOnly = false;
    SilentDiagnosticsScope diagnostics{unit_};
    auto instance = ASTRewriter::instantiate(
        unit_, typeTemplateArgumentList(unit_, types), commonTypeTemplate_,
        instantiationLoc_, sfinaeContext, argsComplete, declarationOnly);
    diagnostics.finish();
    if (diagnostics.hadError() || !instance) return nullptr;
    return instance->type();
  }

  TranslationUnit* unit_;
  TypeTraits traits_;
  Symbol* commonTypeTemplate_;
  SourceLocation instantiationLoc_;
};

[[nodiscard]] auto instantiateBuiltinCommonType(
    TranslationUnit* unit,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation instantiationLoc, bool sfinaeContext, bool argsComplete,
    bool declarationOnly) -> Symbol* {
  if (templateArguments.size() < 3) return nullptr;

  auto commonTypeTemplate = templateTemplateArgument(templateArguments[0]);
  if (!commonTypeTemplate) return nullptr;

  auto identityTemplate = templateTemplateArgument(templateArguments[1]);
  if (!identityTemplate) return nullptr;

  auto emptyType =
      type_cast<ClassType>(template_argument_type(templateArguments[2]));
  if (!emptyType) return nullptr;

  std::vector<const Type*> types;
  for (const auto& argument : std::span{templateArguments}.subspan(3)) {
    auto type = template_argument_type(argument);
    if (!type) return nullptr;
    types.push_back(type);
  }

  auto commonType =
      BuiltinCommonType{unit, commonTypeTemplate, instantiationLoc}(types);
  if (!commonType) return emptyType->symbol();

  const Type* identityArguments[] = {commonType};
  return ASTRewriter::instantiate(
      unit, typeTemplateArgumentList(unit, identityArguments), identityTemplate,
      instantiationLoc, sfinaeContext, argsComplete, declarationOnly);
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

struct ASTRewriter::Instantiate {
  ASTRewriter& rewriter;
  ScopeSymbol* parentScope = nullptr;
  bool declarationOnly = false;

  void attachDeferredBody(FunctionDefinitionAST* instance,
                          FunctionDefinitionAST* pattern) {
    if (instance->functionBody) return;
    auto fn = symbol_cast<FunctionSymbol>(instance->symbol);
    if (!fn || fn->hasPendingBody()) return;
    fn->setPendingBody(
        rewriter.pendingInstantiationOf(pattern, nullptr, parentScope));
  }

  [[nodiscard]] auto patternClassSpecifier(ClassSymbol* symbol)
      -> ClassSpecifierAST* {
    auto pattern = symbol_cast<ClassSymbol>(symbol->instantiationPattern());
    if (!pattern || pattern == symbol) return nullptr;

    auto definition = pattern->resolvedDefinition();

    return ast_cast<ClassSpecifierAST>(definition->declaration());
  }

  auto operator()(ClassSymbol* symbol) -> Symbol* {
    auto classSpecifier = ast_cast<ClassSpecifierAST>(symbol->declaration());
    if (!classSpecifier) classSpecifier = patternClassSpecifier(symbol);
    if (!classSpecifier) return nullptr;

    auto instance =
        ast_cast<ClassSpecifierAST>(rewriter.specifier(classSpecifier));
    if (!instance) return nullptr;

    return instance->symbol;
  }

  auto operator()(VariableSymbol* symbol) -> Symbol* {
    auto definition = symbol->resolvedDefinition();
    auto templateDecl = definition->templateDeclaration();
    if (!templateDecl) return nullptr;

    rewriter.binder_.setInstantiatingSymbol(definition);

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

    return instance->symbol;
  }

  auto operator()(FunctionSymbol* symbol) -> Symbol* {
    completePendingExceptionSpecification(rewriter.unit_, symbol);
    attachPatternDefinition(rewriter.unit_, symbol);

    auto deferredPattern =
        symbol->hasPendingBody()
            ? ast_cast<FunctionDefinitionAST>(symbol->pendingBody()->pattern)
            : nullptr;
    if (symbol->hasPendingBody() &&
        (!declarationOnly || !symbol->declaration())) {
      rewriter.completePendingBody(symbol);
    }

    auto functionDef = symbol->declaration();
    auto definingSymbol = symbol;

    if (!functionDef) {
      for (auto declaration : symbol->declarations()) {
        if (auto def = declaration->declaration()) {
          functionDef = def;
          definingSymbol = declaration;
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
      rewriter.addSymbolRemap(symbol, instance->symbol);
      if (declarationOnly) {
        attachDeferredBody(instance,
                           deferredPattern ? deferredPattern : functionDef);
        rewriter.completePendingExceptionSpecifiers(0);
      }
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

    if (declarationOnly) rewriter.completePendingExceptionSpecifiers(0);
    return instance->initDeclaratorList->value->symbol;
  }

  auto operator()(Symbol*) -> Symbol* { return nullptr; }
};

auto ASTRewriter::enclosingNonClassScope(Symbol* symbol) -> ScopeSymbol* {
  auto scope = symbol->parent();
  while (scope && scope->isClass()) scope = scope->parent();
  return scope;
}

auto ASTRewriter::pendingBodyOf(FunctionDefinitionAST* definition,
                                std::vector<TemplateArgument> arguments,
                                ScopeSymbol* parentScope, int depth)
    -> std::unique_ptr<PendingInstantiation> {
  return std::make_unique<PendingInstantiation>(
      PendingInstantiation{.pattern = definition,
                           .templateArguments = std::move(arguments),
                           .parentScope = parentScope,
                           .depth = depth});
}

auto ASTRewriter::pendingInstantiationOf(AST* pattern, AST* instance,
                                         ScopeSymbol* parentScope) const
    -> std::unique_ptr<PendingInstantiation> {
  return std::make_unique<PendingInstantiation>(
      PendingInstantiation{.pattern = pattern,
                           .instance = instance,
                           .templateArguments = templateArguments(),
                           .parentScope = parentScope,
                           .depth = depth_});
}

auto ASTRewriter::paste(TranslationUnit* unit, ScopeSymbol* scope,
                        StatementAST* ast) -> StatementAST* {
  TranslationUnit::TemplatedContextScope templatedContext{
      unit,
      isEnclosedInDependentTemplate(unit, scope,
                                    /*stopAtConcreteSpecialization=*/true)};
  auto rewriter = ASTRewriter{unit, scope, {}};
  rewriter.pastingCheckedBody_ = true;
  auto result = rewriter.statement(ast);
  return result;
}

auto ASTRewriter::substituteDefaultTypeId(
    TranslationUnit* unit, TypeIdAST* typeId,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> TypeIdAST* {
  if (!typeId) return nullptr;
  SilentDiagnosticsScope silent{unit};
  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;
  auto substituted = rewriter.typeId(typeId);
  if (silent.hadError()) return nullptr;
  return substituted;
}

auto ASTRewriter::substituteDefaultExpression(
    TranslationUnit* unit, ExpressionAST* expression,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> ExpressionAST* {
  if (!expression) return nullptr;
  SilentDiagnosticsScope silent{unit};
  auto rewriter = ASTRewriter{unit, scope,
                              std::vector<TemplateArgument>(templateArguments)};
  rewriter.depth_ = depth;
  auto substituted = rewriter.expression(expression);
  if (silent.hadError()) return nullptr;
  return substituted;
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
  if (auto spec = specialization_entry_of(primaryTemplate, instantiated)) {
    if (!spec->instantiationErrors.empty()) {
      for (const auto& error : spec->instantiationErrors) {
        switch (error.severity) {
          case Severity::Warning:
            unit->warning(error.location, error.message);
            break;
          case Severity::Note:
          case Severity::Message:
            unit->note(error.location, error.message);
            break;
          case Severity::Error:
          case Severity::Fatal:
            unit->error(error.location, error.message);
            break;
        }
      }
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

auto ASTRewriter::instantiateOverloadCandidate(
    TranslationUnit* unit, List<TemplateArgumentAST*>* deducedArguments,
    FunctionSymbol* function, SourceLocation instantiationLoc,
    bool argsComplete, std::vector<Diagnostic>* substitutionFailure)
    -> FunctionSymbol* {
  return symbol_cast<FunctionSymbol>(
      instantiate(unit, deducedArguments, function, instantiationLoc,
                  /*sfinaeContext=*/true, argsComplete,
                  /*declarationOnly=*/true,
                  /*retainEnclosingTemplateLevels=*/false,
                  /*isOverloadCandidate=*/true, substitutionFailure));
}

void ASTRewriter::instantiateSelectedSpecializationDefinition(
    TranslationUnit* unit, FunctionSymbol* selected,
    List<TemplateArgumentAST*>* deducedArguments) {
  if (!selected || !deducedArguments) return;
  if (!selected->isSpecialization()) return;
  if (!unit->requiresDefinitions()) return;

  auto primary = selected->primaryTemplateSymbol();
  if (!primary) return;

  auto instantiationLoc = selected->location();
  if (auto spec = specialization_entry_of(primary, selected);
      spec && spec->pendingInstantiationLoc) {
    instantiationLoc = spec->pendingInstantiationLoc;
  }

  const auto declarationOnly =
      !definitionNeededAtPointOfReference(unit, selected);

  auto instantiated = instantiate(
      unit, deducedArguments, primary, instantiationLoc,
      /*sfinaeContext=*/false, /*argsComplete=*/true, declarationOnly,
      /*retainEnclosingTemplateLevels=*/false,
      /*isOverloadCandidate=*/false,
      /*substitutionFailure=*/nullptr);

  if (!declarationOnly) return;

  if (auto function = symbol_cast<FunctionSymbol>(instantiated))
    requireFunctionDefinition(unit, function);
}

auto ASTRewriter::instantiate(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    Symbol* symbol, SourceLocation instantiationLoc, bool sfinaeContext,
    bool argsComplete, bool declarationOnly, bool retainEnclosingTemplateLevels,
    bool isOverloadCandidate, std::vector<Diagnostic>* substitutionFailure)
    -> Symbol* {
  if (!symbol) return nullptr;

  if (!unit->config().checkTypes) return nullptr;

  if (auto trace = unit->timeTrace())
    trace->count(TimeTrace::kInstantiationRequests);
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
      unit->report(client, instantiationLoc, Severity::Fatal,
                   std::move(message));
    } else {
      unit->report(unit->diagnosticsClient(), instantiationLoc, Severity::Fatal,
                   std::move(message));
    }

    return nullptr;
  }

  const auto activeClientIsSfinae =
      unit->diagnosticsClient() && unit->diagnosticsClient()->isSfinae();

  if (!sfinaeContext && activeClientIsSfinae) {
    sfinaeContext = true;
  }

  const auto functionBodyIsOdrUsed =
      !symbol_cast<FunctionSymbol>(symbol) || unit->requiresDefinitions();

  if (!functionBodyIsOdrUsed) declarationOnly = true;

  auto templateDecl = template_declaration_of(symbol);
  if (!templateDecl) return nullptr;

  auto declaration = template_declaration_ast(symbol);
  if (!declaration) return nullptr;

  const bool ownsSfinaeClient =
      sfinaeContext && (isOverloadCandidate || !activeClientIsSfinae);

  std::optional<SilentDiagnosticsScope> sfinaeScope;
  if (ownsSfinaeClient) sfinaeScope.emplace(unit);

  struct SubstitutionFailureExporter {
    std::optional<SilentDiagnosticsScope>& scope;
    std::vector<Diagnostic>* out;

    ~SubstitutionFailureExporter() {
      if (!out || !scope.has_value() || !scope->hadError()) return;
      const auto& diagnostics = scope->diagnostics();
      out->assign(diagnostics.begin(), diagnostics.end());
    }
  } substitutionFailureExporter{sfinaeScope, substitutionFailure};

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList,
                                  argsComplete);

  if (!subst) {
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
    return result;
  }

  if (symbol_cast<FunctionSymbol>(symbol) &&
      static_cast<int>(templateArguments.size()) <
          templateParameterCount(templateDecl)) {
    return symbol;
  }

  if (isPrimaryTemplate(templateArguments, templateDecl->depth)) {
    return symbol;
  }

  auto cached = retainEnclosingTemplateLevels
                    ? nullptr
                    : find_specialization(unit, symbol, templateArguments);

  if (cached) {
    if (declarationOnly)
      recordFunctionInstantiationRequest(symbol, cached, instantiationLoc);
    auto cachedClass = symbol_cast<ClassSymbol>(cached);
    if (!cachedClass) {
      if (!declarationOnly) {
        if (auto cachedFn = symbol_cast<FunctionSymbol>(cached);
            cachedFn && cachedFn->hasPendingBody()) {
          auto bodyErrors = ASTRewriter::completePendingBodyFor(
              unit, cachedFn, /*captureBodyErrors=*/true);
          if (!bodyErrors.empty()) {
            if (auto spec = specialization_entry_of(symbol, cached)) {
              spec->instantiationErrors = instantiationErrorsOf(bodyErrors);
            }
          }
        }
      }
      if (!sfinaeContext)
        reportPendingInstantiationErrors(unit, symbol, cached,
                                         instantiationLoc);
      return cached;
    }
    if (cachedClass->declaration()) {
      if (!sfinaeContext)
        reportPendingInstantiationErrors(unit, symbol, cached,
                                         instantiationLoc);
      return cached;
    }
  }

  if (!checkAssociatedConstraints(unit, symbol, templateArguments,
                                  templateDecl->depth)) {
    return nullptr;
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, classSymbol, templateArguments);
    if (partial.handled()) {
      return partial.symbol;
    }
  }

  if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
    auto partial =
        tryPartialSpecialization(unit, variableSymbol, templateArguments);
    if (partial.handled()) {
      return partial.symbol;
    }
  }

  TimeTrace::Scope instantiationTrace{unit->timeTrace(), "Instantiate", symbol};
  if (auto trace = unit->timeTrace()) trace->count(TimeTrace::kInstantiations);
  auto parentScope = symbol->parent();
  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;
  rewriter.inheritEnclosingTemplateArguments(parentScope);
  rewriter.writtenTemplateArgumentList_ = templateArgumentList;
  rewriter.setRetainsEnclosingTemplateLevels(retainEnclosingTemplateLevels);
  rewriter.binder().setInstantiatingSymbol(symbol);
  rewriter.binder().setInstantiationLoc(instantiationLoc);
  if (declarationOnly) rewriter.setRestrictedToDeclarations(true);

  auto inheritTemplateAccess = [&](Symbol* result) {
    if (!result || result == symbol) return;
    result->setAccessSpecifier(symbol->accessSpecifier());
  };

  auto registerFunctionSpecialization = [&](Symbol* result) {
    if (!result || result == symbol) return;
    auto fnTemplate = symbol_cast<FunctionSymbol>(symbol);
    if (!fnTemplate) return;
    auto instance = symbol_cast<FunctionSymbol>(result);
    if (!instance || instance->isSpecialization()) return;

    if (fnTemplate->isFriend()) instance->setFriend(true);

    if (fnTemplate->findSpecialization(unit, templateArguments)) return;
    fnTemplate->addSpecialization(unit, templateArguments, instance);
  };

  if (sfinaeContext) {
    auto instance =
        visit(Instantiate{rewriter, parentScope, declarationOnly}, symbol);
    if (sfinaeScope) {
      sfinaeScope->finish();
      if (sfinaeScope->hadError()) return nullptr;
    }
    if (rewriter.substitutionFailed()) return nullptr;

    inheritTemplateAccess(instance);
    registerFunctionSpecialization(instance);
    if (declarationOnly)
      recordFunctionInstantiationRequest(symbol, instance, instantiationLoc);

    auto bodyErrors = rewriter.takeBodyErrors();
    if (!bodyErrors.empty()) {
      if (reportOutsideImmediateContext(unit, bodyErrors)) {
        noteInstantiationRequestedHere(unit, symbol, templateArguments,
                                       instantiationLoc);
      } else if (auto spec = specialization_entry_of(symbol, instance)) {
        spec->instantiationErrors = instantiationErrorsOf(bodyErrors);
      }
    }

    return instance;
  }

  CapturingDiagnosticsScope capturing{unit};
  capturing.forwardToPreviousClient();

  auto instantiatedSymbol =
      visit(Instantiate{rewriter, parentScope, declarationOnly}, symbol);

  capturing.finish();

  inheritTemplateAccess(instantiatedSymbol);
  registerFunctionSpecialization(instantiatedSymbol);
  if (declarationOnly)
    recordFunctionInstantiationRequest(symbol, instantiatedSymbol,
                                       instantiationLoc);

  auto bodyErrors = rewriter.takeBodyErrors();

  auto instantiationErrors = capturing.takeDiagnostics();
  instantiationErrors.insert(instantiationErrors.end(),
                             std::make_move_iterator(bodyErrors.begin()),
                             std::make_move_iterator(bodyErrors.end()));

  if (!instantiationErrors.empty()) {
    if (auto spec = specialization_entry_of(symbol, instantiatedSymbol)) {
      spec->instantiationErrors = instantiationErrorsOf(instantiationErrors);
    }
    noteInstantiationRequestedHere(unit, symbol, templateArguments,
                                   instantiationLoc);
  }

  return instantiatedSymbol;
}

void ASTRewriter::markExplicitInstantiationDeclared(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    Symbol* symbol) {
  if (!symbol) return;
  if (!unit->config().checkTypes) return;

  auto templateDecl = template_declaration_of(symbol);
  if (!templateDecl) return;

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList,
                                  /*argsComplete=*/true);
  if (!subst) return;

  auto templateArguments = std::move(*subst).templateArguments();

  if (isPrimaryTemplate(templateArguments, templateDecl->depth)) return;

  add_extern_instantiation_declaration(symbol, std::move(templateArguments));
}

auto ASTRewriter::ensureCompleteClass(TranslationUnit* unit,
                                      ClassSymbol* classSymbol) -> bool {
  if (!classSymbol) return false;
  if (classSymbol->resolvedDefinition()->isComplete()) return true;
  if (auto memberPattern =
          symbol_cast<ClassSymbol>(classSymbol->instantiationPattern());
      memberPattern && !classSymbol->isSpecialization()) {
    auto pattern = memberPattern->resolvedDefinition();
    auto declaration = ast_cast<ClassSpecifierAST>(pattern->declaration());
    if (!declaration) return false;
    InstantiationDepthGuard depthGuard{unit};
    if (depthGuard.exceeded()) return false;
    ASTRewriter rewriter{unit, classSymbol->parent(),
                         classSymbol->instantiationSubstitutionArguments()};
    rewriter.depth_ = classSymbol->instantiationSubstitutionDepth();
    rewriter.inheritEnclosingTemplateArguments(classSymbol->parent());
    rewriter.classInstanceToComplete_ = classSymbol;
    rewriter.binder_.setInstantiatingSymbol(pattern);
    for (auto oldScope = pattern->parent(), newScope = classSymbol->parent();
         oldScope && newScope && oldScope->isClass() && newScope->isClass();
         oldScope = oldScope->parent(), newScope = newScope->parent()) {
      rewriter.remapScopeMembers(oldScope, newScope);
    }
    rewriter.addSymbolRemap(pattern, classSymbol);
    rewriter.specifier(declaration);
    rewriter.instantiateOutOfClassMemberDefinitions(pattern, classSymbol);
    return classSymbol->isComplete();
  }
  if (!classSymbol->isSpecialization()) return false;

  auto primaryTemplate = classSymbol->primaryTemplateSymbol();
  if (!primaryTemplate) return false;

  auto templateDefinition = primaryTemplate->resolvedDefinition();
  const bool hasPrimaryDefinition =
      ast_cast<ClassSpecifierAST>(templateDefinition->declaration()) != nullptr;
  auto instantiationTemplate = primaryTemplate;
  if (hasPrimaryDefinition) instantiationTemplate = templateDefinition;

  auto spec = primaryTemplate->specializationEntry(classSymbol);
  if (!spec || !spec->isPendingInstantiation) return false;

  auto pendingArgList = spec->pendingArgumentList;
  auto pendingLoc = spec->pendingInstantiationLoc;
  primaryTemplate->setPendingInstantiation(classSymbol, nullptr, pendingLoc,
                                           false);

  auto result = instantiate(unit, pendingArgList, instantiationTemplate,
                            pendingLoc, false);

  if (!result && !hasPrimaryDefinition) {
    primaryTemplate->setPendingInstantiation(classSymbol, pendingArgList,
                                             pendingLoc, true);
  }

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

void ASTRewriter::requireVirtualMemberDefinitions(TranslationUnit* unit,
                                                  ClassSymbol* classSymbol) {
  if (!unit || !classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  for (auto member : classSymbol->members() | views::virtual_functions) {
    if (member->isPure()) continue;
    requireFunctionDefinition(unit, member);
  }
}

void ASTRewriter::requireVTableForKeyFunction(TranslationUnit* unit,
                                              FunctionSymbol* function) {
  if (!unit || !function || !function->isVirtual()) return;
  auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  auto vtableLayout = classSymbol->vtableLayout();
  if (!vtableLayout || !vtableLayout->keyFunction) return;
  if (vtableLayout->keyFunction->canonical() != function->canonical()) return;
  requireVirtualMemberDefinitions(unit, classSymbol);
}

struct ASTRewriter::OutOfClassMemberDefinitions {
  ASTRewriter& rewrite;
  ClassSymbol* pattern;
  ClassSymbol* instanceClass;
  TemplateDeclarationAST* classTemplateDecl;

  void instantiate() {
    attachMemberDefinitions();

    for (auto memberClass : pattern->members() | views::classes)
      instantiateMemberClassDefinition(memberClass);

    for (auto member : pattern->members()) {
      if (auto field = symbol_cast<FieldSymbol>(member))
        instantiateStaticDataMemberDefinition(field);
      if (auto variable = symbol_cast<VariableSymbol>(member))
        instantiateMemberVariableTemplateDefinition(variable);
    }
  }

 private:
  void instantiateDefinition(Symbol* instantiating, Symbol* definition,
                             DeclarationAST* declaration, int depth) {
    auto rewriter =
        ASTRewriter{rewrite.unit_, enclosingNonClassScope(definition),
                    std::vector<TemplateArgument>(rewrite.templateArguments_)};
    rewriter.depth_ = depth;
    rewriter.binder_.setInstantiatingSymbol(instantiating);
    rewriter.remapScopeMembers(pattern, instanceClass);
    (void)rewriter.declaration(declaration);
  }

  void attachMemberDefinitions() {
    for (auto member : instanceClass->members()) {
      for (auto function : views::declared_functions(member))
        attachDefinitions(function);
    }
    for (auto constructor : instanceClass->declaredConstructors())
      attachDefinitions(constructor);
  }

  void attachDefinitions(FunctionSymbol* function) {
    attachPatternDefinition(rewrite.unit_, function);
    for (const auto& entry : function->specializations()) {
      auto specialization = symbol_cast<FunctionSymbol>(entry.symbol);
      if (!specialization || !specialization->isDefinitionRequired()) continue;
      attachPatternDefinition(rewrite.unit_, specialization);
    }
  }

  [[nodiscard]] auto hasCompleteInstance(ClassSymbol* memberClass) const
      -> bool {
    return std::ranges::any_of(
        instanceClass->find(memberClass->name()) | views::classes,
        &ClassSymbol::isComplete);
  }

  void instantiateMemberClassDefinition(ClassSymbol* memberClass) {
    if (hasCompleteInstance(memberClass)) return;

    if (auto memberTemplateDecl =
            outOfClassMemberClassTemplateDefinition(memberClass)) {
      instantiateDefinition(memberClass, memberClass, memberTemplateDecl,
                            classTemplateDecl->depth);
      return;
    }

    auto definition = symbol_cast<ClassSymbol>(memberClass->definition());
    if (!definition || definition == memberClass) return;
    auto templateDecl = definition->templateDeclaration();
    if (!templateDecl) return;
    auto declaration =
        ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
    if (!declaration) return;
    instantiateDefinition(definition, definition, declaration,
                          templateDecl->depth);
  }

  [[nodiscard]] auto outOfClassMemberClassTemplateDefinition(
      ClassSymbol* memberClass) const -> TemplateDeclarationAST* {
    if (!classTemplateDecl) return nullptr;
    if (!memberClass->templateParameters()) return nullptr;
    auto classSpecifier =
        ast_cast<ClassSpecifierAST>(memberClass->declaration());
    if (!classSpecifier) return nullptr;
    if (!classSpecifier->nestedNameSpecifier) return nullptr;
    return memberClass->templateDeclaration();
  }

  void instantiateStaticDataMemberDefinition(FieldSymbol* field) {
    if (!field->isStatic()) return;
    auto instanceField = symbol_cast<FieldSymbol>(rewrite.remapSymbol(field));
    if (!instanceField || instanceField == field) return;
    if (!instanceField->isDefinitionRequired()) return;
    if (instanceField->definition()) return;
    auto definition = field->definition();
    if (!definition) return;
    auto templateDecl = definition->templateDeclaration();
    if (!templateDecl) return;
    auto declaration =
        ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
    if (!declaration) return;
    if (definition->findSpecialization(rewrite.unit_,
                                       rewrite.templateArguments_))
      return;
    instantiateDefinition(definition, definition, declaration,
                          templateDecl->depth);
  }

  [[nodiscard]] auto instanceIsDefined(VariableSymbol* member) const -> bool {
    auto instance = symbol_cast<VariableSymbol>(rewrite.remapSymbol(member));
    if (!instance || instance == member) return false;
    return instance->canonical()->definition() != nullptr;
  }

  void instantiateMemberVariableTemplateDefinition(VariableSymbol* member) {
    if (!classTemplateDecl) return;
    if (!member->isTemplatePattern()) return;
    auto definition = member->definition();
    if (!definition || definition == member) return;
    if (instanceIsDefined(member)) return;
    auto templateDecl = definition->templateDeclaration();
    if (!templateDecl) return;
    instantiateDefinition(pattern, definition, templateDecl,
                          classTemplateDecl->depth);
  }
};

auto ASTRewriter::instantiationTarget(ClassSymbol* pattern,
                                      ClassSymbol* instanceClass)
    -> ClassSymbol* {
  if (!instanceClass) return nullptr;
  if (instanceClass->templateParameters()) return nullptr;

  auto instantiatedFrom = [pattern](ClassSymbol* candidate) {
    auto instantiationPattern = candidate->instantiationPattern();
    if (!instantiationPattern) return true;
    return instantiationPattern->canonical() == pattern->canonical();
  };

  if (!instantiatedFrom(instanceClass)) return nullptr;

  auto definition = symbol_cast<ClassSymbol>(instanceClass->definition());
  if (!definition || definition == instanceClass) return instanceClass;
  if (!instantiatedFrom(definition)) return nullptr;
  return definition;
}

void ASTRewriter::instantiateOutOfClassMemberDefinitions(
    ClassSymbol* pattern, ClassSymbol* instanceClass) {
  if (!pattern) return;
  requireVirtualMemberDefinitions(unit_, instanceClass);
  if (templateArguments_.empty()) return;

  instanceClass = instantiationTarget(pattern, instanceClass);
  if (!instanceClass) return;

  unit_->addPendingMemberInstantiation(instanceClass);
  remapScopeMembers(pattern, instanceClass);

  OutOfClassMemberDefinitions{*this, pattern, instanceClass,
                              pattern->templateDeclaration()}
      .instantiate();
}

auto ASTRewriter::pendingBodyFromPattern(FunctionSymbol* function)
    -> std::unique_ptr<PendingInstantiation> {
  if (function->isFriend()) return nullptr;
  if (auto primary = function->primaryTemplateSymbol())
    return pendingSpecializationBody(function, primary);
  return pendingMemberBody(function);
}

auto ASTRewriter::pendingMemberBody(FunctionSymbol* member)
    -> std::unique_ptr<PendingInstantiation> {
  auto instanceClass = symbol_cast<ClassSymbol>(member->parent());
  if (!instanceClass || instanceClass->templateParameters()) return nullptr;

  auto pattern = symbol_cast<FunctionSymbol>(member->instantiationPattern());
  if (!pattern || pattern->isFriend()) return nullptr;

  auto definition = outOfClassDefinitionOf(pattern);
  if (!definition) return nullptr;

  auto arguments = instanceClass->instantiationSubstitutionArguments();
  return pendingBodyOf(
      ast_cast<FunctionDefinitionAST>(definition->declaration()),
      {arguments.begin(), arguments.end()}, enclosingNonClassScope(definition),
      instanceClass->instantiationSubstitutionDepth());
}

auto ASTRewriter::pendingSpecializationBody(FunctionSymbol* specialization,
                                            FunctionSymbol* primary)
    -> std::unique_ptr<PendingInstantiation> {
  auto entry = specialization_entry_of(primary, specialization);
  if (!entry) return nullptr;

  if (auto definition = outOfClassDefinitionOf(primary)) {
    auto templateDeclaration = definition->templateDeclaration();
    if (!templateDeclaration) return nullptr;
    return pendingBodyOf(
        ast_cast<FunctionDefinitionAST>(definition->declaration()),
        entry->arguments, enclosingNonClassScope(definition),
        templateDeclaration->depth);
  }

  auto memberTemplatePattern =
      symbol_cast<FunctionSymbol>(primary->instantiationPattern());
  if (!memberTemplatePattern) return nullptr;

  auto definition = outOfClassDefinitionOf(memberTemplatePattern);
  if (!definition) return nullptr;

  auto memberTemplateDeclaration = primary->templateDeclaration();
  if (!memberTemplateDeclaration) return nullptr;

  return pendingBodyOf(
      ast_cast<FunctionDefinitionAST>(definition->declaration()),
      entry->arguments, primary->parent(), memberTemplateDeclaration->depth);
}

void ASTRewriter::attachPatternDefinition(TranslationUnit* unit,
                                          FunctionSymbol* function) {
  if (!function || function->hasPendingBody()) return;
  if (function->declaration() || function->isDefined()) return;

  auto pendingBody = pendingBodyFromPattern(function);
  if (!pendingBody) return;

  function->setPendingBody(std::move(pendingBody));
  unit->addPendingBodyCompletion(function);
}

void ASTRewriter::notePendingBodyInstantiation(TranslationUnit* unit,
                                               FunctionSymbol* function) {
  auto client = unit->reportingDiagnosticsClient();
  if (!client) return;
  std::vector<FunctionSymbol*> visited;
  while (function && !std::ranges::contains(visited, function)) {
    visited.push_back(function);
    auto [location, caller] = unit->pendingBodyCompletionRequest(function);
    if (auto primary = function->primaryTemplateSymbol()) {
      if (auto spec = specialization_entry_of(primary, function))
        noteInstantiationRequestedHere(unit, primary, spec->arguments,
                                       spec->pendingInstantiationLoc);
    } else {
      auto classSymbol = symbol_cast<ClassSymbol>(function->parent());
      if (!location || !classSymbol) return;
      unit->report(
          client, location, Severity::Note,
          std::format(
              "in instantiation of member function '{}::{}' requested here",
              to_string(classSymbol->type()), to_string(function->name())));
    }
    function = caller;
  }
}

void ASTRewriter::completePendingMemberInstantiations(TranslationUnit* unit) {
  if (!unit || !unit->config().checkTypes) return;

  auto stopRequested = [unit] {
    const auto& stopParsing = unit->config().stopParsingPredicate;
    return stopParsing && stopParsing();
  };

  auto completeBodies = [&] {
    if (stopRequested()) return false;
    bool progressed = false;
    for (auto function : unit->takePendingBodyCompletions()) {
      attachPatternDefinition(unit, function);
      if (!function->hasPendingBody()) continue;
      auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
      progressed = true;
      auto bodyErrors =
          rewriter.completePendingBody(function, /*captureBodyErrors=*/true);
      if (reportOutsideImmediateContext(unit, bodyErrors))
        notePendingBodyInstantiation(unit, function);
    }
    return progressed;
  };

  auto instantiateMembers = [&] {
    if (stopRequested()) return false;
    bool progressed = false;
    for (auto instance : unit->takePendingMemberInstantiations()) {
      if (!instance->isComplete()) continue;
      if (!unit->beginMemberInstantiation(instance)) continue;
      auto pattern = instance->instantiationTemplate();
      if (!pattern) continue;
      auto args = instance->instantiationSubstitutionArguments();
      if (args.empty()) continue;
      progressed = true;
      auto rewriter =
          ASTRewriter{unit, unit->globalScope(),
                      std::vector<TemplateArgument>(args.begin(), args.end())};
      rewriter.instantiateOutOfClassMemberDefinitions(pattern, instance);
    }
    return progressed;
  };

  for (bool progressed = true; progressed;) {
    progressed = completeBodies();
    progressed = instantiateMembers() || progressed;
  }
}
}  // namespace cxx
