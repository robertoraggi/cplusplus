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
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <format>

namespace cxx {
namespace {
struct IsPackParameter {
  auto operator()(TypenameTypeParameterAST* parameter) -> bool {
    return parameter->isPack;
  }

  auto operator()(NonTypeTemplateParameterAST* parameter) -> bool {
    return parameter->declaration && parameter->declaration->isPack;
  }

  auto operator()(TemplateTypeParameterAST* parameter) -> bool {
    return parameter->isPack;
  }

  auto operator()(ConstraintTypeParameterAST* parameter) -> bool {
    return static_cast<bool>(parameter->ellipsisLoc);
  }
};

struct HasDefaultTemplateArgument {
  auto operator()(TypenameTypeParameterAST* parameter) -> bool {
    return parameter->typeId && parameter->typeId->type;
  }

  auto operator()(NonTypeTemplateParameterAST* parameter) -> bool {
    return parameter->declaration && parameter->declaration->equalLoc &&
           parameter->declaration->expression;
  }

  auto operator()(TemplateTypeParameterAST* parameter) -> bool {
    return parameter->idExpression != nullptr;
  }

  auto operator()(ConstraintTypeParameterAST* parameter) -> bool {
    return parameter->typeId && parameter->typeId->type;
  }
};
}  // namespace

auto isPackParameter(TemplateParameterAST* parameter) -> bool {
  if (!parameter) return false;
  return visit(IsPackParameter{}, parameter);
}

auto hasDefaultTemplateArgument(TemplateParameterAST* parameter) -> bool {
  if (!parameter) return false;
  return visit(HasDefaultTemplateArgument{}, parameter);
}

auto isPackExpansion(TypeIdAST* typeId) -> bool {
  if (!typeId || !typeId->declarator) return false;
  return ast_cast<ParameterPackAST>(typeId->declarator->coreDeclarator) !=
         nullptr;
}

auto isPackExpansionTemplateArgument(TemplateArgumentAST* argument) -> bool {
  if (auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument))
    return isPackExpansion(typeArgument->typeId);

  if (auto expressionArgument =
          ast_cast<ExpressionTemplateArgumentAST>(argument)) {
    return ast_cast<PackExpansionExpressionAST>(
               expressionArgument->expression) != nullptr;
  }

  return false;
}

auto lastTemplateArgument(List<TemplateArgumentAST*>* templateArgumentList)
    -> TemplateArgumentAST* {
  TemplateArgumentAST* last = nullptr;
  for (auto argument : ListView{templateArgumentList}) last = argument;
  return last;
}

auto hasPackExpansionTemplateArgument(
    List<TemplateArgumentAST*>* templateArgumentList) -> bool {
  for (auto argument : ListView{templateArgumentList}) {
    if (isPackExpansionTemplateArgument(argument)) return true;
  }
  return false;
}

auto computeTemplateArity(TemplateDeclarationAST* templateDecl)
    -> TemplateArity {
  TemplateArity arity;
  if (!templateDecl) return arity;

  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    ++arity.maxArgs;

    if (isPackParameter(parameter)) {
      ++arity.packCount;
      arity.hasParameterPack = true;
      continue;
    }

    if (!hasDefaultTemplateArgument(parameter)) {
      ++arity.minArgs;
    }
  }

  return arity;
}

auto templateArgumentCount(List<TemplateArgumentAST*>* templateArgumentList)
    -> int {
  int count = 0;
  for (auto argument : ListView{templateArgumentList}) {
    (void)argument;
    ++count;
  }
  return count;
}

auto isTemplateArityMatch(TemplateDeclarationAST* templateDecl,
                          List<TemplateArgumentAST*>* templateArgumentList,
                          bool isFunctionTemplate) -> bool {
  if (!templateDecl) return true;

  auto arity = computeTemplateArity(templateDecl);
  auto argc = templateArgumentCount(templateArgumentList);

  const bool expandsAnUnknownNumberOfArguments =
      hasPackExpansionTemplateArgument(templateArgumentList);

  if (!isFunctionTemplate && !expandsAnUnknownNumberOfArguments &&
      argc < arity.minArgs)
    return false;
  if (!arity.hasParameterPack && argc > arity.maxArgs) return false;

  return true;
}

struct Substitution::CollectRawTemplateArgument {
  Substitution& subst;

  [[nodiscard]] auto isInTemplateScope(Symbol* symbol) -> bool {
    return isEnclosedInDependentTemplate(subst.unit_, symbol->parent(),
                                         /*stopAtConcreteSpecialization=*/true);
  }

  auto operator()(ExpressionTemplateArgumentAST* ast) -> std::optional<Symbol*>;

  auto operator()(TypeTemplateArgumentAST* ast) -> std::optional<Symbol*>;
};

struct Substitution::MakeDefaultTemplateArgument {
  Substitution& subst;

  [[nodiscard]] auto control() -> Control* { return subst.unit_->control(); }

  auto operator()(TemplateTypeParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  auto operator()(NonTypeTemplateParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  auto operator()(TypenameTypeParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  auto operator()(ConstraintTypeParameterAST* parameter)
      -> std::optional<TemplateArgument>;
};

auto Substitution::MakeDefaultTemplateArgument::operator()(
    TemplateTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->idExpression || !parameter->idExpression->symbol) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }
  return parameter->idExpression->symbol;
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    NonTypeTemplateParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->declaration || !parameter->declaration->expression) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }

  const Type* declaredType = parameter->declaration->type;

  if (declaredType && isDependent(subst.unit_, declaredType) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    auto typeId = TypeIdAST::create(subst.unit_->arena());
    typeId->typeSpecifierList = parameter->declaration->typeSpecifierList;
    typeId->declarator = parameter->declaration->declarator;

    auto substituted = ASTRewriter::substituteDefaultTypeId(
        subst.unit_, typeId, subst.templateArguments_,
        subst.templateDecl_->depth, subst.templateDecl_->symbol);

    if (!substituted || !substituted->type ||
        type_cast<UnresolvedNameType>(substituted->type)) {
      return std::nullopt;
    }

    declaredType = substituted->type;
  }

  auto expression = parameter->declaration->expression;

  if (isDependent(subst.unit_, expression) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    if (auto substituted = ASTRewriter::substituteDefaultExpression(
            subst.unit_, expression, subst.templateArguments_,
            subst.templateDecl_->depth, subst.templateDecl_->symbol)) {
      expression = substituted;
    }
  }

  auto interp = ASTInterpreter{subst.unit_};
  auto value = interp.evaluate(expression);

  if (!value.has_value()) {
    if (isDependent(subst.unit_, expression)) return std::nullopt;

    subst.maybeReportInvalidConstantExpression(
        parameter->firstSourceLocation());

    return std::nullopt;
  }

  auto argument = control()->newVariableSymbol(nullptr, {});
  argument->setInitializer(expression);
  argument->setConstexpr(true);
  argument->setConstValue(value.value());

  const Type* argumentType = declaredType;
  if (!argumentType && expression) argumentType = expression->type;

  if (!argumentType) {
    if (isDependent(subst.unit_, expression)) return std::nullopt;

    subst.maybeReportMalformedTemplateArgument(
        parameter->firstSourceLocation());

    return std::nullopt;
  }

  argument->setType(argumentType);
  return argument;
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    TypenameTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  const auto loc = parameter->firstSourceLocation();

  if (!parameter->typeId) {
    subst.error(loc, "missing default template argument");
    return std::nullopt;
  }

  auto typeId = parameter->typeId;

  if ((!typeId->type || isDependent(subst.unit_, typeId->type)) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        subst.unit_, typeId, subst.templateArguments_,
        subst.templateDecl_->depth, subst.templateDecl_->symbol);
    if (substituted && substituted->type) {
      typeId = substituted;
    }
  }

  if (!typeId->type) {
    subst.error(loc, "missing default template argument");
    return std::nullopt;
  }

  auto argument = control()->newTypeAliasSymbol(nullptr, {});
  argument->setType(typeId->type);
  return argument;
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    ConstraintTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->typeId || !parameter->typeId->type) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }

  auto typeId = parameter->typeId;

  if (isDependent(subst.unit_, typeId->type) &&
      !subst.templateArguments_.empty() && subst.templateDecl_) {
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        subst.unit_, typeId, subst.templateArguments_,
        subst.templateDecl_->depth, subst.templateDecl_->symbol);
    if (substituted && substituted->type) {
      typeId = substituted;
    }
  }

  auto argument = control()->newTypeAliasSymbol(nullptr, {});
  argument->setType(typeId->type);
  return argument;
}

auto Substitution::CollectRawTemplateArgument::operator()(
    ExpressionTemplateArgumentAST* ast) -> std::optional<Symbol*> {
  auto loc = ast->firstSourceLocation();

  auto expression = ast->expression;

  if (!expression) {
    subst.maybeReportMalformedTemplateArgument(loc);
    return std::nullopt;
  }

  auto unit = subst.unit_;
  auto control = unit->control();

  auto interp = ASTInterpreter{unit};

  const auto isDependent = isDependentTemplateArgument(subst.unit_, ast);

  auto value = isDependent ? std::nullopt : interp.evaluate(expression);

  if (!value.has_value()) {
    if (isDependent) {
      auto expandedPattern = expression;
      if (auto packExpansion =
              ast_cast<PackExpansionExpressionAST>(expandedPattern)) {
        expandedPattern = packExpansion->expression;
      }

      if (auto idExpr = ast_cast<IdExpressionAST>(expandedPattern)) {
        if (auto nttp = symbol_cast<NonTypeParameterSymbol>(idExpr->symbol)) {
          return nttp;
        }
        if (auto var = symbol_cast<VariableSymbol>(idExpr->symbol);
            var && !var->parent()) {
          return var;
        }
      }
      auto templateArgument = control->newVariableSymbol(nullptr, {});
      templateArgument->setInitializer(expression);
      if (expression->type) {
        templateArgument->setType(expression->type);
      }
      return templateArgument;
    }

    subst.maybeReportInvalidConstantExpression(loc);

    return std::nullopt;
  }

  auto templateArgument = control->newVariableSymbol(nullptr, {});
  templateArgument->setInitializer(expression);

  auto argumentType = expression->type;

  if (!argumentType) {
    templateArgument->setConstexpr(true);
    templateArgument->setConstValue(value);
    return templateArgument;
  }

  if (!subst.unit_->typeTraits().is_scalar(argumentType)) {
    argumentType = subst.unit_->typeTraits().add_pointer(expression->type);
  }

  templateArgument->setType(argumentType);
  templateArgument->setConstexpr(true);
  templateArgument->setConstValue(value);
  return templateArgument;
}

auto Substitution::CollectRawTemplateArgument::operator()(
    TypeTemplateArgumentAST* ast) -> std::optional<Symbol*> {
  if (!ast->typeId) {
    return std::nullopt;
  }

  auto loc = ast->firstSourceLocation();

  auto unit = subst.unit_;
  auto control = unit->control();

  for (auto spec : ListView{ast->typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!named) continue;
    if (auto pack = symbol_cast<ParameterPackSymbol>(named->symbol))
      return pack;
    if (!ast_cast<NameIdAST>(named->unqualifiedId)) break;
    if (auto alias = symbol_cast<TypeAliasSymbol>(named->symbol)) {
      if (alias->templateParameters()) return alias;
    }
    if (auto classSymbol = symbol_cast<ClassSymbol>(named->symbol)) {
      if (classSymbol->templateParameters()) return classSymbol;
    }
    if (auto templateParameter =
            symbol_cast<TemplateTypeParameterSymbol>(named->symbol)) {
      return templateParameter;
    }
    break;
  }

  if (!ast->typeId->type) {
    if (isDependentTemplateArgument(subst.unit_, ast)) {
      auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
      return templateArgument;
    }
    subst.maybeReportMalformedTemplateArgument(loc);
    return std::nullopt;
  }

  auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
  templateArgument->setType(ast->typeId->type);
  return templateArgument;
}

Substitution::Substitution(TranslationUnit* unit,
                           TemplateDeclarationAST* templateDecl,
                           List<TemplateArgumentAST*>* templateArgumentList,
                           bool argsComplete, bool fillDefaults)
    : unit_(unit),
      templateDecl_(templateDecl),
      templateArgumentList_(templateArgumentList),
      argsComplete_(argsComplete),
      fillDefaults_(fillDefaults) {
  doMake();
}

auto Substitution::make(TranslationUnit* unit,
                        TemplateDeclarationAST* templateDecl,
                        List<TemplateArgumentAST*>* templateArgumentList,
                        bool argsComplete) -> std::optional<Substitution> {
  Substitution subst{unit, templateDecl, templateArgumentList, argsComplete};
  if (subst.hadError_) return std::nullopt;
  return std::optional<Substitution>{std::move(subst)};
}

auto Substitution::makePartial(TranslationUnit* unit,
                               TemplateDeclarationAST* templateDecl,
                               List<TemplateArgumentAST*>* templateArgumentList)
    -> std::optional<Substitution> {
  Substitution subst{unit, templateDecl, templateArgumentList, false, false};
  if (subst.hadError_) return std::nullopt;
  return std::optional<Substitution>{std::move(subst)};
}

void Substitution::doMake() {
  if (!templateDecl_) {
    cxx_runtime_error("no template declaration");
  }

  auto control = unit_->control();

  std::vector<Symbol*> collectedArguments;
  std::vector<bool> collectedIsPackExpansion;
  for (auto argument : ListView{templateArgumentList_}) {
    auto arg = visit(CollectRawTemplateArgument{*this}, argument);
    if (!arg.has_value()) return;
    collectedArguments.push_back(*arg);
    collectedIsPackExpansion.push_back(
        isPackExpansionTemplateArgument(argument));
  }

  std::vector<TemplateParameterAST*> parameters;
  for (auto parameter : ListView{templateDecl_->templateParameterList}) {
    parameters.push_back(parameter);
  }

  const int paramCount = static_cast<int>(parameters.size());
  const int argCount = static_cast<int>(collectedArguments.size());

  int packIndex = -1;
  int packSize = 0;

  for (int i = 0; i < paramCount; ++i) {
    if (!isPackParameter(parameters[i])) continue;
    packIndex = i;

    if (argsComplete_) {
      int nonPackCount = 0;
      for (int j = 0; j < paramCount; ++j)
        if (!isPackParameter(parameters[j])) ++nonPackCount;
      packSize = std::max(0, argCount - nonPackCount);
      break;
    }

    int trailingRequired = 0;
    for (int j = i + 1; j < paramCount; ++j) {
      if (isPackParameter(parameters[j])) continue;
      if (hasDefaultTemplateArgument(parameters[j])) continue;
      ++trailingRequired;
    }

    int availableForPack = argCount - packIndex - trailingRequired;
    packSize = std::max(0, availableForPack);
    break;
  }

  int argumentIndex = 0;
  bool argumentCountIsKnown = true;

  auto deducedPackAt = [&](int index) -> ParameterPackSymbol* {
    if (index >= argCount) return nullptr;
    return symbol_cast<ParameterPackSymbol>(collectedArguments[index]);
  };

  for (int i = 0; i < paramCount; ++i) {
    auto parameter = parameters[i];

    if (isPackParameter(parameter)) {
      if (auto deducedPack = deducedPackAt(argumentIndex)) {
        ++argumentIndex;
        templateArguments_.push_back(deducedPack);
        continue;
      }
      if (argumentIndex >= argCount) {
        auto pack = control->newParameterPackSymbol(nullptr, {});
        templateArguments_.push_back(pack);
        continue;
      }
    }

    if (i == packIndex) {
      auto pack = control->newParameterPackSymbol(nullptr, {});
      auto nonTypeParam = ast_cast<NonTypeTemplateParameterAST>(parameter);

      for (int k = 0; k < packSize && argumentIndex < argCount; ++k) {
        auto symbol = collectedArguments[argumentIndex++];
        symbol = normalizeNonTypeArgument(nonTypeParam, symbol);
        pack->addElement(symbol);
      }

      templateArguments_.push_back(pack);
      continue;
    }

    if (argumentIndex < argCount) {
      if (collectedIsPackExpansion[argumentIndex]) argumentCountIsKnown = false;
      auto symbol = collectedArguments[argumentIndex++];
      auto nonTypeParam = ast_cast<NonTypeTemplateParameterAST>(parameter);
      if (nonTypeParam && !checkNonTypeParameterType(nonTypeParam)) return;
      symbol = normalizeNonTypeArgument(nonTypeParam, symbol);
      templateArguments_.push_back(symbol);
      continue;
    }

    if (!fillDefaults_ || !argumentCountIsKnown) break;

    if (auto defaultArg = getDefaultTemplateArgument(parameter)) {
      templateArguments_.push_back(defaultArg.value());
      continue;
    }

    hadError_ = true;
    return;
  }
}

void Substitution::maybeReportInvalidConstantExpression(SourceLocation loc) {
  error(loc, "template argument is not a constant expression");
}

void Substitution::maybeReportMalformedTemplateArgument(SourceLocation loc) {
  error(loc, "malformed template argument");
}

void Substitution::maybeReportMissingTemplateArgument(SourceLocation loc) {
  error(loc, "missing template argument");
}

void Substitution::error(SourceLocation loc, std::string message) {
  hadError_ = true;
  auto unit = unit_;
  if (!unit->config().checkTypes) return;
  unit->error(loc, std::move(message));
}

void Substitution::warning(SourceLocation loc, std::string message) {
  auto unit = unit_;
  if (!unit->config().checkTypes) return;
  unit->warning(loc, std::move(message));
}

auto Substitution::checkNonTypeParameterType(
    NonTypeTemplateParameterAST* parameter) -> bool {
  if (!parameter->declaration) return true;

  const Type* declaredType = parameter->declaration->type;
  if (!declaredType) return true;
  if (!isDependent(unit_, declaredType)) return true;
  if (templateArguments_.empty() || !templateDecl_) return true;

  auto typeId = TypeIdAST::create(unit_->arena());
  typeId->typeSpecifierList = parameter->declaration->typeSpecifierList;
  typeId->declarator = parameter->declaration->declarator;

  auto substituted = ASTRewriter::substituteDefaultTypeId(
      unit_, typeId, templateArguments_, templateDecl_->depth,
      templateDecl_->symbol);

  if (!substituted || !substituted->type ||
      type_cast<UnresolvedNameType>(substituted->type)) {
    error(parameter->firstSourceLocation(),
          "substitution failure in the type of a non-type template "
          "parameter");
    return false;
  }

  return true;
}

auto Substitution::normalizeNonTypeArgument(
    NonTypeTemplateParameterAST* parameter, Symbol* argument) -> Symbol* {
  if (!parameter) return argument;

  auto unit = unit_;
  auto control = unit->control();

  auto variableArgument = symbol_cast<VariableSymbol>(argument);
  if (!variableArgument) {
    auto typeAliasArgument = symbol_cast<TypeAliasSymbol>(argument);
    if (!typeAliasArgument || !typeAliasArgument->type()) return argument;
    if (typeAliasArgument->templateParameters()) return argument;
    if (!isDependent(unit, typeAliasArgument->type())) return argument;
    if (type_cast<ClassType>(typeAliasArgument->type())) return argument;

    auto normalizedArgument = control->newVariableSymbol(nullptr, {});
    const Type* targetType = typeAliasArgument->type();
    if (parameter && parameter->declaration && parameter->declaration->type) {
      targetType = parameter->declaration->type;
    }
    normalizedArgument->setType(targetType);
    return normalizedArgument;
  }

  auto normalizedArgument = control->newVariableSymbol(nullptr, {});
  normalizedArgument->setInitializer(variableArgument->initializer());
  normalizedArgument->setConstexpr(variableArgument->isConstexpr());
  normalizedArgument->setConstValue(variableArgument->constValue());

  const Type* targetType = variableArgument->type();

  if (!type_cast<TypeParameterType>(targetType) &&
      !type_cast<TemplateTypeParameterType>(targetType)) {
    if (parameter && parameter->declaration && parameter->declaration->type) {
      const Type* declaredType = parameter->declaration->type;
      if (!isDependent(unit, declaredType)) {
        targetType = declaredType;
      } else if (templateDecl_) {
        auto typeId = TypeIdAST::create(unit->arena());
        typeId->typeSpecifierList = parameter->declaration->typeSpecifierList;
        typeId->declarator = parameter->declaration->declarator;

        auto substituted = ASTRewriter::substituteDefaultTypeId(
            unit, typeId, templateArguments_, templateDecl_->depth,
            templateDecl_->symbol);

        if (substituted && substituted->type &&
            !type_cast<UnresolvedNameType>(substituted->type) &&
            !isDependent(unit, substituted->type)) {
          targetType = substituted->type;
        }
      }
    }
  }

  normalizedArgument->setType(targetType);
  return normalizedArgument;
}

auto Substitution::getDefaultTemplateArgument(TemplateParameterAST* parameter)
    -> std::optional<TemplateArgument> {
  return visit(MakeDefaultTemplateArgument{*this}, parameter);
}
}  // namespace cxx
