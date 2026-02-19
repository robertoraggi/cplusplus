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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTAsemaLITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cxx/substitution.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct Substitution::IsPackParameter {
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
    return false;
  }
};

struct Substitution::HasDefaultTemplateArgument {
  auto operator()(TypenameTypeParameterAST* parameter) -> bool {
    return parameter->typeId && parameter->typeId->type;
  }

  auto operator()(NonTypeTemplateParameterAST* parameter) -> bool {
    return parameter->declaration && parameter->declaration->expression;
  }

  auto operator()(TemplateTypeParameterAST* parameter) -> bool {
    return parameter->idExpression != nullptr;
  }

  auto operator()(ConstraintTypeParameterAST* parameter) -> bool {
    return parameter->typeId && parameter->typeId->type;
  }
};

struct Substitution::CollectRawTemplateArgument {
  Substitution& subst;

  [[nodiscard]] auto isInTemplateScope(Symbol* symbol) -> bool {
    for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
      if (scope->isTemplateParameters()) return true;
    }
    return false;
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
  subst.error(parameter->firstSourceLocation(),
              "default template argument for template template "
              "parameters is not implemented");
  return std::nullopt;
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    NonTypeTemplateParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->declaration || !parameter->declaration->expression) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }

  auto expression = parameter->declaration->expression;

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

  const Type* argumentType = parameter->declaration->type;
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

  if (!parameter->typeId || !parameter->typeId->type) {
    subst.error(loc, "missing default template argument");
    return std::nullopt;
  }

  auto argument = control()->newTypeAliasSymbol(nullptr, {});
  argument->setType(parameter->typeId->type);
  return argument;
}

auto Substitution::MakeDefaultTemplateArgument::operator()(
    ConstraintTypeParameterAST* parameter) -> std::optional<TemplateArgument> {
  if (!parameter->typeId || !parameter->typeId->type) {
    subst.maybeReportMissingTemplateArgument(parameter->firstSourceLocation());
    return std::nullopt;
  }

  auto argument = control()->newTypeAliasSymbol(nullptr, {});
  argument->setType(parameter->typeId->type);
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

  auto value = interp.evaluate(expression);

  if (!value.has_value()) {
    if (subst.isDependentExpressionArgument(ast)) {
      // For NTTP parameter references (e.g., X in S<X, 0>), preserve the
      // parameter identity using TypeParameterType so partial spec
      // deduction can recognize it as a deducible position.
      if (auto idExpr = ast_cast<IdExpressionAST>(expression)) {
        if (auto nttp = symbol_cast<NonTypeParameterSymbol>(idExpr->symbol)) {
          auto templateArgument = control->newVariableSymbol(nullptr, {});
          auto paramType = control->getTypeParameterType(
              nttp->index(), nttp->depth(), nttp->isParameterPack());
          templateArgument->setType(paramType);
          return templateArgument;
        }
      }
      // Complex dependent expression (e.g., !is_array<_Tp>::value).
      // Add a placeholder to keep the argument count correct.
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
    if (!subst.isDependentExpressionArgument(ast)) {
      templateArgument->setConstexpr(true);
      templateArgument->setConstValue(value);
      return templateArgument;
    }
    return std::nullopt;
  }

  if (!control->is_scalar(argumentType)) {
    argumentType = control->add_pointer(expression->type);
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

  if (!ast->typeId->type) {
#if true
    auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
    return templateArgument;
#else
    subst.warning(loc, "there is no type for this thing");
    if (subst.isDependentTypeArgument(ast)) {
      // Dependent type argument with unresolved type (e.g., _tp parameter).
      // Add placeholder to maintain argument count.
      auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
      return templateArgument;
    }
    subst.maybeReportMalformedTemplateArgument(loc);
    return std::nullopt;
#endif
  }

  auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
  templateArgument->setType(ast->typeId->type);
  return templateArgument;
}

Substitution::Substitution(TranslationUnit* unit,
                           TemplateDeclarationAST* templateDecl,
                           List<TemplateArgumentAST*>* templateArgumentList)
    : unit_(unit),
      templateDecl_(templateDecl),
      templateArgumentList_(templateArgumentList) {
  make();
}

void Substitution::make() {
  if (!templateDecl_) {
    cxx_runtime_error("no template declaration");
  }

  auto control = unit_->control();

  std::vector<Symbol*> collectedArguments;
  for (auto argument : ListView{templateArgumentList_}) {
    auto arg = visit(CollectRawTemplateArgument{*this}, argument);
    if (!arg.has_value()) return;
    collectedArguments.push_back(*arg);
  }

  // Gather parameter list.
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

    // Count trailing non-pack parameters that don't have defaults —
    // they need their own arguments, so the pack can't consume those.
    int trailingRequired = 0;
    for (int j = i + 1; j < paramCount; ++j) {
      if (isPackParameter(parameters[j])) continue;
      if (hasDefaultTemplateArgument(parameters[j])) continue;
      ++trailingRequired;
    }

    // Non-pack parameters before the pack each consume one argument.
    int availableForPack = argCount - packIndex - trailingRequired;
    packSize = std::max(0, availableForPack);
    break;
  }

  int argumentIndex = 0;

  for (int i = 0; i < paramCount; ++i) {
    auto parameter = parameters[i];

    if (i == packIndex) {
      // Pack parameter: consume packSize collected arguments.
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

    // Non-pack parameter: use collected argument if available.
    if (argumentIndex < argCount) {
      auto symbol = collectedArguments[argumentIndex++];
      auto nonTypeParam = ast_cast<NonTypeTemplateParameterAST>(parameter);
      symbol = normalizeNonTypeArgument(nonTypeParam, symbol);
      templateArguments_.push_back(symbol);
      continue;
    }

    // No collected argument left — fill from default.
    if (auto defaultArg = getDefaultTemplateArgument(parameter)) {
      templateArguments_.push_back(defaultArg.value());
    }
  }
}

auto Substitution::isDependentTypeArgument(TypeTemplateArgumentAST* typeArg)
    -> bool {
  if (!typeArg || !typeArg->typeId) return false;
  return isDependent(unit_, typeArg->typeId);
}

auto Substitution::isDependentExpressionArgument(
    ExpressionTemplateArgumentAST* ast) -> bool {
  if (!ast) return false;
  return isDependent(unit_, ast->expression);
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
  auto unit = unit_;
  if (!unit->config().checkTypes) return;
  unit->error(loc, std::move(message));
}

void Substitution::warning(SourceLocation loc, std::string message) {
  auto unit = unit_;
  if (!unit->config().checkTypes) return;
  unit->warning(loc, std::move(message));
}

auto Substitution::normalizeNonTypeArgument(
    NonTypeTemplateParameterAST* parameter, Symbol* argument) -> Symbol* {
  auto unit = unit_;
  auto control = unit->control();

  auto variableArgument = symbol_cast<VariableSymbol>(argument);
  if (!variableArgument) {
    auto typeAliasArgument = symbol_cast<TypeAliasSymbol>(argument);
    if (!typeAliasArgument || !typeAliasArgument->type()) return argument;
    if (!isDependent(unit, typeAliasArgument->type())) return argument;

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

  // Preserve TypeParameterType for NTTP partial specialization patterns.
  // Without this, the pattern's type info used for deduction would be lost.
  if (!type_cast<TypeParameterType>(targetType) &&
      !type_cast<TemplateTypeParameterType>(targetType)) {
    if (parameter && parameter->declaration && parameter->declaration->type) {
      targetType = parameter->declaration->type;
    }
  }

  normalizedArgument->setType(targetType);
  return normalizedArgument;
}

auto Substitution::isPackParameter(TemplateParameterAST* parameter) -> bool {
  if (!parameter) return false;
  return visit(IsPackParameter{}, parameter);
}

auto Substitution::hasDefaultTemplateArgument(TemplateParameterAST* parameter)
    -> bool {
  if (!parameter) return false;
  return visit(HasDefaultTemplateArgument{}, parameter);
}

auto Substitution::getDefaultTemplateArgument(TemplateParameterAST* parameter)
    -> std::optional<TemplateArgument> {
  return visit(MakeDefaultTemplateArgument{*this}, parameter);
}

}  // namespace cxx
