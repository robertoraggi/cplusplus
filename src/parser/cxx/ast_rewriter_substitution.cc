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

namespace cxx {

namespace {

auto shouldReportSubstitutionDiagnostic(TranslationUnit* unit,
                                        SourceLocation loc) -> bool {
  if (!unit) return false;
  if (!loc) return false;

  auto preprocessor = unit->preprocessor();
  if (!preprocessor) return false;

  const auto& token = unit->tokenAt(loc);
  if (!token) return false;

  return !preprocessor->isSystemHeader(token.fileId());
}

auto isDependentTypeArgument(TypeTemplateArgumentAST* typeArg) -> bool {
  if (!typeArg || !typeArg->typeId) return false;
  return isDependentTypeId(typeArg->typeId);
}

auto isDependentExpressionArgument(ExpressionTemplateArgumentAST* expressionArg)
    -> bool {
  if (!expressionArg) return false;
  return ContainsDependentExpr{}.check(expressionArg->expression);
}

void maybeReportInvalidConstantExpression(TranslationUnit* unit,
                                          SourceLocation loc) {
  if (!shouldReportSubstitutionDiagnostic(unit, loc)) return;
  unit->error(loc, "template argument is not a constant expression");
}

void maybeReportMalformedTemplateArgument(TranslationUnit* unit,
                                          SourceLocation loc) {
  if (!shouldReportSubstitutionDiagnostic(unit, loc)) return;
  unit->error(loc, "malformed template argument");
}

void maybeReportMissingTemplateArgument(TranslationUnit* unit,
                                        SourceLocation loc) {
  if (!shouldReportSubstitutionDiagnostic(unit, loc)) return;
  unit->error(loc, "missing template argument");
}

auto normalizeNonTypeArgument(Control* control,
                              NonTypeTemplateParameterAST* parameter,
                              Symbol* argument) -> Symbol* {
  auto variableArgument = symbol_cast<VariableSymbol>(argument);
  if (!variableArgument) {
    auto typeAliasArgument = symbol_cast<TypeAliasSymbol>(argument);
    if (!typeAliasArgument || !typeAliasArgument->type()) return argument;
    if (!isDependentType(typeAliasArgument->type())) return argument;

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

auto isPackParameter(TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    return typeParameter->isPack;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return nonTypeParameter->declaration &&
           nonTypeParameter->declaration->isPack;
  }

  if (auto templateTypeParameter =
          ast_cast<TemplateTypeParameterAST>(parameter)) {
    return templateTypeParameter->isPack;
  }

  if (auto constraintParameter =
          ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return static_cast<bool>(constraintParameter->ellipsisLoc);
  }

  return false;
}

auto hasDefaultTemplateArgument(TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    return typeParameter->typeId && typeParameter->typeId->type;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return nonTypeParameter->declaration &&
           nonTypeParameter->declaration->expression;
  }

  if (auto templateTypeParameter =
          ast_cast<TemplateTypeParameterAST>(parameter)) {
    return templateTypeParameter->idExpression != nullptr;
  }

  if (auto constraintParameter =
          ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return constraintParameter->typeId && constraintParameter->typeId->type;
  }

  return false;
}

auto collectRawTemplateArguments(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList)
    -> std::vector<Symbol*> {
  if (!unit) return {};

  auto control = unit->control();
  if (!control) return {};

  auto interpreter = ASTInterpreter{unit};

  std::vector<Symbol*> collectedArguments;

  for (auto argument : ListView{templateArgumentList}) {
    if (auto expressionArgument =
            ast_cast<ExpressionTemplateArgumentAST>(argument)) {
      auto expression = expressionArgument->expression;
      if (!expression) {
        maybeReportMalformedTemplateArgument(unit,
                                             argument->firstSourceLocation());
        continue;
      }

      auto value = interpreter.evaluate(expression);
      if (!value.has_value()) {
        if (isDependentExpressionArgument(expressionArgument)) {
          // For NTTP parameter references (e.g., X in S<X, 0>), preserve the
          // parameter identity using TypeParameterType so partial spec
          // deduction can recognize it as a deducible position.
          if (auto idExpr = ast_cast<IdExpressionAST>(expression)) {
            if (auto nttp =
                    symbol_cast<NonTypeParameterSymbol>(idExpr->symbol)) {
              auto templateArgument = control->newVariableSymbol(nullptr, {});
              auto paramType = control->getTypeParameterType(
                  nttp->index(), nttp->depth(), nttp->isParameterPack());
              templateArgument->setType(paramType);
              collectedArguments.push_back(templateArgument);
              continue;
            }
          }
          // Complex dependent expression (e.g., !is_array<_Tp>::value).
          // Add a placeholder to keep the argument count correct.
          auto templateArgument = control->newVariableSymbol(nullptr, {});
          templateArgument->setInitializer(expression);
          if (expression->type) {
            templateArgument->setType(expression->type);
          }
          collectedArguments.push_back(templateArgument);
          continue;
        }
        maybeReportInvalidConstantExpression(unit,
                                             argument->firstSourceLocation());
        continue;
      }

      auto templateArgument = control->newVariableSymbol(nullptr, {});
      templateArgument->setInitializer(expression);

      auto argumentType = expression->type;
      if (!argumentType) {
        if (!isDependentExpressionArgument(expressionArgument)) {
          templateArgument->setConstexpr(true);
          templateArgument->setConstValue(value);
          collectedArguments.push_back(templateArgument);
        }
        continue;
      }

      if (!control->is_scalar(argumentType)) {
        argumentType = control->add_pointer(expression->type);
      }

      templateArgument->setType(argumentType);
      templateArgument->setConstexpr(true);
      templateArgument->setConstValue(value);
      collectedArguments.push_back(templateArgument);
      continue;
    }

    auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
    if (!typeArgument || !typeArgument->typeId || !typeArgument->typeId->type) {
      if (typeArgument && isDependentTypeArgument(typeArgument)) {
        // Dependent type argument with unresolved type (e.g., _Tp parameter).
        // Add placeholder to maintain argument count.
        auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
        collectedArguments.push_back(templateArgument);
        continue;
      }
      maybeReportMalformedTemplateArgument(unit,
                                           argument->firstSourceLocation());
      continue;
    }

    auto templateArgument = control->newTypeAliasSymbol(nullptr, {});
    templateArgument->setType(typeArgument->typeId->type);
    collectedArguments.push_back(templateArgument);
  }

  return collectedArguments;
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

void appendDefaultTemplateArgument(TranslationUnit* unit, Control* control,
                                   TemplateParameterAST* parameter,
                                   std::vector<TemplateArgument>& result) {
  auto missingLoc =
      parameter ? parameter->firstSourceLocation() : SourceLocation{};

  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    if (!typeParameter->typeId || !typeParameter->typeId->type) {
      maybeReportMissingTemplateArgument(unit, missingLoc);
      return;
    }

    auto argument = control->newTypeAliasSymbol(nullptr, {});
    argument->setType(typeParameter->typeId->type);
    result.push_back(argument);
    return;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    if (!nonTypeParameter->declaration ||
        !nonTypeParameter->declaration->expression) {
      maybeReportMissingTemplateArgument(unit, missingLoc);
      return;
    }

    auto expression = nonTypeParameter->declaration->expression;
    auto value = ASTInterpreter{unit}.evaluate(expression);
    if (!value.has_value()) {
      if (ContainsDependentExpr{}.check(expression)) return;
      maybeReportInvalidConstantExpression(unit, missingLoc);
      return;
    }

    auto argument = control->newVariableSymbol(nullptr, {});
    argument->setInitializer(expression);
    argument->setConstexpr(true);
    argument->setConstValue(value.value());

    const Type* argumentType = nonTypeParameter->declaration->type;
    if (!argumentType && expression) argumentType = expression->type;
    if (!argumentType) {
      if (ContainsDependentExpr{}.check(expression)) return;
      maybeReportMalformedTemplateArgument(unit, missingLoc);
      return;
    }

    argument->setType(argumentType);
    result.push_back(argument);
    return;
  }

  auto constraintParameter = ast_cast<ConstraintTypeParameterAST>(parameter);
  if (!constraintParameter || !constraintParameter->typeId ||
      !constraintParameter->typeId->type) {
    maybeReportMissingTemplateArgument(unit, missingLoc);
    return;
  }

  auto argument = control->newTypeAliasSymbol(nullptr, {});
  argument->setType(constraintParameter->typeId->type);
  result.push_back(argument);
}

}  // namespace

auto ASTRewriter::make_substitution(
    TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
    List<TemplateArgumentAST*>* templateArgumentList)
    -> std::vector<TemplateArgument> {
  return Substitution{sema, templateDecl, templateArgumentList}
      .templateArguments();
}

Substitution::Substitution(Semantics* sema,
                           TemplateDeclarationAST* templateDecl,
                           List<TemplateArgumentAST*>* templateArgumentList) {
  auto unit = sema->translationUnit();
  auto control = unit->control();

  auto collectedArguments =
      collectRawTemplateArguments(unit, templateArgumentList);

  int rawArgumentCount = templateArgumentCount(templateArgumentList);

  if (!templateDecl) {
    for (auto symbol : collectedArguments) {
      templateArguments_.push_back(symbol);
    }

    return;
  }

  int argumentIndex = 0;
  std::vector<TemplateParameterAST*> parameters;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    parameters.push_back(parameter);
  }

  for (int parameterIndex = 0;
       parameterIndex < static_cast<int>(parameters.size()); ++parameterIndex) {
    auto parameter = parameters[parameterIndex];
    if (isPackParameter(parameter)) {
      auto pack = control->newParameterPackSymbol(nullptr, {});
      auto nonTypeParameter = ast_cast<NonTypeTemplateParameterAST>(parameter);

      int minTrailingRequired = 0;
      for (int j = parameterIndex + 1; j < static_cast<int>(parameters.size());
           ++j) {
        if (isPackParameter(parameters[j])) continue;
        if (hasDefaultTemplateArgument(parameters[j])) continue;
        ++minTrailingRequired;
      }

      int availableCollected =
          static_cast<int>(collectedArguments.size()) - argumentIndex;
      int packConsumeCount = availableCollected - minTrailingRequired;
      if (packConsumeCount < 0) packConsumeCount = 0;

      while (packConsumeCount-- > 0 &&
             argumentIndex < static_cast<int>(collectedArguments.size())) {
        auto symbol = collectedArguments[argumentIndex];
        symbol = normalizeNonTypeArgument(control, nonTypeParameter, symbol);
        pack->addElement(symbol);
        ++argumentIndex;
      }

      if (argumentIndex < rawArgumentCount &&
          argumentIndex >= static_cast<int>(collectedArguments.size())) {
        argumentIndex = rawArgumentCount;
      }

      templateArguments_.push_back(pack);
      continue;
    }

    if (argumentIndex < static_cast<int>(collectedArguments.size())) {
      auto symbol = collectedArguments[argumentIndex];
      auto nonTypeParameter = ast_cast<NonTypeTemplateParameterAST>(parameter);
      symbol = normalizeNonTypeArgument(control, nonTypeParameter, symbol);
      templateArguments_.push_back(symbol);
      ++argumentIndex;
      continue;
    }

    if (argumentIndex < rawArgumentCount) {
      ++argumentIndex;
      continue;
    }

    getDefaultTemplateArgument(unit, control, parameter, templateArguments_);
  }
}

}  // namespace cxx
