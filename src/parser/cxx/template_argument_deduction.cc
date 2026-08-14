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
#include <cxx/diagnostics_client.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

namespace cxx {
namespace {

[[nodiscard]] auto parameterTemplateArgumentType(TypeIdAST* typeId)
    -> const Type* {
  if (!typeId) return nullptr;
  if (typeId->type) return typeId->type;

  for (auto spec : ListView{typeId->typeSpecifierList}) {
    auto namedSpec = ast_cast<NamedTypeSpecifierAST>(spec);
    if (namedSpec && namedSpec->symbol) return namedSpec->symbol->type();
  }

  return nullptr;
}

}  // namespace

TemplateArgumentDeduction::TemplateArgumentDeduction(TranslationUnit* unit)
    : unit_(unit),
      traits(unit),
      control_(unit->control()),
      arena_(unit->arena()) {}

auto TemplateArgumentDeduction::deduce(
    FunctionSymbol* func, List<ExpressionAST*>* args,
    List<TemplateArgumentAST*>* explicitTemplateArgs)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;
  templateDecl_ = templateDecl;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  collectTemplateParameters(templateDecl);

  if (!substituteExplicitTemplateArguments(explicitTemplateArgs))
    return std::nullopt;

  parameterDeclarations_ = nullptr;
  if (templateDecl->declaration) {
    if (auto clause = getParameterClause(templateDecl->declaration))
      parameterDeclarations_ = clause->parameterDeclarationList;
  }

  if (!deduceFromCall(functionType, args)) return std::nullopt;

  if (!checkDeducedArguments()) return std::nullopt;

  return buildTemplateArgumentList();
}

auto TemplateArgumentDeduction::deduceForGuide(
    TemplateDeclarationAST* templateDecl, const FunctionType* functionType,
    ParameterDeclarationClauseAST* parameters, List<ExpressionAST*>* args)
    -> std::optional<List<TemplateArgumentAST*>*> {
  if (!templateDecl || !functionType) return std::nullopt;

  templateDecl_ = templateDecl;

  collectTemplateParameters(templateDecl);

  parameterDeclarations_ =
      parameters ? parameters->parameterDeclarationList : nullptr;

  if (!deduceFromCall(functionType, args)) return std::nullopt;
  if (!checkDeducedArguments()) return std::nullopt;

  return buildTemplateArgumentList();
}

auto TemplateArgumentDeduction::deduceFromConversionTarget(
    FunctionSymbol* func, const Type* targetType)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;
  templateDecl_ = templateDecl;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  collectTemplateParameters(templateDecl);
  parameterDeclarations_ = nullptr;

  auto P = traits.remove_reference(functionType->returnType());
  auto A = targetType;

  if (traits.is_reference(A)) {
    A = traits.remove_reference(A);
  } else if (traits.is_array(P) || traits.is_function(P)) {
    P = traits.decay(P);
  } else {
    P = traits.remove_cv(P);
  }

  A = traits.remove_cv(A);

  if (!deduceTypeFromType(P, A)) return std::nullopt;
  if (!checkDeducedArguments()) return std::nullopt;

  return buildTemplateArgumentList();
}

auto TemplateArgumentDeduction::deduceFromTargetType(
    FunctionSymbol* func, const FunctionType* targetType)
    -> std::optional<List<TemplateArgumentAST*>*> {
  auto templateDecl = func->templateDeclaration();
  if (!templateDecl) return std::nullopt;
  templateDecl_ = templateDecl;

  auto functionType = type_cast<FunctionType>(func->type());
  if (!functionType) return std::nullopt;

  collectTemplateParameters(templateDecl);

  parameterDeclarations_ = nullptr;
  if (templateDecl->declaration) {
    if (auto clause = getParameterClause(templateDecl->declaration))
      parameterDeclarations_ = clause->parameterDeclarationList;
  }

  auto params = functionType->parameterTypes();
  auto targetParams = targetType->parameterTypes();
  if (params.size() != targetParams.size()) return std::nullopt;
  if (functionType->isVariadic() != targetType->isVariadic())
    return std::nullopt;

  auto paramDeclIt = parameterDeclarations_;
  auto targetIt = targetParams.begin();
  for (auto param : params) {
    auto targetParamType = *targetIt;

    if (!deduceTypeFromType(param, targetParamType)) return std::nullopt;
    if (!deduceFromClassTemplateParam(
            paramDeclIt ? paramDeclIt->value : nullptr, targetParamType, param))
      return std::nullopt;

    ++targetIt;
    if (paramDeclIt) paramDeclIt = paramDeclIt->next;
    beginParameterDeduction();
  }

  if (!checkDeducedArguments()) return std::nullopt;

  return buildTemplateArgumentList();
}

void TemplateArgumentDeduction::collectTemplateParameters(
    TemplateDeclarationAST* templateDecl) {
  templateParams_.clear();

  int position = 0;

  for (auto p : ListView{templateDecl->templateParameterList}) {
    TemplateParameterInfo info;
    info.depth = templateDecl->depth;
    info.index = position++;

    if (auto sym = p->symbol) {
      if (auto paramInfo = template_parameter_info(sym)) {
        info.depth = paramInfo->depth;
        info.index = paramInfo->index;
      }

      info.typeParameterType = type_cast<TypeParameterType>(sym->type());

      if (auto typeParam = symbol_cast<TypeParameterSymbol>(sym)) {
        (void)typeParam;
        info.kind = TemplateParameterInfo::Kind::kType;
        if (info.typeParameterType)
          info.isPack = info.typeParameterType->isParameterPack();
      } else if (auto nonTypeParam = symbol_cast<NonTypeParameterSymbol>(sym)) {
        info.kind = TemplateParameterInfo::Kind::kNonType;
        info.isPack = nonTypeParam->isParameterPack();
      } else if (auto constraintParam =
                     symbol_cast<ConstraintTypeParameterSymbol>(sym)) {
        info.kind = TemplateParameterInfo::Kind::kConstraint;
        info.isPack = constraintParam->isParameterPack();
      } else if (symbol_cast<TemplateTypeParameterSymbol>(sym)) {
        info.kind = TemplateParameterInfo::Kind::kTemplate;
      }
    }

    if (auto t = ast_cast<TypenameTypeParameterAST>(p)) {
      info.isPack = info.isPack || t->isPack;
      info.hasDefault = t->typeId && t->typeId->type;
    } else if (auto n = ast_cast<NonTypeTemplateParameterAST>(p)) {
      info.isPack = info.isPack || (n->declaration && n->declaration->isPack);
      info.hasDefault = n->declaration && n->declaration->expression;
    } else if (auto c = ast_cast<ConstraintTypeParameterAST>(p)) {
      info.isPack = info.isPack || static_cast<bool>(c->ellipsisLoc);
      info.hasDefault = c->typeId && c->typeId->type;
    } else if (auto tt = ast_cast<TemplateTypeParameterAST>(p)) {
      info.isPack = info.isPack || tt->isPack;
      info.hasDefault = tt->idExpression != nullptr;
    }

    templateParams_.push_back(info);
    templateParams_.back().parameterAST = p;
  }

  auto n = templateParams_.size();
  explicitParamArg_.assign(n, nullptr);
  explicitPackArgs_.assign(n, {});
  deducedTypes_.assign(n, nullptr);
  deducedValues_.assign(n, std::nullopt);
  deducedPacks_.assign(n, {});
  packElementCursor_.assign(n, 0);
  deducedValuePacks_.assign(n, {});
}

auto TemplateArgumentDeduction::parameterSlot(int depth, int index) const
    -> int {
  for (std::size_t slot = 0; slot < templateParams_.size(); ++slot) {
    const auto& info = templateParams_[slot];
    if (info.depth == depth && info.index == index) return int(slot);
  }
  return -1;
}

auto TemplateArgumentDeduction::parameterSlot(
    const TypeParameterType* type) const -> int {
  if (!type) return -1;
  return parameterSlot(type->depth(), type->index());
}

auto TemplateArgumentDeduction::substituteExplicitTemplateArguments(
    List<TemplateArgumentAST*>* explicitTemplateArgs) -> bool {
  std::vector<TemplateArgumentAST*> explicitArgs;
  for (auto arg : ListView{explicitTemplateArgs}) {
    explicitArgs.push_back(arg);
  }

  int explicitIndex = 0;
  for (int i = 0; i < static_cast<int>(templateParams_.size()); ++i) {
    if (explicitIndex >= static_cast<int>(explicitArgs.size())) break;

    if (templateParams_[i].isPack) {
      while (explicitIndex < static_cast<int>(explicitArgs.size())) {
        explicitPackArgs_[i].push_back(explicitArgs[explicitIndex]);
        ++explicitIndex;
      }
      break;
    }

    explicitParamArg_[i] = explicitArgs[explicitIndex];
    ++explicitIndex;
  }

  return explicitIndex == static_cast<int>(explicitArgs.size());
}

auto TemplateArgumentDeduction::isExplicitArgumentCompatible(
    const TemplateParameterInfo& info, TemplateArgumentAST* arg) -> bool {
  if (!arg) return false;

  switch (info.kind) {
    case TemplateParameterInfo::Kind::kType:
    case TemplateParameterInfo::Kind::kTemplate:
    case TemplateParameterInfo::Kind::kConstraint: {
      auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg);
      return typeArg && typeArg->typeId && typeArg->typeId->type;
    }

    case TemplateParameterInfo::Kind::kNonType: {
      auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg);
      return exprArg && exprArg->expression;
    }

    case TemplateParameterInfo::Kind::kUnknown:
      return false;
  }

  return false;
}

auto TemplateArgumentDeduction::isForwardingReference(const Type* paramType)
    -> bool {
  auto rrefParam = type_cast<RvalueReferenceType>(paramType);
  if (!rrefParam) return false;

  auto paramTpt = type_cast<TypeParameterType>(rrefParam->elementType());
  if (!paramTpt) return false;

  return true;
}

auto TemplateArgumentDeduction::deduceTypeFromType(const Type* P, const Type* A)
    -> bool {
  auto bareParam = traits.remove_cvref(P);
  auto tpt = type_cast<TypeParameterType>(bareParam);

  auto bareArg = traits.remove_cv(traits.remove_reference(A));

  if (!tpt) {
    if (auto ptrParam = type_cast<PointerType>(traits.remove_cv(bareParam))) {
      CvQualifiers cvP = CvQualifiers::kNone;
      const Type* paramElemBase = ptrParam->elementType();
      if (auto qual = type_cast<QualType>(paramElemBase)) {
        cvP = qual->cvQualifiers();
        paramElemBase = qual->elementType();
      }

      if (auto elemTpt = type_cast<TypeParameterType>(paramElemBase)) {
        const Type* argElemType = nullptr;
        if (auto ptrArg = type_cast<PointerType>(
                traits.remove_cv(traits.remove_reference(A)))) {
          argElemType = ptrArg->elementType();
        }

        if (argElemType) {
          CvQualifiers cvA = CvQualifiers::kNone;
          const Type* argElemBase = argElemType;
          if (auto qual = type_cast<QualType>(argElemType)) {
            cvA = qual->cvQualifiers();
            argElemBase = qual->elementType();
          }

          CvQualifiers cvT = CvQualifiers::kNone;
          if (is_const(cvA) && !is_const(cvP)) cvT = cvT | CvQualifiers::kConst;
          if (is_volatile(cvA) && !is_volatile(cvP))
            cvT = cvT | CvQualifiers::kVolatile;

          const Type* deducedT = cvT != CvQualifiers::kNone
                                     ? control_->getQualType(argElemBase, cvT)
                                     : argElemBase;

          auto idx = parameterSlot(elemTpt);
          if (idx >= 0) {
            if (!deducedTypes_[idx]) {
              deducedTypes_[idx] = deducedT;
            } else if (!traits.is_same(deducedTypes_[idx], deducedT)) {
              return false;
            }
            return true;
          }
        }
      }

      if (auto ptrArg = type_cast<PointerType>(bareArg))
        return deduceTypeFromType(ptrParam->elementType(),
                                  ptrArg->elementType());

      return true;
    }

    if (deduceArrayBound(bareParam, bareArg)) return true;

    if (auto fnParam = type_cast<FunctionType>(traits.remove_cv(bareParam))) {
      auto fnArg = type_cast<FunctionType>(bareArg);
      if (!fnArg) return true;
      if (fnParam->isVariadic() != fnArg->isVariadic()) return false;
      if (fnParam->parameterTypes().size() != fnArg->parameterTypes().size())
        return false;
      if (!deduceTypeFromType(fnParam->returnType(), fnArg->returnType()))
        return false;
      auto argParamIt = fnArg->parameterTypes().begin();
      for (auto paramType : fnParam->parameterTypes()) {
        if (!deduceTypeFromType(paramType, *argParamIt)) return false;
        ++argParamIt;
      }
      return true;
    }

    return true;
  }

  auto idx = parameterSlot(tpt);
  if (idx < 0) return false;

  const Type* deducedArg = A;

  if (auto qualP = type_cast<QualType>(traits.remove_reference(P))) {
    auto cvP = qualP->cvQualifiers();

    CvQualifiers cvA = CvQualifiers::kNone;
    const Type* argBase = deducedArg;
    if (auto qualA = type_cast<QualType>(deducedArg)) {
      cvA = qualA->cvQualifiers();
      argBase = qualA->elementType();
    }

    CvQualifiers cvT = CvQualifiers::kNone;
    if (is_const(cvA) && !is_const(cvP)) cvT = cvT | CvQualifiers::kConst;
    if (is_volatile(cvA) && !is_volatile(cvP))
      cvT = cvT | CvQualifiers::kVolatile;

    deducedArg = cvT != CvQualifiers::kNone
                     ? control_->getQualType(argBase, cvT)
                     : argBase;
  }

  if (templateParams_[idx].isPack) {
    if (!explicitPackArgs_[idx].empty()) {
      auto explicitPackIndex = static_cast<int>(deducedPacks_[idx].size());
      if (explicitPackIndex >=
          static_cast<int>(explicitPackArgs_[idx].size())) {
        return false;
      }
      if (!isExplicitArgumentCompatible(
              templateParams_[idx],
              explicitPackArgs_[idx][explicitPackIndex])) {
        return false;
      }

      auto explicitTypeArg = ast_cast<TypeTemplateArgumentAST>(
          explicitPackArgs_[idx][explicitPackIndex]);
      if (!explicitTypeArg || !explicitTypeArg->typeId ||
          !explicitTypeArg->typeId->type) {
        return false;
      }

      if (!traits.is_same(explicitTypeArg->typeId->type, deducedArg)) {
        return false;
      }
    }

    auto& elements = deducedPacks_[idx];
    auto& cursor = packElementCursor_[idx];

    if (cursor < elements.size()) {
      if (!traits.is_same(elements[cursor], deducedArg)) return false;
    } else {
      elements.push_back(deducedArg);
    }

    ++cursor;
    return true;
  }

  if (auto explicitArg = explicitParamArg_[idx]) {
    if (!isExplicitArgumentCompatible(templateParams_[idx], explicitArg))
      return false;

    auto explicitTypeArg = ast_cast<TypeTemplateArgumentAST>(explicitArg);
    if (!explicitTypeArg || !explicitTypeArg->typeId ||
        !explicitTypeArg->typeId->type) {
      return false;
    }

    deducedTypes_[idx] = explicitTypeArg->typeId->type;
    return true;
  }

  if (!deducedTypes_[idx]) {
    deducedTypes_[idx] = deducedArg;
  } else if (!traits.is_same(deducedTypes_[idx], deducedArg)) {
    return false;
  }

  return true;
}

auto TemplateArgumentDeduction::deduceTemplateId(
    SimpleTemplateIdAST* pattern,
    std::span<const TemplateArgument> argumentSpan,
    std::span<TemplateArgumentAST* const> substitutions) -> bool {
  auto arguments = expand_template_arguments(argumentSpan);
  std::size_t argumentIndex = 0;

  auto substitutedType = [&](const Type* type) {
    auto info = getTypeParamInfo(type);
    if (!info || info->index < 0 ||
        info->index >= static_cast<int>(substitutions.size()))
      return type;
    auto argument =
        ast_cast<TypeTemplateArgumentAST>(substitutions[info->index]);
    auto replacement =
        argument ? parameterTemplateArgumentType(argument->typeId) : nullptr;
    return replacement ? replacement : type;
  };

  auto substitutedValueParameter = [&](ExpressionAST*& expression) -> int {
    auto expansion = ast_cast<PackExpansionExpressionAST>(expression);
    if (expansion) expression = expansion->expression;
    auto id = ast_cast<IdExpressionAST>(expression);
    auto parameter =
        id ? symbol_cast<NonTypeParameterSymbol>(id->symbol) : nullptr;
    auto info = template_parameter_info(parameter);
    if (!info) return -1;
    if (substitutions.empty()) return nonTypeParameterIndex(expression);
    if (info->index < 0 ||
        info->index >= static_cast<int>(substitutions.size()))
      return -1;
    auto replacement =
        ast_cast<ExpressionTemplateArgumentAST>(substitutions[info->index]);
    if (!replacement) return -1;
    expression = replacement->expression;
    if (auto replacementExpansion =
            ast_cast<PackExpansionExpressionAST>(expression))
      expression = replacementExpansion->expression;
    return nonTypeParameterIndex(expression);
  };

  for (auto patternArgument : ListView{pattern->templateArgumentList}) {
    if (auto typeArgument =
            ast_cast<TypeTemplateArgumentAST>(patternArgument)) {
      auto patternType = parameterTemplateArgumentType(typeArgument->typeId);
      if (!patternType) {
        if (argumentIndex >= arguments.size()) return false;
        ++argumentIndex;
        continue;
      }
      patternType = substitutedType(patternType);
      auto info = getTypeParamInfo(patternType);
      const auto end =
          info && info->isPack ? arguments.size() : argumentIndex + 1;
      for (; argumentIndex < end; ++argumentIndex) {
        if (argumentIndex >= arguments.size()) return false;
        auto argumentType = template_argument_type(arguments[argumentIndex]);
        if (!argumentType) return false;
        bool matchedTemplateId = false;
        for (auto specifier :
             ListView{typeArgument->typeId->typeSpecifierList}) {
          auto named = ast_cast<NamedTypeSpecifierAST>(specifier);
          auto nested =
              named ? ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)
                    : nullptr;
          if (!nested) continue;
          auto argumentClass =
              type_cast<ClassType>(traits.remove_cvref(argumentType));
          if (!argumentClass) return false;
          auto patternTemplate = symbol_cast<ClassSymbol>(nested->symbol);
          auto argumentTemplate = argumentClass->symbol();
          if (patternTemplate && patternTemplate->isSpecialization())
            patternTemplate = patternTemplate->primaryTemplateSymbol();
          if (argumentTemplate->isSpecialization())
            argumentTemplate = argumentTemplate->primaryTemplateSymbol();
          if (patternTemplate && patternTemplate != argumentTemplate)
            return false;
          if (!deduceTemplateId(nested,
                                argumentClass->symbol()->templateArguments(),
                                substitutions))
            return false;
          matchedTemplateId = true;
        }
        if (matchedTemplateId) continue;
        if (getTypeParamInfo(patternType)) {
          if (!deduceTypeFromType(patternType, argumentType)) return false;
        } else if (!traits.is_same(patternType, argumentType)) {
          return false;
        }
      }
      continue;
    }

    auto expressionArgument =
        ast_cast<ExpressionTemplateArgumentAST>(patternArgument);
    if (!expressionArgument) return false;
    auto expression = expressionArgument->expression;
    const auto sourceExpansion =
        ast_cast<PackExpansionExpressionAST>(expression) != nullptr;
    auto parameterIndex = substitutedValueParameter(expression);
    if (parameterIndex < 0) {
      if (argumentIndex >= arguments.size()) return false;
      auto patternValue = ASTInterpreter{unit_}.evaluate(expression);
      auto argumentValue = template_argument_value(arguments[argumentIndex]);
      if (patternValue && (!argumentValue || *patternValue != *argumentValue))
        return false;
      ++argumentIndex;
      continue;
    }
    const auto isPack =
        sourceExpansion || templateParams_[parameterIndex].isPack;
    const auto end = isPack ? arguments.size() : argumentIndex + 1;
    for (; argumentIndex < end; ++argumentIndex) {
      auto value = template_argument_value(arguments[argumentIndex]);
      if (!value || !recordDeducedValue(parameterIndex, *value, isPack))
        return false;
    }
  }

  return argumentIndex == arguments.size();
}

auto TemplateArgumentDeduction::deduceFromCallArgument(const Type* P,
                                                       const Type* A,
                                                       ExpressionAST* argExpr)
    -> bool {
  if (isForwardingReference(P) && argExpr &&
      argExpr->valueCategory == ValueCategory::kLValue) {
    return deduceTypeFromType(
        P, traits.add_lvalue_reference(traits.remove_reference(A)));
  }

  if (traits.is_reference(P))
    return deduceTypeFromType(P, traits.remove_reference(A));

  A = traits.remove_reference(A);

  if (traits.is_array(A) || traits.is_function(A))
    return deduceTypeFromType(P, traits.decay(A));

  return deduceTypeFromType(P, traits.remove_cv(A));
}

void TemplateArgumentDeduction::beginParameterDeduction() {
  packElementCursor_.assign(packElementCursor_.size(), 0);
}

auto TemplateArgumentDeduction::deduceFromCall(const FunctionType* functionType,
                                               List<ExpressionAST*>* args)
    -> bool {
  auto paramIt = functionType->parameterTypes().begin();
  auto paramEnd = functionType->parameterTypes().end();
  auto paramDeclIt = parameterDeclarations_;

  for (auto argIt = args; argIt; argIt = argIt->next) {
    auto argType = argIt->value ? argIt->value->type : nullptr;
    if (!argType) return false;

    if (paramIt == paramEnd) {
      if (functionType->isVariadic()) break;
      return false;
    }

    auto P = *paramIt;

    if (!deduceFromCallArgument(P, argType, argIt->value)) return false;

    if (paramDeclIt &&
        !deduceFromClassTemplateParam(paramDeclIt->value, argType, P))
      return false;

    auto bareParam = traits.remove_cvref(P);
    auto packSlot = parameterSlot(type_cast<TypeParameterType>(bareParam));
    if (packSlot < 0 || !templateParams_[packSlot].isPack) {
      ++paramIt;
      if (paramDeclIt) paramDeclIt = paramDeclIt->next;
      beginParameterDeduction();
    }
  }

  return true;
}

auto TemplateArgumentDeduction::nonTypeParameterIndex(ExpressionAST* expr) const
    -> int {
  auto idExpression = ast_cast<IdExpressionAST>(expr);
  if (!idExpression) return -1;

  auto parameter = symbol_cast<NonTypeParameterSymbol>(idExpression->symbol);
  if (!parameter) return -1;

  auto info = template_parameter_info(parameter);
  const int depth =
      info ? info->depth : (templateDecl_ ? templateDecl_->depth : 0);

  return parameterSlot(depth, parameter->index());
}

auto TemplateArgumentDeduction::deduceArrayBound(const Type* P, const Type* A)
    -> bool {
  auto unresolvedParam =
      type_cast<UnresolvedBoundedArrayType>(traits.remove_cv(P));
  if (!unresolvedParam) return false;

  auto boundedArg = type_cast<BoundedArrayType>(traits.remove_cv(A));
  if (!boundedArg) return false;

  auto index = nonTypeParameterIndex(unresolvedParam->size());
  if (index < 0) return false;

  const auto bound = static_cast<std::uint64_t>(boundedArg->size());

  if (deducedValues_[index] && *deducedValues_[index] != bound) return false;
  deducedValues_[index] = bound;

  return deduceTypeFromType(unresolvedParam->elementType(),
                            boundedArg->elementType());
}

auto TemplateArgumentDeduction::checkDeducedArguments() -> bool {
  for (int i = 0; i < static_cast<int>(templateParams_.size()); ++i) {
    if (templateParams_[i].isPack) continue;
    if (templateParams_[i].hasDefault) continue;
    if (explicitParamArg_[i]) continue;
    if (deducedValues_[i]) continue;
    if (!deducedTypes_[i]) return false;
  }
  return true;
}

auto TemplateArgumentDeduction::collectDeducedSoFar(
    List<TemplateArgumentAST*>* argumentsSoFar)
    -> std::optional<std::vector<TemplateArgument>> {
  if (!argumentsSoFar) return std::vector<TemplateArgument>{};

  SilentDiagnosticsScope silent{unit_};
  auto substitution =
      Substitution::makePartial(unit_, templateDecl_, argumentsSoFar);

  if (!substitution.has_value() || substitution->hadError())
    return std::nullopt;

  return std::move(*substitution).templateArguments();
}

auto TemplateArgumentDeduction::substituteDefaultTypeId(
    TypeIdAST* typeId, const std::vector<TemplateArgument>& arguments)
    -> TypeIdAST* {
  SilentDiagnosticsScope silent{unit_};
  return ASTRewriter::substituteDefaultTypeId(
      unit_, typeId, arguments, templateDecl_->depth, templateDecl_->symbol);
}

auto TemplateArgumentDeduction::substituteDefaultExpression(
    ExpressionAST* expression, const std::vector<TemplateArgument>& arguments)
    -> ExpressionAST* {
  SilentDiagnosticsScope silent{unit_};
  return ASTRewriter::substituteDefaultExpression(unit_, expression, arguments,
                                                  templateDecl_->depth,
                                                  templateDecl_->symbol);
}

auto TemplateArgumentDeduction::recordDeducedValue(int index,
                                                   const ConstValue& value,
                                                   bool isPack) -> bool {
  auto number = std::get_if<std::intmax_t>(&value);
  if (!number) return false;

  auto deduced = static_cast<std::uint64_t>(*number);

  if (isPack) {
    deducedValuePacks_[index].push_back(deduced);
    return true;
  }

  deducedValues_[index] = deduced;
  return true;
}

auto TemplateArgumentDeduction::makeTypePackElement(const Type* elementType)
    -> Symbol* {
  auto element = control_->newTypeAliasSymbol(nullptr, {});
  element->setType(elementType);
  return element;
}

auto TemplateArgumentDeduction::deducedTypeArgument(int parameterIndex) const
    -> const Type* {
  if (parameterIndex < 0 ||
      parameterIndex >= static_cast<int>(templateParams_.size())) {
    return nullptr;
  }

  if (auto deduced = deducedTypes_[parameterIndex]) return deduced;

  auto explicitArg = explicitParamArg_[parameterIndex];
  auto typeArg = ast_cast<TypeTemplateArgumentAST>(explicitArg);
  if (typeArg && typeArg->typeId) return typeArg->typeId->type;

  return nullptr;
}

auto TemplateArgumentDeduction::nonTypeParameterType(int parameterIndex) const
    -> const Type* {
  auto parameterAST = templateParams_[parameterIndex].parameterAST;
  auto nonTypeParam = ast_cast<NonTypeTemplateParameterAST>(parameterAST);
  if (!nonTypeParam || !nonTypeParam->declaration) return nullptr;

  auto declaredType = nonTypeParam->declaration->type;
  if (!declaredType) return nullptr;

  auto declaredParam = getTypeParamInfo(declaredType);
  if (!declaredParam) return declaredType;
  if (templateDecl_ && declaredParam->depth != templateDecl_->depth)
    return declaredType;

  return deducedTypeArgument(declaredParam->index);
}

auto TemplateArgumentDeduction::makeValuePackElement(const ConstValue& value,
                                                     const Type* elementType)
    -> Symbol* {
  auto element = control_->newVariableSymbol(nullptr, {});
  element->setType(elementType);
  element->setConstexpr(true);
  element->setConstValue(value);
  return element;
}

auto TemplateArgumentDeduction::makeExplicitPackElement(
    TemplateArgumentAST* explicitArg, int parameterIndex) -> Symbol* {
  if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(explicitArg)) {
    auto elementType = parameterTemplateArgumentType(typeArg->typeId);
    if (!elementType) return nullptr;
    return makeTypePackElement(elementType);
  }

  auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(explicitArg);
  if (!exprArg || !exprArg->expression) return nullptr;

  auto value = ASTInterpreter{unit_}.evaluate(exprArg->expression);
  if (!value.has_value()) return nullptr;

  auto elementType = nonTypeParameterType(parameterIndex);
  if (!elementType) return nullptr;

  return makeValuePackElement(*value, elementType);
}

auto TemplateArgumentDeduction::makePackArgument(int parameterIndex)
    -> TemplateArgumentAST* {
  auto pack = control_->newParameterPackSymbol(nullptr, {});

  for (auto explicitArg : explicitPackArgs_[parameterIndex]) {
    auto element = makeExplicitPackElement(explicitArg, parameterIndex);
    if (!element) return nullptr;
    pack->addElement(element);
  }

  if (explicitPackArgs_[parameterIndex].empty()) {
    for (auto elementType : deducedPacks_[parameterIndex])
      pack->addElement(makeTypePackElement(elementType));

    if (!deducedValuePacks_[parameterIndex].empty()) {
      auto elementType = nonTypeParameterType(parameterIndex);
      if (!elementType) return nullptr;

      for (auto elementValue : deducedValuePacks_[parameterIndex]) {
        pack->addElement(makeValuePackElement(
            ConstValue{static_cast<std::intmax_t>(elementValue)}, elementType));
      }
    }
  }

  auto namedSpec = NamedTypeSpecifierAST::create(arena_);
  namedSpec->symbol = pack;

  auto typeId = TypeIdAST::create(arena_);
  typeId->typeSpecifierList = make_list_node<SpecifierAST>(arena_, namedSpec);

  auto typeArg = TypeTemplateArgumentAST::create(arena_);
  typeArg->typeId = typeId;

  return typeArg;
}

auto TemplateArgumentDeduction::buildTemplateArgumentList()
    -> std::optional<List<TemplateArgumentAST*>*> {
  List<TemplateArgumentAST*>* templArgList = nullptr;
  auto argListIt = &templArgList;

  for (int i = 0; i < static_cast<int>(templateParams_.size()); ++i) {
    if (templateParams_[i].isPack) {
      if (!explicitPackArgs_[i].empty()) {
        if (!deducedPacks_[i].empty() &&
            deducedPacks_[i].size() != explicitPackArgs_[i].size()) {
          return std::nullopt;
        }

        for (auto explicitArg : explicitPackArgs_[i]) {
          if (!isExplicitArgumentCompatible(templateParams_[i], explicitArg))
            return std::nullopt;
        }
      }

      auto packArgument = makePackArgument(i);
      if (!packArgument) return std::nullopt;

      *argListIt = make_list_node<TemplateArgumentAST>(arena_, packArgument);
      argListIt = &(*argListIt)->next;
      continue;
    }

    if (auto explicitArg = explicitParamArg_[i]) {
      if (!isExplicitArgumentCompatible(templateParams_[i], explicitArg))
        return std::nullopt;
      *argListIt = make_list_node<TemplateArgumentAST>(arena_, explicitArg);
      argListIt = &(*argListIt)->next;
      continue;
    }

    if (!deducedTypes_[i] && deducedValues_[i]) {
      auto literal =
          control_->integerLiteral(std::to_string(*deducedValues_[i]));
      auto value = IntLiteralExpressionAST::create(
          arena_, literal, ValueCategory::kPrValue, control_->getSizeType());

      auto exprArg = ExpressionTemplateArgumentAST::create(arena_);
      exprArg->expression = value;
      *argListIt = make_list_node<TemplateArgumentAST>(arena_, exprArg);
      argListIt = &(*argListIt)->next;
      continue;
    }

    if (!deducedTypes_[i]) {
      if (!templateParams_[i].hasDefault) return std::nullopt;
      auto p = templateParams_[i].parameterAST;
      auto deducedSoFar = collectDeducedSoFar(templArgList);
      if (!deducedSoFar.has_value()) return std::nullopt;
      auto defaultArgument = defaultTemplateArgument(p, *deducedSoFar);
      if (!defaultArgument) return std::nullopt;
      *argListIt = make_list_node<TemplateArgumentAST>(arena_, defaultArgument);
      argListIt = &(*argListIt)->next;
      continue;
    }

    auto typeId = TypeIdAST::create(arena_);
    typeId->type = deducedTypes_[i];
    auto typeArg = TypeTemplateArgumentAST::create(arena_);
    typeArg->typeId = typeId;
    *argListIt = make_list_node<TemplateArgumentAST>(arena_, typeArg);
    argListIt = &(*argListIt)->next;
  }

  return templArgList;
}

auto TemplateArgumentDeduction::defaultTemplateArgument(
    TemplateParameterAST* parameter,
    const std::vector<TemplateArgument>& argumentsSoFar)
    -> TemplateArgumentAST* {
  if (auto nonType = ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    if (!nonType->declaration || !nonType->declaration->expression) {
      return nullptr;
    }
    auto expression = nonType->declaration->expression;
    if (isDependent(unit_, expression) && !argumentsSoFar.empty()) {
      expression = substituteDefaultExpression(expression, argumentsSoFar);
      if (!expression) return nullptr;
    }
    auto argument = ExpressionTemplateArgumentAST::create(arena_);
    argument->expression = expression;
    return argument;
  }

  auto type = ast_cast<TypenameTypeParameterAST>(parameter);
  if (!type || !type->typeId) return nullptr;
  auto typeId = type->typeId;
  if ((!typeId->type || isDependent(unit_, typeId->type)) &&
      !argumentsSoFar.empty() && templateDecl_) {
    auto substituted = substituteDefaultTypeId(typeId, argumentsSoFar);
    if (!substituted || !substituted->type ||
        type_cast<UnresolvedNameType>(substituted->type)) {
      return nullptr;
    }
    typeId = substituted;
  }
  auto argument = TypeTemplateArgumentAST::create(arena_);
  argument->typeId = typeId;
  return argument;
}

auto TemplateArgumentDeduction::getParameterClause(DeclarationAST* decl)
    -> ParameterDeclarationClauseAST* {
  DeclaratorAST* declarator = nullptr;
  if (auto funcDef = ast_cast<FunctionDefinitionAST>(decl))
    declarator = funcDef->declarator;
  else if (auto simpleDecl = ast_cast<SimpleDeclarationAST>(decl))
    if (simpleDecl->initDeclaratorList)
      declarator = simpleDecl->initDeclaratorList->value->declarator;
  if (!declarator) return nullptr;
  for (auto chunk : ListView{declarator->declaratorChunkList})
    if (auto fc = ast_cast<FunctionDeclaratorChunkAST>(chunk))
      return fc->parameterDeclarationClause;
  return nullptr;
}

auto TemplateArgumentDeduction::deduceFromClassTemplateParam(
    ParameterDeclarationAST* paramDecl, const Type* argType, const Type* P)
    -> bool {
  if (!paramDecl) return true;

  auto bareP = traits.remove_cvref(P);
  auto bareA = traits.remove_cvref(argType);

  while (true) {
    auto ptrP = type_cast<PointerType>(bareP);
    auto ptrA = type_cast<PointerType>(bareA);
    if (!ptrP || !ptrA) break;
    bareP = traits.remove_cv(ptrP->elementType());
    bareA = traits.remove_cv(ptrA->elementType());
  }

  auto paramClassType = type_cast<ClassType>(bareP);
  if (!paramClassType) return true;
  auto paramClass = paramClassType->symbol();
  if (!paramClass) return true;
  if (paramClass->isSpecialization())
    paramClass = paramClass->primaryTemplateSymbol();
  if (!paramClass || !paramClass->templateDeclaration()) return true;

  auto argClassType = type_cast<ClassType>(bareA);
  if (!argClassType) return false;
  auto argClass = argClassType->symbol();
  if (!argClass) return false;

  auto findSpecializationOfPrimary = [&](ClassSymbol* cls,
                                         auto&& self) -> ClassSymbol* {
    if (!cls) return nullptr;
    cls = cls->resolvedDefinition();
    if (cls->isSpecialization() && cls->primaryTemplateSymbol() == paramClass) {
      return cls;
    }
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (auto found = self(baseClass, self)) return found;
    }
    return nullptr;
  };

  argClass = findSpecializationOfPrimary(argClass, findSpecializationOfPrimary);
  if (!argClass) return false;

  auto argArgs = expand_template_arguments(argClass->templateArguments());

  for (auto spec : ListView{paramDecl->typeSpecifierList}) {
    auto namedSpec = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!namedSpec) continue;
    auto templateId = ast_cast<SimpleTemplateIdAST>(namedSpec->unqualifiedId);
    if (!templateId) continue;

    auto alias = symbol_cast<TypeAliasSymbol>(templateId->symbol);
    if (!alias) alias = symbol_cast<TypeAliasSymbol>(namedSpec->symbol);
    if (alias) {
      if (alias->isSpecialization()) alias = alias->primaryTemplateSymbol();
      auto declaration = alias->declaration();
      if (!declaration || !declaration->typeId) return false;
      std::vector<TemplateArgumentAST*> substitutions;
      for (auto argument : ListView{templateId->templateArgumentList})
        substitutions.push_back(argument);
      for (auto associatedSpecifier :
           ListView{declaration->typeId->typeSpecifierList}) {
        auto associatedNamed =
            ast_cast<NamedTypeSpecifierAST>(associatedSpecifier);
        auto associatedId =
            associatedNamed
                ? ast_cast<SimpleTemplateIdAST>(associatedNamed->unqualifiedId)
                : nullptr;
        if (associatedId)
          return deduceTemplateId(associatedId, argClass->templateArguments(),
                                  substitutions);
      }
      return false;
    }
    return deduceTemplateId(templateId, argArgs);
  }

  return true;
}
}  // namespace cxx
