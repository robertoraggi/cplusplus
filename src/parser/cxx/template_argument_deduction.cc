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
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

namespace cxx {
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
  }

  if (!checkDeducedArguments()) return std::nullopt;

  return buildTemplateArgumentList();
}

void TemplateArgumentDeduction::collectTemplateParameters(
    TemplateDeclarationAST* templateDecl) {
  templateParams_.clear();

  for (auto p : ListView{templateDecl->templateParameterList}) {
    TemplateParameterInfo info;

    if (auto sym = p->symbol) {
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

          auto idx = elemTpt->index();
          if (idx >= 0 && idx < static_cast<int>(templateParams_.size())) {
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

  auto idx = tpt->index();
  if (idx < 0 || idx >= static_cast<int>(templateParams_.size())) return false;

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

    deducedPacks_[idx].push_back(deducedArg);
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
    auto tpt = type_cast<TypeParameterType>(bareParam);
    if (!tpt || !templateParams_[tpt->index()].isPack) {
      ++paramIt;
      if (paramDeclIt) paramDeclIt = paramDeclIt->next;
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

  auto index = parameter->index();
  if (index < 0 || index >= static_cast<int>(templateParams_.size())) return -1;

  return index;
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

auto TemplateArgumentDeduction::collectDeducedSoFar(int i)
    -> std::optional<std::vector<TemplateArgument>> {
  std::vector<TemplateArgument> deducedSoFar;
  for (int j = 0; j < i; ++j) {
    const Type* argType = deducedTypes_[j];
    if (!argType && explicitParamArg_[j]) {
      if (auto ta = ast_cast<TypeTemplateArgumentAST>(explicitParamArg_[j]);
          ta && ta->typeId) {
        argType = ta->typeId->type;
      }
    }
    if (!argType) return std::nullopt;
    auto alias = control_->newTypeAliasSymbol(nullptr, {});
    alias->setType(argType);
    deducedSoFar.push_back(alias);
  }
  return deducedSoFar;
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
          *argListIt = make_list_node<TemplateArgumentAST>(arena_, explicitArg);
          argListIt = &(*argListIt)->next;
        }
        continue;
      }

      for (auto& packType : deducedPacks_[i]) {
        auto typeId = TypeIdAST::create(arena_);
        typeId->type = packType;
        auto typeArg = TypeTemplateArgumentAST::create(arena_);
        typeArg->typeId = typeId;
        *argListIt = make_list_node<TemplateArgumentAST>(arena_, typeArg);
        argListIt = &(*argListIt)->next;
      }
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
      if (auto n = ast_cast<NonTypeTemplateParameterAST>(p)) {
        if (!n->declaration || !n->declaration->expression) return std::nullopt;

        if (auto declaredType = n->declaration->type;
            declaredType && isDependent(unit_, declaredType) && templateDecl_) {
          if (auto deducedSoFar = collectDeducedSoFar(i);
              deducedSoFar && !deducedSoFar->empty()) {
            auto typeId = TypeIdAST::create(arena_);
            typeId->typeSpecifierList = n->declaration->typeSpecifierList;
            typeId->declarator = n->declaration->declarator;

            SilentDiagnosticsClient silent;
            auto saved = unit_->changeDiagnosticsClient(&silent);
            auto substituted = ASTRewriter::substituteDefaultTypeId(
                unit_, typeId, *deducedSoFar, templateDecl_->depth,
                templateDecl_->symbol);
            (void)unit_->changeDiagnosticsClient(saved);

            if (!substituted || !substituted->type ||
                type_cast<UnresolvedNameType>(substituted->type)) {
              return std::nullopt;
            }
          }
        }

        auto exprArg = ExpressionTemplateArgumentAST::create(arena_);
        exprArg->expression = n->declaration->expression;
        *argListIt = make_list_node<TemplateArgumentAST>(arena_, exprArg);
        argListIt = &(*argListIt)->next;
      } else if (auto t = ast_cast<TypenameTypeParameterAST>(p)) {
        if (!t->typeId) return std::nullopt;
        auto typeId = t->typeId;
        if (!typeId->type || isDependent(unit_, typeId->type)) {
          if (auto deducedSoFar = collectDeducedSoFar(i);
              deducedSoFar && !deducedSoFar->empty() && templateDecl_) {
            SilentDiagnosticsClient silent;
            auto saved = unit_->changeDiagnosticsClient(&silent);
            auto substituted = ASTRewriter::substituteDefaultTypeId(
                unit_, typeId, *deducedSoFar, templateDecl_->depth,
                templateDecl_->symbol);
            (void)unit_->changeDiagnosticsClient(saved);
            if (substituted && substituted->type &&
                !isDependent(unit_, substituted->type)) {
              typeId = substituted;
            }
          }
        }
        auto typeArg = TypeTemplateArgumentAST::create(arena_);
        typeArg->typeId = typeId;
        *argListIt = make_list_node<TemplateArgumentAST>(arena_, typeArg);
        argListIt = &(*argListIt)->next;
      }
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
  if (!paramClass || !paramClass->templateDeclaration() ||
      paramClass->isSpecialization())
    return true;

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

  auto argArgs = argClass->templateArguments();

  for (auto spec : ListView{paramDecl->typeSpecifierList}) {
    auto namedSpec = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!namedSpec) continue;
    auto templateId = ast_cast<SimpleTemplateIdAST>(namedSpec->unqualifiedId);
    if (!templateId) continue;

    int i = 0;
    for (auto arg : ListView{templateId->templateArgumentList}) {
      if (i >= static_cast<int>(argArgs.size())) break;
      auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg);
      if (!typeArg || !typeArg->typeId) {
        ++i;
        continue;
      }

      const Type* pType = typeArg->typeId->type;
      if (!pType) {
        for (auto s : ListView{typeArg->typeId->typeSpecifierList}) {
          if (auto ns = ast_cast<NamedTypeSpecifierAST>(s)) {
            if (ns->symbol) {
              pType = ns->symbol->type();
              break;
            }
          }
        }
      }

      const Type* aType = nullptr;
      if (auto sym = std::get_if<Symbol*>(&argArgs[i]))
        aType = (*sym)->type();
      else if (auto t = std::get_if<const Type*>(&argArgs[i]))
        aType = *t;

      if (pType && aType) (void)deduceTypeFromType(pType, aType);
      ++i;
    }
    return true;
  }

  return true;
}
}  // namespace cxx
