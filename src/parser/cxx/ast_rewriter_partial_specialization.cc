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
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <functional>
#include <map>
#include <span>

namespace cxx {
namespace {
struct DeducedArguments {
  std::vector<Symbol*> values;

  explicit DeducedArguments(size_t size) : values(size, nullptr) {}

  auto set(int pos, Symbol* symbol) -> bool {
    if (pos < 0 || pos >= static_cast<int>(values.size())) return false;
    values[pos] = symbol;
    return true;
  }

  [[nodiscard]] auto get(int pos) const -> Symbol* {
    if (pos < 0 || pos >= static_cast<int>(values.size())) return nullptr;
    return values[pos];
  }

  [[nodiscard]] auto complete() const -> bool {
    for (auto value : values) {
      if (!value) return false;
    }
    return true;
  }

  [[nodiscard]] auto toTemplateArguments() const
      -> std::vector<TemplateArgument> {
    std::vector<TemplateArgument> result;
    result.reserve(values.size());
    for (auto value : values) {
      result.push_back(value);
    }
    return result;
  }
};

struct NestedTemplatePattern {
  SimpleTemplateIdAST* root = nullptr;
  std::map<const SimpleTemplateIdAST*, std::vector<SimpleTemplateIdAST*>>
      childrenByTemplateId;

  [[nodiscard]] auto child(const SimpleTemplateIdAST* id, size_t argPos) const
      -> SimpleTemplateIdAST* {
    if (!id) return nullptr;
    auto it = childrenByTemplateId.find(id);
    if (it == childrenByTemplateId.end()) return nullptr;
    if (argPos >= it->second.size()) return nullptr;
    return it->second[argPos];
  }
};

auto findTemplateIdInTypeId(TypeIdAST* typeId) -> SimpleTemplateIdAST* {
  if (!typeId) return nullptr;
  for (auto sp : ListView{typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(sp);
    if (!named) continue;
    if (auto templId = ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)) {
      return templId;
    }
  }
  return nullptr;
}

auto extractDirectNestedTemplateIds(SimpleTemplateIdAST* templId)
    -> std::vector<SimpleTemplateIdAST*> {
  std::vector<SimpleTemplateIdAST*> nested;
  for (auto arg : ListView{templId->templateArgumentList}) {
    auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg);
    if (!typeArg || !typeArg->typeId) {
      nested.push_back(nullptr);
      continue;
    }

    nested.push_back(findTemplateIdInTypeId(typeArg->typeId));
  }
  return nested;
}

void buildNestedTemplatePattern(SimpleTemplateIdAST* templId,
                                NestedTemplatePattern& pattern) {
  if (!templId) return;
  if (pattern.childrenByTemplateId.contains(templId)) return;

  auto direct = extractDirectNestedTemplateIds(templId);
  pattern.childrenByTemplateId.emplace(templId, direct);

  for (auto nested : direct) {
    buildNestedTemplatePattern(nested, pattern);
  }
}

auto extractNestedTemplatePattern(ClassSpecifierAST* specBody)
    -> std::optional<NestedTemplatePattern> {
  auto root = ast_cast<SimpleTemplateIdAST>(specBody->unqualifiedId);
  if (!root) return std::nullopt;

  NestedTemplatePattern pattern;
  pattern.root = root;
  buildNestedTemplatePattern(root, pattern);
  return pattern;
}

auto asSymbolArgument(const TemplateArgument& argument) -> Symbol* {
  auto symbol = std::get_if<Symbol*>(&argument);
  if (!symbol) return nullptr;
  return *symbol;
}

struct PartialSpecMatcher {
  TranslationUnit* unit = nullptr;
  const NestedTemplatePattern* pattern = nullptr;
  DeducedArguments& deducedArgs;
  std::function<int(int depth, int index)> paramPosition;
  int score = 0;
  int packMatches = 0;
  int exactTypeMatches = 0;
  int nestedMatches = 0;
  int nonTypeMatches = 0;
  int deducedParamMatches = 0;

  [[nodiscard]] auto control() const -> Control* { return unit->control(); }

  auto sameArgument(Symbol* lhs, Symbol* rhs) const -> bool {
    if (lhs == rhs) return true;
    if (!lhs || !rhs) return false;

    auto lhsPack = symbol_cast<ParameterPackSymbol>(lhs);
    auto rhsPack = symbol_cast<ParameterPackSymbol>(rhs);
    if (lhsPack && rhsPack) {
      const auto& lhsElements = lhsPack->elements();
      const auto& rhsElements = rhsPack->elements();
      if (lhsElements.size() != rhsElements.size()) return false;
      for (size_t i = 0; i < lhsElements.size(); ++i) {
        if (!sameArgument(lhsElements[i], rhsElements[i])) return false;
      }
      return true;
    }

    auto lhsVar = symbol_cast<VariableSymbol>(lhs);
    auto rhsVar = symbol_cast<VariableSymbol>(rhs);
    if (lhsVar && rhsVar && lhsVar->constValue().has_value() &&
        rhsVar->constValue().has_value()) {
      return lhsVar->constValue().value() == rhsVar->constValue().value();
    }

    auto lhsType = lhs->type();
    auto rhsType = rhs->type();
    if (lhsType && rhsType && unit->typeTraits().is_same(lhsType, rhsType)) {
      return true;
    }

    return false;
  }

  auto deduceOrCheck(int pos, Symbol* newSymbol, bool countAsParamMatch)
      -> bool {
    if (pos < 0) return true;
    if (!newSymbol) return false;

    auto existingSymbol = deducedArgs.get(pos);
    if (!existingSymbol) {
      deducedArgs.set(pos, newSymbol);
      if (countAsParamMatch) ++deducedParamMatches;
      return true;
    }

    return sameArgument(existingSymbol, newSymbol);
  }

  auto deduceArgumentList(const std::vector<Symbol*>& patArgs,
                          const std::vector<Symbol*>& concArgs,
                          const SimpleTemplateIdAST* patTemplId,
                          size_t writtenBase) -> bool {
    for (size_t patIdx = 0; patIdx < patArgs.size(); ++patIdx) {
      auto patArg = patArgs[patIdx];
      if (!patArg) return false;

      auto patInfo = template_parameter_info(patArg);

      if (patInfo && patInfo->isPack) {
        if (patIdx + 1 != patArgs.size()) return false;

        auto deducedPack = control()->newParameterPackSymbol(nullptr, {});
        for (size_t concIdx = patIdx; concIdx < concArgs.size(); ++concIdx) {
          if (!concArgs[concIdx]) return false;
          deducedPack->addElement(concArgs[concIdx]);
        }

        return deduceOrCheck(paramPosition(patInfo->depth, patInfo->index),
                             deducedPack, /*countAsParamMatch=*/false);
      }

      if (patIdx >= concArgs.size()) return false;

      if (!deduceArgument(patArg, concArgs[patIdx], patTemplId,
                          writtenBase + patIdx)) {
        return false;
      }
    }

    return patArgs.size() == concArgs.size();
  }

  auto deduceArgument(Symbol* patSym, Symbol* concSym,
                      const SimpleTemplateIdAST* patTemplId, size_t argPos)
      -> bool {
    if (!patSym || !concSym) return false;

    if (auto patInfo = template_parameter_info(patSym)) {
      return deduceOrCheck(paramPosition(patInfo->depth, patInfo->index),
                           concSym, /*countAsParamMatch=*/true);
    }

    auto patPack = symbol_cast<ParameterPackSymbol>(patSym);
    auto concPack = symbol_cast<ParameterPackSymbol>(concSym);
    if (patPack && concPack) {
      if (!deduceArgumentList(patPack->elements(), concPack->elements(),
                              patTemplId, argPos)) {
        return false;
      }
      ++packMatches;
      return true;
    }

    auto patType = patSym->type();
    auto concType = concSym->type();
    if (!patType || !concType) {
      if (patType != concType) return false;
      ++score;
      ++exactTypeMatches;
      return true;
    }

    if (type_cast<UnresolvedNameType>(patType)) return true;

    auto patVar = symbol_cast<VariableSymbol>(patSym);
    auto concVar = symbol_cast<VariableSymbol>(concSym);
    if (patVar && concVar &&
        (patVar->constValue().has_value() ||
         concVar->constValue().has_value())) {
      if (!concVar->constValue().has_value()) return false;
      if (!patVar->constValue().has_value()) return true;

      if (patVar->constValue().value() != concVar->constValue().value()) {
        return false;
      }

      ++score;
      ++nonTypeMatches;
      return true;
    }

    return deduceType(patType, concType,
                      pattern ? pattern->child(patTemplId, argPos) : nullptr);
  }

  auto matchArg(const TemplateArgument& pat, const TemplateArgument& conc,
                size_t argPos) -> bool {
    return deduceArgument(asSymbolArgument(pat), asSymbolArgument(conc),
                          pattern ? pattern->root : nullptr, argPos);
  }

  auto deduceType(const Type* patType, const Type* concType,
                  SimpleTemplateIdAST* patTemplId) -> bool {
    if (!patType || !concType) return false;

    if (auto patParamInfo = getTypeParamInfo(patType)) {
      auto pos = paramPosition(patParamInfo->depth, patParamInfo->index);
      auto argument = control()->newTypeAliasSymbol(nullptr, {});
      argument->setType(concType);
      return deduceOrCheck(pos, argument, /*countAsParamMatch=*/true);
    }

    if (auto patQual = type_cast<QualType>(patType)) {
      auto concQual = type_cast<QualType>(concType);
      if (!concQual) return false;
      if (!cv_is_subset_of(patQual->cvQualifiers(), concQual->cvQualifiers()))
        return false;

      auto remainder = CvQualifiers::kNone;
      if (is_const(concQual->cvQualifiers()) &&
          !is_const(patQual->cvQualifiers()))
        remainder = remainder | CvQualifiers::kConst;
      if (is_volatile(concQual->cvQualifiers()) &&
          !is_volatile(patQual->cvQualifiers()))
        remainder = remainder | CvQualifiers::kVolatile;

      const Type* concElement = concQual->elementType();
      if (remainder != CvQualifiers::kNone)
        concElement = control()->getQualType(concElement, remainder);

      ++score;
      return deduceType(patQual->elementType(), concElement, patTemplId);
    }

    if (auto patPointer = type_cast<PointerType>(patType)) {
      auto concPointer = type_cast<PointerType>(concType);
      if (!concPointer) return false;
      ++score;
      return deduceType(patPointer->elementType(), concPointer->elementType(),
                        patTemplId);
    }

    if (auto patRef = type_cast<LvalueReferenceType>(patType)) {
      auto concRef = type_cast<LvalueReferenceType>(concType);
      if (!concRef) return false;
      ++score;
      return deduceType(patRef->elementType(), concRef->elementType(),
                        patTemplId);
    }

    if (auto patRef = type_cast<RvalueReferenceType>(patType)) {
      auto concRef = type_cast<RvalueReferenceType>(concType);
      if (!concRef) return false;
      ++score;
      return deduceType(patRef->elementType(), concRef->elementType(),
                        patTemplId);
    }

    if (auto patArray = type_cast<BoundedArrayType>(patType)) {
      auto concArray = type_cast<BoundedArrayType>(concType);
      if (!concArray) return false;
      if (patArray->size() != concArray->size()) return false;
      ++score;
      return deduceType(patArray->elementType(), concArray->elementType(),
                        patTemplId);
    }

    if (auto patArray = type_cast<UnboundedArrayType>(patType)) {
      auto concArray = type_cast<UnboundedArrayType>(concType);
      if (!concArray) return false;
      ++score;
      return deduceType(patArray->elementType(), concArray->elementType(),
                        patTemplId);
    }

    if (auto patArray = type_cast<UnresolvedBoundedArrayType>(patType)) {
      auto concArray = type_cast<BoundedArrayType>(concType);
      if (!concArray) return false;

      auto idExpr = ast_cast<IdExpressionAST>(patArray->size());
      if (!idExpr) return false;

      auto nttp = symbol_cast<NonTypeParameterSymbol>(idExpr->symbol);
      if (!nttp) return false;

      auto pos = paramPosition(nttp->depth(), nttp->index());

      auto value = control()->newVariableSymbol(nullptr, {});
      value->setType(nttp->objectType());
      value->setConstexpr(true);
      value->setConstValue(
          ConstValue(static_cast<std::intmax_t>(concArray->size())));

      if (!deduceOrCheck(pos, value, /*countAsParamMatch=*/true)) return false;

      ++score;
      return deduceType(patArray->elementType(), concArray->elementType(),
                        patTemplId);
    }

    if (auto patFunction = type_cast<FunctionType>(patType)) {
      auto concFunction = type_cast<FunctionType>(concType);
      if (!concFunction) return false;
      if (patFunction->isVariadic() != concFunction->isVariadic()) return false;
      if (patFunction->cvQualifiers() != concFunction->cvQualifiers())
        return false;
      if (patFunction->refQualifier() != concFunction->refQualifier())
        return false;

      const auto& patParams = patFunction->parameterTypes();
      const auto& concParams = concFunction->parameterTypes();
      if (patParams.size() != concParams.size()) return false;

      if (!deduceType(patFunction->returnType(), concFunction->returnType(),
                      patTemplId))
        return false;

      for (size_t i = 0; i < patParams.size(); ++i) {
        if (!deduceType(patParams[i], concParams[i], patTemplId)) return false;
      }

      ++score;
      return true;
    }

    if (auto patClassType = type_cast<ClassType>(patType)) {
      if (auto concClassType = type_cast<ClassType>(concType)) {
        auto patClassSym = patClassType->symbol();
        auto concClassSym = concClassType->symbol();
        if (!patClassSym || !concClassSym) return false;

        if (concClassSym->isSpecialization() &&
            concClassSym->primaryTemplateSymbol() == patClassSym &&
            patClassSym->templateDeclaration()) {
          if (!patTemplId) return false;

          if (!matchNestedWithPattern(patTemplId, patClassSym,
                                      concClassSym->templateArguments())) {
            return false;
          }

          ++score;
          ++nestedMatches;
          return true;
        }

        if (patClassSym->isSpecialization() &&
            concClassSym->isSpecialization() &&
            patClassSym->primaryTemplateSymbol() ==
                concClassSym->primaryTemplateSymbol()) {
          auto primary = patClassSym->primaryTemplateSymbol();
          if (!primary->templateDeclaration()) return false;
          if (!patTemplId) return false;

          if (!matchNestedWithPattern(patTemplId, primary,
                                      concClassSym->templateArguments())) {
            return false;
          }

          ++score;
          ++nestedMatches;
          return true;
        }
      }
    }

    if (patType != concType && !unit->typeTraits().is_same(patType, concType)) {
      return false;
    }

    ++score;
    ++exactTypeMatches;
    return true;
  }

 private:
  static auto extractClassSymbol(const TemplateArgument& arg) -> ClassSymbol* {
    const Type* type = nullptr;

    if (auto sym = std::get_if<Symbol*>(&arg)) {
      type = *sym ? (*sym)->type() : nullptr;
    } else if (auto tp = std::get_if<const Type*>(&arg)) {
      type = *tp;
    }

    if (!type) return nullptr;

    auto classType = type_cast<ClassType>(type);
    if (!classType) return nullptr;

    return classType->symbol();
  }

  auto remapAliasParameter(Symbol* symbol, int aliasDepth,
                           const std::vector<TemplateArgument>& aliasArguments)
      -> Symbol* {
    if (!symbol) return symbol;

    if (auto pack = symbol_cast<ParameterPackSymbol>(symbol)) {
      auto remapped = control()->newParameterPackSymbol(nullptr, {});
      for (auto element : pack->elements()) {
        auto mapped = remapAliasParameter(element, aliasDepth, aliasArguments);
        if (auto mappedPack = symbol_cast<ParameterPackSymbol>(mapped)) {
          for (auto mappedElement : mappedPack->elements())
            remapped->addElement(mappedElement);
        } else if (mapped) {
          remapped->addElement(mapped);
        }
      }
      return remapped;
    }

    auto info = template_parameter_info(symbol);
    if (!info || info->depth != aliasDepth) return symbol;
    if (info->index < 0 ||
        info->index >= static_cast<int>(aliasArguments.size())) {
      return symbol;
    }
    return asSymbolArgument(aliasArguments[info->index]);
  }

  auto aliasExpandedArguments(SimpleTemplateIdAST* patTemplId,
                              ClassSymbol* primarySym)
      -> std::optional<std::vector<Symbol*>> {
    auto alias = symbol_cast<TypeAliasSymbol>(patTemplId->symbol);
    if (!alias) return std::nullopt;

    auto aliasTemplateDecl = alias->templateDeclaration();
    if (!aliasTemplateDecl) return std::nullopt;

    auto aliasDeclaration =
        ast_cast<AliasDeclarationAST>(aliasTemplateDecl->declaration);
    if (!aliasDeclaration) return std::nullopt;

    auto underlying = findTemplateIdInTypeId(aliasDeclaration->typeId);
    if (!underlying) return std::nullopt;

    auto aliasArguments =
        Substitution(unit, aliasTemplateDecl, patTemplId->templateArgumentList)
            .templateArguments();

    auto underlyingArgs = Substitution(unit, primarySym->templateDeclaration(),
                                       underlying->templateArgumentList)
                              .templateArguments();

    std::vector<Symbol*> result;
    result.reserve(underlyingArgs.size());
    for (const auto& arg : underlyingArgs) {
      result.push_back(remapAliasParameter(
          asSymbolArgument(arg), aliasTemplateDecl->depth, aliasArguments));
    }
    return result;
  }

  auto matchNestedWithPattern(SimpleTemplateIdAST* patTemplId,
                              ClassSymbol* primarySym,
                              std::span<const TemplateArgument> concArgs)
      -> bool {
    std::vector<Symbol*> patSymbols;

    if (auto expanded = aliasExpandedArguments(patTemplId, primarySym)) {
      patSymbols = std::move(*expanded);
    } else {
      auto patInnerArgs = Substitution(unit, primarySym->templateDeclaration(),
                                       patTemplId->templateArgumentList)
                              .templateArguments();
      patSymbols.reserve(patInnerArgs.size());
      for (const auto& arg : patInnerArgs) {
        patSymbols.push_back(asSymbolArgument(arg));
      }
    }

    std::vector<Symbol*> concSymbols;
    concSymbols.reserve(concArgs.size());
    for (const auto& arg : concArgs) {
      concSymbols.push_back(asSymbolArgument(arg));
    }

    return deduceArgumentList(patSymbols, concSymbols, patTemplId, 0);
  }
};

struct Candidate {
  ClassSymbol* specClass = nullptr;
  VariableSymbol* specVar = nullptr;
  TemplateDeclarationAST* specTemplateDecl = nullptr;
  ClassSpecifierAST* specBody = nullptr;
  int declarationOrder = -1;
  int packParameterCount = 0;
  std::vector<TemplateArgument> deducedArgs;
  int score = -1;
  int packMatches = 0;
  int exactTypeMatches = 0;
  int nestedMatches = 0;
  int nonTypeMatches = 0;
  int deducedParamMatches = 0;
};

auto makeParamPosition(TemplateDeclarationAST* templateDecl)
    -> std::pair<int, std::function<int(int depth, int index)>> {
  int paramCount = 0;
  std::map<std::pair<int, int>, int> paramPositionMap;

  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    paramPositionMap[{parameter->depth, parameter->index}] = paramCount;
    ++paramCount;
  }

  auto position = [map = std::move(paramPositionMap)](int depth,
                                                      int index) -> int {
    auto it = map.find({depth, index});
    if (it == map.end()) return -1;
    return it->second;
  };

  return {paramCount, std::move(position)};
}

auto countPackTemplateParameters(TemplateDeclarationAST* templateDecl) -> int {
  int count = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
      if (typeParameter->isPack) ++count;
      continue;
    }

    if (auto nonTypeParameter =
            ast_cast<NonTypeTemplateParameterAST>(parameter)) {
      if (nonTypeParameter->declaration &&
          nonTypeParameter->declaration->isPack)
        ++count;
      continue;
    }

    if (auto templateTypeParameter =
            ast_cast<TemplateTypeParameterAST>(parameter)) {
      if (templateTypeParameter->isPack) ++count;
      continue;
    }

    if (auto constraintParameter =
            ast_cast<ConstraintTypeParameterAST>(parameter)) {
      if (constraintParameter->ellipsisLoc) ++count;
      continue;
    }
  }
  return count;
}

auto reevaluateCollapsedExpressionArgument(
    TranslationUnit* unit, ExpressionTemplateArgumentAST* exprArg,
    const TemplateArgument& storedArg, const TemplateArgument& concreteArg,
    const std::vector<TemplateArgument>& deduced,
    TemplateDeclarationAST* specTemplateDecl, ScopeSymbol* enclosingScope)
    -> bool {
  if (!exprArg->expression) return true;

  auto storedSym = std::get_if<Symbol*>(&storedArg);
  auto storedVar = storedSym && *storedSym
                       ? symbol_cast<VariableSymbol>(*storedSym)
                       : nullptr;
  if (!storedVar || storedVar->constValue().has_value()) return true;

  auto concreteSym = std::get_if<Symbol*>(&concreteArg);
  auto concreteVar = concreteSym && *concreteSym
                         ? symbol_cast<VariableSymbol>(*concreteSym)
                         : nullptr;
  if (!concreteVar || !concreteVar->constValue().has_value()) return true;

  SilentDiagnosticsClient client;
  auto savedClient = unit->changeDiagnosticsClient(&client);
  auto substituted = ASTRewriter::substituteDefaultExpression(
      unit, exprArg->expression, deduced, specTemplateDecl->depth,
      enclosingScope);
  std::optional<ConstValue> value;
  if (substituted) value = ASTInterpreter{unit}.evaluate(substituted);
  (void)unit->changeDiagnosticsClient(savedClient);

  if (client.hadError() || !value.has_value()) return false;
  return value.value() == concreteVar->constValue().value();
}

auto checkCollapsedSpecArguments(TranslationUnit* unit,
                                 SimpleTemplateIdAST* specId,
                                 std::span<const TemplateArgument> patternArgs,
                                 std::span<const TemplateArgument> concreteArgs,
                                 const std::vector<TemplateArgument>& deduced,
                                 TemplateDeclarationAST* specTemplateDecl,
                                 ScopeSymbol* enclosingScope) -> bool {
  if (!specId) return true;

  size_t index = 0;
  for (auto argAst : ListView{specId->templateArgumentList}) {
    if (index >= patternArgs.size()) break;
    const auto& storedArg = patternArgs[index];
    const auto argPos = index;
    ++index;

    if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(argAst)) {
      if (argPos >= concreteArgs.size()) continue;
      if (!reevaluateCollapsedExpressionArgument(
              unit, exprArg, storedArg, concreteArgs[argPos], deduced,
              specTemplateDecl, enclosingScope)) {
        return false;
      }
      continue;
    }

    auto typeArg = ast_cast<TypeTemplateArgumentAST>(argAst);
    if (!typeArg || !typeArg->typeId) continue;
    if (!isDependent(unit, typeArg->typeId)) continue;

    bool storedIsDeducedParam = false;
    if (auto storedType = std::get_if<const Type*>(&storedArg)) {
      storedIsDeducedParam = *storedType && getTypeParamInfo(*storedType);
    } else if (auto storedSym = std::get_if<Symbol*>(&storedArg)) {
      if (!*storedSym) continue;
      if (symbol_cast<ParameterPackSymbol>(*storedSym)) continue;
      auto storedSymType = (*storedSym)->type();
      storedIsDeducedParam = storedSymType && getTypeParamInfo(storedSymType);
    }
    if (storedIsDeducedParam) continue;

    SilentDiagnosticsClient client;
    auto savedClient = unit->changeDiagnosticsClient(&client);
    auto substituted = ASTRewriter::substituteDefaultTypeId(
        unit, typeArg->typeId, deduced, specTemplateDecl->depth,
        enclosingScope);
    (void)unit->changeDiagnosticsClient(savedClient);

    if (client.hadError() || !substituted || !substituted->type ||
        type_cast<UnresolvedNameType>(substituted->type)) {
      return false;
    }
  }

  return true;
}

template <typename SpecEntry>
auto collectCandidate(TranslationUnit* unit, const SpecEntry& spec,
                      const std::vector<TemplateArgument>& templateArguments,
                      int declarationOrder) -> std::optional<Candidate> {
  auto specClass = symbol_cast<ClassSymbol>(spec.symbol);
  if (!specClass) return std::nullopt;
  specClass = specClass->resolvedDefinition();

  auto specTemplateDecl = specClass->templateDeclaration();
  if (!specTemplateDecl) return std::nullopt;

  const auto& patternArgs = spec.arguments;
  if (patternArgs.size() != templateArguments.size()) return std::nullopt;

  auto specBody = ast_cast<ClassSpecifierAST>(specClass->declaration());
  if (!specBody) return std::nullopt;

  auto pattern = extractNestedTemplatePattern(specBody);
  if (!pattern) return std::nullopt;

  auto [specParamCount, paramPosition] = makeParamPosition(specTemplateDecl);
  DeducedArguments deducedArgs(specParamCount);

  PartialSpecMatcher matcher{unit, &*pattern, deducedArgs, paramPosition};

  for (size_t i = 0; i < patternArgs.size(); ++i) {
    if (!matcher.matchArg(patternArgs[i], templateArguments[i], i)) {
      return std::nullopt;
    }
  }

  if (!deducedArgs.complete()) return std::nullopt;

  if (!checkCollapsedSpecArguments(
          unit, ast_cast<SimpleTemplateIdAST>(specBody->unqualifiedId),
          patternArgs, templateArguments, deducedArgs.toTemplateArguments(),
          specTemplateDecl, specClass->enclosingNonTemplateParametersScope())) {
    return std::nullopt;
  }

  return Candidate{
      .specClass = specClass,
      .specTemplateDecl = specTemplateDecl,
      .specBody = specBody,
      .declarationOrder = declarationOrder,
      .packParameterCount = countPackTemplateParameters(specTemplateDecl),
      .deducedArgs = deducedArgs.toTemplateArguments(),
      .score = matcher.score,
      .packMatches = matcher.packMatches,
      .exactTypeMatches = matcher.exactTypeMatches,
      .nestedMatches = matcher.nestedMatches,
      .nonTypeMatches = matcher.nonTypeMatches,
      .deducedParamMatches = matcher.deducedParamMatches};
}

template <typename SpecEntry>
auto collectVariableCandidate(
    TranslationUnit* unit, const SpecEntry& spec,
    const std::vector<TemplateArgument>& templateArguments,
    int declarationOrder) -> std::optional<Candidate> {
  auto specVar = symbol_cast<VariableSymbol>(spec.symbol);
  if (!specVar) return std::nullopt;

  auto specTemplateDecl = specVar->templateDeclaration();
  if (!specTemplateDecl) return std::nullopt;

  const auto& patternArgs = spec.arguments;
  if (patternArgs.size() != templateArguments.size()) return std::nullopt;

  auto [specParamCount, paramPosition] = makeParamPosition(specTemplateDecl);
  DeducedArguments deducedArgs(specParamCount);

  std::optional<NestedTemplatePattern> pattern;
  if (auto simpleDecl =
          ast_cast<SimpleDeclarationAST>(specTemplateDecl->declaration);
      simpleDecl && simpleDecl->initDeclaratorList &&
      simpleDecl->initDeclaratorList->value) {
    if (auto declId = getDeclaratorId(
            simpleDecl->initDeclaratorList->value->declarator)) {
      if (auto root = ast_cast<SimpleTemplateIdAST>(declId->unqualifiedId)) {
        NestedTemplatePattern p;
        p.root = root;
        buildNestedTemplatePattern(root, p);
        pattern = std::move(p);
      }
    }
  }

  PartialSpecMatcher matcher{unit, pattern ? &*pattern : nullptr, deducedArgs,
                             paramPosition};

  for (size_t i = 0; i < patternArgs.size(); ++i) {
    if (!matcher.matchArg(patternArgs[i], templateArguments[i], i)) {
      return std::nullopt;
    }
  }

  if (!deducedArgs.complete()) return std::nullopt;

  if (pattern && !checkCollapsedSpecArguments(
                     unit, pattern->root, patternArgs, templateArguments,
                     deducedArgs.toTemplateArguments(), specTemplateDecl,
                     specVar->enclosingNonTemplateParametersScope())) {
    return std::nullopt;
  }

  return Candidate{
      .specVar = specVar,
      .specTemplateDecl = specTemplateDecl,
      .declarationOrder = declarationOrder,
      .packParameterCount = countPackTemplateParameters(specTemplateDecl),
      .deducedArgs = deducedArgs.toTemplateArguments(),
      .score = matcher.score,
      .packMatches = matcher.packMatches,
      .exactTypeMatches = matcher.exactTypeMatches,
      .nestedMatches = matcher.nestedMatches,
      .nonTypeMatches = matcher.nonTypeMatches,
      .deducedParamMatches = matcher.deducedParamMatches};
}

auto isLessSpecific(const Candidate& lhs, const Candidate& rhs) -> bool {
  if (lhs.score != rhs.score) return lhs.score < rhs.score;
  if (lhs.exactTypeMatches != rhs.exactTypeMatches)
    return lhs.exactTypeMatches < rhs.exactTypeMatches;
  if (lhs.nestedMatches != rhs.nestedMatches)
    return lhs.nestedMatches < rhs.nestedMatches;
  if (lhs.nonTypeMatches != rhs.nonTypeMatches)
    return lhs.nonTypeMatches < rhs.nonTypeMatches;
  if (lhs.packParameterCount != rhs.packParameterCount)
    return lhs.packParameterCount > rhs.packParameterCount;
  if (lhs.packMatches != rhs.packMatches)
    return lhs.packMatches > rhs.packMatches;
  if (lhs.deducedParamMatches != rhs.deducedParamMatches)
    return lhs.deducedParamMatches < rhs.deducedParamMatches;
  return lhs.declarationOrder > rhs.declarationOrder;
}

auto hasEqualSpecificity(const Candidate& lhs, const Candidate& rhs) -> bool {
  return lhs.score == rhs.score &&
         lhs.exactTypeMatches == rhs.exactTypeMatches &&
         lhs.nestedMatches == rhs.nestedMatches &&
         lhs.nonTypeMatches == rhs.nonTypeMatches &&
         lhs.packParameterCount == rhs.packParameterCount &&
         lhs.packMatches == rhs.packMatches &&
         lhs.deducedParamMatches == rhs.deducedParamMatches;
}

auto bestCandidate(std::vector<Candidate>& candidates)
    -> std::vector<Candidate>::iterator {
  if (candidates.empty()) return candidates.end();

  return std::max_element(candidates.begin(), candidates.end(), isLessSpecific);
}

template <typename Specializations, typename Collect>
auto selectCandidate(TranslationUnit* unit,
                     const Specializations& specializations,
                     const std::vector<TemplateArgument>& templateArguments,
                     SourceLocation fallbackLocation, Collect collect)
    -> std::optional<Candidate> {
  std::vector<Candidate> candidates;
  int declarationOrder = 0;
  for (const auto& specialization : specializations) {
    auto candidate =
        collect(specialization, templateArguments, declarationOrder);
    ++declarationOrder;
    if (candidate) candidates.push_back(std::move(*candidate));
  }

  auto best = bestCandidate(candidates);
  if (best == candidates.end()) return std::nullopt;
  for (auto it = candidates.begin(); it != candidates.end(); ++it) {
    if (it == best || !hasEqualSpecificity(*best, *it)) continue;
    auto location = best->specBody ? best->specBody->firstSourceLocation()
                                   : fallbackLocation;
    unit->error(location, "partial specialization is ambiguous");
    return std::nullopt;
  }
  return std::move(*best);
}
}  // namespace

auto ASTRewriter::findPartialSpecializationPattern(
    TranslationUnit* unit, ClassSymbol* primary,
    List<TemplateArgumentAST*>* templateArgumentList) -> ClassSymbol* {
  if (primary && primary->isSpecialization())
    primary = primary->primaryTemplateSymbol();
  auto primaryDeclaration = primary ? primary->templateDeclaration() : nullptr;
  if (!primaryDeclaration) return nullptr;
  auto templateArguments =
      Substitution(unit, primaryDeclaration, templateArgumentList)
          .templateArguments();
  for (auto& argument : templateArguments) {
    if (auto type = std::get_if<const Type*>(&argument)) {
      auto symbol = unit->control()->newTypeAliasSymbol(nullptr, {});
      symbol->setType(*type);
      Symbol* value = symbol;
      argument = value;
      continue;
    }
    if (auto value = std::get_if<ConstValue>(&argument)) {
      auto symbol = unit->control()->newVariableSymbol(nullptr, {});
      symbol->setConstexpr(true);
      symbol->setConstValue(*value);
      Symbol* materialized = symbol;
      argument = materialized;
      continue;
    }
    auto expression = std::get_if<ExpressionAST*>(&argument);
    if (!expression || !*expression) continue;
    if (auto id = ast_cast<IdExpressionAST>(*expression); id && id->symbol) {
      argument = id->symbol;
      continue;
    }
    if (auto value = ASTInterpreter{unit}.evaluate(*expression)) {
      auto symbol = unit->control()->newVariableSymbol(nullptr, {});
      symbol->setType((*expression)->type);
      symbol->setConstexpr(true);
      symbol->setConstValue(*value);
      Symbol* materialized = symbol;
      argument = materialized;
    }
  }
  auto selected = selectCandidate(
      unit, primary->specializations(), templateArguments, primary->location(),
      [unit](const auto& specialization, const auto& arguments, int order) {
        return collectCandidate(unit, specialization, arguments, order);
      });
  return selected ? selected->specClass->resolvedDefinition() : nullptr;
}

auto ASTRewriter::tryPartialSpecialization(
    TranslationUnit* unit, ClassSymbol* classSymbol,
    const std::vector<TemplateArgument>& templateArguments) -> Symbol* {
  auto candidate = selectCandidate(
      unit, classSymbol->specializations(), templateArguments,
      classSymbol->location(),
      [unit](const auto& specialization, const auto& arguments, int order) {
        return collectCandidate(unit, specialization, arguments, order);
      });
  if (!candidate) return nullptr;
  auto& selected = *candidate;

  if (auto cached =
          selected.specClass->findSpecialization(selected.deducedArgs)) {
    if (auto cachedClass = symbol_cast<ClassSymbol>(cached)) {
      classSymbol->addSpecialization(templateArguments, cachedClass);
    }
    return cached;
  }

  auto specParentScope =
      selected.specClass->enclosingNonTemplateParametersScope();
  auto specRewriter = ASTRewriter{unit, specParentScope, selected.deducedArgs};
  specRewriter.depth_ = selected.specTemplateDecl->depth;
  specRewriter.binder().setInstantiatingSymbol(selected.specClass);

  auto pendingInstance = symbol_cast<ClassSymbol>(
      classSymbol->findSpecialization(templateArguments));
  if (pendingInstance && !pendingInstance->isComplete()) {
    specRewriter.setClassInstanceToComplete(pendingInstance);
  }

  auto instance =
      ast_cast<ClassSpecifierAST>(specRewriter.specifier(selected.specBody));
  if (!instance || !instance->symbol) return nullptr;

  if (auto instanceClass = symbol_cast<ClassSymbol>(instance->symbol)) {
    classSymbol->addSpecialization(templateArguments, instanceClass);
  }

  return instance->symbol;
}

auto ASTRewriter::tryPartialSpecialization(
    TranslationUnit* unit, VariableSymbol* variableSymbol,
    const std::vector<TemplateArgument>& templateArguments) -> Symbol* {
  auto candidate = selectCandidate(
      unit, variableSymbol->specializations(), templateArguments,
      variableSymbol->location(),
      [unit](const auto& specialization, const auto& arguments, int order) {
        return collectVariableCandidate(unit, specialization, arguments, order);
      });
  if (!candidate) return nullptr;
  auto& selected = *candidate;

  if (auto cached =
          selected.specVar->findSpecialization(selected.deducedArgs)) {
    if (auto cachedVar = symbol_cast<VariableSymbol>(cached)) {
      variableSymbol->addSpecialization(templateArguments, cachedVar);
    }
    return cached;
  }

  auto specTemplateDecl = selected.specVar->templateDeclaration();
  if (!specTemplateDecl) return nullptr;

  auto simpleDecl =
      ast_cast<SimpleDeclarationAST>(specTemplateDecl->declaration);
  if (!simpleDecl) return nullptr;

  auto specParentScope =
      selected.specVar->enclosingNonTemplateParametersScope();
  auto specRewriter = ASTRewriter{unit, specParentScope, selected.deducedArgs};
  specRewriter.depth_ = selected.specTemplateDecl->depth;
  specRewriter.binder().setInstantiatingSymbol(selected.specVar);

  auto instance =
      ast_cast<SimpleDeclarationAST>(specRewriter.declaration(simpleDecl));
  if (!instance || !instance->initDeclaratorList ||
      !instance->initDeclaratorList->value) {
    return nullptr;
  }

  auto instantiatedSymbol = instance->initDeclaratorList->value->symbol;
  if (!instantiatedSymbol) return nullptr;

  if (auto instanceVar = symbol_cast<VariableSymbol>(instantiatedSymbol)) {
    variableSymbol->addSpecialization(templateArguments, instanceVar);
  }

  return instantiatedSymbol;
}
}  // namespace cxx
