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
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <functional>
#include <map>
#include <ranges>
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

auto expansionTypeIdOfAlias(NamedTypeSpecifierAST* named) -> TypeIdAST* {
  auto alias = symbol_cast<TypeAliasSymbol>(named->symbol);
  if (!alias || alias->templateDeclaration()) return nullptr;
  return alias->expansionTypeId();
}

auto findTemplateIdInTypeId(TypeIdAST* typeId,
                            std::vector<TypeIdAST*>& expandedAliases)
    -> SimpleTemplateIdAST* {
  if (!typeId) return nullptr;
  if (std::ranges::contains(expandedAliases, typeId)) return nullptr;
  expandedAliases.push_back(typeId);

  for (auto sp : ListView{typeId->typeSpecifierList}) {
    auto named = ast_cast<NamedTypeSpecifierAST>(sp);
    if (!named) continue;

    if (auto expansion = expansionTypeIdOfAlias(named)) {
      if (auto templId = findTemplateIdInTypeId(expansion, expandedAliases))
        return templId;
    }

    if (auto templId = ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)) {
      return templId;
    }
  }
  return nullptr;
}

auto findTemplateIdInTypeId(TypeIdAST* typeId) -> SimpleTemplateIdAST* {
  std::vector<TypeIdAST*> expandedAliases;
  return findTemplateIdInTypeId(typeId, expandedAliases);
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

struct HasDefaultTemplateArgument {
  auto operator()(TypenameTypeParameterAST* p) const -> bool {
    return p->typeId != nullptr;
  }
  auto operator()(TemplateTypeParameterAST* p) const -> bool {
    return p->idExpression != nullptr;
  }
  auto operator()(ConstraintTypeParameterAST* p) const -> bool {
    return p->typeId != nullptr;
  }
  auto operator()(NonTypeTemplateParameterAST* p) const -> bool {
    return p->declaration && p->declaration->expression != nullptr;
  }
};

auto hasDefaultsFromPosition(TemplateDeclarationAST* templateDecl,
                             size_t fromPosition) -> bool {
  size_t position = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    if (position >= fromPosition &&
        !visit(HasDefaultTemplateArgument{}, parameter)) {
      return false;
    }
    ++position;
  }
  return true;
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

  [[nodiscard]] auto control() const -> Control* { return unit->control(); }

  static auto toSymbolVector(std::span<const TemplateArgument> args)
      -> std::vector<Symbol*> {
    std::vector<Symbol*> result;
    result.reserve(args.size());

    for (const auto& arg : args) result.push_back(asSymbolArgument(arg));

    return result;
  }

  auto collectWrittenTemplateArgumentSymbols(List<TemplateArgumentAST*>* args)
      -> std::optional<std::vector<Symbol*>> {
    std::vector<Symbol*> symbols;
    for (auto arg : ListView{args}) {
      if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
        if (!typeArg->typeId || !typeArg->typeId->type) return std::nullopt;
        auto wrapper = control()->newTypeAliasSymbol(nullptr, {});
        wrapper->setType(typeArg->typeId->type);
        symbols.push_back(wrapper);
        continue;
      }

      if (auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(arg)) {
        auto idExpr = ast_cast<IdExpressionAST>(exprArg->expression);
        if (!idExpr || !idExpr->symbol) return std::nullopt;
        symbols.push_back(idExpr->symbol);
        continue;
      }

      return std::nullopt;
    }
    return symbols;
  }

  auto finishNestedMatch(const std::vector<Symbol*>& patSymbols,
                         const std::vector<Symbol*>& concSymbols,
                         SimpleTemplateIdAST* patTemplId) -> bool {
    if (!deduceArgumentList(patSymbols, concSymbols, patTemplId, 0)) {
      return false;
    }
    return true;
  }

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

  auto deduceOrCheck(int pos, Symbol* newSymbol) -> bool {
    if (pos < 0) return true;
    if (!newSymbol) return false;

    auto existingSymbol = deducedArgs.get(pos);
    if (!existingSymbol) {
      deducedArgs.set(pos, newSymbol);
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
                             deducedPack);
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

    auto childTemplateId =
        pattern ? pattern->child(patTemplId, argPos) : nullptr;

    if (auto patTTP = type_cast<TemplateTypeParameterType>(patSym->type());
        patTTP && childTemplateId && childTemplateId->templateArgumentList) {
      return deduceTemplateTemplateParameter(patTTP, concSym->type(),
                                             childTemplateId);
    }

    if (auto patInfo = template_parameter_info(patSym)) {
      auto concInfo = template_parameter_info(concSym);
      if (!patInfo->isPack && concInfo && concInfo->isPack) return false;
      return deduceOrCheck(paramPosition(patInfo->depth, patInfo->index),
                           concSym);
    }

    auto patPack = symbol_cast<ParameterPackSymbol>(patSym);
    auto concPack = symbol_cast<ParameterPackSymbol>(concSym);
    if (patPack && concPack) {
      if (!deduceArgumentList(patPack->elements(), concPack->elements(),
                              patTemplId, argPos)) {
        return false;
      }
      return true;
    }

    auto patType = patSym->type();
    auto concType = concSym->type();
    if (!patType || !concType) {
      if (patType != concType) return false;
      return true;
    }

    if (type_cast<UnresolvedNameType>(patType)) return true;

    if (auto decided = deduceConstantArgument(patSym, concSym)) return *decided;

    return deduceType(patType, concType, childTemplateId);
  }

  auto deduceConstantArgument(Symbol* patSym, Symbol* concSym)
      -> std::optional<bool> {
    auto patVar = symbol_cast<VariableSymbol>(patSym);
    auto concVar = symbol_cast<VariableSymbol>(concSym);
    if (!patVar || !concVar) return std::nullopt;
    if (!patVar->constValue().has_value() && !concVar->constValue().has_value())
      return std::nullopt;

    if (!concVar->constValue().has_value()) return false;
    if (!patVar->constValue().has_value()) return true;
    if (patVar->constValue().value() != concVar->constValue().value())
      return false;

    return true;
  }

  auto matchArg(const TemplateArgument& pat, const TemplateArgument& conc,
                size_t argPos) -> bool {
    return deduceArgument(asSymbolArgument(pat), asSymbolArgument(conc),
                          pattern ? pattern->root : nullptr, argPos);
  }

  auto matchTemplateIdentity(SimpleTemplateIdAST* pat,
                             SimpleTemplateIdAST* conc) -> bool {
    if (!pat || !conc || !pat->symbol || !conc->symbol) return false;

    auto identity = [](Symbol* symbol) -> Symbol* {
      if (auto cls = symbol_cast<ClassSymbol>(symbol)) {
        if (cls->isSpecialization()) return cls->primaryTemplateSymbol();
      }
      return symbol;
    };

    auto patIdentity = identity(pat->symbol);
    auto concIdentity = identity(conc->symbol);
    if (auto info = template_parameter_info(patIdentity)) {
      return deduceOrCheck(paramPosition(info->depth, info->index),
                           concIdentity);
    }
    return patIdentity == concIdentity;
  }

  auto deduceTemplateTemplateParameter(const TemplateTypeParameterType* patTTP,
                                       const Type* concType,
                                       SimpleTemplateIdAST* patTemplId)
      -> bool {
    auto concClassType = type_cast<ClassType>(concType);
    if (!concClassType) return false;

    auto concClassSym = concClassType->symbol();
    if (!concClassSym || !concClassSym->isSpecialization()) return false;

    auto primary = concClassSym->primaryTemplateSymbol();
    if (!primary || !primary->templateDeclaration()) return false;

    auto pos = paramPosition(patTTP->depth(), patTTP->index());
    if (!deduceOrCheck(pos, primary)) return false;

    auto patSymbols =
        collectWrittenTemplateArgumentSymbols(patTemplId->templateArgumentList);
    if (!patSymbols) return false;

    auto concSymbols = toSymbolVector(concClassSym->templateArguments());

    auto patInfo = patSymbols->empty()
                       ? std::nullopt
                       : template_parameter_info(patSymbols->back());
    auto patHasTrailingPack = patInfo && patInfo->isPack;
    if (!patHasTrailingPack && patSymbols->size() < concSymbols.size()) {
      if (!hasDefaultsFromPosition(primary->templateDeclaration(),
                                   patSymbols->size())) {
        return false;
      }
      concSymbols.resize(patSymbols->size());
    }

    return finishNestedMatch(*patSymbols, concSymbols, patTemplId);
  }

  auto deduceType(const Type* patType, const Type* concType,
                  SimpleTemplateIdAST* patTemplId) -> bool {
    if (!patType || !concType) return false;

    if (auto patTTP = type_cast<TemplateTypeParameterType>(patType);
        patTTP && patTemplId && patTemplId->templateArgumentList) {
      return deduceTemplateTemplateParameter(patTTP, concType, patTemplId);
    }

    if (auto patParamInfo = getTypeParamInfo(patType)) {
      auto pos = paramPosition(patParamInfo->depth, patParamInfo->index);
      auto argument = control()->newTypeAliasSymbol(nullptr, {});
      argument->setType(concType);
      return deduceOrCheck(pos, argument);
    }

    if (auto decided = deduceQualifiedType(patType, concType, patTemplId))
      return *decided;

    if (auto decided = deduceIndirectionType(patType, concType, patTemplId))
      return *decided;

    if (auto decided = deduceArrayType(patType, concType, patTemplId))
      return *decided;

    if (auto decided = deduceFunctionType(patType, concType, patTemplId))
      return *decided;

    if (auto decided = deduceClassType(patType, concType, patTemplId))
      return *decided;

    if (patType != concType && !unit->typeTraits().is_same(patType, concType)) {
      return false;
    }

    return true;
  }

  auto deduceQualifiedType(const Type* patType, const Type* concType,
                           SimpleTemplateIdAST* patTemplId)
      -> std::optional<bool> {
    auto patQual = type_cast<QualType>(patType);
    if (!patQual) return std::nullopt;

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

    return deduceType(patQual->elementType(), concElement, patTemplId);
  }

  template <typename T>
  auto deduceElementType(const Type* patType, const Type* concType,
                         SimpleTemplateIdAST* patTemplId)
      -> std::optional<bool> {
    auto pat = type_cast<T>(patType);
    if (!pat) return std::nullopt;

    auto conc = type_cast<T>(concType);
    if (!conc) return false;

    return deduceType(pat->elementType(), conc->elementType(), patTemplId);
  }

  auto deduceIndirectionType(const Type* patType, const Type* concType,
                             SimpleTemplateIdAST* patTemplId)
      -> std::optional<bool> {
    if (auto decided =
            deduceElementType<PointerType>(patType, concType, patTemplId))
      return decided;

    if (auto decided = deduceElementType<LvalueReferenceType>(patType, concType,
                                                              patTemplId))
      return decided;

    return deduceElementType<RvalueReferenceType>(patType, concType,
                                                  patTemplId);
  }

  auto deduceArrayType(const Type* patType, const Type* concType,
                       SimpleTemplateIdAST* patTemplId) -> std::optional<bool> {
    if (auto patArray = type_cast<BoundedArrayType>(patType)) {
      auto concArray = type_cast<BoundedArrayType>(concType);
      if (!concArray || patArray->size() != concArray->size()) return false;
      return deduceElementType<BoundedArrayType>(patType, concType, patTemplId);
    }

    if (auto decided = deduceElementType<UnboundedArrayType>(patType, concType,
                                                             patTemplId))
      return decided;

    return deduceUnresolvedArrayType(patType, concType, patTemplId);
  }

  auto deduceUnresolvedArrayType(const Type* patType, const Type* concType,
                                 SimpleTemplateIdAST* patTemplId)
      -> std::optional<bool> {
    auto patArray = type_cast<UnresolvedBoundedArrayType>(patType);
    if (!patArray) return std::nullopt;

    auto concArray = type_cast<BoundedArrayType>(concType);
    if (!concArray) return false;

    auto idExpr = ast_cast<IdExpressionAST>(patArray->size());
    if (!idExpr) return false;

    auto nttp = symbol_cast<NonTypeParameterSymbol>(idExpr->symbol);
    if (!nttp) return false;

    auto value = control()->newVariableSymbol(nullptr, {});
    value->setType(nttp->objectType());
    value->setConstexpr(true);
    value->setConstValue(
        ConstValue(static_cast<std::intmax_t>(concArray->size())));

    if (!deduceOrCheck(paramPosition(nttp->depth(), nttp->index()), value))
      return false;

    return deduceType(patArray->elementType(), concArray->elementType(),
                      patTemplId);
  }

  auto deduceFunctionType(const Type* patType, const Type* concType,
                          SimpleTemplateIdAST* patTemplId)
      -> std::optional<bool> {
    auto patFunction = type_cast<FunctionType>(patType);
    if (!patFunction) return std::nullopt;

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

    return true;
  }

  auto deduceClassType(const Type* patType, const Type* concType,
                       SimpleTemplateIdAST* patTemplId) -> std::optional<bool> {
    auto patClassType = type_cast<ClassType>(patType);
    auto concClassType = type_cast<ClassType>(concType);
    if (!patClassType || !concClassType) return std::nullopt;

    auto patClassSym = patClassType->symbol();
    auto concClassSym = concClassType->symbol();
    if (!patClassSym || !concClassSym) return false;
    if (!concClassSym->isSpecialization()) return std::nullopt;

    auto primary = patClassSym->isSpecialization()
                       ? patClassSym->primaryTemplateSymbol()
                       : patClassSym;

    if (concClassSym->primaryTemplateSymbol() != primary) return std::nullopt;
    if (!primary->templateDeclaration()) return false;
    if (!patTemplId) return false;

    return matchNestedWithPattern(patTemplId, primary,
                                  concClassSym->templateArguments());
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
      patSymbols = toSymbolVector(patInnerArgs);
    }

    return finishNestedMatch(patSymbols, toSymbolVector(concArgs), patTemplId);
  }
};
}  // namespace

struct ASTRewriter::RewritePartialSpecialization {
  TranslationUnit* unit = nullptr;

  explicit RewritePartialSpecialization(TranslationUnit* unit) : unit(unit) {}

  [[nodiscard]] auto control() const -> Control* { return unit->control(); }

  struct Candidate {
    ClassSymbol* specClass = nullptr;
    VariableSymbol* specVar = nullptr;
    TemplateDeclarationAST* specTemplateDecl = nullptr;
    ClassSpecifierAST* specBody = nullptr;
    SimpleTemplateIdAST* patternRoot = nullptr;
    std::vector<TemplateArgument> patternArguments;
    std::vector<TemplateArgument> deducedArgs;

    [[nodiscard]] auto symbol() const -> Symbol* {
      if (specClass) return specClass;
      return specVar;
    }
  };

  struct Selection {
    std::optional<Candidate> candidate;
    bool ambiguous = false;
  };

  using Collect = std::optional<Candidate> (RewritePartialSpecialization::*)(
      const TemplateSpecialization&, const std::vector<TemplateArgument>&);

  [[nodiscard]] auto findPattern(
      ClassSymbol* primary, List<TemplateArgumentAST*>* templateArgumentList)
      -> ClassSymbol*;

  [[nodiscard]] auto apply(
      ClassSymbol* classSymbol,
      const std::vector<TemplateArgument>& templateArguments)
      -> PartialSpecializationResult;

  [[nodiscard]] auto apply(
      VariableSymbol* variableSymbol,
      const std::vector<TemplateArgument>& templateArguments)
      -> PartialSpecializationResult;

 private:
  [[nodiscard]] auto collectClassCandidate(
      const TemplateSpecialization& spec,
      const std::vector<TemplateArgument>& templateArguments)
      -> std::optional<Candidate>;

  [[nodiscard]] auto collectVariableCandidate(
      const TemplateSpecialization& spec,
      const std::vector<TemplateArgument>& templateArguments)
      -> std::optional<Candidate>;

  [[nodiscard]] auto select(
      std::span<const TemplateSpecialization> specializations,
      const std::vector<TemplateArgument>& templateArguments,
      SourceLocation fallbackLocation, Collect collect) -> Selection;

  [[nodiscard]] auto isMoreSpecialized(const Candidate& lhs,
                                       const Candidate& rhs) const -> bool;

  [[nodiscard]] auto isAtLeastAsSpecialized(const Candidate& lhs,
                                            const Candidate& rhs) const -> bool;

  [[nodiscard]] auto hasEquivalentTransformedType(const Candidate& lhs,
                                                  const Candidate& rhs) const
      -> bool;

  [[nodiscard]] auto checkCollapsedSpecArguments(
      SimpleTemplateIdAST* specId,
      std::span<const TemplateArgument> patternArgs,
      std::span<const TemplateArgument> concreteArgs,
      const std::vector<TemplateArgument>& deduced,
      TemplateDeclarationAST* specTemplateDecl, ScopeSymbol* enclosingScope)
      -> bool;

  [[nodiscard]] auto reevaluateCollapsedExpressionArgument(
      ExpressionTemplateArgumentAST* exprArg, const TemplateArgument& storedArg,
      const TemplateArgument& concreteArg,
      const std::vector<TemplateArgument>& deduced,
      TemplateDeclarationAST* specTemplateDecl, ScopeSymbol* enclosingScope)
      -> bool;

  [[nodiscard]] static auto makeParamPosition(
      TemplateDeclarationAST* templateDecl)
      -> std::pair<int, std::function<int(int depth, int index)>>;

  [[nodiscard]] auto materializeTemplateArguments(
      std::vector<TemplateArgument> templateArguments) const
      -> std::vector<TemplateArgument>;
};

auto ASTRewriter::RewritePartialSpecialization::makeParamPosition(
    TemplateDeclarationAST* templateDecl)
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

auto ASTRewriter::RewritePartialSpecialization::
    reevaluateCollapsedExpressionArgument(
        ExpressionTemplateArgumentAST* exprArg,
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

auto ASTRewriter::RewritePartialSpecialization::checkCollapsedSpecArguments(
    SimpleTemplateIdAST* specId, std::span<const TemplateArgument> patternArgs,
    std::span<const TemplateArgument> concreteArgs,
    const std::vector<TemplateArgument>& deduced,
    TemplateDeclarationAST* specTemplateDecl, ScopeSymbol* enclosingScope)
    -> bool {
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
              exprArg, storedArg, concreteArgs[argPos], deduced,
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

auto ASTRewriter::RewritePartialSpecialization::collectClassCandidate(
    const TemplateSpecialization& spec,
    const std::vector<TemplateArgument>& templateArguments)
    -> std::optional<Candidate> {
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
          ast_cast<SimpleTemplateIdAST>(specBody->unqualifiedId), patternArgs,
          templateArguments, deducedArgs.toTemplateArguments(),
          specTemplateDecl, specClass->parent())) {
    return std::nullopt;
  }

  if (!ASTRewriter::checkAssociatedConstraints(
          unit, specClass, deducedArgs.toTemplateArguments(),
          specTemplateDecl->depth)) {
    return std::nullopt;
  }

  return Candidate{.specClass = specClass,
                   .specTemplateDecl = specTemplateDecl,
                   .specBody = specBody,
                   .patternRoot = pattern->root,
                   .patternArguments = patternArgs,
                   .deducedArgs = deducedArgs.toTemplateArguments()};
}

auto ASTRewriter::RewritePartialSpecialization::collectVariableCandidate(
    const TemplateSpecialization& spec,
    const std::vector<TemplateArgument>& templateArguments)
    -> std::optional<Candidate> {
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
                     pattern->root, patternArgs, templateArguments,
                     deducedArgs.toTemplateArguments(), specTemplateDecl,
                     specVar->parent())) {
    return std::nullopt;
  }

  if (!ASTRewriter::checkAssociatedConstraints(
          unit, specVar, deducedArgs.toTemplateArguments(),
          specTemplateDecl->depth)) {
    return std::nullopt;
  }

  return Candidate{.specVar = specVar,
                   .specTemplateDecl = specTemplateDecl,
                   .patternRoot = pattern ? pattern->root : nullptr,
                   .patternArguments = patternArgs,
                   .deducedArgs = deducedArgs.toTemplateArguments()};
}

auto ASTRewriter::RewritePartialSpecialization::isAtLeastAsSpecialized(
    const Candidate& lhs, const Candidate& rhs) const -> bool {
  if (!rhs.patternRoot) return false;

  auto [parameterCount, parameterPosition] =
      makeParamPosition(rhs.specTemplateDecl);
  DeducedArguments deduced(parameterCount);
  NestedTemplatePattern pattern;
  pattern.root = rhs.patternRoot;
  buildNestedTemplatePattern(rhs.patternRoot, pattern);
  PartialSpecMatcher matcher{unit, &pattern, deduced, parameterPosition};

  auto rhsNested = extractDirectNestedTemplateIds(rhs.patternRoot);
  auto lhsNested = extractDirectNestedTemplateIds(lhs.patternRoot);
  if (rhs.patternArguments.size() != lhs.patternArguments.size() ||
      rhsNested.size() != lhsNested.size())
    return false;

  for (std::size_t i = 0; i < rhs.patternArguments.size(); ++i) {
    if (rhsNested[i] || lhsNested[i]) {
      if (!rhsNested[i] || !lhsNested[i]) return false;
      if (!matcher.matchTemplateIdentity(rhsNested[i], lhsNested[i]))
        return false;
      auto rhsArguments = matcher.collectWrittenTemplateArgumentSymbols(
          rhsNested[i]->templateArgumentList);
      auto lhsArguments = matcher.collectWrittenTemplateArgumentSymbols(
          lhsNested[i]->templateArgumentList);
      if (!rhsArguments || !lhsArguments ||
          !matcher.finishNestedMatch(*rhsArguments, *lhsArguments,
                                     rhsNested[i]))
        return false;
      continue;
    }

    if (!matcher.matchArg(rhs.patternArguments[i], lhs.patternArguments[i], i))
      return false;
  }

  return deduced.complete();
}

auto ASTRewriter::RewritePartialSpecialization::hasEquivalentTransformedType(
    const Candidate& lhs, const Candidate& rhs) const -> bool {
  if (!areTemplateParameterListsEquivalentForPartialOrdering(
          unit, lhs.specTemplateDecl->templateParameterList,
          rhs.specTemplateDecl->templateParameterList))
    return false;

  if (lhs.patternArguments.size() != rhs.patternArguments.size()) return false;

  for (std::size_t i = 0; i < lhs.patternArguments.size(); ++i) {
    auto lhsType = template_argument_type(lhs.patternArguments[i]);
    auto rhsType = template_argument_type(rhs.patternArguments[i]);
    if (lhsType || rhsType) {
      if (!lhsType || !rhsType ||
          !areTypesEquivalentForPartialOrdering(unit, lhsType, rhsType,
                                                lhs.specTemplateDecl,
                                                rhs.specTemplateDecl))
        return false;
      continue;
    }

    auto lhsValue = template_argument_value(lhs.patternArguments[i]);
    auto rhsValue = template_argument_value(rhs.patternArguments[i]);
    if (lhsValue || rhsValue) {
      if (!lhsValue || !rhsValue || *lhsValue != *rhsValue) return false;
      continue;
    }

    auto lhsSymbol = asSymbolArgument(lhs.patternArguments[i]);
    auto rhsSymbol = asSymbolArgument(rhs.patternArguments[i]);
    auto lhsInfo = template_parameter_info(lhsSymbol);
    auto rhsInfo = template_parameter_info(rhsSymbol);
    if (!lhsInfo || !rhsInfo || lhsInfo->index != rhsInfo->index ||
        lhsInfo->isPack != rhsInfo->isPack)
      return false;
  }

  return true;
}

auto ASTRewriter::RewritePartialSpecialization::isMoreSpecialized(
    const Candidate& lhs, const Candidate& rhs) const -> bool {
  auto lhsAtLeast = isAtLeastAsSpecialized(lhs, rhs);
  auto rhsAtLeast = isAtLeastAsSpecialized(rhs, lhs);
  if (lhsAtLeast != rhsAtLeast) return lhsAtLeast;
  if (!lhsAtLeast || !hasEquivalentTransformedType(lhs, rhs)) return false;
  return ASTRewriter::isMoreConstrained(unit, lhs.symbol(), rhs.symbol());
}

auto ASTRewriter::RewritePartialSpecialization::select(
    std::span<const TemplateSpecialization> specializations,
    const std::vector<TemplateArgument>& templateArguments,
    SourceLocation fallbackLocation, Collect collect) -> Selection {
  std::vector<TemplateSpecialization> stableSpecializations(
      specializations.begin(), specializations.end());
  std::vector<Candidate> candidates;
  for (const auto& specialization : stableSpecializations) {
    auto candidate = (this->*collect)(specialization, templateArguments);
    if (candidate) candidates.push_back(std::move(*candidate));
  }

  if (candidates.empty()) return {};
  if (candidates.size() == 1)
    return {.candidate = std::move(candidates.front())};

  auto best = candidates.end();
  for (auto it = candidates.begin(); it != candidates.end(); ++it) {
    bool dominates = true;
    for (auto other = candidates.begin(); other != candidates.end(); ++other) {
      if (it != other && !isMoreSpecialized(*it, *other)) {
        dominates = false;
        break;
      }
    }
    if (!dominates) continue;
    if (best != candidates.end()) {
      best = candidates.end();
      break;
    }
    best = it;
  }

  if (best == candidates.end()) {
    auto location = candidates.front().specBody
                        ? candidates.front().specBody->firstSourceLocation()
                        : fallbackLocation;
    unit->error(location, "partial specialization is ambiguous");
    return {.ambiguous = true};
  }

  return {.candidate = std::move(*best)};
}

auto ASTRewriter::RewritePartialSpecialization::materializeTemplateArguments(
    std::vector<TemplateArgument> templateArguments) const
    -> std::vector<TemplateArgument> {
  for (auto& argument : templateArguments) {
    if (auto type = std::get_if<const Type*>(&argument)) {
      auto symbol = control()->newTypeAliasSymbol(nullptr, {});
      symbol->setType(*type);
      Symbol* value = symbol;
      argument = value;
      continue;
    }
    if (auto value = std::get_if<ConstValue>(&argument)) {
      auto symbol = control()->newVariableSymbol(nullptr, {});
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
      auto symbol = control()->newVariableSymbol(nullptr, {});
      symbol->setType((*expression)->type);
      symbol->setConstexpr(true);
      symbol->setConstValue(*value);
      Symbol* materialized = symbol;
      argument = materialized;
    }
  }
  return templateArguments;
}

auto ASTRewriter::RewritePartialSpecialization::findPattern(
    ClassSymbol* primary, List<TemplateArgumentAST*>* templateArgumentList)
    -> ClassSymbol* {
  if (primary && primary->isSpecialization())
    primary = primary->primaryTemplateSymbol();

  auto primaryDeclaration = primary ? primary->templateDeclaration() : nullptr;
  if (!primaryDeclaration) return nullptr;

  auto templateArguments = materializeTemplateArguments(
      Substitution(unit, primaryDeclaration, templateArgumentList)
          .templateArguments());

  auto selected =
      select(primary->specializations(), templateArguments, primary->location(),
             &RewritePartialSpecialization::collectClassCandidate);

  if (!selected.candidate) return nullptr;
  return selected.candidate->specClass->resolvedDefinition();
}

auto ASTRewriter::RewritePartialSpecialization::apply(
    ClassSymbol* classSymbol,
    const std::vector<TemplateArgument>& templateArguments)
    -> PartialSpecializationResult {
  auto selection = select(classSymbol->specializations(), templateArguments,
                          classSymbol->location(),
                          &RewritePartialSpecialization::collectClassCandidate);

  if (!selection.candidate) return {.resolutionFailed = selection.ambiguous};
  auto& selected = *selection.candidate;

  if (auto cached =
          selected.specClass->findSpecialization(unit, selected.deducedArgs)) {
    if (auto cachedClass = symbol_cast<ClassSymbol>(cached)) {
      classSymbol->addSpecialization(unit, templateArguments, cachedClass);
    }
    return {.symbol = cached};
  }

  auto specParentScope = selected.specClass->parent();
  auto specRewriter = ASTRewriter{unit, specParentScope, selected.deducedArgs};
  specRewriter.depth_ = selected.specTemplateDecl->depth;
  specRewriter.binder().setInstantiatingSymbol(selected.specClass);

  auto pendingInstance = symbol_cast<ClassSymbol>(
      classSymbol->findSpecialization(unit, templateArguments));
  if (pendingInstance && !pendingInstance->isComplete()) {
    specRewriter.setClassInstanceToComplete(pendingInstance);
  }

  auto instance =
      ast_cast<ClassSpecifierAST>(specRewriter.specifier(selected.specBody));
  if (!instance || !instance->symbol) return {.resolutionFailed = true};

  if (auto instanceClass = symbol_cast<ClassSymbol>(instance->symbol)) {
    classSymbol->addSpecialization(unit, templateArguments, instanceClass);
  }

  return {.symbol = instance->symbol};
}

auto ASTRewriter::RewritePartialSpecialization::apply(
    VariableSymbol* variableSymbol,
    const std::vector<TemplateArgument>& templateArguments)
    -> PartialSpecializationResult {
  auto selection =
      select(variableSymbol->specializations(), templateArguments,
             variableSymbol->location(),
             &RewritePartialSpecialization::collectVariableCandidate);

  if (!selection.candidate) return {.resolutionFailed = selection.ambiguous};
  auto& selected = *selection.candidate;

  if (auto cached =
          selected.specVar->findSpecialization(unit, selected.deducedArgs)) {
    if (auto cachedVar = symbol_cast<VariableSymbol>(cached)) {
      variableSymbol->addSpecialization(unit, templateArguments, cachedVar);
    }
    return {.symbol = cached};
  }

  auto specTemplateDecl = selected.specVar->templateDeclaration();
  if (!specTemplateDecl) return {.resolutionFailed = true};

  auto simpleDecl =
      ast_cast<SimpleDeclarationAST>(specTemplateDecl->declaration);
  if (!simpleDecl) return {.resolutionFailed = true};

  auto specParentScope = selected.specVar->parent();
  auto specRewriter = ASTRewriter{unit, specParentScope, selected.deducedArgs};
  specRewriter.depth_ = selected.specTemplateDecl->depth;
  specRewriter.binder().setInstantiatingSymbol(selected.specVar);

  auto instance =
      ast_cast<SimpleDeclarationAST>(specRewriter.declaration(simpleDecl));
  if (!instance || !instance->initDeclaratorList ||
      !instance->initDeclaratorList->value) {
    return {.resolutionFailed = true};
  }

  auto instantiatedSymbol = instance->initDeclaratorList->value->symbol;
  if (!instantiatedSymbol) return {.resolutionFailed = true};

  if (auto instanceVar = symbol_cast<VariableSymbol>(instantiatedSymbol)) {
    variableSymbol->addSpecialization(unit, templateArguments, instanceVar);
  }

  return {.symbol = instantiatedSymbol};
}

auto ASTRewriter::findPartialSpecializationPattern(
    TranslationUnit* unit, ClassSymbol* primary,
    List<TemplateArgumentAST*>* templateArgumentList) -> ClassSymbol* {
  return RewritePartialSpecialization{unit}.findPattern(primary,
                                                        templateArgumentList);
}

auto ASTRewriter::tryPartialSpecialization(
    TranslationUnit* unit, ClassSymbol* classSymbol,
    const std::vector<TemplateArgument>& templateArguments)
    -> PartialSpecializationResult {
  return RewritePartialSpecialization{unit}.apply(classSymbol,
                                                  templateArguments);
}

auto ASTRewriter::tryPartialSpecialization(
    TranslationUnit* unit, VariableSymbol* variableSymbol,
    const std::vector<TemplateArgument>& templateArguments)
    -> PartialSpecializationResult {
  return RewritePartialSpecialization{unit}.apply(variableSymbol,
                                                  templateArguments);
}
}  // namespace cxx
