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

#include <cxx/ast_rewriter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

// std
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

auto extractDirectNestedTemplateIds(SimpleTemplateIdAST* templId)
    -> std::vector<SimpleTemplateIdAST*> {
  std::vector<SimpleTemplateIdAST*> nested;
  for (auto arg : ListView{templId->templateArgumentList}) {
    auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg);
    if (!typeArg || !typeArg->typeId) {
      nested.push_back(nullptr);
      continue;
    }

    SimpleTemplateIdAST* innerTemplId = nullptr;
    for (auto sp : ListView{typeArg->typeId->typeSpecifierList}) {
      auto named = ast_cast<NamedTypeSpecifierAST>(sp);
      if (!named) continue;

      innerTemplId = ast_cast<SimpleTemplateIdAST>(named->unqualifiedId);
      if (innerTemplId) break;
    }

    nested.push_back(innerTemplId);
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

struct PackDeducer {
  Control* control;
  DeducedArguments& deducedArgs;
  std::function<int(int depth, int index)> paramPosition;

  void setDeduced(int pos, Symbol* symbol) {
    if (pos < 0) return;
    deducedArgs.set(pos, symbol);
  }

  auto deducePackElements(const std::vector<Symbol*>& patElems,
                          const std::vector<Symbol*>& concElems) -> bool {
    size_t patIdx = 0;
    size_t concIdx = 0;

    while (patIdx < patElems.size() && concIdx < concElems.size()) {
      auto patElem = patElems[patIdx];
      auto concElem = concElems[concIdx];
      if (!patElem || !concElem) return false;

      auto patElemType = patElem->type();
      auto concElemType = concElem->type();
      if (!patElemType || !concElemType) {
        if (patElemType != concElemType) return false;
        ++patIdx;
        ++concIdx;
        continue;
      }

      auto patElemInfo = getTypeParamInfo(patElemType);

      if (patElemInfo && patElemInfo->isPack) {
        auto deducedPack = control->newParameterPackSymbol(nullptr, {});
        while (concIdx < concElems.size()) {
          deducedPack->addElement(concElems[concIdx]);
          ++concIdx;
        }
        setDeduced(paramPosition(patElemInfo->depth, patElemInfo->index),
                   deducedPack);
        ++patIdx;
        continue;
      }

      if (patElemInfo) {
        setDeduced(paramPosition(patElemInfo->depth, patElemInfo->index),
                   concElems[concIdx]);
        ++patIdx;
        ++concIdx;
        continue;
      }

      if (patElemType != concElemType) return false;
      ++patIdx;
      ++concIdx;
    }

    while (patIdx < patElems.size()) {
      auto patElem = patElems[patIdx];
      if (!patElem) return false;

      auto patElemInfo = getTypeParamInfo(patElem->type());
      if (!patElemInfo || !patElemInfo->isPack) return false;

      auto deducedPack = control->newParameterPackSymbol(nullptr, {});
      setDeduced(paramPosition(patElemInfo->depth, patElemInfo->index),
                 deducedPack);
      ++patIdx;
    }

    return concIdx == concElems.size();
  }
};

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

  auto deduceOrCheck(int pos, Symbol* newSymbol) -> bool {
    if (pos < 0) return true;
    if (!newSymbol) return false;

    auto existingSymbol = deducedArgs.get(pos);
    if (!existingSymbol) {
      deducedArgs.set(pos, newSymbol);
      ++deducedParamMatches;
      return true;
    }

    if (existingSymbol == newSymbol) return true;

    auto leftType = existingSymbol->type();
    auto rightType = newSymbol->type();
    if (leftType && rightType && control()->is_same(leftType, rightType)) {
      return true;
    }

    auto leftVar = symbol_cast<VariableSymbol>(existingSymbol);
    auto rightVar = symbol_cast<VariableSymbol>(newSymbol);
    if (!leftVar || !rightVar) return false;

    if (!leftVar->constValue().has_value() ||
        !rightVar->constValue().has_value()) {
      return false;
    }

    return leftVar->constValue().value() == rightVar->constValue().value();
  }

  auto matchArg(const TemplateArgument& pat, const TemplateArgument& conc,
                size_t argPos) -> bool {
    auto patSym = asSymbolArgument(pat);
    auto concSym = asSymbolArgument(conc);
    if (!patSym || !concSym) return false;

    auto patPack = symbol_cast<ParameterPackSymbol>(patSym);
    auto concPack = symbol_cast<ParameterPackSymbol>(concSym);
    if (patPack && concPack) {
      PackDeducer deducer{control(), deducedArgs, paramPosition};
      if (!deducer.deducePackElements(patPack->elements(),
                                      concPack->elements())) {
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

    if (auto patParamInfo = getTypeParamInfo(patType)) {
      auto pos = paramPosition(patParamInfo->depth, patParamInfo->index);
      return deduceOrCheck(pos, concSym);
    }

    auto patVar = symbol_cast<VariableSymbol>(patSym);
    auto concVar = symbol_cast<VariableSymbol>(concSym);
    if (patVar && concVar) {
      if (!patVar->constValue().has_value() ||
          !concVar->constValue().has_value()) {
        return false;
      }

      if (patVar->constValue().value() != concVar->constValue().value()) {
        return false;
      }

      ++score;
      ++nonTypeMatches;
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
          auto innerTemplId =
              pattern ? pattern->child(pattern->root, argPos) : nullptr;
          if (!innerTemplId) return false;

          if (!matchNestedWithPattern(innerTemplId, patClassSym,
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

          auto innerTemplId =
              pattern ? pattern->child(pattern->root, argPos) : nullptr;
          if (!innerTemplId) return false;

          if (!matchNestedWithPattern(innerTemplId, primary,
                                      concClassSym->templateArguments())) {
            return false;
          }

          ++score;
          ++nestedMatches;
          return true;
        }
      }
    }

    if (patType != concType) return false;

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

  auto matchNestedWithPattern(SimpleTemplateIdAST* patTemplId,
                              ClassSymbol* primarySym,
                              std::span<const TemplateArgument> concArgs)
      -> bool {
    auto patInnerArgs = Substitution(unit, primarySym->templateDeclaration(),
                                     patTemplId->templateArgumentList)
                            .templateArguments();

    if (patInnerArgs.size() != concArgs.size()) return false;

    std::vector<TemplateArgument> concVec(concArgs.begin(), concArgs.end());

    for (size_t j = 0; j < patInnerArgs.size(); ++j) {
      auto ipSym = asSymbolArgument(patInnerArgs[j]);
      auto icSym = asSymbolArgument(concVec[j]);
      if (!ipSym || !icSym) return false;

      auto ipPack = symbol_cast<ParameterPackSymbol>(ipSym);
      auto icPack = symbol_cast<ParameterPackSymbol>(icSym);
      if (ipPack && icPack) {
        PackDeducer deducer{control(), deducedArgs, paramPosition};
        if (!deducer.deducePackElements(ipPack->elements(),
                                        icPack->elements())) {
          return false;
        }

        ++packMatches;
        continue;
      }

      auto ipType = ipSym->type();
      auto icType = icSym->type();
      if (!ipType || !icType) {
        if (ipType != icType) return false;
        ++score;
        ++exactTypeMatches;
        continue;
      }

      if (auto info = getTypeParamInfo(ipType)) {
        auto pos = paramPosition(info->depth, info->index);
        if (!deduceOrCheck(pos, icSym)) return false;
        continue;
      }

      auto patInner = extractClassSymbol(patInnerArgs[j]);
      auto concInner = extractClassSymbol(concVec[j]);

      if (patInner && concInner && concInner->isSpecialization() &&
          concInner->primaryTemplateSymbol() == patInner &&
          patInner->templateDeclaration()) {
        auto nestedTemplId = pattern ? pattern->child(patTemplId, j) : nullptr;
        if (!nestedTemplId) return false;

        if (!matchNestedWithPattern(nestedTemplId, patInner,
                                    concInner->templateArguments())) {
          return false;
        }

        ++score;
        ++nestedMatches;
        continue;
      }

      if (patInner && concInner && patInner->isSpecialization() &&
          concInner->isSpecialization() &&
          patInner->primaryTemplateSymbol() ==
              concInner->primaryTemplateSymbol()) {
        auto primary = patInner->primaryTemplateSymbol();
        auto nestedTemplId = pattern ? pattern->child(patTemplId, j) : nullptr;
        if (!nestedTemplId || !primary->templateDeclaration()) return false;

        if (!matchNestedWithPattern(nestedTemplId, primary,
                                    concInner->templateArguments())) {
          return false;
        }

        ++score;
        ++nestedMatches;
        continue;
      }

      if (ipType != icType) return false;
      ++score;
      ++exactTypeMatches;
    }

    return true;
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

template <typename SpecEntry>
auto collectCandidate(TranslationUnit* unit, const SpecEntry& spec,
                      const std::vector<TemplateArgument>& templateArguments,
                      int declarationOrder) -> std::optional<Candidate> {
  auto specClass = symbol_cast<ClassSymbol>(spec.symbol);
  if (!specClass) return std::nullopt;

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

  PartialSpecMatcher matcher{unit, nullptr, deducedArgs, paramPosition};

  for (size_t i = 0; i < patternArgs.size(); ++i) {
    if (!matcher.matchArg(patternArgs[i], templateArguments[i], i)) {
      return std::nullopt;
    }
  }

  if (!deducedArgs.complete()) return std::nullopt;

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

}  // namespace

auto ASTRewriter::tryPartialSpecialization(
    TranslationUnit* unit, ClassSymbol* classSymbol,
    const std::vector<TemplateArgument>& templateArguments) -> Symbol* {
  std::vector<Candidate> candidates;
  int declarationOrder = 0;
  for (const auto& spec : classSymbol->specializations()) {
    auto candidate =
        collectCandidate(unit, spec, templateArguments, declarationOrder);
    ++declarationOrder;
    if (!candidate) continue;
    candidates.push_back(std::move(*candidate));
  }

  auto best = bestCandidate(candidates);
  if (best == candidates.end()) return nullptr;

  for (auto it = candidates.begin(); it != candidates.end(); ++it) {
    if (it == best) continue;
    if (!hasEqualSpecificity(*best, *it)) continue;

    auto loc = classSymbol ? classSymbol->location() : SourceLocation{};
    if (best->specBody) loc = best->specBody->firstSourceLocation();
    unit->error(loc, "partial specialization is ambiguous");
    return nullptr;
  }

  auto& selected = *best;

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
  std::vector<Candidate> candidates;
  int declarationOrder = 0;
  for (const auto& spec : variableSymbol->specializations()) {
    auto candidate = collectVariableCandidate(unit, spec, templateArguments,
                                              declarationOrder);
    ++declarationOrder;
    if (!candidate) continue;
    candidates.push_back(std::move(*candidate));
  }

  auto best = bestCandidate(candidates);
  if (best == candidates.end()) return nullptr;

  for (auto it = candidates.begin(); it != candidates.end(); ++it) {
    if (it == best) continue;
    if (!hasEqualSpecificity(*best, *it)) continue;

    auto loc = variableSymbol ? variableSymbol->location() : SourceLocation{};
    unit->error(loc, "partial specialization is ambiguous");
    return nullptr;
  }

  auto& selected = *best;

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
