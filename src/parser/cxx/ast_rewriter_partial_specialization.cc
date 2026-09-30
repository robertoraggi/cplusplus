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
#include <cxx/decl.h>
#include <cxx/diagnostics_client.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_deduction.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <span>

namespace cxx {
namespace {
[[nodiscard]] auto declaredSpecializationTemplate(Symbol* symbol)
    -> TemplateDeclarationAST* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol))
    return classSymbol->resolvedDefinition()->templateDeclaration();
  if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol))
    return variableSymbol->templateDeclaration();
  return nullptr;
}

[[nodiscard]] auto writtenTemplateId(ClassSymbol* specialization)
    -> SimpleTemplateIdAST* {
  auto body = ast_cast<ClassSpecifierAST>(specialization->declaration());
  return body ? ast_cast<SimpleTemplateIdAST>(body->unqualifiedId) : nullptr;
}

[[nodiscard]] auto writtenTemplateId(TemplateDeclarationAST* templateDecl)
    -> SimpleTemplateIdAST* {
  auto declaration = ast_cast<SimpleDeclarationAST>(templateDecl->declaration);
  if (!declaration || !declaration->initDeclaratorList) return nullptr;
  auto declaratorId =
      getDeclaratorId(declaration->initDeclaratorList->value->declarator);
  return declaratorId
             ? ast_cast<SimpleTemplateIdAST>(declaratorId->unqualifiedId)
             : nullptr;
}
}  // namespace

struct ASTRewriter::RewritePartialSpecialization {
  TranslationUnit* unit = nullptr;

  explicit RewritePartialSpecialization(TranslationUnit* unit) : unit(unit) {}

  struct Candidate {
    Symbol* symbol = nullptr;
    TemplateDeclarationAST* specTemplateDecl = nullptr;
    SimpleTemplateIdAST* templateId = nullptr;
    std::vector<TemplateArgument> patternArguments;
    std::vector<TemplateArgument> deducedArgs;
  };

  struct Selection {
    std::optional<Candidate> candidate;
    bool ambiguous = false;
  };

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
  [[nodiscard]] auto candidate(
      Symbol* primary, const TemplateSpecialization& specialization,
      const std::vector<TemplateArgument>& templateArguments)
      -> std::optional<Candidate>;

  [[nodiscard]] auto reproducesArguments(
      Symbol* primary, const Candidate& candidate,
      const std::vector<TemplateArgument>& templateArguments) -> bool;

  [[nodiscard]] auto select(
      Symbol* primary, std::span<const TemplateSpecialization> specializations,
      const std::vector<TemplateArgument>& templateArguments) -> Selection;

  [[nodiscard]] auto isMoreSpecialized(const Candidate& lhs,
                                       const Candidate& rhs) const -> bool;

  [[nodiscard]] auto isAtLeastAsSpecialized(const Candidate& lhs,
                                            const Candidate& rhs) const -> bool;

  [[nodiscard]] auto hasEquivalentTransformedType(const Candidate& lhs,
                                                  const Candidate& rhs) const
      -> bool;

  [[nodiscard]] auto hasEquivalentArgument(
      const TemplateArgument& lhs, const TemplateArgument& rhs,
      TemplateDeclarationAST* lhsTemplateDecl,
      TemplateDeclarationAST* rhsTemplateDecl) const -> bool;
};

auto ASTRewriter::RewritePartialSpecialization::candidate(
    Symbol* primary, const TemplateSpecialization& specialization,
    const std::vector<TemplateArgument>& templateArguments)
    -> std::optional<Candidate> {
  auto specTemplateDecl = declaredSpecializationTemplate(specialization.symbol);
  if (!specTemplateDecl) return std::nullopt;

  auto symbol = specialization.symbol;
  SimpleTemplateIdAST* templateId = nullptr;
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    symbol = classSymbol->resolvedDefinition();
    templateId = writtenTemplateId(classSymbol->resolvedDefinition());
  } else {
    templateId = writtenTemplateId(specTemplateDecl);
  }

  TypeDeduction deduction{unit, specTemplateDecl};
  if (!deduction.deduce(specialization.arguments, templateArguments)) {
    return std::nullopt;
  }

  auto deduced = deduction.templateArguments();
  if (!deduced) return std::nullopt;

  Candidate result{.symbol = symbol,
                   .specTemplateDecl = specTemplateDecl,
                   .templateId = templateId,
                   .patternArguments = specialization.arguments,
                   .deducedArgs = std::move(*deduced)};

  if (!reproducesArguments(primary, result, templateArguments)) {
    return std::nullopt;
  }

  if (!ASTRewriter::checkAssociatedConstraints(unit, symbol, result.deducedArgs,
                                               specTemplateDecl->depth))
    return std::nullopt;

  return result;
}

auto ASTRewriter::RewritePartialSpecialization::reproducesArguments(
    Symbol* primary, const Candidate& candidate,
    const std::vector<TemplateArgument>& templateArguments) -> bool {
  auto primaryDeclaration = template_declaration_of(primary);
  if (!candidate.templateId || !primaryDeclaration) return false;

  SilentDiagnosticsScope silent{unit};
  auto scope = candidate.symbol->parent();
  auto rewriter = ASTRewriter{unit, scope, candidate.deducedArgs};
  rewriter.depth_ = candidate.specTemplateDecl->depth;
  rewriter.inheritEnclosingTemplateArguments(scope);
  auto substituted = rewriter.rewriteTemplateArgumentList(
      candidate.templateId->templateArgumentList);
  if (silent.hadError()) return false;

  auto substitution = Substitution::make(unit, primaryDeclaration, substituted);
  if (!substitution || silent.hadError()) return false;

  return compare_args(unit, substitution->templateArguments(),
                      templateArguments);
}

auto ASTRewriter::RewritePartialSpecialization::isAtLeastAsSpecialized(
    const Candidate& lhs, const Candidate& rhs) const -> bool {
  TypeDeduction deduction{unit, rhs.specTemplateDecl};
  if (!deduction.deduce(rhs.patternArguments, lhs.patternArguments))
    return false;
  return deduction.templateArguments().has_value();
}

auto ASTRewriter::RewritePartialSpecialization::hasEquivalentArgument(
    const TemplateArgument& lhs, const TemplateArgument& rhs,
    TemplateDeclarationAST* lhsTemplateDecl,
    TemplateDeclarationAST* rhsTemplateDecl) const -> bool {
  auto lhsType = template_argument_type(lhs);
  auto rhsType = template_argument_type(rhs);
  if (lhsType || rhsType) {
    if (!lhsType || !rhsType) return false;
    return TemplateEquivalence{unit}.sameForOrdering(
        lhsType, rhsType, lhsTemplateDecl, rhsTemplateDecl);
  }

  auto lhsValue = template_argument_value(lhs);
  auto rhsValue = template_argument_value(rhs);
  if (lhsValue || rhsValue) {
    if (!lhsValue || !rhsValue) return false;
    return *lhsValue == *rhsValue;
  }

  auto lhsSymbol = std::get_if<Symbol*>(&lhs);
  auto rhsSymbol = std::get_if<Symbol*>(&rhs);
  if (!lhsSymbol || !rhsSymbol) return false;

  auto lhsPack = symbol_cast<ParameterPackSymbol>(*lhsSymbol);
  auto rhsPack = symbol_cast<ParameterPackSymbol>(*rhsSymbol);
  if (lhsPack || rhsPack) {
    if (!lhsPack || !rhsPack) return false;
    if (lhsPack->elements().size() != rhsPack->elements().size()) return false;
    for (std::size_t i = 0; i < lhsPack->elements().size(); ++i) {
      if (!hasEquivalentArgument(lhsPack->elements()[i], rhsPack->elements()[i],
                                 lhsTemplateDecl, rhsTemplateDecl))
        return false;
    }
    return true;
  }

  auto lhsInfo = template_parameter_info(*lhsSymbol);
  auto rhsInfo = template_parameter_info(*rhsSymbol);
  if (!lhsInfo || !rhsInfo) return false;

  return lhsInfo->index == rhsInfo->index && lhsInfo->isPack == rhsInfo->isPack;
}

auto ASTRewriter::RewritePartialSpecialization::hasEquivalentTransformedType(
    const Candidate& lhs, const Candidate& rhs) const -> bool {
  if (!TemplateEquivalence{unit}.sameForOrdering(
          lhs.specTemplateDecl->templateParameterList,
          rhs.specTemplateDecl->templateParameterList))
    return false;

  if (lhs.patternArguments.size() != rhs.patternArguments.size()) return false;

  for (std::size_t i = 0; i < lhs.patternArguments.size(); ++i) {
    if (!hasEquivalentArgument(lhs.patternArguments[i], rhs.patternArguments[i],
                               lhs.specTemplateDecl, rhs.specTemplateDecl))
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
  return ASTRewriter::isMoreConstrained(unit, lhs.symbol, rhs.symbol);
}

auto ASTRewriter::RewritePartialSpecialization::select(
    Symbol* primary, std::span<const TemplateSpecialization> specializations,
    const std::vector<TemplateArgument>& templateArguments) -> Selection {
  std::vector<Candidate> candidates;
  for (const auto& specialization : specializations) {
    if (auto trace = unit->timeTrace())
      trace->count(TimeTrace::kPartialSpecializationVisits);
    if (auto match = candidate(primary, specialization, templateArguments))
      candidates.push_back(std::move(*match));
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
    auto location = candidates.front().templateId
                        ? candidates.front().templateId->firstSourceLocation()
                        : primary->location();
    unit->error(location, "partial specialization is ambiguous");
    return {.ambiguous = true};
  }

  return {.candidate = std::move(*best)};
}

auto ASTRewriter::RewritePartialSpecialization::findPattern(
    ClassSymbol* primary, List<TemplateArgumentAST*>* templateArgumentList)
    -> ClassSymbol* {
  if (primary && primary->isSpecialization())
    primary = primary->primaryTemplateSymbol();

  if (!primary) return nullptr;

  for (const auto& specialization :
       primary->declaredSpecializations(unit->timeTrace())) {
    auto classSymbol = symbol_cast<ClassSymbol>(specialization.symbol);
    if (!classSymbol) continue;
    auto definition = classSymbol->resolvedDefinition();
    if (!definition->templateDeclaration()) continue;
    auto templateId = writtenTemplateId(definition);
    if (!templateId) continue;
    if (TemplateEquivalence{unit}.same(templateId->templateArgumentList,
                                       templateArgumentList))
      return definition;
  }

  return nullptr;
}

auto ASTRewriter::RewritePartialSpecialization::apply(
    ClassSymbol* classSymbol,
    const std::vector<TemplateArgument>& templateArguments)
    -> PartialSpecializationResult {
  auto selection = select(
      classSymbol, classSymbol->declaredSpecializations(unit->timeTrace()),
      templateArguments);

  if (!selection.candidate) return {.resolutionFailed = selection.ambiguous};
  auto& selected = *selection.candidate;
  auto specClass = symbol_cast<ClassSymbol>(selected.symbol);
  auto specBody = ast_cast<ClassSpecifierAST>(specClass->declaration());
  if (!specBody) return {.resolutionFailed = true};

  TimeTrace::Scope trace{unit->timeTrace(), "Instantiate", specClass};
  if (auto trace = unit->timeTrace()) trace->count(TimeTrace::kInstantiations);
  auto specParentScope = specClass->parent();
  auto specRewriter = ASTRewriter{unit, specParentScope, selected.deducedArgs};
  specRewriter.depth_ = selected.specTemplateDecl->depth;
  specRewriter.inheritEnclosingTemplateArguments(specParentScope);
  specRewriter.binder().setInstantiatingSymbol(specClass);

  auto pendingInstance = symbol_cast<ClassSymbol>(
      classSymbol->findSpecialization(unit, templateArguments));
  if (!pendingInstance) {
    pendingInstance = unit->control()->newClassSymbol(classSymbol->parent(),
                                                      classSymbol->location());
    pendingInstance->setName(classSymbol->name());
    pendingInstance->setType(unit->control()->getClassType(pendingInstance));
    classSymbol->addSpecialization(unit, templateArguments, pendingInstance);
  }
  pendingInstance->setInstantiationPattern(specClass);
  specRewriter.setClassInstanceToComplete(pendingInstance);

  auto instance = ast_cast<ClassSpecifierAST>(specRewriter.specifier(specBody));
  if (!instance || !instance->symbol) return {.resolutionFailed = true};

  if (auto instanceClass = symbol_cast<ClassSymbol>(instance->symbol)) {
    instanceClass->setInstantiationPattern(specClass);
    classSymbol->addSpecialization(unit, templateArguments, instanceClass);
  }

  return {.symbol = instance->symbol};
}

auto ASTRewriter::RewritePartialSpecialization::apply(
    VariableSymbol* variableSymbol,
    const std::vector<TemplateArgument>& templateArguments)
    -> PartialSpecializationResult {
  auto selection =
      select(variableSymbol,
             variableSymbol->declaredSpecializations(unit->timeTrace()),
             templateArguments);

  if (!selection.candidate) return {.resolutionFailed = selection.ambiguous};
  auto& selected = *selection.candidate;
  auto specVar = symbol_cast<VariableSymbol>(selected.symbol);

  if (auto cached = specVar->findSpecialization(unit, selected.deducedArgs)) {
    if (auto cachedVar = symbol_cast<VariableSymbol>(cached)) {
      variableSymbol->addSpecialization(unit, templateArguments, cachedVar);
    }
    return {.symbol = cached};
  }

  auto specTemplateDecl = specVar->templateDeclaration();
  if (!specTemplateDecl) return {.resolutionFailed = true};

  auto simpleDecl =
      ast_cast<SimpleDeclarationAST>(specTemplateDecl->declaration);
  if (!simpleDecl) return {.resolutionFailed = true};

  TimeTrace::Scope trace{unit->timeTrace(), "Instantiate", specVar};
  if (auto trace = unit->timeTrace()) trace->count(TimeTrace::kInstantiations);
  auto specParentScope = specVar->parent();
  auto specRewriter = ASTRewriter{unit, specParentScope, selected.deducedArgs};
  specRewriter.depth_ = selected.specTemplateDecl->depth;
  specRewriter.inheritEnclosingTemplateArguments(specParentScope);
  specRewriter.binder().setInstantiatingSymbol(specVar);

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

auto ASTRewriter::findUndeducedPartialSpecializationParameter(
    TranslationUnit* unit, TemplateDeclarationAST* templateDeclaration,
    const std::vector<TemplateArgument>& templateArguments)
    -> TemplateParameterAST* {
  if (!templateDeclaration) return nullptr;
  TypeDeduction deduction{unit, templateDeclaration};
  (void)deduction.deduce(templateArguments, templateArguments);
  return deduction.undeducedParameter();
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
