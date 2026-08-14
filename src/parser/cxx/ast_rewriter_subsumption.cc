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
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/diagnostics_client.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <optional>
#include <set>

namespace cxx {

struct ASTRewriter::ConstraintSubsumption {
  TranslationUnit* unit = nullptr;

  explicit ConstraintSubsumption(TranslationUnit* unit) : unit(unit) {}

  struct Atom {
    ExpressionAST* expression = nullptr;
    std::vector<TemplateArgument> parameterMapping;

    [[nodiscard]] auto isIdenticalTo(TranslationUnit* unit,
                                     const Atom& other) const -> bool {
      if (expression != other.expression) return false;
      return compare_args(unit, parameterMapping, other.parameterMapping);
    }
  };

  struct Constraint {
    enum class Form {
      kAtomic,
      kConceptDependent,
      kFoldExpanded,
      kConjunction,
      kDisjunction
    };

    Form form = Form::kAtomic;
    Atom atom;
    TokenKind foldOperator = TokenKind::T_EOF_SYMBOL;
    std::vector<std::pair<int, int>> unexpandedPacks;
    std::vector<Constraint> operands;
  };

  using Clause = std::vector<const Constraint*>;
  using NormalForm = std::vector<Clause>;

  [[nodiscard]] auto isMoreConstrained(Symbol* symbol, Symbol* other) -> bool;

 private:
  [[nodiscard]] auto normalize(ScopeSymbol* parentScope,
                               ExpressionAST* expression,
                               const std::vector<TemplateArgument>& mapping,
                               int depth) -> std::optional<Constraint>;

  [[nodiscard]] auto normalizeAssociatedConstraints(Symbol* symbol)
      -> std::optional<Constraint>;

  [[nodiscard]] static auto normalForm(const Constraint& constraint,
                                       bool disjunctive)
      -> std::optional<NormalForm>;

  [[nodiscard]] static auto distribute(const NormalForm& lhs,
                                       const NormalForm& rhs)
      -> std::optional<NormalForm>;

  [[nodiscard]] auto subsumes(const std::optional<Constraint>& lhs,
                              const std::optional<Constraint>& rhs) const
      -> std::optional<bool>;

  [[nodiscard]] auto clauseSubsumes(const Clause& lhs, const Clause& rhs) const
      -> bool;

  [[nodiscard]] auto leafSubsumes(const Constraint& lhs,
                                  const Constraint& rhs) const -> bool;

  [[nodiscard]] static auto containsConceptDependent(
      const Constraint& constraint) -> bool;

  [[nodiscard]] auto normalizeFold(ScopeSymbol* parentScope,
                                   ExpressionAST* expression, TokenKind op,
                                   const std::vector<TemplateArgument>& mapping,
                                   int depth) -> std::optional<Constraint>;

  [[nodiscard]] static auto conceptIdOf(ExpressionAST* expression)
      -> std::pair<ConceptSymbol*, SimpleTemplateIdAST*>;

  [[nodiscard]] static auto isIdentityMapping(
      const std::vector<TemplateArgument>& mapping, int depth) -> bool;

  [[nodiscard]] static auto restrictMapping(
      ExpressionAST* expression, const std::vector<TemplateArgument>& mapping,
      int depth) -> std::vector<TemplateArgument>;
};

namespace {
struct ReferencedTemplateParameters final : ASTVisitor {
  int depth = 0;
  std::set<int> indices;
  std::vector<std::pair<int, int>> packs;

  void record(Symbol* symbol) {
    if (!symbol) return;
    auto info = template_parameter_info(symbol);
    if (!info) return;
    if (info->depth == depth) indices.insert(info->index);
    if (info->isPack) {
      auto key = std::pair{info->depth, info->index};
      if (std::ranges::find(packs, key) == packs.end()) packs.push_back(key);
    }
  }

  void visit(NamedTypeSpecifierAST* ast) override {
    record(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(SimpleNestedNameSpecifierAST* ast) override {
    record(ast->symbol);
    ASTVisitor::visit(ast);
  }

  void visit(IdExpressionAST* ast) override {
    record(ast->symbol);
    ASTVisitor::visit(ast);
  }
};
}  // namespace

auto ASTRewriter::ConstraintSubsumption::restrictMapping(
    ExpressionAST* expression, const std::vector<TemplateArgument>& mapping,
    int depth) -> std::vector<TemplateArgument> {
  if (mapping.empty()) return mapping;

  ReferencedTemplateParameters referenced;
  referenced.depth = depth;
  referenced.accept(expression);

  std::vector<TemplateArgument> restricted;
  restricted.reserve(referenced.indices.size());

  for (auto index : referenced.indices) {
    if (index < 0 || index >= static_cast<int>(mapping.size())) continue;
    restricted.push_back(mapping[index]);
  }

  return restricted;
}

auto ASTRewriter::ConstraintSubsumption::isIdentityMapping(
    const std::vector<TemplateArgument>& mapping, int depth) -> bool {
  for (std::size_t index = 0; index < mapping.size(); ++index) {
    const auto& argument = mapping[index];

    std::optional<TypeParamInfo> info;
    if (auto symbol = std::get_if<Symbol*>(&argument)) {
      info = *symbol ? template_parameter_info(*symbol) : std::nullopt;
    } else if (auto type = std::get_if<const Type*>(&argument)) {
      info = getTypeParamInfo(*type);
    }

    if (!info || info->isPack) return false;
    if (info->depth != depth) return false;
    if (info->index != static_cast<int>(index)) return false;
  }
  return true;
}

auto ASTRewriter::ConstraintSubsumption::conceptIdOf(ExpressionAST* expression)
    -> std::pair<ConceptSymbol*, SimpleTemplateIdAST*> {
  auto id = ast_cast<IdExpressionAST>(expression);
  if (!id) return {};
  auto conceptSymbol = symbol_cast<ConceptSymbol>(id->symbol);
  if (!conceptSymbol) return {};
  return {conceptSymbol, ast_cast<SimpleTemplateIdAST>(id->unqualifiedId)};
}

auto ASTRewriter::ConstraintSubsumption::normalize(
    ScopeSymbol* parentScope, ExpressionAST* expression,
    const std::vector<TemplateArgument>& mapping, int depth)
    -> std::optional<Constraint> {
  if (!expression) return std::nullopt;

  if (auto nested = ast_cast<NestedExpressionAST>(expression)) {
    return normalize(parentScope, nested->expression, mapping, depth);
  }

  if (auto binary = ast_cast<BinaryExpressionAST>(expression);
      binary && (binary->op == TokenKind::T_AMP_AMP ||
                 binary->op == TokenKind::T_BAR_BAR)) {
    auto lhs = normalize(parentScope, binary->leftExpression, mapping, depth);
    if (!lhs) return std::nullopt;

    auto rhs = normalize(parentScope, binary->rightExpression, mapping, depth);
    if (!rhs) return std::nullopt;

    Constraint result;
    result.form = binary->op == TokenKind::T_AMP_AMP
                      ? Constraint::Form::kConjunction
                      : Constraint::Form::kDisjunction;
    result.operands.push_back(std::move(*lhs));
    result.operands.push_back(std::move(*rhs));
    return result;
  }

  if (auto fold = ast_cast<RightFoldExpressionAST>(expression);
      fold &&
      (fold->op == TokenKind::T_AMP_AMP || fold->op == TokenKind::T_BAR_BAR)) {
    return normalizeFold(parentScope, fold->expression, fold->op, mapping,
                         depth);
  }

  if (auto fold = ast_cast<LeftFoldExpressionAST>(expression);
      fold &&
      (fold->op == TokenKind::T_AMP_AMP || fold->op == TokenKind::T_BAR_BAR)) {
    return normalizeFold(parentScope, fold->expression, fold->op, mapping,
                         depth);
  }

  if (auto fold = ast_cast<FoldExpressionAST>(expression);
      fold &&
      (fold->op == TokenKind::T_AMP_AMP || fold->op == TokenKind::T_BAR_BAR)) {
    ReferencedTemplateParameters referenced;
    referenced.depth = depth;
    referenced.accept(fold->leftExpression);
    const bool packOnLeft = !referenced.packs.empty();

    auto folded = normalizeFold(
        parentScope, packOnLeft ? fold->leftExpression : fold->rightExpression,
        packOnLeft ? fold->op : fold->foldOp, mapping, depth);
    auto initial = normalize(
        parentScope, packOnLeft ? fold->rightExpression : fold->leftExpression,
        mapping, depth);
    if (!folded || !initial) return std::nullopt;

    Constraint result;
    result.form = fold->op == TokenKind::T_AMP_AMP
                      ? Constraint::Form::kConjunction
                      : Constraint::Form::kDisjunction;
    if (packOnLeft) {
      result.operands.push_back(std::move(*folded));
      result.operands.push_back(std::move(*initial));
    } else {
      result.operands.push_back(std::move(*initial));
      result.operands.push_back(std::move(*folded));
    }
    return result;
  }

  auto [conceptSymbol, templateId] = conceptIdOf(expression);

  if (!conceptSymbol) {
    auto id = ast_cast<IdExpressionAST>(expression);
    if (id && ast_cast<SimpleTemplateIdAST>(id->unqualifiedId) &&
        symbol_cast<TemplateTypeParameterSymbol>(id->symbol)) {
      Constraint result;
      result.form = Constraint::Form::kConceptDependent;
      result.atom.expression = expression;
      result.atom.parameterMapping =
          restrictMapping(expression, mapping, depth);
      return result;
    }
  }

  if (conceptSymbol && templateId) {
    auto definition = conceptSymbol->declaration();
    auto templateDecl = conceptSymbol->templateDeclaration();

    if (definition && definition->expression && templateDecl) {
      auto argumentList = templateId->templateArgumentList;

      if (!isIdentityMapping(mapping, depth)) {
        argumentList = ASTRewriter::substituteTemplateArgumentList(
            unit, argumentList, mapping, depth, parentScope);
      }

      auto subst = Substitution::make(unit, templateDecl, argumentList);
      if (!subst) return std::nullopt;

      return normalize(conceptSymbol->parent(), definition->expression,
                       std::move(*subst).templateArguments(),
                       templateDecl->depth);
    }
  }

  Constraint result;
  result.form = Constraint::Form::kAtomic;
  result.atom.expression = expression;
  result.atom.parameterMapping = restrictMapping(expression, mapping, depth);
  return result;
}

auto ASTRewriter::ConstraintSubsumption::normalizeFold(
    ScopeSymbol* parentScope, ExpressionAST* expression, TokenKind op,
    const std::vector<TemplateArgument>& mapping, int depth)
    -> std::optional<Constraint> {
  auto inner = normalize(parentScope, expression, mapping, depth);
  if (!inner) return std::nullopt;

  ReferencedTemplateParameters referenced;
  referenced.depth = depth;
  referenced.accept(expression);

  Constraint result;
  result.form = Constraint::Form::kFoldExpanded;
  result.foldOperator = op;
  result.unexpandedPacks = std::move(referenced.packs);
  result.operands.push_back(std::move(*inner));
  return result;
}

auto ASTRewriter::ConstraintSubsumption::normalizeAssociatedConstraints(
    Symbol* symbol) -> std::optional<Constraint> {
  auto parentScope = symbol->parent();
  auto templateDeclaration = template_declaration_of(symbol);
  const int depth = templateDeclaration ? templateDeclaration->depth : 0;

  std::optional<Constraint> result;

  for (auto constraint : ASTRewriter::associatedConstraints(unit, symbol)) {
    auto normalized = normalize(parentScope, constraint, {}, depth);
    if (!normalized) return std::nullopt;

    if (!result) {
      result = std::move(normalized);
      continue;
    }

    Constraint conjunction;
    conjunction.form = Constraint::Form::kConjunction;
    conjunction.operands.push_back(std::move(*result));
    conjunction.operands.push_back(std::move(*normalized));
    result = std::move(conjunction);
  }

  return result;
}

auto ASTRewriter::ConstraintSubsumption::distribute(const NormalForm& lhs,
                                                    const NormalForm& rhs)
    -> std::optional<NormalForm> {
  NormalForm result;
  result.reserve(lhs.size() * rhs.size());

  for (const auto& left : lhs) {
    for (const auto& right : rhs) {
      Clause clause = left;
      clause.insert(clause.end(), right.begin(), right.end());
      result.push_back(std::move(clause));
    }
  }

  return result;
}

auto ASTRewriter::ConstraintSubsumption::normalForm(
    const Constraint& constraint, bool disjunctive)
    -> std::optional<NormalForm> {
  if (constraint.form == Constraint::Form::kAtomic)
    return NormalForm{Clause{&constraint}};
  if (constraint.form == Constraint::Form::kConceptDependent ||
      constraint.form == Constraint::Form::kFoldExpanded)
    return NormalForm{Clause{&constraint}};

  auto lhs = normalForm(constraint.operands[0], disjunctive);
  if (!lhs) return std::nullopt;

  auto rhs = normalForm(constraint.operands[1], disjunctive);
  if (!rhs) return std::nullopt;

  const bool isDisjunction = constraint.form == Constraint::Form::kDisjunction;

  if (isDisjunction != disjunctive) return distribute(*lhs, *rhs);

  lhs->insert(lhs->end(), rhs->begin(), rhs->end());
  return lhs;
}

auto ASTRewriter::ConstraintSubsumption::clauseSubsumes(const Clause& lhs,
                                                        const Clause& rhs) const
    -> bool {
  return std::ranges::any_of(lhs, [&](const Constraint* leaf) {
    return std::ranges::any_of(rhs, [&](const Constraint* other) {
      return leafSubsumes(*leaf, *other);
    });
  });
}

auto ASTRewriter::ConstraintSubsumption::leafSubsumes(
    const Constraint& lhs, const Constraint& rhs) const -> bool {
  if (lhs.form == Constraint::Form::kAtomic &&
      rhs.form == Constraint::Form::kAtomic)
    return lhs.atom.isIdenticalTo(unit, rhs.atom);

  if (lhs.form != Constraint::Form::kFoldExpanded ||
      rhs.form != Constraint::Form::kFoldExpanded)
    return false;
  if (lhs.foldOperator != rhs.foldOperator) return false;

  const bool compatible =
      std::ranges::any_of(lhs.unexpandedPacks, [&](const auto& pack) {
        return std::ranges::find(rhs.unexpandedPacks, pack) !=
               rhs.unexpandedPacks.end();
      });
  if (!compatible) return false;

  return subsumes(lhs.operands.front(), rhs.operands.front()).value_or(false);
}

auto ASTRewriter::ConstraintSubsumption::containsConceptDependent(
    const Constraint& constraint) -> bool {
  if (constraint.form == Constraint::Form::kConceptDependent) return true;
  return std::ranges::any_of(constraint.operands, [](const Constraint& child) {
    return containsConceptDependent(child);
  });
}

auto ASTRewriter::ConstraintSubsumption::subsumes(
    const std::optional<Constraint>& lhs,
    const std::optional<Constraint>& rhs) const -> std::optional<bool> {
  if (!rhs.has_value()) return true;
  if (!lhs.has_value()) return false;

  auto disjunctive = normalForm(*lhs, /*disjunctive=*/true);
  if (!disjunctive) return std::nullopt;

  auto conjunctive = normalForm(*rhs, /*disjunctive=*/false);
  if (!conjunctive) return std::nullopt;

  return std::ranges::all_of(*disjunctive, [&](const Clause& clause) {
    return std::ranges::all_of(*conjunctive, [&](const Clause& other) {
      return clauseSubsumes(clause, other);
    });
  });
}

auto ASTRewriter::ConstraintSubsumption::isMoreConstrained(Symbol* symbol,
                                                           Symbol* other)
    -> bool {
  SilentDiagnosticsClient silent;
  auto saved = unit->changeDiagnosticsClient(&silent);

  auto lhs = normalizeAssociatedConstraints(symbol);
  auto rhs = normalizeAssociatedConstraints(other);

  (void)unit->changeDiagnosticsClient(saved);

  auto atLeastAsConstrained = [this](const std::optional<Constraint>& first,
                                     const std::optional<Constraint>& second) {
    if (!second) return true;
    if (!first || containsConceptDependent(*first)) return false;
    return subsumes(first, second).value_or(false);
  };

  return atLeastAsConstrained(lhs, rhs) && !atLeastAsConstrained(rhs, lhs);
}

auto ASTRewriter::isMoreConstrained(TranslationUnit* unit, Symbol* symbol,
                                    Symbol* other) -> bool {
  return ConstraintSubsumption{unit}.isMoreConstrained(symbol, other);
}

auto ASTRewriter::substituteTemplateArgumentList(
    TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList,
    const std::vector<TemplateArgument>& templateArguments, int depth,
    ScopeSymbol* scope) -> List<TemplateArgumentAST*>* {
  auto rewriter = ASTRewriter{unit, scope, templateArguments};
  rewriter.depth_ = depth;

  List<TemplateArgumentAST*>* result = nullptr;
  auto out = &result;

  for (auto argument : ListView{templateArgumentList}) {
    *out = make_list_node(unit->arena(), rewriter.templateArgument(argument));
    out = &(*out)->next;
  }

  return result;
}

}  // namespace cxx
