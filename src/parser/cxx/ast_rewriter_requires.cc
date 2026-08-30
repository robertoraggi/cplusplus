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
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>

#include <ranges>

namespace cxx {
namespace {
auto functionParameterClause(FunctionSymbol* function)
    -> ParameterDeclarationClauseAST* {
  if (!function) return nullptr;

  auto declaration = template_declaration_ast(function);
  if (auto definition = ast_cast<FunctionDefinitionAST>(declaration)) {
    if (auto prototype = getFunctionPrototype(definition->declarator))
      return prototype->parameterDeclarationClause;
  }

  auto simpleDeclaration = ast_cast<SimpleDeclarationAST>(declaration);
  if (!simpleDeclaration) return nullptr;
  for (auto initDeclarator : ListView{simpleDeclaration->initDeclaratorList}) {
    auto candidate = symbol_cast<FunctionSymbol>(initDeclarator->symbol);
    if (!candidate || candidate->canonical() != function->canonical()) continue;
    if (auto prototype = getFunctionPrototype(initDeclarator->declarator))
      return prototype->parameterDeclarationClause;
  }
  return nullptr;
}
}  // namespace

ASTRewriter::ImmediateContextGuard::ImmediateContextGuard(ASTRewriter& rewrite)
    : rewrite_(rewrite),
      silent_(rewrite.unit_),
      substitutionFailed_(std::exchange(rewrite.substitutionFailed_, false)) {
  ++rewrite_.immediateContextDepth_;
}

ASTRewriter::ImmediateContextGuard::~ImmediateContextGuard() {
  --rewrite_.immediateContextDepth_;
  rewrite_.substitutionFailed_ = substitutionFailed_;
}

auto ASTRewriter::shouldReportCheckErrors() const -> bool {
  return immediateContextDepth_ == 0 && !pastingCheckedBody_ &&
         binder_.reportErrors();
}

auto ASTRewriter::shouldCaptureBodyErrors() const -> bool {
  return rewritingFunctionBody_ &&
         symbol_cast<FunctionSymbol>(binder_.instantiatingSymbol()) &&
         shouldReportCheckErrors();
}

auto ASTRewriter::typeChecker() -> TypeChecker {
  auto typeChecker = TypeChecker{unit_};
  typeChecker.setScope(binder_.scope());
  typeChecker.setReportErrors(shouldReportCheckErrors());
  return typeChecker;
}

ASTRewriter::BodyErrorScope::BodyErrorScope(ASTRewriter& rewrite)
    : rewrite_(rewrite),
      rewritingFunctionBody_(
          std::exchange(rewrite.rewritingFunctionBody_, true)) {
  if (rewrite_.shouldCaptureBodyErrors()) capture_.emplace(rewrite_.unit_);
}

ASTRewriter::BodyErrorScope::~BodyErrorScope() {
  if (capture_.has_value()) {
    capture_->finish();
    auto diagnostics = capture_->takeDiagnostics();
    auto& bodyErrors = rewrite_.bodyErrors_;
    bodyErrors.insert(bodyErrors.end(),
                      std::make_move_iterator(diagnostics.begin()),
                      std::make_move_iterator(diagnostics.end()));
  }
  rewrite_.rewritingFunctionBody_ = rewritingFunctionBody_;
}

auto ASTRewriter::typeConstraintExpression(
    TranslationUnit* unit, ConstraintTypeParameterSymbol* parameter)
    -> ExpressionAST* {
  if (!parameter) return nullptr;
  if (auto cached = parameter->constraintExpression()) return cached;

  auto typeConstraint = parameter->typeConstraint();
  if (!typeConstraint || !typeConstraint->symbol) return nullptr;

  auto arena = unit->arena();

  auto constrainedSpecifier = NamedTypeSpecifierAST::create(arena);
  constrainedSpecifier->unqualifiedId =
      NameIdAST::create(arena, name_cast<Identifier>(parameter->name()));
  constrainedSpecifier->symbol = parameter;

  auto constrainedTypeId = TypeIdAST::create(arena);
  constrainedTypeId->typeSpecifierList =
      make_list_node<SpecifierAST>(arena, constrainedSpecifier);
  constrainedTypeId->type = parameter->type();

  auto constrainedArgument = TypeTemplateArgumentAST::create(arena);
  constrainedArgument->typeId = constrainedTypeId;

  auto templateId = SimpleTemplateIdAST::create(arena);
  templateId->identifierLoc = typeConstraint->identifierLoc;
  templateId->lessLoc = typeConstraint->lessLoc;
  templateId->greaterLoc = typeConstraint->greaterLoc;
  templateId->identifier = typeConstraint->identifier;
  templateId->symbol = typeConstraint->symbol;

  auto out = &templateId->templateArgumentList;
  *out = make_list_node<TemplateArgumentAST>(arena, constrainedArgument);
  out = &(*out)->next;

  for (auto argument : ListView{typeConstraint->templateArgumentList}) {
    *out = make_list_node(arena, argument);
    out = &(*out)->next;
  }

  auto idExpression = IdExpressionAST::create(arena);
  idExpression->nestedNameSpecifier = typeConstraint->nestedNameSpecifier;
  idExpression->unqualifiedId = templateId;
  idExpression->symbol = typeConstraint->symbol;
  idExpression->type = unit->control()->getBoolType();
  idExpression->valueCategory = ValueCategory::kPrValue;

  parameter->setConstraintExpression(idExpression);

  return idExpression;
}

auto ASTRewriter::associatedConstraints(TranslationUnit* unit, Symbol* symbol)
    -> std::vector<ExpressionAST*> {
  std::vector<ExpressionAST*> constraints;

  auto templateDeclaration = template_declaration_of(symbol);

  for (auto parameter :
       ListView{templateDeclaration ? templateDeclaration->templateParameterList
                                    : nullptr}) {
    auto constrained =
        symbol_cast<ConstraintTypeParameterSymbol>(parameter->symbol);
    if (!constrained) continue;
    if (auto constraint = typeConstraintExpression(unit, constrained))
      constraints.push_back(constraint);
  }

  auto append = [&](RequiresClauseAST* clause) {
    if (clause && clause->expression) constraints.push_back(clause->expression);
  };

  if (templateDeclaration) append(templateDeclaration->requiresClause);

  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    if (auto clause = function->trailingRequiresClause()) {
      append(clause);
    } else if (auto declaration = function->declaration()) {
      append(declaration->requiresClause);
    }
  }

  return constraints;
}

namespace {

auto constrainedPrimaryTemplateOf(Symbol* symbol) -> FunctionSymbol* {
  auto function = symbol_cast<FunctionSymbol>(symbol);
  if (!function) return nullptr;
  if (!function->isSpecialization()) return nullptr;
  auto primary = function->primaryTemplateSymbol();
  if (!primary) return nullptr;
  if (!primary->templateDeclaration()) return nullptr;
  return primary;
}

}  // namespace

auto ASTRewriter::evaluateSpecializationConstraints(TranslationUnit* unit,
                                                    FunctionSymbol* symbol,
                                                    FunctionSymbol* primary)
    -> std::optional<bool> {
  if (associatedConstraints(unit, primary).empty()) return true;

  auto arguments = symbol->templateArguments();
  std::vector<TemplateArgument> templateArguments{arguments.begin(),
                                                  arguments.end()};

  return checkAssociatedConstraints(unit, primary, templateArguments,
                                    primary->templateDeclaration()->depth);
}

auto ASTRewriter::evaluateAssociatedConstraints(TranslationUnit* unit,
                                                Symbol* symbol)
    -> std::optional<bool> {
  if (auto primary = constrainedPrimaryTemplateOf(symbol)) {
    return evaluateSpecializationConstraints(
        unit, symbol_cast<FunctionSymbol>(symbol), primary);
  }

  auto constraints = associatedConstraints(unit, symbol);
  if (constraints.empty()) return true;
  if (isDependent(unit, symbol->type())) return std::nullopt;

  auto interp = ASTInterpreter{unit};
  std::optional<bool> conjunction = true;

  for (auto constraint : constraints) {
    std::optional<ConstValue> value;
    bool hadError = false;

    {
      SilentDiagnosticsScope silent{unit};

      if (!constraint->type) {
        auto typeChecker = TypeChecker{unit};
        typeChecker.setScope(symbol->parent());
        typeChecker.setReportErrors(false);
        typeChecker.check(constraint);
      }

      value = interp.evaluate(constraint);
      hadError = silent.hadError();
    }

    if (hadError) return false;

    std::optional<bool> satisfied;
    if (value.has_value()) satisfied = interp.toBool(*value);

    if (!satisfied.has_value()) {
      conjunction = std::nullopt;
      continue;
    }
    if (!*satisfied) return false;
  }

  return conjunction;
}

auto ASTRewriter::checkConstraintExpression(
    TranslationUnit* unit, Symbol* symbol, ExpressionAST* constraint,
    const std::vector<TemplateArgument>& templateArguments, int depth)
    -> std::optional<bool> {
  if (!constraint) return true;

  while (auto nested = ast_cast<NestedExpressionAST>(constraint))
    constraint = nested->expression;

  if (auto binary = ast_cast<BinaryExpressionAST>(constraint)) {
    const bool isConjunction = binary->op == TokenKind::T_AMP_AMP;
    const bool isDisjunction = binary->op == TokenKind::T_BAR_BAR;
    if (isConjunction || isDisjunction) {
      auto left = checkConstraintExpression(
          unit, symbol, binary->leftExpression, templateArguments, depth);

      if (left.has_value()) {
        if (isConjunction) {
          if (!*left) return false;
        } else if (*left) {
          return true;
        }
      }

      auto right = checkConstraintExpression(
          unit, symbol, binary->rightExpression, templateArguments, depth);
      if (right.has_value()) {
        if (isConjunction) {
          if (!*right) return false;
        } else if (*right) {
          return true;
        }
      }

      if (!left.has_value()) return std::nullopt;
      if (!right.has_value()) return std::nullopt;
      if (isConjunction) return *left && *right;
      return *left || *right;
    }
  }

  auto parentScope = symbol->parent();
  auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
  reqRewriter.depth_ = depth;
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    (void)reqRewriter.parameterDeclarationClause(
        functionParameterClause(function));
  }

  auto substituteIntoConstraint = [&]() -> ExpressionAST* {
    ExpressionAST* rewritten = nullptr;
    bool hadError = false;

    {
      SilentDiagnosticsScope silent{unit};

      rewritten = reqRewriter.expression(constraint);
      if (rewritten) reqRewriter.check(rewritten);

      hadError = silent.hadError();
    }

    if (hadError) return nullptr;
    if (reqRewriter.substitutionFailed()) return nullptr;
    return rewritten;
  };

  auto rewritten = substituteIntoConstraint();
  if (!rewritten) return false;

  if (isDependent(unit, rewritten)) return std::nullopt;

  auto interp = ASTInterpreter{unit};
  auto val = interp.evaluate(rewritten);
  if (!val.has_value()) return false;

  auto boolVal = interp.toBool(*val);
  return boolVal.value_or(false);
}

auto ASTRewriter::checkAssociatedConstraints(
    TranslationUnit* unit, Symbol* symbol,
    const std::vector<TemplateArgument>& templateArguments, int depth) -> bool {
  auto constraints = associatedConstraints(unit, symbol);
  if (constraints.empty()) return true;

  const bool cacheable = std::ranges::none_of(
      templateArguments, [&](const TemplateArgument& argument) {
        return isDependentTemplateArgument(unit, argument);
      });

  if (cacheable) {
    auto cached = unit->cachedConstraintSatisfaction(symbol, constraints,
                                                     templateArguments);
    if (cached.has_value()) return *cached;
  }

  bool determinate = true;
  for (auto constraint : constraints) {
    auto result = checkConstraintExpression(unit, symbol, constraint,
                                            templateArguments, depth);
    if (!result.has_value()) {
      determinate = false;
      continue;
    }
    if (!*result) {
      if (cacheable)
        unit->cacheConstraintSatisfaction(symbol, std::move(constraints),
                                          templateArguments, false);
      return false;
    }
  }

  if (cacheable) {
    if (determinate)
      unit->cacheConstraintSatisfaction(symbol, std::move(constraints),
                                        templateArguments, true);
  }
  return true;
}

auto ASTRewriter::evaluateConstraintExpression(
    TranslationUnit* unit, ScopeSymbol* parentScope, ExpressionAST* expression,
    const std::vector<TemplateArgument>& templateArguments, int depth)
    -> std::optional<bool> {
  if (auto binary = ast_cast<BinaryExpressionAST>(expression);
      binary && (binary->op == TokenKind::T_AMP_AMP ||
                 binary->op == TokenKind::T_BAR_BAR)) {
    const bool isConjunction = binary->op == TokenKind::T_AMP_AMP;

    auto left = evaluateConstraintExpression(
        unit, parentScope, binary->leftExpression, templateArguments, depth);

    if (left.has_value() && *left != isConjunction) return left;

    auto right = evaluateConstraintExpression(
        unit, parentScope, binary->rightExpression, templateArguments, depth);

    if (!left.has_value() || !right.has_value()) return std::nullopt;

    if (isConjunction) return *left && *right;
    return *left || *right;
  }

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = depth;

  ExpressionAST* constraint = nullptr;

  {
    SilentDiagnosticsScope silent{unit};

    constraint = rewriter.expression(expression);
    if (constraint) rewriter.check(constraint);
  }

  if (!constraint) return std::nullopt;
  if (rewriter.substitutionFailed()) return false;

  auto interp = ASTInterpreter{unit};
  auto value = interp.evaluate(constraint);
  if (!value.has_value()) return std::nullopt;

  return interp.toBool(*value);
}

auto ASTRewriter::constraintOperandDeterminesResult(ExpressionAST* operand,
                                                    TokenKind op) const
    -> bool {
  if (op != TokenKind::T_AMP_AMP && op != TokenKind::T_BAR_BAR) return false;
  if (!operand) return false;

  auto interpreter = ASTInterpreter{unit_};
  auto value = interpreter.evaluate(operand);
  if (!value.has_value()) return false;
  auto result = interpreter.toBool(*value);
  if (!result.has_value()) return false;

  if (op == TokenKind::T_AMP_AMP) return !*result;
  return *result;
}

auto ASTRewriter::evaluateConcept(
    TranslationUnit* unit, ConceptSymbol* conceptSymbol,
    List<TemplateArgumentAST*>* templateArgumentList) -> std::optional<bool> {
  if (!conceptSymbol) return std::nullopt;

  auto definition = conceptSymbol->declaration();
  if (!definition || !definition->expression) return std::nullopt;

  auto templateDecl = conceptSymbol->templateDeclaration();
  if (!templateDecl) return std::nullopt;

  auto subst = Substitution::make(unit, templateDecl, templateArgumentList);
  if (!subst) return std::nullopt;

  auto templateArguments = std::move(*subst).templateArguments();
  const bool cacheable = std::ranges::none_of(
      templateArguments, [&](const TemplateArgument& argument) {
        return isDependentTemplateArgument(unit, argument);
      });
  std::vector<ExpressionAST*> constraints{definition->expression};

  if (cacheable) {
    auto cached = unit->cachedConstraintSatisfaction(conceptSymbol, constraints,
                                                     templateArguments);
    if (cached.has_value()) return cached;
  }

  auto result = evaluateConstraintExpression(
      unit, conceptSymbol->parent(), definition->expression, templateArguments,
      templateDecl->depth);
  if (cacheable) {
    if (result.has_value())
      unit->cacheConstraintSatisfaction(conceptSymbol, std::move(constraints),
                                        std::move(templateArguments), *result);
  }
  return result;
}

void ASTRewriter::check(ExpressionAST* ast) {
  if (!ast) return;
  if (isDependent(unit_, ast)) return;

  TranslationUnit::PotentiallyEvaluatedScope evaluated{
      unit_, unevaluatedOperandDepth_ == 0};
  auto checker = typeChecker();
  checker.check(ast);
}
}  // namespace cxx
