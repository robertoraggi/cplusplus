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
  return immediateContextDepth_ == 0 &&
         symbol_cast<FunctionSymbol>(binder_.instantiatingSymbol()) &&
         binder_.reportErrors();
}

auto ASTRewriter::shouldCaptureBodyErrors() const -> bool {
  return rewritingFunctionBody_ && shouldReportCheckErrors();
}

void ASTRewriter::typeCheckAndCapture(std::function<void()> checkFn) {
  if (shouldCaptureBodyErrors()) {
    CapturingDiagnosticsClient capture;
    auto saved = unit_->changeDiagnosticsClient(&capture);
    checkFn();
    (void)unit_->changeDiagnosticsClient(saved);
    bodyErrors_.insert(bodyErrors_.end(),
                       std::make_move_iterator(capture.diagnostics.begin()),
                       std::make_move_iterator(capture.diagnostics.end()));
  } else {
    checkFn();
  }
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

auto ASTRewriter::evaluateAssociatedConstraints(TranslationUnit* unit,
                                                Symbol* symbol)
    -> std::optional<bool> {
  auto constraints = associatedConstraints(unit, symbol);
  if (constraints.empty()) return true;
  if (isDependent(unit, symbol->type())) return std::nullopt;

  SilentDiagnosticsClient silent;
  auto saved = unit->changeDiagnosticsClient(&silent);

  auto interp = ASTInterpreter{unit};
  std::optional<bool> conjunction = true;

  for (auto constraint : constraints) {
    if (!constraint->type) {
      auto typeChecker = TypeChecker{unit};
      typeChecker.setScope(symbol->enclosingNonTemplateParametersScope());
      typeChecker.setReportErrors(false);
      typeChecker.check(constraint);
    }

    auto value = interp.evaluate(constraint);
    auto satisfied = value.has_value() ? interp.toBool(*value) : std::nullopt;

    if (!satisfied.has_value()) {
      conjunction = std::nullopt;
      continue;
    }

    if (!*satisfied) {
      (void)unit->changeDiagnosticsClient(saved);
      return false;
    }
  }

  (void)unit->changeDiagnosticsClient(saved);

  return conjunction;
}

auto ASTRewriter::checkConstraintExpression(
    TranslationUnit* unit, Symbol* symbol, ExpressionAST* constraint,
    const std::vector<TemplateArgument>& templateArguments, int depth) -> bool {
  if (!constraint) return true;

  while (auto nested = ast_cast<NestedExpressionAST>(constraint))
    constraint = nested->expression;

  if (auto binary = ast_cast<BinaryExpressionAST>(constraint);
      binary && (binary->op == TokenKind::T_AMP_AMP ||
                 binary->op == TokenKind::T_BAR_BAR)) {
    const bool left = checkConstraintExpression(
        unit, symbol, binary->leftExpression, templateArguments, depth);

    if (binary->op == TokenKind::T_AMP_AMP && !left) return false;
    if (binary->op == TokenKind::T_BAR_BAR && left) return true;

    return checkConstraintExpression(unit, symbol, binary->rightExpression,
                                     templateArguments, depth);
  }

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
  reqRewriter.depth_ = depth;
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    (void)reqRewriter.parameterDeclarationClause(
        functionParameterClause(function));
  }
  auto rewritten = reqRewriter.expression(constraint);
  if (!rewritten) return true;

  reqRewriter.check(rewritten);

  if (isDependent(unit, rewritten)) return true;

  auto interp = ASTInterpreter{unit};
  auto val = interp.evaluate(rewritten);
  if (!val.has_value()) return false;

  auto boolVal = interp.toBool(*val);
  return boolVal.value_or(false);
}

auto ASTRewriter::checkAssociatedConstraints(
    TranslationUnit* unit, Symbol* symbol,
    const std::vector<TemplateArgument>& templateArguments, int depth) -> bool {
  for (auto constraint : associatedConstraints(unit, symbol)) {
    if (!checkConstraintExpression(unit, symbol, constraint, templateArguments,
                                   depth))
      return false;
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

    return isConjunction ? (*left && *right) : (*left || *right);
  }

  SilentDiagnosticsClient silent;
  auto saved = unit->changeDiagnosticsClient(&silent);

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = depth;

  auto constraint = rewriter.expression(expression);
  if (constraint) rewriter.check(constraint);

  (void)unit->changeDiagnosticsClient(saved);

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

  return evaluateConstraintExpression(
      unit, conceptSymbol->parent(), definition->expression,
      std::move(*subst).templateArguments(), templateDecl->depth);
}

void ASTRewriter::check(ExpressionAST* ast) {
  if (!ast) return;
  if (isDependent(unit_, ast)) return;

  auto typeChecker = TypeChecker{unit_};
  typeChecker.setScope(binder_.scope());
  typeChecker.setReportErrors(shouldReportCheckErrors());
  typeChecker.setPotentiallyEvaluated(unevaluatedOperandDepth_ == 0);
  typeCheckAndCapture([&] { typeChecker.check(ast); });
}

void ASTRewriter::checkUnevaluated(ExpressionAST* ast) {
  ++unevaluatedOperandDepth_;
  check(ast);
  --unevaluatedOperandDepth_;
}
}  // namespace cxx
