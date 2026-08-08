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
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>

namespace cxx {
auto ASTRewriter::shouldReportCheckErrors() const -> bool {
  return symbol_cast<FunctionSymbol>(binder_.instantiatingSymbol()) &&
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
    if (auto declaration = function->declaration())
      append(declaration->requiresClause);
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

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
  reqRewriter.depth_ = depth;
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

  auto parentScope = conceptSymbol->parent();

  SilentDiagnosticsClient silent;
  auto saved = unit->changeDiagnosticsClient(&silent);

  auto rewriter = ASTRewriter{unit, parentScope, templateArguments};
  rewriter.depth_ = templateDecl->depth;

  auto constraint = rewriter.expression(definition->expression);
  if (constraint) rewriter.check(constraint);

  (void)unit->changeDiagnosticsClient(saved);

  if (!constraint) return std::nullopt;
  if (rewriter.substitutionFailed()) return false;

  auto interp = ASTInterpreter{unit};
  auto value = interp.evaluate(constraint);
  if (!value.has_value()) return std::nullopt;

  return interp.toBool(*value);
}

void ASTRewriter::check(ExpressionAST* ast) {
  if (!ast) return;
  if (isDependent(unit_, ast)) return;

  auto typeChecker = TypeChecker{unit_};
  typeChecker.setScope(binder_.scope());
  typeChecker.setReportErrors(shouldReportCheckErrors());
  typeCheckAndCapture([&] { typeChecker.check(ast); });
}
}  // namespace cxx
