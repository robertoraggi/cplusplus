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
#include <cxx/ast_interpreter.h>
#include <cxx/dependent_types.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

namespace cxx {

auto ASTRewriter::shouldCaptureBodyErrors() const -> bool {
  return symbol_cast<FunctionSymbol>(binder_.instantiatingSymbol()) &&
         binder_.reportErrors();
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

auto ASTRewriter::checkRequiresClause(
    TranslationUnit* unit, Symbol* symbol, RequiresClauseAST* clause,
    const std::vector<TemplateArgument>& templateArguments, int depth) -> bool {
  if (!clause) return true;

  auto parentScope = symbol->enclosingNonTemplateParametersScope();
  auto reqRewriter = ASTRewriter{unit, parentScope, templateArguments};
  reqRewriter.depth_ = depth;
  auto rewrittenClause = reqRewriter.requiresClause(clause);
  if (!rewrittenClause || !rewrittenClause->expression) return true;

  reqRewriter.check(rewrittenClause->expression);
  auto interp = ASTInterpreter{unit};
  auto val = interp.evaluate(rewrittenClause->expression);
  if (!val.has_value()) return true;

  auto boolVal = interp.toBool(*val);
  if (boolVal.has_value() && !*boolVal) return false;

  return true;
}

void ASTRewriter::check(ExpressionAST* ast) {
  if (!ast) return;
  if (isDependent(unit_, ast)) return;

  auto typeChecker = TypeChecker{unit_};
  typeChecker.setScope(binder_.scope());
  typeChecker.setReportErrors(shouldCaptureBodyErrors());
  typeCheckAndCapture([&] { typeChecker.check(ast); });
}

}  // namespace cxx
