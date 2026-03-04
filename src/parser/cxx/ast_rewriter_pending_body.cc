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
#include <cxx/decl.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>

namespace cxx {

void ASTRewriter::remapScopeMembers(ScopeSymbol* oldScope,
                                    ScopeSymbol* newScope) {
  if (!oldScope || !newScope || oldScope == newScope) return;
  auto& oldMembers = oldScope->members();
  auto& newMembers = newScope->members();
  auto n = std::min(oldMembers.size(), newMembers.size());
  for (std::size_t i = 0; i < n; ++i) {
    addSymbolRemap(oldMembers[i], newMembers[i]);
    if (auto oldNested = symbol_cast<ClassSymbol>(oldMembers[i])) {
      if (auto newNested = symbol_cast<ClassSymbol>(newMembers[i])) {
        remapScopeMembers(oldNested, newNested);
      }
    }
  }
}

void ASTRewriter::completePendingBody(FunctionSymbol* func) {
  if (!func || !func->hasPendingBody()) return;

  auto pending = func->pendingBody();

  auto newAst = func->declaration();
  if (!newAst) {
    func->clearPendingBody();
    return;
  }

  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  auto originalDef = pending->originalDefinition;
  func->clearPendingBody();

  auto rewriter = ASTRewriter{unit_, parentScope, templateArguments};
  rewriter.depth_ = depth;

  if (auto oldFunc = symbol_cast<FunctionSymbol>(originalDef->symbol)) {
    auto oldClass = symbol_cast<ClassSymbol>(oldFunc->parent());
    auto newClass = symbol_cast<ClassSymbol>(func->parent());
    if (oldClass && newClass && oldClass != newClass) {
      rewriter.remapScopeMembers(oldClass, newClass);
    }

    if (auto oldParams = oldFunc->functionParameters()) {
      if (auto newParams = func->functionParameters()) {
        auto& oldPMembers = oldParams->members();
        auto& newPMembers = newParams->members();
        auto n = std::min(oldPMembers.size(), newPMembers.size());
        for (std::size_t i = 0; i < n; ++i) {
          rewriter.addSymbolRemap(oldPMembers[i], newPMembers[i]);
        }
      }
    }
  }

  auto functionDeclarator = getFunctionPrototype(newAst->declarator);
  if (!functionDeclarator) {
    rewriter.binder_.setScope(func);
  } else if (auto params = functionDeclarator->parameterDeclarationClause) {
    rewriter.binder_.setScope(params->functionParametersSymbol);
  } else {
    rewriter.binder_.setScope(func);
  }

  newAst->functionBody = rewriter.functionBody(originalDef->functionBody);

  auto compoundBody =
      ast_cast<CompoundStatementFunctionBodyAST>(newAst->functionBody);
  if (!compoundBody || !compoundBody->memInitializerList) return;

  TypeChecker check{unit_};
  check.setScope(func);
  check.check_mem_initializers(compoundBody);
}

}  // namespace cxx
