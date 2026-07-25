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
#include <cxx/decl.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <iterator>
#include <optional>

namespace cxx {
namespace {
auto memberFunctionKey(FunctionSymbol* fn) -> std::pair<bool, std::size_t> {
  bool isConst = false;
  std::size_t arity = 0;
  if (auto ft = type_cast<FunctionType>(fn->type())) {
    isConst = ft->cvQualifiers() == CvQualifiers::kConst ||
              ft->cvQualifiers() == CvQualifiers::kConstVolatile;
    arity = ft->parameterTypes().size();
  }
  return {isConst, arity};
}

void collectFunctions(Symbol* member, std::vector<FunctionSymbol*>& out) {
  std::ranges::copy(views::each_function(member), std::back_inserter(out));
}
}  // namespace

void ASTRewriter::remapScopeMembers(ScopeSymbol* oldScope,
                                    ScopeSymbol* newScope) {
  if (!oldScope || !newScope || oldScope == newScope) return;

  if (auto oldClass = symbol_cast<ClassSymbol>(oldScope)) {
    oldScope = oldClass->resolvedDefinition();
  }
  if (auto newClass = symbol_cast<ClassSymbol>(newScope)) {
    newScope = newClass->resolvedDefinition();
  }
  if (oldScope == newScope) return;

  addSymbolRemap(oldScope, newScope);
  auto& oldMembers = oldScope->members();
  auto& newMembers = newScope->members();

  std::unordered_map<const Name*, std::vector<Symbol*>> newByName;
  for (auto newMember : newMembers) {
    newByName[newMember->name()].push_back(newMember);
  }

  std::unordered_map<const Name*, std::size_t> nextIndex;
  for (auto oldMember : oldMembers) {
    auto it = newByName.find(oldMember->name());
    if (it == newByName.end()) continue;
    auto& candidates = it->second;
    auto& index = nextIndex[oldMember->name()];
    if (index >= candidates.size()) continue;
    auto newMember = candidates[index++];
    addSymbolRemap(oldMember, newMember);
    if (symbol_cast<OverloadSetSymbol>(oldMember) ||
        symbol_cast<OverloadSetSymbol>(newMember)) {
      std::vector<FunctionSymbol*> oldFns, newFns;
      collectFunctions(oldMember, oldFns);
      collectFunctions(newMember, newFns);
      std::vector<bool> used(newFns.size(), false);
      for (auto oldFn : oldFns) {
        auto key = memberFunctionKey(oldFn);
        FunctionSymbol* fallback = nullptr;
        FunctionSymbol* chosen = nullptr;
        for (std::size_t i = 0; i < newFns.size(); ++i) {
          if (memberFunctionKey(newFns[i]) != key) continue;
          if (!fallback) fallback = newFns[i];
          if (!used[i]) {
            used[i] = true;
            chosen = newFns[i];
            break;
          }
        }
        if (!chosen) chosen = fallback;
        if (chosen) addSymbolRemap(oldFn, chosen);
      }
    }
    if (auto oldNested = symbol_cast<ClassSymbol>(oldMember)) {
      if (auto newNested = symbol_cast<ClassSymbol>(newMember)) {
        remapScopeMembers(oldNested, newNested);
      }
    } else if (auto oldEnum = symbol_cast<EnumSymbol>(oldMember)) {
      if (auto newEnum = symbol_cast<EnumSymbol>(newMember)) {
        remapScopeMembers(oldEnum, newEnum);
      }
    } else if (auto oldScopedEnum = symbol_cast<ScopedEnumSymbol>(oldMember)) {
      if (auto newScopedEnum = symbol_cast<ScopedEnumSymbol>(newMember)) {
        remapScopeMembers(oldScopedEnum, newScopedEnum);
      }
    }
  }
}

auto ASTRewriter::completePendingBodyFor(TranslationUnit* unit,
                                         FunctionSymbol* function,
                                         bool captureBodyErrors)
    -> std::vector<Diagnostic> {
  if (!unit || !function || !function->hasPendingBody()) return {};
  auto rewriter = ASTRewriter{unit, unit->globalScope(), {}};
  return rewriter.completePendingBody(function, captureBodyErrors);
}

auto ASTRewriter::completePendingBody(FunctionSymbol* func,
                                      bool captureBodyErrors)
    -> std::vector<Diagnostic> {
  if (!func || !func->hasPendingBody()) return {};

  auto pending = func->pendingBody();

  std::optional<CapturingDiagnosticsClient> capture;
  DiagnosticsClient* savedClient = nullptr;
  if (captureBodyErrors) {
    capture.emplace();
    savedClient = unit_->changeDiagnosticsClient(&*capture);
  }
  auto finish = [&](std::vector<Diagnostic> extra = {}) {
    if (captureBodyErrors) {
      (void)unit_->changeDiagnosticsClient(savedClient);
      extra.insert(extra.end(),
                   std::make_move_iterator(capture->diagnostics.begin()),
                   std::make_move_iterator(capture->diagnostics.end()));
    }
    return extra;
  };

  auto newAst = func->declaration();
  if (!newAst) {
    auto originalDef = pending->originalDefinition;
    auto classArguments = std::move(pending->templateArguments);
    auto parentScope = pending->parentScope;
    auto depth = pending->depth;
    func->clearPendingBody();

    if (!originalDef || !originalDef->symbol) return finish();

    auto rewriter = ASTRewriter{unit_, parentScope, std::move(classArguments)};
    rewriter.depth_ = depth;
    rewriter.binder_.setInstantiatingSymbol(originalDef->symbol);

    auto patternClass = symbol_cast<ClassSymbol>(
        originalDef->symbol->enclosingNonTemplateParametersScope());
    auto instanceClass =
        symbol_cast<ClassSymbol>(func->enclosingNonTemplateParametersScope());
    if (patternClass && instanceClass) {
      rewriter.remapScopeMembers(patternClass, instanceClass);
    }

    auto patternTemplateDecl = originalDef->symbol->templateDeclaration();
    auto rewrittenDecl = patternTemplateDecl
                             ? rewriter.declaration(patternTemplateDecl)
                             : rewriter.declaration(originalDef);

    auto copy = ast_cast<FunctionDefinitionAST>(rewrittenDecl);
    if (!copy) {
      if (auto rewrittenTemplateDecl =
              ast_cast<TemplateDeclarationAST>(rewrittenDecl)) {
        copy =
            ast_cast<FunctionDefinitionAST>(rewrittenTemplateDecl->declaration);
      }
    }

    if (!func->declaration() && copy) func->setDeclaration(copy);
    return finish();
  }

  auto templateArguments = std::move(pending->templateArguments);
  auto parentScope = pending->parentScope;
  auto depth = pending->depth;
  auto originalDef = pending->originalDefinition;
  func->clearPendingBody();

  auto rewriter = ASTRewriter{unit_, parentScope, templateArguments};
  rewriter.depth_ = depth;
  rewriter.binder_.setInstantiatingSymbol(func);

  if (auto oldFunc = symbol_cast<FunctionSymbol>(originalDef->symbol)) {
    auto oldClass = symbol_cast<ClassSymbol>(
        oldFunc->enclosingNonTemplateParametersScope());
    auto newClass =
        symbol_cast<ClassSymbol>(func->enclosingNonTemplateParametersScope());

    while (oldClass && newClass) {
      auto oldUp = symbol_cast<ClassSymbol>(
          oldClass->enclosingNonTemplateParametersScope());
      auto newUp = symbol_cast<ClassSymbol>(
          newClass->enclosingNonTemplateParametersScope());
      if (!oldUp || !newUp) break;
      oldClass = oldUp;
      newClass = newUp;
    }

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

  auto bodyErrors = rewriter.takeBodyErrors();

  rewriter.binder_.synthesizeCompleteObjectCtor(func);

  if (ast_cast<DefaultFunctionBodyAST>(newAst->functionBody)) {
    rewriter.binder_.synthesizeDefaultedMemberBody(func);
    return finish(std::move(bodyErrors));
  }

  auto compoundBody =
      ast_cast<CompoundStatementFunctionBodyAST>(newAst->functionBody);
  if (!compoundBody) {
    return finish(std::move(bodyErrors));
  }

  TypeChecker check{unit_};
  check.setScope(func);
  check.check_mem_initializers(compoundBody);

  return finish(std::move(bodyErrors));
}
}  // namespace cxx
