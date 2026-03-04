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
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/views/symbol_chain.h>

namespace cxx {

ASTRewriter::ASTRewriter(TranslationUnit* unit, ScopeSymbol* scope,
                         std::vector<TemplateArgument> templateArguments)
    : unit_(unit),
      templateArguments_(std::move(templateArguments)),
      binder_(unit) {
  binder_.setScope(scope);
}

ASTRewriter::~ASTRewriter() {}

void ASTRewriter::addSymbolRemap(Symbol* oldSym, Symbol* newSym) {
  if (oldSym && newSym && oldSym != newSym) {
    symbolRemap_[oldSym] = newSym;
  }
}

auto ASTRewriter::remapSymbol(Symbol* sym) const -> Symbol* {
  if (!sym) return nullptr;
  auto it = symbolRemap_.find(sym);
  if (it != symbolRemap_.end()) return it->second;

  if (auto ovl = symbol_cast<OverloadSetSymbol>(sym)) {
    for (auto func : ovl->functions()) {
      auto remappedIt = symbolRemap_.find(func);
      if (remappedIt == symbolRemap_.end()) continue;
      auto remapped = remappedIt->second;
      for (auto scope = remapped->parent(); scope; scope = scope->parent()) {
        if (scope->isTemplateParameters()) continue;
        for (auto candidate : scope->find(ovl->name())) {
          if (auto newOvl = symbol_cast<OverloadSetSymbol>(candidate)) {
            return newOvl;
          }
        }
        break;
      }
      return remapped;
    }
  }

  return sym;
}

auto ASTRewriter::control() const -> Control* { return unit_->control(); }

auto ASTRewriter::arena() const -> Arena* { return unit_->arena(); }

auto ASTRewriter::restrictedToDeclarations() const -> bool {
  return restrictedToDeclarations_;
}

void ASTRewriter::setRestrictedToDeclarations(bool restrictedToDeclarations) {
  restrictedToDeclarations_ = restrictedToDeclarations;
}

void ASTRewriter::warning(SourceLocation loc, std::string message) {
  binder_.warning(loc, std::move(message));
}

void ASTRewriter::note(SourceLocation loc, std::string message) {
  binder_.note(loc, std::move(message));
}

void ASTRewriter::error(SourceLocation loc, std::string message) {
  binder_.error(loc, std::move(message));
}

}  // namespace cxx
