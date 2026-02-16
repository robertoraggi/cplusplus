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

#include <cxx/binder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/decl_specs.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

// std
#include <format>

namespace cxx {

struct [[nodiscard]] Binder::BindClass {
  Binder& binder;
  ClassSpecifierAST* ast;
  DeclSpecs& declSpecs;

  auto control() const -> Control* { return binder.control(); }

  void bind();
  void check_optional_nested_name_specifier();
  auto check_template_specialization() -> bool;
};

void Binder::bind(ClassSpecifierAST* ast, DeclSpecs& declSpecs) {
  BindClass{*this, ast, declSpecs}.bind();
}

void Binder::BindClass::bind() {
  check_optional_nested_name_specifier();

  if (check_template_specialization()) return;

  // get the component anme
  const Identifier* className = nullptr;

  if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId)) {
    className = nameId->identifier;
  }

  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->classLoc;

  ClassSymbol* classSymbol = nullptr;

  if (className) {
    for (auto candidate :
         binder.declaringScope()->find(className) | views::classes) {
      classSymbol = candidate;
      break;
    }
  }

  if (classSymbol && classSymbol->isComplete()) {
    // not a template-id, but a class with the same name already exists
    binder.error(location, std::format("redefinition of class '{}'",
                                       to_string(className)));
    classSymbol = nullptr;
  }

  if (classSymbol && classSymbol->isHidden()) {
    classSymbol->setHidden(false);
  }

  if (!classSymbol) {
    const auto isUnion = ast->classKey == TokenKind::T_UNION;
    classSymbol = control()->newClassSymbol(binder.scope(), location);
    classSymbol->setIsUnion(isUnion);
    classSymbol->setName(className);

    binder.declaringScope()->addSymbol(classSymbol);
  }

  ast->symbol = classSymbol;

  ast->symbol->setDeclaration(ast);

  auto classCanon = ast->symbol->canonical();
  classCanon->setDefinition(ast->symbol);

  if (declSpecs.templateHead) {
    ast->symbol->setTemplateDeclaration(declSpecs.templateHead);
  }

  ast->symbol->setFinal(ast->isFinal);

  declSpecs.setTypeSpecifier(ast);
  declSpecs.setType(ast->symbol->type());
}

void Binder::BindClass::check_optional_nested_name_specifier() {
  if (!ast->nestedNameSpecifier) return;

  auto parent = ast->nestedNameSpecifier->symbol;

  if (!parent || !parent->isClassOrNamespace()) {
    binder.error(ast->nestedNameSpecifier->firstSourceLocation(),
                 "nested name specifier must be a class or namespace");
    return;
  }

  binder.setScope(parent);
}

auto Binder::BindClass::check_template_specialization() -> bool {
  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
  if (!templateId) return false;

  const auto location = templateId->identifierLoc;

  ClassSymbol* primaryTemplateSymbol = nullptr;

  for (auto candidate :
       binder.declaringScope()->find(templateId->identifier) | views::classes) {
    primaryTemplateSymbol = candidate;
    break;
  }

  if (!primaryTemplateSymbol || !primaryTemplateSymbol->templateParameters()) {
    binder.error(location,
                 std::format("specialization of undeclared template '{}'",
                             templateId->identifier->name()));
    // return true;
  }

  std::vector<TemplateArgument> templateArguments;
  ClassSymbol* specialization = nullptr;

  if (primaryTemplateSymbol) {
    templateArguments = ASTRewriter::make_substitution(
        binder.unit_, primaryTemplateSymbol->templateDeclaration(),
        templateId->templateArgumentList);

    specialization =
        primaryTemplateSymbol
            ? symbol_cast<ClassSymbol>(
                  primaryTemplateSymbol->findSpecialization(templateArguments))
            : nullptr;

    if (specialization) {
      bool isTrueRedefinition = true;
      if (declSpecs.templateHead) {
        auto existingSpec = symbol_cast<ClassSymbol>(specialization);

        auto existingTemplDecl =
            existingSpec ? existingSpec->templateDeclaration() : nullptr;

        if (existingTemplDecl) {
          int existingCount = 0, newCount = 0;
          bool existingHasPack = false, newHasPack = false;

          for (auto p : ListView{existingTemplDecl->templateParameterList}) {
            ++existingCount;

            if (auto tp = ast_cast<TypenameTypeParameterAST>(p))
              if (tp->isPack) existingHasPack = true;
          }

          for (auto p :
               ListView{declSpecs.templateHead->templateParameterList}) {
            ++newCount;

            if (auto tp = ast_cast<TypenameTypeParameterAST>(p))
              if (tp->isPack) newHasPack = true;
          }

          if (existingCount != newCount || existingHasPack != newHasPack) {
            isTrueRedefinition = false;
          }

          if (existingTemplDecl->requiresClause ||
              declSpecs.templateHead->requiresClause) {
            isTrueRedefinition = false;
          }
        }
      }

      if (isTrueRedefinition) {
        binder.error(location,
                     std::format("redefinition of specialization '{}'",
                                 templateId->identifier->name()));
      }

      // return true;
    }
  }

  const auto isUnion = ast->classKey == TokenKind::T_UNION;

  auto classSymbol = control()->newClassSymbol(binder.scope(), location);
  ast->symbol = classSymbol;

  classSymbol->setIsUnion(isUnion);
  classSymbol->setName(templateId->identifier);
  ast->symbol->setDeclaration(ast);
  ast->symbol->setFinal(ast->isFinal);

  declSpecs.setTypeSpecifier(ast);
  declSpecs.setType(ast->symbol->type());

  if (declSpecs.templateHead) {
    ast->symbol->setTemplateDeclaration(declSpecs.templateHead);
  }

  if (primaryTemplateSymbol) {
    primaryTemplateSymbol->addSpecialization(std::move(templateArguments),
                                             classSymbol);
  }

  return true;
}

}  // namespace cxx