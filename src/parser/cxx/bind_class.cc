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
  auto className() const -> const Identifier*;
  auto classLocation() const -> SourceLocation;

  auto findExistingClass(const Identifier* className) const -> ClassSymbol*;
  auto createClassSymbol(const Identifier* className, SourceLocation location)
      -> ClassSymbol*;
  void initializeClassSymbol(ClassSymbol* classSymbol);

  auto findPrimaryTemplateSymbol(SimpleTemplateIdAST* templateId) const
      -> ClassSymbol*;
  auto templateParameterCount(TemplateDeclarationAST* templateDecl) const
      -> int;
  auto hasPackTemplateParameter(TemplateDeclarationAST* templateDecl) const
      -> bool;
  auto isTrueRedefinition(ClassSymbol* specialization) const -> bool;

  void bind();
  void check_optional_nested_name_specifier();
  auto check_template_specialization() -> bool;
};

void Binder::bind(ClassSpecifierAST* ast, DeclSpecs& declSpecs) {
  BindClass{*this, ast, declSpecs}.bind();
}

auto Binder::BindClass::className() const -> const Identifier* {
  auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId);
  if (!nameId) return nullptr;
  return nameId->identifier;
}

auto Binder::BindClass::classLocation() const -> SourceLocation {
  if (ast->unqualifiedId) return ast->unqualifiedId->firstSourceLocation();
  return ast->classLoc;
}

auto Binder::BindClass::findExistingClass(const Identifier* className) const
    -> ClassSymbol* {
  if (!className) return nullptr;

  for (auto candidate :
       binder.declaringScope()->find(className) | views::classes) {
    return candidate;
  }

  return nullptr;
}

auto Binder::BindClass::createClassSymbol(const Identifier* className,
                                          SourceLocation location)
    -> ClassSymbol* {
  const auto isUnion = ast->classKey == TokenKind::T_UNION;
  auto classSymbol = control()->newClassSymbol(binder.scope(), location);
  classSymbol->setIsUnion(isUnion);
  classSymbol->setName(className);
  return classSymbol;
}

void Binder::BindClass::initializeClassSymbol(ClassSymbol* classSymbol) {
  ast->symbol = classSymbol;
  ast->symbol->setDeclaration(ast);
  ast->symbol->setFinal(ast->isFinal);

  if (declSpecs.templateHead) {
    ast->symbol->setTemplateDeclaration(declSpecs.templateHead);
  }

  auto classCanon = ast->symbol->canonical();
  classCanon->setDefinition(ast->symbol);

  declSpecs.setTypeSpecifier(ast);
  declSpecs.setType(ast->symbol->type());
}

auto Binder::BindClass::findPrimaryTemplateSymbol(
    SimpleTemplateIdAST* templateId) const -> ClassSymbol* {
  for (auto candidate :
       binder.declaringScope()->find(templateId->identifier) | views::classes) {
    return candidate;
  }

  return nullptr;
}

auto Binder::BindClass::templateParameterCount(
    TemplateDeclarationAST* templateDecl) const -> int {
  if (!templateDecl) return 0;

  int count = 0;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    (void)parameter;
    ++count;
  }
  return count;
}

auto Binder::BindClass::hasPackTemplateParameter(
    TemplateDeclarationAST* templateDecl) const -> bool {
  if (!templateDecl) return false;

  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter);
    if (typeParameter && typeParameter->isPack) return true;
  }

  return false;
}

auto Binder::BindClass::isTrueRedefinition(ClassSymbol* specialization) const
    -> bool {
  if (!specialization) return false;

  bool isRedefinition = true;
  if (!declSpecs.templateHead) return isRedefinition;

  auto existingTemplateDecl = specialization->templateDeclaration();
  if (!existingTemplateDecl) return isRedefinition;

  if (templateParameterCount(existingTemplateDecl) !=
      templateParameterCount(declSpecs.templateHead)) {
    return false;
  }

  if (hasPackTemplateParameter(existingTemplateDecl) !=
      hasPackTemplateParameter(declSpecs.templateHead)) {
    return false;
  }

  if (existingTemplateDecl->requiresClause ||
      declSpecs.templateHead->requiresClause) {
    return false;
  }

  return isRedefinition;
}

void Binder::BindClass::bind() {
  check_optional_nested_name_specifier();

  if (check_template_specialization()) return;

  const auto* name = className();
  const auto location = classLocation();
  auto classSymbol = findExistingClass(name);

  if (classSymbol && classSymbol->isComplete()) {
    // not a template-id, but a class with the same name already exists
    binder.error(location,
                 std::format("redefinition of class '{}'", to_string(name)));
    classSymbol = nullptr;
  }

  if (classSymbol && classSymbol->isHidden()) classSymbol->setHidden(false);

  if (!classSymbol) {
    classSymbol = createClassSymbol(name, location);
    binder.declaringScope()->addSymbol(classSymbol);
  }

  initializeClassSymbol(classSymbol);
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
  auto primaryTemplateSymbol = findPrimaryTemplateSymbol(templateId);

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

    specialization = symbol_cast<ClassSymbol>(
        primaryTemplateSymbol->findSpecialization(templateArguments));
    if (specialization && isTrueRedefinition(specialization)) {
      binder.error(location, std::format("redefinition of specialization '{}'",
                                         templateId->identifier->name()));
    }
  }

  auto classSymbol = createClassSymbol(templateId->identifier, location);
  initializeClassSymbol(classSymbol);

  if (primaryTemplateSymbol) {
    primaryTemplateSymbol->addSpecialization(std::move(templateArguments),
                                             classSymbol);
  }

  return true;
}

}  // namespace cxx