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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl_specs.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

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
  [[nodiscard]] auto isTrueRedefinition(
      ClassSymbol* specialization, SimpleTemplateIdAST* newTemplateId) const
      -> bool;

  void bind();
  void check_optional_nested_name_specifier();
  auto check_template_specialization() -> bool;
  auto bindOutOfClassNestedDefinition(ClassSymbol* declared) -> bool;
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
  auto classSymbol =
      control()->newClassSymbol(binder.declaringScope(), location);
  classSymbol->setIsUnion(isUnion);
  classSymbol->setName(className);
  return classSymbol;
}

void Binder::BindClass::initializeClassSymbol(ClassSymbol* classSymbol) {
  ast->symbol = classSymbol;
  ast->symbol->setDeclaration(ast);
  ast->symbol->setFinal(ast->isFinal);

  binder.applyAbiTags(ast->symbol, ast->attributeList);

  if (declSpecs.templateHead) {
    if (auto oldDecl = ast->symbol->templateDeclaration()) {
      auto mergeDefault = [](TemplateParameterAST* src,
                             TemplateParameterAST* dst) {
        if (auto s = ast_cast<TypenameTypeParameterAST>(src)) {
          auto d = ast_cast<TypenameTypeParameterAST>(dst);
          if (d && s->typeId && !d->typeId) {
            d->equalLoc = s->equalLoc;
            d->typeId = s->typeId;
          }
        } else if (auto s = ast_cast<NonTypeTemplateParameterAST>(src)) {
          auto d = ast_cast<NonTypeTemplateParameterAST>(dst);
          if (d && s->declaration && d->declaration &&
              s->declaration->expression && !d->declaration->expression) {
            d->declaration->equalLoc = s->declaration->equalLoc;
            d->declaration->expression = s->declaration->expression;
          }
        } else if (auto s = ast_cast<TemplateTypeParameterAST>(src)) {
          auto d = ast_cast<TemplateTypeParameterAST>(dst);
          if (d && s->idExpression && !d->idExpression) {
            d->equalLoc = s->equalLoc;
            d->idExpression = s->idExpression;
          }
        } else if (auto s = ast_cast<ConstraintTypeParameterAST>(src)) {
          auto d = ast_cast<ConstraintTypeParameterAST>(dst);
          if (d && s->typeId && !d->typeId) {
            d->equalLoc = s->equalLoc;
            d->typeId = s->typeId;
          }
        }
      };

      auto oldParams = ListView{oldDecl->templateParameterList};
      auto newParams = ListView{declSpecs.templateHead->templateParameterList};
      auto newIt = newParams.begin();
      for (auto oldIt = oldParams.begin();
           oldIt != oldParams.end() && newIt != newParams.end();
           ++oldIt, ++newIt) {
        mergeDefault(*oldIt, *newIt);
      }
    }

    ast->symbol->setTemplateDeclaration(declSpecs.templateHead);
    ast->symbol->setTemplateParameters(declSpecs.templateHead->symbol);
  }

  auto classCanon = ast->symbol->canonical();
  classCanon->setDefinition(ast->symbol);

  declSpecs.setTypeSpecifier(ast);
  declSpecs.setType(ast->symbol->type());

  if (classSymbol->name() && binder.isCxx()) {
    auto injected = control()->newInjectedClassNameSymbol(
        classSymbol, classSymbol->location());
    injected->setName(classSymbol->name());
    injected->setType(classSymbol->type());
    injected->setClassSymbol(classSymbol);
    classSymbol->addSymbol(injected);
  }
}

auto Binder::BindClass::findPrimaryTemplateSymbol(
    SimpleTemplateIdAST* templateId) const -> ClassSymbol* {
  auto isClassTemplate = [](Symbol* symbol) {
    auto classSymbol = symbol_cast<ClassSymbol>(symbol);
    return classSymbol && classSymbol->templateParameters();
  };

  return symbol_cast<ClassSymbol>(qualifiedLookup(
      binder.declaringScope(), templateId->identifier, isClassTemplate));
}

auto Binder::BindClass::isTrueRedefinition(
    ClassSymbol* specialization, SimpleTemplateIdAST* newTemplateId) const
    -> bool {
  if (!specialization) return false;

  if (!specialization->isComplete()) return false;

  bool isRedefinition = true;
  if (!declSpecs.templateHead) return isRedefinition;

  auto existingTemplateDecl = specialization->templateDeclaration();

  if (!existingTemplateDecl) return false;

  if (!areTemplateHeadsEquivalentForRedeclaration(
          binder.unit_, existingTemplateDecl, declSpecs.templateHead)) {
    return false;
  }

  if (newTemplateId) {
    auto existingClassSpec =
        ast_cast<ClassSpecifierAST>(specialization->declaration());
    if (existingClassSpec) {
      auto existingTemplateId =
          ast_cast<SimpleTemplateIdAST>(existingClassSpec->unqualifiedId);
      if (existingTemplateId) {
        if (!areTemplateArgumentListsSyntacticallyEquivalent(
                binder.unit_, existingTemplateId->templateArgumentList,
                newTemplateId->templateArgumentList)) {
          return false;
        }
      }
    }
  }

  return isRedefinition;
}

void Binder::BindClass::bind() {
  check_optional_nested_name_specifier();

  if (check_template_specialization()) return;

  const auto name = className();
  const auto location = classLocation();
  auto classSymbol = findExistingClass(name);

  if (classSymbol && classSymbol->isComplete()) {
    binder.error(location,
                 std::format("redefinition of class '{}'", to_string(name)));
    classSymbol = nullptr;
  }

  if (classSymbol && classSymbol->isHidden()) classSymbol->setHidden(false);

  if (classSymbol && bindOutOfClassNestedDefinition(classSymbol)) return;

  if (!classSymbol) {
    classSymbol = createClassSymbol(name, location);
    binder.declaringScope()->addSymbol(classSymbol);
  } else {
    classSymbol->setParent(binder.declaringScope());
  }

  initializeClassSymbol(classSymbol);
}

auto Binder::BindClass::bindOutOfClassNestedDefinition(ClassSymbol* declared)
    -> bool {
  if (!ast->nestedNameSpecifier) return false;
  if (!declSpecs.templateHead) return false;

  auto enclosing = symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol);
  if (!enclosing) return false;

  auto enclosingHead = enclosing->templateDeclaration();
  if (!enclosingHead) return false;

  if (declSpecs.templateHead->depth != enclosingHead->depth) return false;

  auto defSymbol = createClassSymbol(className(), classLocation());
  defSymbol->setIsUnion(ast->classKey == TokenKind::T_UNION);
  declared->canonical()->addRedeclaration(defSymbol);

  initializeClassSymbol(defSymbol);
  return true;
}

void Binder::BindClass::check_optional_nested_name_specifier() {
  if (!ast->nestedNameSpecifier) return;

  auto parent = ast->nestedNameSpecifier->symbol;

  if (!parent || !parent->isClassOrNamespace()) {
    if (!binder.inTemplate()) {
      binder.error(ast->nestedNameSpecifier->firstSourceLocation(),
                   "nested name specifier must be a class or namespace");
    }
    return;
  }

  binder.setScope(parent->asScopeSymbol());
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
  }

  std::vector<TemplateArgument> templateArguments;
  ClassSymbol* specialization = nullptr;
  if (primaryTemplateSymbol) {
    templateArguments =
        Substitution(binder.unit_, primaryTemplateSymbol->templateDeclaration(),
                     templateId->templateArgumentList)
            .templateArguments();

    specialization =
        symbol_cast<ClassSymbol>(primaryTemplateSymbol->findSpecialization(
            binder.unit_, templateArguments));
    if (specialization && isTrueRedefinition(specialization, templateId)) {
      binder.error(location, std::format("redefinition of specialization '{}'",
                                         templateId->identifier->name()));
    }
  }

  ClassSymbol* classSymbol = nullptr;

  if (specialization && !specialization->isComplete()) {
    classSymbol = specialization;
    classSymbol->setIsUnion(ast->classKey == TokenKind::T_UNION);
    primaryTemplateSymbol->clearPendingInstantiation(specialization);
  } else {
    classSymbol = createClassSymbol(templateId->identifier, location);
    if (primaryTemplateSymbol) {
      primaryTemplateSymbol->addSpecialization(
          binder.unit_, std::move(templateArguments), classSymbol);
    }
  }

  initializeClassSymbol(classSymbol);

  return true;
}
}  // namespace cxx
