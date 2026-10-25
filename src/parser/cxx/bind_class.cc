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
  auto enclosingClassOfDeclaration() const -> ClassSymbol*;
  void initializeClassSymbol(ClassSymbol* classSymbol);

  auto findPrimaryTemplateSymbol(SimpleTemplateIdAST* templateId) const
      -> ClassSymbol*;
  [[nodiscard]] static auto isImplicitInstantiation(ClassSymbol* specialization)
      -> bool {
    return specialization->isComplete() &&
           specialization->instantiationPattern() != nullptr;
  }

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

auto Binder::resolveClassSpecialization(ClassSymbol* primary,
                                        SimpleTemplateIdAST* templateId,
                                        TemplateDeclarationAST* templateHead)
    -> std::optional<ClassSpecialization> {
  const auto location = templateId->identifierLoc;

  if (!primary || !primary->templateParameters()) {
    error(location, std::format("specialization of undeclared template '{}'",
                                templateId->identifier->name()));
    return std::nullopt;
  }

  auto arguments = Substitution(unit_, primary->templateDeclaration(),
                                templateId->templateArgumentList)
                       .templateArguments();

  if (auto parameter = ASTRewriter::findUndeducedPartialSpecializationParameter(
          unit_, templateHead, arguments)) {
    error(parameter->firstSourceLocation(),
          "class template partial specialization contains a template "
          "parameter that cannot be deduced");
  }

  auto specialization =
      symbol_cast<ClassSymbol>(primary->findSpecialization(unit_, arguments));
  specialization =
      specializeMemberClassTemplate(specialization, templateHead, location);

  return ClassSpecialization{specialization, std::move(arguments)};
}

void Binder::bindClassSpecializationDeclaration(ElaboratedTypeSpecifierAST* ast,
                                                DeclSpecs& declSpecs,
                                                ClassSymbol* primary) {
  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
  auto resolved =
      resolveClassSpecialization(primary, templateId, declSpecs.templateHead);
  if (!resolved) return;

  const auto location = templateId->identifierLoc;
  auto specialization = resolved->symbol;

  if (!specialization) {
    specialization = control()->newClassSymbol(declaringScope(), location);
    specialization->setName(templateId->identifier);
    specialization->setIsUnion(ast->classKey == TokenKind::T_UNION);
    applyAccessSpecifier(specialization);

    primary->addSpecialization(unit_, std::move(resolved->arguments),
                               specialization);
  } else if (specialization->isComplete() &&
             specialization->instantiationPattern() &&
             !specialization->templateDeclaration()) {
    error(location,
          std::format("explicit specialization of '{}' after instantiation",
                      to_diag_string(specialization->type())));
  }

  if (!specialization->isComplete()) {
    specialization->setDeclaration(ast);
    specialization->setTemplateDeclaration(declSpecs.templateHead);
    specialization->setTemplateParameters(declSpecs.templateHead->symbol);
    primary->clearPendingInstantiation(specialization);
  }

  ast->symbol = specialization;
  ast->declaredSymbol = specialization;
  declSpecs.setTypeSpecifier(ast);
  declSpecs.setType(specialization->type());
}

auto Binder::specializeMemberClassTemplate(ClassSymbol* declared,
                                           TemplateDeclarationAST* templateHead,
                                           SourceLocation location)
    -> ClassSymbol* {
  if (!declared || !declared->instantiationPattern() || !templateHead ||
      !templateHead->templateParameterList)
    return declared;

  auto explicitHead =
      symbol_cast<TemplateParametersSymbol>(templateHead->symbol->parent());
  if (!explicitHead || !explicitHead->isExplicitTemplateSpecialization())
    return declared;

  auto specializesEnclosingClass = false;
  for (auto enclosing = declared->enclosingClass(); enclosing;
       enclosing = enclosing->enclosingClass()) {
    if (enclosing->templateDeclaration() || !enclosing->isSpecialization())
      continue;

    auto primary = enclosing->primaryTemplateSymbol();
    auto head = primary ? primary->templateDeclaration() : nullptr;
    if (head && head->depth + 1 == templateHead->depth) {
      specializesEnclosingClass = true;
      break;
    }
  }

  if (!specializesEnclosingClass) return declared;

  auto primary = declared->primaryTemplateSymbol();
  auto specializations =
      primary ? primary->specializations() : declared->specializations();

  for (const auto& entry : specializations) {
    auto instance = symbol_cast<ClassSymbol>(entry.symbol);
    if (!instance || instance->templateDeclaration() || !instance->isComplete())
      continue;
    if (primary && instance->instantiationPattern() != declared) continue;

    error(location, std::format("explicit specialization of member class "
                                "template '{}' after instantiation",
                                to_diag_string(declared->name())));
    break;
  }

  auto parent = declared->parent();
  auto specialized = control()->newClassSymbol(parent, location);
  specialized->setName(declared->name());
  specialized->setIsUnion(declared->isUnion());
  specialized->setAccessSpecifier(declared->accessSpecifier());
  inheritDeclarationAttributes(specialized, declared);

  auto canonical = declared->canonical();
  addRedeclaration(canonical, specialized);
  setDefinition(canonical, static_cast<ClassSymbol*>(nullptr));

  if (primary) {
    recordSpeculativeMutation([primary, declared, specialized] {
      primary->replaceSpecialization(specialized, declared);
    });
    primary->replaceSpecialization(declared, specialized);
  } else {
    recordSpeculativeMutation([parent, declared, specialized] {
      parent->replaceSymbol(specialized, declared);
    });
    parent->replaceSymbol(declared, specialized);
  }

  return specialized;
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

auto Binder::BindClass::enclosingClassOfDeclaration() const -> ClassSymbol* {
  if (!ast->nestedNameSpecifier) return nullptr;
  return symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol);
}

void Binder::BindClass::initializeClassSymbol(ClassSymbol* classSymbol) {
  ast->symbol = classSymbol;
  binder.applyAccessSpecifier(classSymbol);
  ast->symbol->setDeclaration(ast);
  ast->symbol->setFinal(ast->isFinal);

  binder.applyDeclarationAttributes(ast->symbol, ast->attributeList);

  classSymbol->setPackAlignment(
      binder.translationUnit()->packAlignmentAt(classLocation()));

  const auto declaredAlignment = classSymbol->explicitAlignment();
  if (auto alignment =
          binder.explicitAlignment(ast->attributeList, classLocation())) {
    binder.checkRedeclaredAlignment(classSymbol, *alignment, classLocation());
  } else if (declaredAlignment) {
    binder.error(classLocation(),
                 "'alignas' must be specified on the definition if it is "
                 "specified on any declaration");
  }

  if (declSpecs.templateHead) {
    if (is_templated_class(enclosingClassOfDeclaration())) {
      binder.rejectDefaultTemplateArguments(
          declSpecs.templateHead,
          "a default template argument cannot be specified on the "
          "out-of-class definition of a member of a class template");
    } else if (declSpecs.isFriend) {
      binder.rejectDefaultTemplateArguments(
          declSpecs.templateHead,
          "a default template argument cannot be specified on a friend "
          "template declaration");
    }

    binder.setTemplateHead(classSymbol, declSpecs.templateHead);

    binder.checkTemplateParameterDefaultOrder(
        ast->symbol->templateParameters());
  }

  binder.setDefinition(ast->symbol->canonical(), ast->symbol);

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
  if (!declSpecs.templateHead) return true;

  auto existingTemplateDecl = specialization->templateDeclaration();

  if (!existingTemplateDecl) return false;

  if (!TemplateEquivalence{binder.unit_}.same(existingTemplateDecl,
                                              declSpecs.templateHead)) {
    return false;
  }

  if (newTemplateId) {
    auto existingClassSpec =
        ast_cast<ClassSpecifierAST>(specialization->declaration());
    if (existingClassSpec) {
      auto existingTemplateId =
          ast_cast<SimpleTemplateIdAST>(existingClassSpec->unqualifiedId);
      if (existingTemplateId) {
        if (!TemplateEquivalence{binder.unit_}.sameWritten(
                existingTemplateId->templateArgumentList,
                newTemplateId->templateArgumentList)) {
          return false;
        }
      }
    }
  }

  return true;
}

void Binder::BindClass::bind() {
  check_optional_nested_name_specifier();

  if (check_template_specialization()) return;

  const auto name = className();
  const auto location = classLocation();
  auto classSymbol = findExistingClass(name);
  classSymbol = binder.specializeMemberClassTemplate(
      classSymbol, declSpecs.templateHead, location);

  if (classSymbol && classSymbol->isComplete()) {
    binder.error(location, std::format("redefinition of class '{}'",
                                       to_diag_string(name)));
    classSymbol = nullptr;
  }

  if (classSymbol && classSymbol->isHidden()) classSymbol->setHidden(false);

  if (classSymbol && bindOutOfClassNestedDefinition(classSymbol)) return;

  if (!classSymbol) {
    classSymbol = createClassSymbol(name, location);
    binder.declaringScope()->addSymbol(classSymbol);
    binder.recordStandardLibraryType(classSymbol);
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
  binder.addRedeclaration(declared->canonical(), defSymbol);

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

  auto resolved = binder.resolveClassSpecialization(
      primaryTemplateSymbol, templateId, declSpecs.templateHead);
  auto specialization = resolved ? resolved->symbol : nullptr;
  bool specializesInstantiation = false;

  if (specialization) {
    if (isTrueRedefinition(specialization, templateId)) {
      binder.error(location, std::format("redefinition of specialization '{}'",
                                         templateId->identifier->name()));
    } else if (isImplicitInstantiation(specialization)) {
      specializesInstantiation = true;
      binder.error(location,
                   std::format("explicit specialization of '{}' after "
                               "instantiation",
                               to_diag_string(specialization->type())));
    }
  }

  ClassSymbol* classSymbol = nullptr;

  if (specialization && !specialization->isComplete()) {
    classSymbol = specialization;
    classSymbol->setIsUnion(ast->classKey == TokenKind::T_UNION);
    primaryTemplateSymbol->clearPendingInstantiation(specialization);
  } else {
    classSymbol = createClassSymbol(templateId->identifier, location);
    if (resolved && !specializesInstantiation) {
      primaryTemplateSymbol->addSpecialization(
          binder.unit_, std::move(resolved->arguments), classSymbol);
    }
  }

  initializeClassSymbol(classSymbol);

  return true;
}
}  // namespace cxx
