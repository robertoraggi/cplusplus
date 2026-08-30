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
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
struct [[nodiscard]] Binder::ResolveUnqualifiedId {
  Binder& binder;
  NestedNameSpecifierAST* nestedNameSpecifier;
  UnqualifiedIdAST* unqualifiedId;
  bool checkTemplates;
  Symbol* resolvedType = nullptr;

  auto control() const -> Control* { return binder.control(); }
  auto inTemplate() const -> bool { return binder.inTemplate(); }

  auto shouldKeepTemplateIdAsDependent(SimpleTemplateIdAST* templateId) const
      -> bool;
  auto resolveClassTemplateId(SimpleTemplateIdAST* templateId,
                              ClassSymbol* classSymbol) -> Symbol*;
  auto resolveTypeAliasTemplateId(SimpleTemplateIdAST* templateId,
                                  TypeAliasSymbol* typeAliasSymbol) -> Symbol*;

  auto operator()(SimpleTemplateIdAST* templateId) -> Symbol*;

  auto operator()(NameIdAST* nameId) -> Symbol*;

  auto operator()(UnqualifiedIdAST*) -> Symbol* {
    binder.error(
        unqualifiedId->firstSourceLocation(),
        "unable to resolve unqualified-id: not a NameId or SimpleTemplateId");
    return nullptr;
  }
};

auto Binder::resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                     UnqualifiedIdAST* unqualifiedId, bool checkTemplates,
                     Symbol* resolvedType) -> Symbol* {
  return visit(ResolveUnqualifiedId{*this, nestedNameSpecifier, unqualifiedId,
                                    checkTemplates, resolvedType},
               unqualifiedId);
}

auto Binder::ResolveUnqualifiedId::shouldKeepTemplateIdAsDependent(
    SimpleTemplateIdAST* templateId) const -> bool {
  if (!inTemplate()) return false;
  if (symbol_cast<TemplateTypeParameterSymbol>(templateId->symbol)) {
    return true;
  }
  if (!hasDependentTemplateArguments(binder.unit_, templateId)) return false;

  auto resolved = templateId->symbol;
  if (auto injected = symbol_cast<InjectedClassNameSymbol>(resolved))
    resolved = injected->classSymbol();

  auto classSymbol = symbol_cast<ClassSymbol>(resolved);
  if (!classSymbol) return true;

  return names_template_head_parameters(templateId, classSymbol);
}

auto Binder::ResolveUnqualifiedId::resolveClassTemplateId(
    SimpleTemplateIdAST* templateId, ClassSymbol* classSymbol) -> Symbol* {
  if (!isTemplateArityMatch(classSymbol->templateDeclaration(),
                            templateId->templateArgumentList)) {
    return nullptr;
  }

  if (!classSymbol->templateDeclaration()) return nullptr;

  auto subst =
      Substitution::make(binder.unit_, classSymbol->templateDeclaration(),
                         templateId->templateArgumentList);

  if (!subst) return nullptr;

  auto templateArgs = std::move(*subst).templateArguments();

  if (auto cached =
          classSymbol->findSpecialization(binder.unit_, templateArgs)) {
    return cached;
  }

  auto parentScope = classSymbol->parent();
  auto spec = control()->newClassSymbol(parentScope, classSymbol->location());
  spec->setName(classSymbol->name());
  spec->setType(control()->getClassType(spec));
  classSymbol->addSpecialization(binder.unit_, std::move(templateArgs), spec);

  classSymbol->setPendingInstantiation(
      spec, templateId->templateArgumentList, templateId->identifierLoc,
      !isDependent(binder.unit_, spec->type()));

  return spec;
}

auto Binder::ResolveUnqualifiedId::resolveTypeAliasTemplateId(
    SimpleTemplateIdAST* templateId, TypeAliasSymbol* typeAliasSymbol)
    -> Symbol* {
  if (typeAliasSymbol->isSpecialization()) {
    typeAliasSymbol = typeAliasSymbol->primaryTemplateSymbol();
  }

  if (!isTemplateArityMatch(typeAliasSymbol->templateDeclaration(),
                            templateId->templateArgumentList)) {
    return nullptr;
  }

  const auto retainEnclosingTemplateLevels =
      inTemplate() && hasDependentTemplateArguments(binder.unit_, templateId);

  return ASTRewriter::instantiate(
      binder.unit_, templateId->templateArgumentList, typeAliasSymbol,
      templateId->identifierLoc, /*sfinaeContext=*/false,
      /*argsComplete=*/false, /*declarationOnly=*/false,
      retainEnclosingTemplateLevels);
}

auto Binder::ResolveUnqualifiedId::operator()(SimpleTemplateIdAST* templateId)
    -> Symbol* {
  if (!checkTemplates) return templateId->symbol;

  if (templateId->identifier && templateId->identifier->builtinTemplate() !=
                                    BuiltinTemplateKind::T_NONE) {
    if (inTemplate() &&
        hasDependentTemplateArguments(binder.unit_, templateId)) {
      auto placeholder = control()->newTypeAliasSymbol(nullptr, {});
      placeholder->setName(templateId->identifier);
      if (auto alias = symbol_cast<TypeAliasSymbol>(templateId->symbol))
        placeholder->setTemplateDeclaration(alias->templateDeclaration());
      return placeholder;
    }
  }

  auto resolvedSymbol = templateId->symbol;
  if (auto injected = symbol_cast<InjectedClassNameSymbol>(resolvedSymbol)) {
    resolvedSymbol = injected->classSymbol();
  }

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(resolvedSymbol)) {
    return resolveTypeAliasTemplateId(templateId, typeAliasSymbol);
  }

  if (shouldKeepTemplateIdAsDependent(templateId)) return templateId->symbol;

  if (auto classSymbol = symbol_cast<ClassSymbol>(resolvedSymbol)) {
    if (auto spec = resolveClassTemplateId(templateId, classSymbol))
      return spec;
    if (inTemplate() && hasDependentTemplateArguments(binder.unit_, templateId))
      return templateId->symbol;
    return nullptr;
  }

  return templateId->symbol;
}

auto Binder::ResolveUnqualifiedId::operator()(NameIdAST* nameId) -> Symbol* {
  Symbol* symbol = resolvedType;
  if (!symbol && nestedNameSpecifier && nestedNameSpecifier->symbol) {
    symbol =
        qualifiedLookupType(nestedNameSpecifier->symbol, nameId->identifier);
  }

  if (!is_type(symbol)) return nullptr;

  if (auto injected = symbol_cast<InjectedClassNameSymbol>(symbol)) {
    if (auto cls = injected->classSymbol()) return cls;
  }

  return symbol;
}
}  // namespace cxx
