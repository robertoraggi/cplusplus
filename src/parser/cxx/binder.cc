// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

Binder::Binder(TranslationUnit* unit) : unit_(unit) {}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

void Binder::setTranslationUnit(TranslationUnit* unit) { unit_ = unit; }

auto Binder::control() const -> Control* {
  return unit_ ? unit_->control() : nullptr;
}

auto Binder::reportErrors() const -> bool { return reportErrors_; }

void Binder::setReportErrors(bool reportErrors) {
  reportErrors_ = reportErrors;
}

void Binder::error(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->error(loc, std::move(message));
}

void Binder::warning(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->warning(loc, std::move(message));
}

auto Binder::inTemplate() const -> bool { return inTemplate_; }

auto Binder::currentTemplateParameters() const -> TemplateParametersSymbol* {
  auto templateParameters =
      symbol_cast<TemplateParametersSymbol>(scope()->owner());

  return templateParameters;
}

auto Binder::declaringScope() const -> Scope* {
  if (!scope_) return nullptr;
  if (!scope_->isTemplateParametersScope()) return scope_;
  return scope_->enclosingNonTemplateParametersScope();
}

auto Binder::scope() const -> Scope* { return scope_; }

void Binder::setScope(Scope* scope) {
  scope_ = scope;
  inTemplate_ = false;

  for (auto current = scope_; current; current = current->parent()) {
    if (current->isTemplateParametersScope()) {
      inTemplate_ = true;
      break;
    }
  }
}

void Binder::setScope(ScopedSymbol* symbol) { setScope(symbol->scope()); }

auto Binder::enterBlock(SourceLocation loc) -> BlockSymbol* {
  auto blockSymbol = control()->newBlockSymbol(scope_, loc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol->scope());
  return blockSymbol;
}

void Binder::bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpecs) {
  const auto underlyingType = underlyingTypeSpecs.getType();

  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->lbraceLoc;

  auto enumName = get_name(control(), ast->unqualifiedId);

  if (ast->classLoc) {
    auto enumSymbol = control()->newScopedEnumSymbol(scope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  } else {
    auto enumSymbol = control()->newEnumSymbol(scope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  }
}

void Binder::bind(ParameterDeclarationAST* ast, const Decl& decl,
                  bool inTemplateParameters) {
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.getType());

  if (auto declId = decl.declaratorId; declId && declId->unqualifiedId) {
    auto paramName = get_name(control(), declId->unqualifiedId);
    if (auto identifier = name_cast<Identifier>(paramName)) {
      ast->identifier = identifier;
    } else {
      error(declId->unqualifiedId->firstSourceLocation(),
            "expected an identifier");
    }
  }

  if (!inTemplateParameters) {
    auto parameterSymbol =
        control()->newParameterSymbol(scope_, decl.location());
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(ast->type);
    scope_->addSymbol(parameterSymbol);
  }
}

void Binder::bind(EnumeratorAST* ast, const Type* type,
                  std::optional<ConstValue> value) {
  auto symbol = control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
  ast->symbol = symbol;

  symbol->setName(ast->identifier);
  symbol->setType(type);
  ast->symbol->setValue(value);
  scope()->addSymbol(symbol);

  if (auto enumSymbol = symbol_cast<EnumSymbol>(scope()->owner())) {
    auto enumeratorSymbol =
        control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
    enumeratorSymbol->setName(ast->identifier);
    enumeratorSymbol->setType(type);
    enumeratorSymbol->setValue(value);

    auto parentScope = enumSymbol->enclosingScope();
    parentScope->addSymbol(enumeratorSymbol);
  }
}

auto Binder::declareTypeAlias(SourceLocation identifierLoc, TypeIdAST* typeId)
    -> TypeAliasSymbol* {
  auto name = unit_->identifier(identifierLoc);
  auto symbol = control()->newTypeAliasSymbol(scope(), identifierLoc);
  symbol->setName(name);
  if (typeId) symbol->setType(typeId->type);
  symbol->setTemplateParameters(currentTemplateParameters());
  declaringScope()->addSymbol(symbol);
  return symbol;
}

auto Binder::declareTypedef(DeclaratorAST* declarator, const Decl& decl)
    -> TypeAliasSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());
  auto symbol = control()->newTypeAliasSymbol(scope(), decl.location());
  symbol->setName(name);
  symbol->setType(type);
  scope()->addSymbol(symbol);
  return symbol;
}

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());

  auto parentScope = scope();

  if (parentScope->isBlockScope()) {
    parentScope = parentScope->enclosingNamespaceScope();
  }

  auto functionSymbol = control()->newFunctionSymbol(scope(), decl.location());
  applySpecifiers(functionSymbol, decl.specs);
  functionSymbol->setName(name);
  functionSymbol->setType(type);
  functionSymbol->setTemplateParameters(currentTemplateParameters());

  if (isConstructor(functionSymbol)) {
    auto enclosingClass = symbol_cast<ClassSymbol>(scope()->owner());

    if (enclosingClass) {
      enclosingClass->addConstructor(functionSymbol);
    }

    return functionSymbol;
  }

  auto scope = declaringScope();

  OverloadSetSymbol* overloadSet = nullptr;

  for (Symbol* candidate : scope->find(functionSymbol->name())) {
    overloadSet = symbol_cast<OverloadSetSymbol>(candidate);
    if (overloadSet) break;

    if (auto previousFunction = symbol_cast<FunctionSymbol>(candidate)) {
      overloadSet = control()->newOverloadSetSymbol(scope, {});
      overloadSet->setName(functionSymbol->name());
      overloadSet->addFunction(previousFunction);
      scope->replaceSymbol(previousFunction, overloadSet);
      break;
    }
  }

  if (overloadSet) {
    overloadSet->addFunction(functionSymbol);
  } else {
    scope->addSymbol(functionSymbol);
  }

  return functionSymbol;
}

auto Binder::declareField(DeclaratorAST* declarator, const Decl& decl)
    -> FieldSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());
  auto fieldSymbol = control()->newFieldSymbol(scope(), decl.location());
  applySpecifiers(fieldSymbol, decl.specs);
  fieldSymbol->setName(name);
  fieldSymbol->setType(type);
  fieldSymbol->setMutable(decl.specs.isMutable);
  if (auto alignment = control()->memoryLayout()->alignmentOf(type)) {
    fieldSymbol->setAlignment(alignment.value());
  }
  scope()->addSymbol(fieldSymbol);
  return fieldSymbol;
}

auto Binder::declareVariable(DeclaratorAST* declarator, const Decl& decl)
    -> VariableSymbol* {
  auto name = decl.getName();
  auto symbol = control()->newVariableSymbol(scope(), decl.location());
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);
  symbol->setTemplateParameters(currentTemplateParameters());
  declaringScope()->addSymbol(symbol);
  return symbol;
}

auto Binder::declareMemberSymbol(DeclaratorAST* declarator, const Decl& decl)
    -> Symbol* {
  if (decl.specs.isTypedef) return declareTypedef(declarator, decl);

  if (getFunctionPrototype(declarator))
    return declareFunction(declarator, decl);

  return declareField(declarator, decl);
}

void Binder::applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setExtern(specs.isExtern);
  symbol->setFriend(specs.isFriend);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConsteval(specs.isConsteval);
  symbol->setInline(specs.isInline);
  symbol->setVirtual(specs.isVirtual);
  symbol->setExplicit(specs.isExplicit);
}

void Binder::applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setExtern(specs.isExtern);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

void Binder::applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

auto Binder::isConstructor(Symbol* symbol) const -> bool {
  auto functionSymbol = symbol_cast<FunctionSymbol>(symbol);
  if (!functionSymbol) return false;
  if (!functionSymbol->enclosingScope()) return false;
  auto classSymbol =
      symbol_cast<ClassSymbol>(functionSymbol->enclosingScope()->owner());
  if (!classSymbol) return false;
  if (classSymbol->name() != functionSymbol->name()) return false;
  return true;
}

}  // namespace cxx
