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
#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

struct [[nodiscard]] Binder::CompleteClass {
  Binder& binder;
  ClassSpecifierAST* ast;
  ClassSymbol* classSymbol;
  Arena* pool;

  CompleteClass(Binder& b, ClassSpecifierAST* a)
      : binder(b), ast(a), classSymbol(a->symbol), pool(b.unit_->arena()) {}

  auto control() const -> Control* { return binder.control(); }

  void complete();

  void markComplete();
  auto shouldSynthesizeSpecialMembers() const -> bool;
  void synthesizeSpecialMembers();
  auto hasVirtualBaseDestructor() const -> bool;

  auto newDefaultedFunction(const Name* name, const Type* type)
      -> FunctionSymbol*;
  void attachDeclaration(FunctionSymbol* symbol, UnqualifiedIdAST* id);
  auto makeCtorNameId() -> NameIdAST*;
  void addDefaultConstructor();
  void addCopyConstructor();
  void addMoveConstructor();
  void addCopyAssignmentOperator();
  void addMoveAssignmentOperator();
  void addDestructor();
};

void Binder::complete(ClassSpecifierAST* ast) {
  CompleteClass{*this, ast}.complete();
}

void Binder::CompleteClass::markComplete() { ast->symbol->setComplete(true); }

auto Binder::CompleteClass::shouldSynthesizeSpecialMembers() const -> bool {
  if (!binder.is_parsing_cxx()) return false;
  if (!classSymbol->name()) return false;
  return true;
}

void Binder::CompleteClass::synthesizeSpecialMembers() {
  addDefaultConstructor();
  addCopyConstructor();
  addMoveConstructor();
  addCopyAssignmentOperator();
  addMoveAssignmentOperator();
  addDestructor();
}

auto Binder::CompleteClass::hasVirtualBaseDestructor() const -> bool {
  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;

    auto dtor = baseClass->destructor();
    if (dtor && dtor->isVirtual()) return true;
  }

  return false;
}

void Binder::CompleteClass::complete() {
  if (binder.inTemplate()) {
    markComplete();
    return;
  }

  if (shouldSynthesizeSpecialMembers()) synthesizeSpecialMembers();

  auto status = classSymbol->buildClassLayout(control());
  if (!status.has_value())
    binder.error(classSymbol->location(), status.error());

  binder.computeClassFlags(classSymbol);
  markComplete();
}

auto Binder::CompleteClass::newDefaultedFunction(const Name* name,
                                                 const Type* type)
    -> FunctionSymbol* {
  auto symbol =
      control()->newFunctionSymbol(classSymbol, classSymbol->location());
  symbol->setName(name);
  symbol->setType(type);
  symbol->setDefined(true);
  symbol->setDefaulted(true);
  symbol->setLanguageLinkage(LanguageKind::kCXX);
  return symbol;
}

void Binder::CompleteClass::attachDeclaration(FunctionSymbol* symbol,
                                              UnqualifiedIdAST* id) {
  auto idDecl = IdDeclaratorAST::create(pool);
  idDecl->unqualifiedId = id;

  auto funcChunk = FunctionDeclaratorChunkAST::create(pool);

  auto declarator = DeclaratorAST::create(
      pool, nullptr, idDecl,
      make_list_node<DeclaratorChunkAST>(pool, funcChunk));

  auto funcDef = FunctionDefinitionAST::create(pool);
  funcDef->declarator = declarator;
  funcDef->functionBody = DefaultFunctionBodyAST::create(pool);
  funcDef->symbol = symbol;
  symbol->setDeclaration(funcDef);
}

auto Binder::CompleteClass::makeCtorNameId() -> NameIdAST* {
  return NameIdAST::create(pool, name_cast<Identifier>(classSymbol->name()));
}

void Binder::CompleteClass::addDefaultConstructor() {
  if (!classSymbol->constructors().empty()) return;

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addCopyConstructor() {
  if (classSymbol->copyConstructor()) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getConstType(classSymbol->type()));

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {constRefType}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addMoveConstructor() {
  if (classSymbol->moveConstructor()) return;

  auto rvalRefType = control()->getRvalueReferenceType(classSymbol->type());

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {rvalRefType}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addCopyAssignmentOperator() {
  if (classSymbol->copyAssignmentOperator()) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getConstType(classSymbol->type()));
  auto retType = control()->getLvalueReferenceType(classSymbol->type());

  auto symbol =
      newDefaultedFunction(control()->getOperatorId(TokenKind::T_EQUAL),
                           control()->getFunctionType(retType, {constRefType}));
  classSymbol->addSymbol(symbol);
  attachDeclaration(symbol,
                    OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL));
}

void Binder::CompleteClass::addMoveAssignmentOperator() {
  if (classSymbol->moveAssignmentOperator()) return;

  auto rvalRefType = control()->getRvalueReferenceType(classSymbol->type());
  auto retType = control()->getLvalueReferenceType(classSymbol->type());

  auto symbol =
      newDefaultedFunction(control()->getOperatorId(TokenKind::T_EQUAL),
                           control()->getFunctionType(retType, {rvalRefType}));
  classSymbol->addSymbol(symbol);
  attachDeclaration(symbol,
                    OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL));
}

void Binder::CompleteClass::addDestructor() {
  if (classSymbol->destructor()) return;

  auto symbol = newDefaultedFunction(
      control()->getDestructorId(classSymbol->name()),
      control()->getFunctionType(control()->getVoidType(), {}));

  if (hasVirtualBaseDestructor()) symbol->setVirtual(true);

  classSymbol->addSymbol(symbol);

  auto dtorId = DestructorIdAST::create(pool);
  if (auto id = name_cast<Identifier>(classSymbol->name()))
    dtorId->id = NameIdAST::create(pool, id);
  attachDeclaration(symbol, dtorId);
}

}  // namespace cxx