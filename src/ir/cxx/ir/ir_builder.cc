// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ir/ir_builder.h>

#include <utility>

namespace cxx::ir {

IRBuilder::IRBuilder(Module* module) : module_(module) {
  factory_ = module_ ? module_->irFactory() : nullptr;
}

void IRBuilder::setModule(Module* module) {
  module_ = module;
  factory_ = module_ ? module_->irFactory() : nullptr;
}

void IRBuilder::setInsertionPoint(Block* block,
                                  const std::list<Stmt*>::iterator& ip) {
  block_ = block;
  ip_ = ip;
}

void IRBuilder::setInsertionPoint(Block* block) {
  setInsertionPoint(block,
                    block ? block->code().end() : std::list<Stmt*>::iterator());
}

auto IRBuilder::emitExpr(Expr* target) -> Expr* { return insert(target); }

auto IRBuilder::emitMove(Expr* target, Expr* source) -> Move* {
  return insert(factory_->createMove(target, source));
}

auto IRBuilder::emitJump(Block* target) -> Jump* {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createJump(target));
}

auto IRBuilder::emitCondJump(Expr* condition, Block* iftrue, Block* iffalse)
    -> CondJump* {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createCondJump(condition, iftrue, iffalse));
}

auto IRBuilder::emitSwitch(Expr* condition) -> Switch* {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createSwitch(condition));
}

auto IRBuilder::emitRet(Expr* result) -> Ret* {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createRet(result));
}

auto IRBuilder::emitRetVoid() -> RetVoid* {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createRetVoid());
}

auto IRBuilder::createThis(const QualifiedType& type) -> This* {
  return factory_->createThis(type);
}

auto IRBuilder::createBoolLiteral(bool value) -> BoolLiteral* {
  return factory_->createBoolLiteral(value);
}

auto IRBuilder::createCharLiteral(const cxx::CharLiteral* value)
    -> CharLiteral* {
  return factory_->createCharLiteral(value);
}

auto IRBuilder::createIntegerLiteral(const IntegerValue& value)
    -> IntegerLiteral* {
  return factory_->createIntegerLiteral(value);
}

auto IRBuilder::createFloatLiteral(const FloatValue& value) -> FloatLiteral* {
  return factory_->createFloatLiteral(value);
}

auto IRBuilder::createNullptrLiteral() -> NullptrLiteral* {
  return factory_->createNullptrLiteral();
}

auto IRBuilder::createStringLiteral(const cxx::StringLiteral* value)
    -> StringLiteral* {
  return factory_->createStringLiteral(value);
}

auto IRBuilder::createUserDefinedStringLiteral(std::string value)
    -> UserDefinedStringLiteral* {
  return factory_->createUserDefinedStringLiteral(std::move(value));
}

auto IRBuilder::createTemp(Local* local) -> Temp* {
  return factory_->createTemp(local);
}

auto IRBuilder::createId(Symbol* symbol) -> Id* {
  return factory_->createId(symbol);
}

auto IRBuilder::createExternalId(std::string name) -> ExternalId* {
  return factory_->createExternalId(std::move(name));
}

auto IRBuilder::createTypeid(Expr* expr) -> Typeid* {
  return factory_->createTypeid(expr);
}

auto IRBuilder::createUnary(UnaryOp op, Expr* expr) -> Unary* {
  return factory_->createUnary(op, expr);
}

auto IRBuilder::createBinary(BinaryOp op, Expr* left, Expr* right) -> Binary* {
  return factory_->createBinary(op, left, right);
}

auto IRBuilder::createCall(Expr* base, std::vector<Expr*> args) -> Call* {
  return factory_->createCall(base, std::move(args));
}

auto IRBuilder::createSubscript(Expr* base, Expr* index) -> Subscript* {
  return factory_->createSubscript(base, index);
}

auto IRBuilder::createAccess(Expr* base, Symbol* member) -> Access* {
  return factory_->createAccess(base, member);
}

auto IRBuilder::createCast(const QualifiedType& type, Expr* expr) -> Cast* {
  return factory_->createCast(type, expr);
}

auto IRBuilder::createStaticCast(const QualifiedType& type, Expr* expr)
    -> StaticCast* {
  return factory_->createStaticCast(type, expr);
}

auto IRBuilder::createDynamicCast(const QualifiedType& type, Expr* expr)
    -> DynamicCast* {
  return factory_->createDynamicCast(type, expr);
}

auto IRBuilder::createReinterpretCast(const QualifiedType& type, Expr* expr)
    -> ReinterpretCast* {
  return factory_->createReinterpretCast(type, expr);
}

auto IRBuilder::createNew(const QualifiedType& type, std::vector<Expr*> args)
    -> New* {
  return factory_->createNew(type, std::move(args));
}

auto IRBuilder::createNewArray(const QualifiedType& type, Expr* size)
    -> NewArray* {
  return factory_->createNewArray(type, size);
}

auto IRBuilder::createDelete(Expr* expr) -> Delete* {
  return factory_->createDelete(expr);
}

auto IRBuilder::createDeleteArray(Expr* expr) -> DeleteArray* {
  return factory_->createDeleteArray(expr);
}

auto IRBuilder::createThrow(Expr* expr) -> Throw* {
  return factory_->createThrow(expr);
}

}  // namespace cxx::ir
