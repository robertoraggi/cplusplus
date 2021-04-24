// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ir_builder.h>

namespace cxx::ir {

IRBuilder::IRBuilder(Module* module) : module_(module) {
  factory_ = module_ ? module_->irFactory() : nullptr;
}

IRBuilder::~IRBuilder() {}

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

Move* IRBuilder::emitMove(Expr* target, Expr* source) {
  return insert(factory_->createMove(target, source));
}

Jump* IRBuilder::emitJump(Block* target) {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createJump(target));
}

CondJump* IRBuilder::emitCondJump(Expr* condition, Block* iftrue,
                                  Block* iffalse) {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createCondJump(condition, iftrue, iffalse));
}

Ret* IRBuilder::emitRet(Expr* result) {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createRet(result));
}

RetVoid* IRBuilder::emitRetVoid() {
  if (blockHasTerminator()) return nullptr;
  return insert(factory_->createRetVoid());
}

This* IRBuilder::createThis(Expr* type) {
  return insert(factory_->createThis(type));
}

BoolLiteral* IRBuilder::createBoolLiteral(bool value) {
  return factory_->createBoolLiteral(value);
}

IntegerLiteral* IRBuilder::createIntegerLiteral(const IntegerValue& value) {
  return factory_->createIntegerLiteral(value);
}

FloatLiteral* IRBuilder::createFloatLiteral(const FloatValue& value) {
  return factory_->createFloatLiteral(value);
}

NullptrLiteral* IRBuilder::createNullptrLiteral() {
  return factory_->createNullptrLiteral();
}

StringLiteral* IRBuilder::createStringLiteral(std::string value) {
  return factory_->createStringLiteral(std::move(value));
}

UserDefinedStringLiteral* IRBuilder::createUserDefinedStringLiteral(
    std::string value) {
  return factory_->createUserDefinedStringLiteral(std::move(value));
}

Load* IRBuilder::createLoad(Local* local) {
  return factory_->createLoad(local);
}

Id* IRBuilder::createId(Symbol* symbol) { return factory_->createId(symbol); }

ExternalId* IRBuilder::createExternalId(std::string name) {
  return factory_->createExternalId(std::move(name));
}

Typeid* IRBuilder::createTypeid(Expr* expr) {
  return factory_->createTypeid(expr);
}

Unary* IRBuilder::createUnary(UnaryOp op, Expr* expr) {
  return factory_->createUnary(op, expr);
}

Binary* IRBuilder::createBinary(BinaryOp op, Expr* left, Expr* right) {
  return factory_->createBinary(op, left, right);
}

Call* IRBuilder::createCall(Expr* base, std::vector<Expr*> args) {
  return factory_->createCall(base, args);
}

Subscript* IRBuilder::createSubscript(Expr* base, Expr* index) {
  return factory_->createSubscript(base, index);
}

Access* IRBuilder::createAccess(Expr* base, Expr* member) {
  return factory_->createAccess(base, member);
}

Cast* IRBuilder::createCast(Expr* type, Expr* expr) {
  return factory_->createCast(type, expr);
}

StaticCast* IRBuilder::createStaticCast(Expr* type, Expr* expr) {
  return factory_->createStaticCast(type, expr);
}

DynamicCast* IRBuilder::createDynamicCast(Expr* type, Expr* expr) {
  return factory_->createDynamicCast(type, expr);
}

ReinterpretCast* IRBuilder::createReinterpretCast(Expr* type, Expr* expr) {
  return factory_->createReinterpretCast(type, expr);
}

New* IRBuilder::createNew(Expr* type, std::vector<Expr*> args) {
  return factory_->createNew(type, std::move(args));
}

NewArray* IRBuilder::createNewArray(Expr* type, Expr* size) {
  return factory_->createNewArray(type, size);
}

Delete* IRBuilder::createDelete(Expr* expr) {
  return factory_->createDelete(expr);
}

DeleteArray* IRBuilder::createDeleteArray(Expr* expr) {
  return factory_->createDeleteArray(expr);
}

Throw* IRBuilder::createThrow(Expr* expr) {
  return factory_->createThrow(expr);
}

}  // namespace cxx::ir
