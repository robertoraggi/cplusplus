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

#include <cxx/ir.h>
#include <cxx/ir_factory.h>

#include <forward_list>

namespace cxx::ir {

struct IRFactory::Private {
  Module* module_ = nullptr;
  std::forward_list<Global> globals_;
  std::forward_list<Function> functions_;
  std::forward_list<Block> blocks_;
  std::forward_list<Move> moves_;
  std::forward_list<Jump> jumps_;
  std::forward_list<CondJump> condJumps_;
  std::forward_list<Ret> rets_;
  std::forward_list<RetVoid> retVoids_;
  std::forward_list<This> this_;
  std::forward_list<BoolLiteral> boolLiterals_;
  std::forward_list<IntegerLiteral> integerLiterals_;
  std::forward_list<FloatLiteral> floatLiterals_;
  std::forward_list<NullptrLiteral> nullptrLiterals_;
  std::forward_list<StringLiteral> stringLiterals_;
  std::forward_list<UserDefinedStringLiteral> userDefinedStringLiterals_;
  std::forward_list<Temp> temps_;
  std::forward_list<Id> ids_;
  std::forward_list<ExternalId> externalIds_;
  std::forward_list<Typeid> typeids_;
  std::forward_list<Unary> unarys_;
  std::forward_list<Binary> binarys_;
  std::forward_list<Call> calls_;
  std::forward_list<Subscript> subscripts_;
  std::forward_list<Access> accesss_;
  std::forward_list<Cast> casts_;
  std::forward_list<StaticCast> staticCasts_;
  std::forward_list<DynamicCast> dynamicCasts_;
  std::forward_list<ReinterpretCast> reinterpretCasts_;
  std::forward_list<New> news_;
  std::forward_list<NewArray> newArrays_;
  std::forward_list<Delete> deletes_;
  std::forward_list<DeleteArray> deleteArrays_;
  std::forward_list<Throw> throws_;
};

IRFactory::IRFactory() : d(std::make_unique<Private>()) {}

IRFactory::~IRFactory() {}

Module* IRFactory::module() const { return d->module_; }

void IRFactory::setModule(Module* module) { d->module_ = module; }

Global* IRFactory::createGlobal(Symbol* symbol) {
  return &d->globals_.emplace_front(d->module_, symbol);
}

Function* IRFactory::createFunction(FunctionSymbol* symbol) {
  return &d->functions_.emplace_front(d->module_, symbol);
}

Block* IRFactory::createBlock(Function* function) {
  return &d->blocks_.emplace_front(function);
}

Move* IRFactory::createMove(Expr* target, Expr* source) {
  return &d->moves_.emplace_front(target, source);
}

Jump* IRFactory::createJump(Block* target) {
  return &d->jumps_.emplace_front(target);
}

CondJump* IRFactory::createCondJump(Expr* condition, Block* iftrue,
                                    Block* iffalse) {
  return &d->condJumps_.emplace_front(condition, iftrue, iffalse);
}

Ret* IRFactory::createRet(Expr* result) {
  return &d->rets_.emplace_front(result);
}

RetVoid* IRFactory::createRetVoid() { return &d->retVoids_.emplace_front(); }

This* IRFactory::createThis(const QualifiedType& type) {
  return &d->this_.emplace_front(type);
}

BoolLiteral* IRFactory::createBoolLiteral(bool value) {
  return &d->boolLiterals_.emplace_front(value);
}

IntegerLiteral* IRFactory::createIntegerLiteral(const IntegerValue& value) {
  return &d->integerLiterals_.emplace_front(value);
}

FloatLiteral* IRFactory::createFloatLiteral(const FloatValue& value) {
  return &d->floatLiterals_.emplace_front(value);
}

NullptrLiteral* IRFactory::createNullptrLiteral() {
  return &d->nullptrLiterals_.emplace_front();
}

StringLiteral* IRFactory::createStringLiteral(std::string value) {
  return &d->stringLiterals_.emplace_front(std::move(value));
}

UserDefinedStringLiteral* IRFactory::createUserDefinedStringLiteral(
    std::string value) {
  return &d->userDefinedStringLiterals_.emplace_front(std::move(value));
}

Temp* IRFactory::createTemp(Local* local) {
  return &d->temps_.emplace_front(local);
}

Id* IRFactory::createId(Symbol* symbol) {
  return &d->ids_.emplace_front(symbol);
}

ExternalId* IRFactory::createExternalId(std::string name) {
  return &d->externalIds_.emplace_front(std::move(name));
}

Typeid* IRFactory::createTypeid(Expr* expr) {
  return &d->typeids_.emplace_front(expr);
}

Unary* IRFactory::createUnary(UnaryOp op, Expr* expr) {
  return &d->unarys_.emplace_front(op, expr);
}

Binary* IRFactory::createBinary(BinaryOp op, Expr* left, Expr* right) {
  return &d->binarys_.emplace_front(op, left, right);
}

Call* IRFactory::createCall(Expr* base, std::vector<Expr*> args) {
  return &d->calls_.emplace_front(base, std::move(args));
}

Subscript* IRFactory::createSubscript(Expr* base, Expr* index) {
  return &d->subscripts_.emplace_front(base, index);
}

Access* IRFactory::createAccess(Expr* base, Expr* member) {
  return &d->accesss_.emplace_front(base, member);
}

Cast* IRFactory::createCast(const QualifiedType& type, Expr* expr) {
  return &d->casts_.emplace_front(type, expr);
}

StaticCast* IRFactory::createStaticCast(const QualifiedType& type, Expr* expr) {
  return &d->staticCasts_.emplace_front(type, expr);
}

DynamicCast* IRFactory::createDynamicCast(const QualifiedType& type,
                                          Expr* expr) {
  return &d->dynamicCasts_.emplace_front(type, expr);
}

ReinterpretCast* IRFactory::createReinterpretCast(const QualifiedType& type,
                                                  Expr* expr) {
  return &d->reinterpretCasts_.emplace_front(type, expr);
}

New* IRFactory::createNew(const QualifiedType& type, std::vector<Expr*> args) {
  return &d->news_.emplace_front(type, std::move(args));
}

NewArray* IRFactory::createNewArray(const QualifiedType& type, Expr* size) {
  return &d->newArrays_.emplace_front(type, size);
}

Delete* IRFactory::createDelete(Expr* expr) {
  return &d->deletes_.emplace_front(expr);
}

DeleteArray* IRFactory::createDeleteArray(Expr* expr) {
  return &d->deleteArrays_.emplace_front(expr);
}

Throw* IRFactory::createThrow(Expr* expr) {
  return &d->throws_.emplace_front(expr);
}

}  // namespace cxx::ir
