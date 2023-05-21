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

#include <cxx/ir/ir.h>
#include <cxx/ir/ir_factory.h>

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
  std::forward_list<Switch> switchs_;
  std::forward_list<Ret> rets_;
  std::forward_list<RetVoid> retVoids_;
  std::forward_list<This> this_;
  std::forward_list<BoolLiteral> boolLiterals_;
  std::forward_list<CharLiteral> charLiterals_;
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

IRFactory::~IRFactory() = default;

auto IRFactory::module() const -> Module* { return d->module_; }

void IRFactory::setModule(Module* module) { d->module_ = module; }

auto IRFactory::createGlobal(Symbol* symbol) -> Global* {
  return &d->globals_.emplace_front(d->module_, symbol);
}

auto IRFactory::createFunction(FunctionSymbol* symbol) -> Function* {
  return &d->functions_.emplace_front(d->module_, symbol);
}

auto IRFactory::createBlock(Function* function) -> Block* {
  return &d->blocks_.emplace_front(function);
}

auto IRFactory::createMove(Expr* target, Expr* source) -> Move* {
  return &d->moves_.emplace_front(target, source);
}

auto IRFactory::createJump(Block* target) -> Jump* {
  return &d->jumps_.emplace_front(target);
}

auto IRFactory::createCondJump(Expr* condition, Block* iftrue, Block* iffalse)
    -> CondJump* {
  return &d->condJumps_.emplace_front(condition, iftrue, iffalse);
}

auto IRFactory::createSwitch(Expr* condition) -> Switch* {
  return &d->switchs_.emplace_front(condition);
}

auto IRFactory::createRet(Expr* result) -> Ret* {
  return &d->rets_.emplace_front(result);
}

auto IRFactory::createRetVoid() -> RetVoid* {
  return &d->retVoids_.emplace_front();
}

auto IRFactory::createThis(const QualifiedType& type) -> This* {
  return &d->this_.emplace_front(type);
}

auto IRFactory::createBoolLiteral(bool value) -> BoolLiteral* {
  return &d->boolLiterals_.emplace_front(value);
}

auto IRFactory::createCharLiteral(const cxx::CharLiteral* value)
    -> CharLiteral* {
  return &d->charLiterals_.emplace_front(value);
}

auto IRFactory::createIntegerLiteral(const IntegerValue& value)
    -> IntegerLiteral* {
  return &d->integerLiterals_.emplace_front(value);
}

auto IRFactory::createFloatLiteral(const FloatValue& value) -> FloatLiteral* {
  return &d->floatLiterals_.emplace_front(value);
}

auto IRFactory::createNullptrLiteral() -> NullptrLiteral* {
  return &d->nullptrLiterals_.emplace_front();
}

auto IRFactory::createStringLiteral(const cxx::StringLiteral* value)
    -> StringLiteral* {
  return &d->stringLiterals_.emplace_front(value);
}

auto IRFactory::createUserDefinedStringLiteral(std::string value)
    -> UserDefinedStringLiteral* {
  return &d->userDefinedStringLiterals_.emplace_front(std::move(value));
}

auto IRFactory::createTemp(Local* local) -> Temp* {
  return &d->temps_.emplace_front(local);
}

auto IRFactory::createId(Symbol* symbol) -> Id* {
  return &d->ids_.emplace_front(symbol);
}

auto IRFactory::createExternalId(std::string name) -> ExternalId* {
  return &d->externalIds_.emplace_front(std::move(name));
}

auto IRFactory::createTypeid(Expr* expr) -> Typeid* {
  return &d->typeids_.emplace_front(expr);
}

auto IRFactory::createUnary(UnaryOp op, Expr* expr) -> Unary* {
  return &d->unarys_.emplace_front(op, expr);
}

auto IRFactory::createBinary(BinaryOp op, Expr* left, Expr* right) -> Binary* {
  return &d->binarys_.emplace_front(op, left, right);
}

auto IRFactory::createCall(Expr* base, std::vector<Expr*> args) -> Call* {
  return &d->calls_.emplace_front(base, std::move(args));
}

auto IRFactory::createSubscript(Expr* base, Expr* index) -> Subscript* {
  return &d->subscripts_.emplace_front(base, index);
}

auto IRFactory::createAccess(Expr* base, Symbol* member) -> Access* {
  return &d->accesss_.emplace_front(base, member);
}

auto IRFactory::createCast(const QualifiedType& type, Expr* expr) -> Cast* {
  return &d->casts_.emplace_front(type, expr);
}

auto IRFactory::createStaticCast(const QualifiedType& type, Expr* expr)
    -> StaticCast* {
  return &d->staticCasts_.emplace_front(type, expr);
}

auto IRFactory::createDynamicCast(const QualifiedType& type, Expr* expr)
    -> DynamicCast* {
  return &d->dynamicCasts_.emplace_front(type, expr);
}

auto IRFactory::createReinterpretCast(const QualifiedType& type, Expr* expr)
    -> ReinterpretCast* {
  return &d->reinterpretCasts_.emplace_front(type, expr);
}

auto IRFactory::createNew(const QualifiedType& type, std::vector<Expr*> args)
    -> New* {
  return &d->news_.emplace_front(type, std::move(args));
}

auto IRFactory::createNewArray(const QualifiedType& type, Expr* size)
    -> NewArray* {
  return &d->newArrays_.emplace_front(type, size);
}

auto IRFactory::createDelete(Expr* expr) -> Delete* {
  return &d->deletes_.emplace_front(expr);
}

auto IRFactory::createDeleteArray(Expr* expr) -> DeleteArray* {
  return &d->deleteArrays_.emplace_front(expr);
}

auto IRFactory::createThrow(Expr* expr) -> Throw* {
  return &d->throws_.emplace_front(expr);
}

}  // namespace cxx::ir
