// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#pragma once

#include <cxx/ir/ir.h>
#include <cxx/ir/ir_factory.h>

#include <list>

namespace cxx::ir {

class IRBuilder {
 public:
  explicit IRBuilder(Module* module = nullptr);
  ~IRBuilder() = default;

  auto module() -> Module* { return module_; }
  void setModule(Module* module);

  void setInsertionPoint(Block* block, const std::list<Stmt*>::iterator& ip);
  void setInsertionPoint(Block* block);

  explicit operator bool() const { return module_ && block_; }

  [[nodiscard]] auto block() const -> Block* { return block_; }

  [[nodiscard]] auto blockHasTerminator() const -> bool {
    if (!block_) return true;

    return !block_->code().empty() ? block_->code().back()->isTerminator()
                                   : false;
  }

  auto emitExpr(Expr* target) -> Expr*;
  auto emitMove(Expr* target, Expr* source) -> Move*;
  auto emitJump(Block* target) -> Jump*;
  auto emitCondJump(Expr* condition, Block* iftrue, Block* iffalse)
      -> CondJump*;
  auto emitSwitch(Expr* condition) -> Switch*;
  auto emitRet(Expr* result) -> Ret*;
  auto emitRetVoid() -> RetVoid*;

  auto createThis(const QualifiedType& type) -> This*;
  auto createBoolLiteral(bool value) -> BoolLiteral*;
  auto createCharLiteral(const cxx::CharLiteral* value) -> CharLiteral*;
  auto createIntegerLiteral(const IntegerValue& value) -> IntegerLiteral*;
  auto createFloatLiteral(const FloatValue& value) -> FloatLiteral*;
  auto createNullptrLiteral() -> NullptrLiteral*;
  auto createStringLiteral(const cxx::StringLiteral* value) -> StringLiteral*;
  auto createUserDefinedStringLiteral(std::string value)
      -> UserDefinedStringLiteral*;
  auto createTemp(Local* local) -> Temp*;
  auto createId(Symbol* symbol) -> Id*;
  auto createExternalId(std::string name) -> ExternalId*;
  auto createTypeid(Expr* expr) -> Typeid*;
  auto createUnary(UnaryOp op, Expr* expr) -> Unary*;
  auto createBinary(BinaryOp op, Expr* left, Expr* right) -> Binary*;
  auto createCall(Expr* base, std::vector<Expr*> args) -> Call*;
  auto createSubscript(Expr* base, Expr* index) -> Subscript*;
  auto createAccess(Expr* base, Symbol* member) -> Access*;
  auto createCast(const QualifiedType& type, Expr* expr) -> Cast*;
  auto createStaticCast(const QualifiedType& type, Expr* expr) -> StaticCast*;
  auto createDynamicCast(const QualifiedType& type, Expr* expr) -> DynamicCast*;
  auto createReinterpretCast(const QualifiedType& type, Expr* expr)
      -> ReinterpretCast*;
  auto createNew(const QualifiedType& type, std::vector<Expr*> args) -> New*;
  auto createNewArray(const QualifiedType& type, Expr* size) -> NewArray*;
  auto createDelete(Expr* expr) -> Delete*;
  auto createDeleteArray(Expr* expr) -> DeleteArray*;
  auto createThrow(Expr* expr) -> Throw*;

 private:
  template <typename T>
  auto insert(T* stmt) -> T* {
    auto it = block_->code().insert(ip_, stmt);
    ip_ = ++it;
    return stmt;
  }

 private:
  Module* module_ = nullptr;
  IRFactory* factory_ = nullptr;
  Block* block_ = nullptr;
  std::list<Stmt*>::iterator ip_;
};

}  // namespace cxx::ir
