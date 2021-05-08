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

#pragma once

#include <cxx/ir.h>
#include <cxx/ir_factory.h>

#include <list>

namespace cxx::ir {

class IRBuilder {
 public:
  explicit IRBuilder(Module* module = nullptr);
  ~IRBuilder();

  Module* module() { return module_; }
  void setModule(Module* module);

  void setInsertionPoint(Block* block, const std::list<Stmt*>::iterator& ip);
  void setInsertionPoint(Block* block);

  explicit operator bool() const { return module_ && block_; }

  Block* block() const { return block_; }

  bool blockHasTerminator() const {
    if (!block_) return true;

    return !block_->code().empty() ? block_->code().back()->isTerminator()
                                   : false;
  }

  Expr* emitExpr(Expr* target);
  Move* emitMove(Expr* target, Expr* source);
  Jump* emitJump(Block* target);
  CondJump* emitCondJump(Expr* condition, Block* iftrue, Block* iffalse);
  Switch* emitSwitch(Expr* condition);
  Ret* emitRet(Expr* result);
  RetVoid* emitRetVoid();

  This* createThis(const QualifiedType& type);
  BoolLiteral* createBoolLiteral(bool value);
  CharLiteral* createCharLiteral(const cxx::CharLiteral* value);
  IntegerLiteral* createIntegerLiteral(const IntegerValue& value);
  FloatLiteral* createFloatLiteral(const FloatValue& value);
  NullptrLiteral* createNullptrLiteral();
  StringLiteral* createStringLiteral(const cxx::StringLiteral* value);
  UserDefinedStringLiteral* createUserDefinedStringLiteral(std::string value);
  Temp* createTemp(Local* local);
  Id* createId(Symbol* symbol);
  ExternalId* createExternalId(std::string name);
  Typeid* createTypeid(Expr* expr);
  Unary* createUnary(UnaryOp op, Expr* expr);
  Binary* createBinary(BinaryOp op, Expr* left, Expr* right);
  Call* createCall(Expr* base, std::vector<Expr*> args);
  Subscript* createSubscript(Expr* base, Expr* index);
  Access* createAccess(Expr* base, Symbol* member);
  Cast* createCast(const QualifiedType& type, Expr* expr);
  StaticCast* createStaticCast(const QualifiedType& type, Expr* expr);
  DynamicCast* createDynamicCast(const QualifiedType& type, Expr* expr);
  ReinterpretCast* createReinterpretCast(const QualifiedType& type, Expr* expr);
  New* createNew(const QualifiedType& type, std::vector<Expr*> args);
  NewArray* createNewArray(const QualifiedType& type, Expr* size);
  Delete* createDelete(Expr* expr);
  DeleteArray* createDeleteArray(Expr* expr);
  Throw* createThrow(Expr* expr);

 private:
  template <typename T>
  T* insert(T* stmt) {
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
