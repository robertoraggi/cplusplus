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

#pragma once

#include <cxx/ir/ir_fwd.h>
#include <cxx/literals_fwd.h>
#include <cxx/symbols_fwd.h>

#include <memory>
#include <vector>

namespace cxx::ir {

class IRFactory {
 public:
  IRFactory();
  ~IRFactory();

  [[nodiscard]] auto module() const -> Module*;
  void setModule(Module* module);

  auto createGlobal(Symbol* symbol) -> Global*;
  auto createFunction(FunctionSymbol* symbol) -> Function*;
  auto createBlock(Function* function) -> Block*;
  auto createMove(Expr* target, Expr* source) -> Move*;
  auto createJump(Block* target) -> Jump*;
  auto createCondJump(Expr* condition, Block* iftrue, Block* iffalse)
      -> CondJump*;
  auto createSwitch(Expr* condition) -> Switch*;
  auto createRet(Expr* result) -> Ret*;
  auto createRetVoid() -> RetVoid*;
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
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx::ir
