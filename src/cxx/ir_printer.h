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

#include <cxx/ir_visitor.h>

#include <iosfwd>
#include <string>

namespace cxx::ir {

class IRPrinter final : IRVisitor {
 public:
  void print(Module* module, std::ostream& out);
  void print(Function* function, std::ostream& out);
  void print(Block* block, std::ostream& out);
  void print(Stmt* stmt, std::ostream& out);

 private:
  std::string toString(Stmt* stmt);
  std::string toString(Block* block) const;
  std::string_view toString(UnaryOp op) const;
  std::string_view toString(BinaryOp op) const;

  std::string quote(const std::string& s) const;

  void accept(Stmt* stmt);

  void visit(Jump*) override;
  void visit(CondJump*) override;
  void visit(Switch*) override;
  void visit(Ret*) override;
  void visit(RetVoid*) override;
  void visit(Move*) override;

  void visit(This*) override;
  void visit(BoolLiteral*) override;
  void visit(IntegerLiteral*) override;
  void visit(FloatLiteral*) override;
  void visit(NullptrLiteral*) override;
  void visit(StringLiteral*) override;
  void visit(UserDefinedStringLiteral*) override;
  void visit(Temp*) override;
  void visit(Id*) override;
  void visit(ExternalId*) override;
  void visit(Typeid*) override;
  void visit(Unary*) override;
  void visit(Binary*) override;
  void visit(Call*) override;
  void visit(Subscript*) override;
  void visit(Access*) override;
  void visit(Cast*) override;
  void visit(StaticCast*) override;
  void visit(DynamicCast*) override;
  void visit(ReinterpretCast*) override;
  void visit(New*) override;
  void visit(NewArray*) override;
  void visit(Delete*) override;
  void visit(DeleteArray*) override;
  void visit(Throw*) override;

 private:
  std::string text_;
};

}  // namespace cxx::ir
