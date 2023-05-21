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

namespace cxx::ir {

class IRVisitor {
 public:
  virtual ~IRVisitor();

  virtual void visit(Jump*) = 0;
  virtual void visit(CondJump*) = 0;
  virtual void visit(Switch*) = 0;
  virtual void visit(Ret*) = 0;
  virtual void visit(RetVoid*) = 0;
  virtual void visit(Move*) = 0;

  virtual void visit(This*) = 0;
  virtual void visit(BoolLiteral*) = 0;
  virtual void visit(CharLiteral*) = 0;
  virtual void visit(IntegerLiteral*) = 0;
  virtual void visit(FloatLiteral*) = 0;
  virtual void visit(NullptrLiteral*) = 0;
  virtual void visit(StringLiteral*) = 0;
  virtual void visit(UserDefinedStringLiteral*) = 0;
  virtual void visit(Temp*) = 0;
  virtual void visit(Id*) = 0;
  virtual void visit(ExternalId*) = 0;
  virtual void visit(Typeid*) = 0;
  virtual void visit(Unary*) = 0;
  virtual void visit(Binary*) = 0;
  virtual void visit(Call*) = 0;
  virtual void visit(Subscript*) = 0;
  virtual void visit(Access*) = 0;
  virtual void visit(Cast*) = 0;
  virtual void visit(StaticCast*) = 0;
  virtual void visit(DynamicCast*) = 0;
  virtual void visit(ReinterpretCast*) = 0;
  virtual void visit(New*) = 0;
  virtual void visit(NewArray*) = 0;
  virtual void visit(Delete*) = 0;
  virtual void visit(DeleteArray*) = 0;
  virtual void visit(Throw*) = 0;
};

}  // namespace cxx::ir
