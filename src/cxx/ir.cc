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
#include <cxx/ir_visitor.h>

namespace cxx::ir {

int Block::id() const {
  auto it = find(begin(function_->blocks()), end(function_->blocks()), this);
  if (it != end(function_->blocks()))
    return std::distance(begin(function_->blocks()), it) + 1;
  return 0;
}

void Jump::accept(IRVisitor* visitor) { visitor->visit(this); }

void CondJump::accept(IRVisitor* visitor) { visitor->visit(this); }

void Ret::accept(IRVisitor* visitor) { visitor->visit(this); }

void RetVoid::accept(IRVisitor* visitor) { visitor->visit(this); }

void This::accept(IRVisitor* visitor) { visitor->visit(this); }

void BoolLiteral::accept(IRVisitor* visitor) { visitor->visit(this); }

void IntegerLiteral::accept(IRVisitor* visitor) { visitor->visit(this); }

void FloatLiteral::accept(IRVisitor* visitor) { visitor->visit(this); }

void NullptrLiteral::accept(IRVisitor* visitor) { visitor->visit(this); }

void StringLiteral::accept(IRVisitor* visitor) { visitor->visit(this); }

void UserDefinedStringLiteral::accept(IRVisitor* visitor) {
  visitor->visit(this);
}

void Id::accept(IRVisitor* visitor) { visitor->visit(this); }

void ExternalId::accept(IRVisitor* visitor) { visitor->visit(this); }

void Sizeof::accept(IRVisitor* visitor) { visitor->visit(this); }

void Typeid::accept(IRVisitor* visitor) { visitor->visit(this); }

void Alignof::accept(IRVisitor* visitor) { visitor->visit(this); }

void Unary::accept(IRVisitor* visitor) { visitor->visit(this); }

void Binary::accept(IRVisitor* visitor) { visitor->visit(this); }

void Assignment::accept(IRVisitor* visitor) { visitor->visit(this); }

void Call::accept(IRVisitor* visitor) { visitor->visit(this); }

void Subscript::accept(IRVisitor* visitor) { visitor->visit(this); }

void Access::accept(IRVisitor* visitor) { visitor->visit(this); }

void Cast::accept(IRVisitor* visitor) { visitor->visit(this); }

void StaticCast::accept(IRVisitor* visitor) { visitor->visit(this); }

void DynamicCast::accept(IRVisitor* visitor) { visitor->visit(this); }

void ReinterpretCast::accept(IRVisitor* visitor) { visitor->visit(this); }

void New::accept(IRVisitor* visitor) { visitor->visit(this); }

void NewArray::accept(IRVisitor* visitor) { visitor->visit(this); }

void Delete::accept(IRVisitor* visitor) { visitor->visit(this); }

void DeleteArray::accept(IRVisitor* visitor) { visitor->visit(this); }

void Throw::accept(IRVisitor* visitor) { visitor->visit(this); }

}  // namespace cxx::ir