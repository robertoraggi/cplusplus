// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#define FOR_EACH_IR_STMT(V) \
  V(Exp)                    \
  V(Move)                   \
  V(Ret)                    \
  V(Jump)                   \
  V(CJump)

#define FOR_EACH_IR_EXPR(V) \
  V(This)                   \
  V(Const)                  \
  V(Temp)                   \
  V(Sym)                    \
  V(Cast)                   \
  V(DynamicCast)            \
  V(StaticCast)             \
  V(ReinterpretCast)        \
  V(ConstCast)              \
  V(Call)                   \
  V(Member)                 \
  V(Subscript)              \
  V(Unop)                   \
  V(Binop)

namespace cxx {

namespace IR {

struct Module;
struct Function;
struct BasicBlock;
struct Terminator;
struct Stmt;
struct Expr;

#define VISIT_IR_STMT(x) struct x;
FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT

#define VISIT_IR_EXPR(x) struct x;
FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR

enum struct StmtKind {
#define VISIT_IR_STMT(x) k##x,
  FOR_EACH_IR_STMT(VISIT_IR_STMT)
#undef VISIT_IR_STMT
};

enum struct ExprKind {
#define VISIT_IR_EXPR(x) k##x,
  FOR_EACH_IR_EXPR(VISIT_IR_EXPR)
#undef VISIT_IR_EXPR
};

}  // end of namespace IR

}  // namespace cxx
