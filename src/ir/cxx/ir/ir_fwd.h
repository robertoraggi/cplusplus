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

#include <cstdint>
#include <variant>

namespace cxx::ir {

class Module;
class Global;
class Local;
class Function;
class Block;
class Stmt;
class Expr;

class IRVisitor;
class IRFactory;
class IRBuilder;

// statements
class Jump;
class CondJump;
class Switch;
class Ret;
class RetVoid;
class Move;

// expressions
class This;
class BoolLiteral;
class CharLiteral;
class IntegerLiteral;
class FloatLiteral;
class NullptrLiteral;
class StringLiteral;
class UserDefinedStringLiteral;
class Temp;
class Id;
class ExternalId;
class Typeid;
class Unary;
class Binary;
class Call;
class Subscript;
class Access;
class Cast;
class StaticCast;
class DynamicCast;
class ReinterpretCast;
class New;
class NewArray;
class Delete;
class DeleteArray;
class Throw;

using IntegerValue = std::variant<std::uint64_t, std::int64_t>;

using FloatValue = std::variant<float, double, long double>;

enum class UnaryOp {
  kStar,
  kAmp,
  kPlus,
  kMinus,
  kExclaim,
  kTilde,
  kPlusPlus,
  kMinusMinus
};

enum class BinaryOp {
  kStar,
  kSlash,
  kPercent,
  kPlus,
  kMinus,
  kGreaterGreater,
  kLessLess,
  kGreater,
  kLess,
  kGreaterEqual,
  kLessEqual,
  kEqualEqual,
  kExclaimEqual,
  kAmp,
  kCaret,
  kBar
};

}  // namespace cxx::ir
