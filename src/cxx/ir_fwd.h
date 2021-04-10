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

#include <cstdint>
#include <variant>

namespace cxx::ir {

class Module;
class Global;
class Function;
class Block;
class Stmt;
class Expr;

class IRVisitor;

// statements
class Jump;
class CondJump;
class Ret;
class RetVoid;

// expressions
class This;
class BoolLiteral;
class IntegerLiteral;
class FloatLiteral;
class NullptrLiteral;
class StringLiteral;
class UserDefinedStringLiteral;
class Id;
class ExternalId;
class Sizeof;
class Typeid;
class Alignof;
class Unary;
class Binary;
class Assignment;
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

using IntegerValue =
    std::variant<std::int8_t, std::int16_t, std::int32_t, std::int64_t,
                 std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;

using FloatValue = std::variant<float, double, long double>;

enum class UnaryOp {
  kStar,
  kAmp,
  kPlus,
  kMinus,
  kExclaim,
  kTilde,
  kPlusPlus,
  kMinusMinus,
  kPostPlusPlus,
  kPostMinusMinus,
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

enum class AssignmentOp {
  kEqual,
  kStarEqual,
  kSlashEqual,
  kPercentEqual,
  kPlusEqual,
  kMinusEqual,
  kGreaterGreaterEqual,
  kLessLessEqual,
  kAmpEqual,
  kCaretEqual,
  kBarEqual,
};

}  // namespace cxx::ir
