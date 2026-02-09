// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/cxx_fwd.h>
#include <cxx/types_fwd.h>

#include <functional>
#include <variant>

namespace cxx {

#define CXX_FOR_EACH_NAME(V) \
  V(Identifier)              \
  V(OperatorId)              \
  V(DestructorId)            \
  V(LiteralOperatorId)       \
  V(ConversionFunctionId)    \
  V(TemplateId)

#define PROCESS_NAME(N) k##N,
enum class NameKind { CXX_FOR_EACH_NAME(PROCESS_NAME) };
#undef PROCESS_NAME

class Name;

#define PROCESS_NAME(N) class N;
CXX_FOR_EACH_NAME(PROCESS_NAME)
#undef PROCESS_NAME

class Symbol;

using TemplateArgument =
    std::variant<const Type*, Symbol*, ConstValue, ExpressionAST*>;

enum class IdentifierInfoKind {
  kTypeTrait,
  kUnaryBuiltinType,
  kBuiltinFunction,
};

class IdentifierInfo;
class TypeTraitIdentifierInfo;
class BuiltinFunctionIdentifierInfo;

auto to_string(const Name* name) -> std::string;

}  // namespace cxx
