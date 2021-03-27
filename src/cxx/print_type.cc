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

#include <cxx/names.h>
#include <cxx/print_type.h>
#include <cxx/qualified_type.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace cxx {

void PrintType::operator()(const QualifiedType& type, std::ostream& out) {
  if (!type) return;
  auto o = &out;
  std::swap(out_, o);
  accept(type);
  std::swap(out_, o);
}

void PrintType::accept(const QualifiedType& type) {
  if (!type) return;
  type->accept(this);
}

void PrintType::visit(const UndefinedType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const ErrorType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const UnresolvedType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const VoidType*) { fmt::print(*out_, "void"); }

void PrintType::visit(const NullptrType*) { fmt::print(*out_, "nullptr_t"); }

void PrintType::visit(const BooleanType*) { fmt::print(*out_, "bool"); }

void PrintType::visit(const CharacterType* type) {
  switch (type->kind()) {
    case CharacterKind::kChar8T:
      fmt::print(*out_, "char8_t");
      break;
    case CharacterKind::kChar16T:
      fmt::print(*out_, "char16_t");
      break;
    case CharacterKind::kChar32T:
      fmt::print(*out_, "char32_t");
      break;
    case CharacterKind::kWCharT:
      fmt::print(*out_, "wchar_t");
      break;
  }  // switch
}

void PrintType::visit(const IntegerType* type) {
  if (type->isUnsigned()) fmt::print(*out_, "unsigned ");
  switch (type->kind()) {
    case IntegerKind::kChar:
      fmt::print(*out_, "char");
      break;
    case IntegerKind::kShort:
      fmt::print(*out_, "short");
      break;
    case IntegerKind::kInt:
      fmt::print(*out_, "int");
      break;
    case IntegerKind::kInt64:
      fmt::print(*out_, "__int64");
      break;
    case IntegerKind::kInt128:
      fmt::print(*out_, "__int128");
      break;
    case IntegerKind::kLong:
      fmt::print(*out_, "long");
      break;
    case IntegerKind::kLongLong:
      fmt::print(*out_, "long long");
      break;
  }  // switch
}

void PrintType::visit(const FloatingPointType* type) {
  switch (type->kind()) {
    case FloatingPointKind::kFloat:
      fmt::print(*out_, "float");
      break;
    case FloatingPointKind::kDouble:
      fmt::print(*out_, "double");
      break;
    case FloatingPointKind::kLongDouble:
      fmt::print(*out_, "long double");
      break;
    case FloatingPointKind::kFloat128:
      fmt::print(*out_, "__float128");
      break;
  }  // switch
}

void PrintType::visit(const EnumType* type) {
  if (auto name = type->symbol()->name())
    fmt::print(*out_, "enum({})", *name);
  else
    fmt::print(*out_, "enum()");
}

void PrintType::visit(const ScopedEnumType* type) {
  auto underlyingType = type->symbol()->underlyingType();
  if (auto name = type->symbol()->name())
    fmt::print(*out_, "enum({}, {})", *name, underlyingType);
  else
    fmt::print(*out_, "enum({})", underlyingType);
}

void PrintType::visit(const PointerType* type) {
  fmt::print(*out_, "ptr(");
  accept(type->elementType());
  fmt::print(*out_, ")");
}

void PrintType::visit(const PointerToMemberType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const ReferenceType* type) {
  fmt::print(*out_, "ref(");
  accept(type->elementType());
  fmt::print(*out_, ")");
}

void PrintType::visit(const RValueReferenceType* type) {
  fmt::print(*out_, "rref(");
  accept(type->elementType());
  fmt::print(*out_, ")");
}

void PrintType::visit(const ArrayType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const UnboundArrayType* type) {
  fmt::print(*out_, "vla(");
  accept(type->elementType());
  fmt::print(*out_, ")");
}

void PrintType::visit(const FunctionType* type) {
  fmt::print(*out_, "fun(");
  const auto& args = type->argumentTypes();
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) fmt::print(*out_, ", ");
    accept(args[i]);
  }
  if (type->isVariadic()) fmt::print(*out_, "...");
  fmt::print(*out_, ") -> ");
  accept(type->returnType());
}

void PrintType::visit(const MemberFunctionType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const NamespaceType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const ClassType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const TemplateType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const TemplateArgumentType* type) {
  throw std::runtime_error("todo");
}

void PrintType::visit(const ConceptType* type) {
  throw std::runtime_error("todo");
}

}  // namespace cxx
