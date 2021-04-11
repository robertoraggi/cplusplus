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
#include <cxx/qualified_type.h>
#include <cxx/symbols.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace cxx {

void TypePrinter::operator()(std::ostream& out, const QualifiedType& type,
                             std::string declarator) {
  out << toString(type, std::move(declarator));
}

std::string TypePrinter::toString(const QualifiedType& type,
                                  std::string declarator) {
  if (!type) return {};
  std::string specifiers;
  std::string ptrOps;
  std::swap(specifiers_, specifiers);
  std::swap(ptrOps_, ptrOps);
  std::swap(declarator_, declarator);
  accept(type);
  std::swap(declarator_, declarator);
  std::swap(ptrOps_, ptrOps);
  std::swap(specifiers_, specifiers);
  if (ptrOps.empty() && declarator.empty()) return specifiers;
  return fmt::format("{} {}{}", specifiers, ptrOps, declarator);
}

void TypePrinter::accept(const QualifiedType& type) {
  if (!type) return;
  addQualifiers(specifiers_, type);
  type->accept(this);
}

void TypePrinter::addQualifiers(std::string& out, const QualifiedType& type) {
  if (type.isConst()) out += "const ";
  if (type.isVolatile()) out += "volatile ";
  if (type.isRestrict()) out += "restrict ";
}

void TypePrinter::visit(const UndefinedType* type) {
  specifiers_ += "__undefined_type__";
}

void TypePrinter::visit(const ErrorType* type) {
  specifiers_ += "__error_type__";
}

void TypePrinter::visit(const UnresolvedType* type) {
  specifiers_ += "__unresolved_type__";
}

void TypePrinter::visit(const VoidType*) { specifiers_ += "void"; }

void TypePrinter::visit(const NullptrType*) { specifiers_ += "nullptr_t"; }

void TypePrinter::visit(const BooleanType*) { specifiers_ += "bool"; }

void TypePrinter::visit(const CharacterType* type) {
  switch (type->kind()) {
    case CharacterKind::kChar8T:
      specifiers_ += "char8_t";
      break;
    case CharacterKind::kChar16T:
      specifiers_ += "char16_t";
      break;
    case CharacterKind::kChar32T:
      specifiers_ += "char32_t";
      break;
    case CharacterKind::kWCharT:
      specifiers_ += "wchar_t";
      break;
  }  // switch
}

void TypePrinter::visit(const IntegerType* type) {
  if (type->isUnsigned()) specifiers_ += "unsigned ";
  switch (type->kind()) {
    case IntegerKind::kChar:
      specifiers_ += "char";
      break;
    case IntegerKind::kShort:
      specifiers_ += "short";
      break;
    case IntegerKind::kInt:
      specifiers_ += "int";
      break;
    case IntegerKind::kInt64:
      specifiers_ += "__int64";
      break;
    case IntegerKind::kInt128:
      specifiers_ += "__int128";
      break;
    case IntegerKind::kLong:
      specifiers_ += "long";
      break;
    case IntegerKind::kLongLong:
      specifiers_ += "long long";
      break;
  }  // switch
}

void TypePrinter::visit(const FloatingPointType* type) {
  switch (type->kind()) {
    case FloatingPointKind::kFloat:
      specifiers_ += "float";
      break;
    case FloatingPointKind::kDouble:
      specifiers_ += "double";
      break;
    case FloatingPointKind::kLongDouble:
      specifiers_ += "long double";
      break;
    case FloatingPointKind::kFloat128:
      specifiers_ += "__float128";
      break;
  }  // switch
}

void TypePrinter::visit(const EnumType* type) {
  if (auto name = type->symbol()->name())
    specifiers_ += fmt::format("enum {}", *name);
  else
    specifiers_ += "enum __anon__";
}

void TypePrinter::visit(const ScopedEnumType* type) {
  if (auto name = type->symbol()->name())
    specifiers_ += fmt::format("enum class {}", *name);
  else
    specifiers_ += "enum class __anon__";
}

void TypePrinter::visit(const PointerType* type) {
  ptrOps_ += "*";
  addQualifiers(ptrOps_, type->elementType());
  accept(type->elementType());
}

void TypePrinter::visit(const PointerToMemberType* type) {
  throw std::runtime_error("todo");
}

void TypePrinter::visit(const ReferenceType* type) {
  ptrOps_ += "&";
  accept(type->elementType());
}

void TypePrinter::visit(const RValueReferenceType* type) {
  ptrOps_ += "&&";
  accept(type->elementType());
}

void TypePrinter::visit(const ArrayType* type) {
  if (!ptrOps_.empty()) {
    declarator_ =
        fmt::format("({}{}[{}])", ptrOps_, declarator_, type->dimension());
    ptrOps_.clear();
  } else {
    declarator_ += fmt::format("[{}]", type->dimension());
  }
  accept(type->elementType());
}

void TypePrinter::visit(const UnboundArrayType* type) {
  if (!ptrOps_.empty()) {
    declarator_ = fmt::format("({}{}[])", ptrOps_, declarator_);
    ptrOps_.clear();
  } else {
    declarator_ += fmt::format("[]");
  }
  accept(type->elementType());
}

void TypePrinter::visit(const FunctionType* type) {
  const auto& args = type->argumentTypes();
  std::string params = "(";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) params += ", ";
    params += toString(args[i]);
  }
  if (type->isVariadic()) params += "...";
  params += ")";
  if (!ptrOps_.empty()) {
    declarator_ = fmt::format("({}{}){}", ptrOps_, declarator_, params);
    ptrOps_.clear();
  } else {
    declarator_ += params;
  }
  accept(type->returnType());
}

void TypePrinter::visit(const MemberFunctionType* type) {
  throw std::runtime_error("todo");
}

void TypePrinter::visit(const NamespaceType* type) {
  throw std::runtime_error("todo");
}

void TypePrinter::visit(const ClassType* type) {
  if (auto name = type->symbol()->name())
    specifiers_ += fmt::format("class {}", *name);
  else
    specifiers_ += "class __anon__";
}

void TypePrinter::visit(const TemplateType* type) {
  throw std::runtime_error("todo");
}

void TypePrinter::visit(const TemplateArgumentType* type) {
  throw std::runtime_error("todo");
}

void TypePrinter::visit(const ConceptType* type) {
  throw std::runtime_error("todo");
}

}  // namespace cxx
