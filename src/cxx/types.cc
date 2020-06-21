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

#include "types.h"

#include <cassert>
#include <sstream>
#include <string>

#include "ir.h"
#include "names.h"
#include "symbols.h"
#include "token.h"

namespace cxx {

FunctionSymbol* OverloadSetType::firstCandidate() const {
  auto n = name();
  for (auto sym = scope()->findSymbol(n); sym; sym = sym->next()) {
    if (sym->unqualifiedName() != n) continue;
    if (auto x = sym->asFunctionSymbol()) return x;
  }
  return nullptr;
}

void TypeToString::accept(QualType type) {
  switch (type->kind()) {
#define VISIT_TYPE(T)           \
  case TypeKind::k##T:          \
    visit(type->as##T##Type()); \
    break;
    FOR_EACH_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
    default:
      assert(!"unreachable");
  }  // switch
  if (type.isUnsigned()) text = "unsigned " + text;
  if (type.isConst()) text = "const " + text;
  if (type.isVolatile()) text = "volatile " + text;
}

std::string TypeToString::print(QualType type, std::string&& decl) {
  std::string text;
  std::swap(this->text, text);
  std::swap(this->decl, decl);
  accept(type);
  std::swap(this->decl, decl);
  std::swap(this->text, text);
  return text;
}

std::string TypeToString::print(QualType type, const Name* name) {
  return print(type, name ? name->toString() : "");
}

std::string TypeToString::operator()(QualType type, const Name* name) {
  return print(type, name);
}

void TypeToString::visit(const IntegerType* type) {
  switch (type->integerKind()) {
#define VISIT_TYPE(t, n)  \
  case IntegerKind::k##t: \
    text = n;             \
    break;
    FOR_EACH_INTEGER_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
    default:
      assert(!"unreachable");
  }  // switch
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const FloatType* type) {
  switch (type->floatKind()) {
#define VISIT_TYPE(t, n) \
  case FloatKind::k##t:  \
    text = n;            \
    break;
    FOR_EACH_FLOAT_TYPE(VISIT_TYPE)
#undef VISIT_TYPE
    default:
      assert(!"unreachable");
  }  // switch
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const UndefinedType* type) {
  //  text = "/*undefined*/";
  //  if (! decl.empty())
  //    text += ' ';
  text += decl;
}

void TypeToString::visit(const AutoType*) {
  text = "auto";
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const VoidType*) {
  text = "void";
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const NullptrType* type) {
  text = "decltype(nullptr)";
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const PointerType* type) {
  std::string prefix, suffix;
  auto elementType = type->elementType();
  if (elementType->asArrayType() || elementType->asFunctionType()) {
    prefix = "(*";
    suffix = ")";
  } else {
    prefix = "*";
  }
  text = print(elementType, prefix + decl + suffix);
}

void TypeToString::visit(const LValueReferenceType* type) {
  std::string prefix, suffix;
  auto elementType = type->elementType();
  if (elementType->asArrayType() || elementType->asFunctionType()) {
    prefix = "(&";
    suffix = ")";
  } else {
    prefix = "&";
  }
  text = print(elementType, prefix + decl + suffix);
}

void TypeToString::visit(const RValueReferenceType* type) {
  std::string prefix, suffix;
  auto elementType = type->elementType();
  if (elementType->asArrayType() || elementType->asFunctionType()) {
    prefix = "(&&";
    suffix = ")";
  } else {
    prefix = "&&";
  }
  text = print(elementType, prefix + decl + suffix);
}

void TypeToString::visit(const ArrayType* type) {
  std::string subscript;
  subscript += '[';
  if (auto size = type->size()) {
    std::ostringstream s;  // ### this is slow!
    size->dump(s);
    subscript += s.str();
  }
  subscript += ']';
  text = print(type->elementType(), decl + subscript);
}

std::string TypeToString::prototype(const FunctionType* type,
                                    const std::vector<const Name*>& actuals) {
  std::string proto;
  proto += '(';
  auto&& argTypes = type->argumentTypes();
  for (size_t i = 0; i < argTypes.size(); ++i) {
    auto&& argTy = argTypes[i];
    if (i != 0) proto += ", ";
    const Name* argName = i < actuals.size() ? actuals[i] : nullptr;
    proto += print(argTy, argName);
  }
  if (type->isVariadic()) proto += "...";
  proto += ')';
  if (type->isConst()) proto += " const";
  return proto;
}

void TypeToString::visit(const FunctionType* type) {
  auto&& proto = prototype(type, std::vector<const Name*>{});
  text = print(type->returnType(), decl + proto);
}

void TypeToString::visit(const OverloadSetType* type) {
  text = "@overload-set-type";
}

void TypeToString::visit(const ClassType* type) {
  text = token_spell[type->symbol()->classKey()];
  if (auto name = type->symbol()->name()) {
    text += ' ';
    text += name->toString();
  }
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const EnumType* type) {
  text = "enum";
  if (auto name = type->symbol()->name()) {
    text += ' ';
    text += name->toString();
  }
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const NamedType* type) {
  text += type->name()->toString();
  if (!decl.empty()) text += ' ';
  text += decl;
}

void TypeToString::visit(const ElaboratedType* type) {
  text += token_spell[type->classKey()];
  text += ' ';
  text += type->name()->toString();
  if (!decl.empty()) text += ' ';
  text += decl;
}

}  // namespace cxx
