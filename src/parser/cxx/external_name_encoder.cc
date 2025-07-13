// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/external_name_encoder.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct ExternalNameEncoder::NameVisitor {
  ExternalNameEncoder& encoder;

  void operator()(const Identifier* name) {}

  void operator()(const OperatorId* name) {}

  void operator()(const DestructorId* name) {}

  void operator()(const LiteralOperatorId* name) {}

  void operator()(const ConversionFunctionId* name) {}

  void operator()(const TemplateId* name) {}
};

struct ExternalNameEncoder::TypeVisitor {
  ExternalNameEncoder& encoder;

  void operator()(const VoidType* type) { encoder.out("v"); }

  void operator()(const NullptrType* type) { encoder.out("Dn"); }

  void operator()(const DecltypeAutoType* type) { encoder.out("Dc"); }

  void operator()(const AutoType* type) { encoder.out("Da"); }

  void operator()(const BoolType* type) { encoder.out("b"); }

  void operator()(const SignedCharType* type) { encoder.out("a"); }

  void operator()(const ShortIntType* type) { encoder.out("s"); }

  void operator()(const IntType* type) { encoder.out("i"); }

  void operator()(const LongIntType* type) { encoder.out("l"); }

  void operator()(const LongLongIntType* type) { encoder.out("x"); }

  void operator()(const Int128Type* type) { encoder.out("n"); }

  void operator()(const UnsignedCharType* type) { encoder.out("h"); }

  void operator()(const UnsignedShortIntType* type) { encoder.out("t"); }

  void operator()(const UnsignedIntType* type) { encoder.out("j"); }

  void operator()(const UnsignedLongIntType* type) { encoder.out("m"); }

  void operator()(const UnsignedLongLongIntType* type) { encoder.out("y"); }

  void operator()(const UnsignedInt128Type* type) { encoder.out("o"); }

  void operator()(const CharType* type) { encoder.out("c"); }

  void operator()(const Char8Type* type) { encoder.out("Du"); }

  void operator()(const Char16Type* type) { encoder.out("Ds"); }

  void operator()(const Char32Type* type) { encoder.out("Di"); }

  void operator()(const WideCharType* type) { encoder.out("w"); }

  void operator()(const FloatType* type) { encoder.out("f"); }

  void operator()(const DoubleType* type) { encoder.out("d"); }

  void operator()(const LongDoubleType* type) { encoder.out("e"); }

  void operator()(const QualType* type) {}

  void operator()(const BoundedArrayType* type) {}

  void operator()(const UnboundedArrayType* type) {}

  void operator()(const PointerType* type) {}

  void operator()(const LvalueReferenceType* type) {}

  void operator()(const RvalueReferenceType* type) {}

  void operator()(const FunctionType* type) {}

  void operator()(const ClassType* type) {}

  void operator()(const EnumType* type) {}

  void operator()(const ScopedEnumType* type) {}

  void operator()(const MemberObjectPointerType* type) {}

  void operator()(const MemberFunctionPointerType* type) {}

  void operator()(const NamespaceType* type) {}

  void operator()(const TypeParameterType* type) {}

  void operator()(const TemplateTypeParameterType* type) {}

  void operator()(const UnresolvedNameType* type) {}

  void operator()(const UnresolvedBoundedArrayType* type) {}

  void operator()(const UnresolvedUnderlyingType* type) {}

  void operator()(const OverloadSetType* type) {}

  void operator()(const BuiltinVaListType* type) {}
};

struct ExternalNameEncoder::SymbolVisitor {
  ExternalNameEncoder& encoder;

  void operator()(NamespaceSymbol* symbol) {}

  void operator()(ConceptSymbol* symbol) {}

  void operator()(ClassSymbol* symbol) {}

  void operator()(EnumSymbol* symbol) {}

  void operator()(ScopedEnumSymbol* symbol) {}

  void operator()(FunctionSymbol* symbol) {}

  void operator()(TypeAliasSymbol* symbol) {}

  void operator()(VariableSymbol* symbol) {}

  void operator()(FieldSymbol* symbol) {}

  void operator()(ParameterSymbol* symbol) {}

  void operator()(ParameterPackSymbol* symbol) {}

  void operator()(EnumeratorSymbol* symbol) {}

  void operator()(FunctionParametersSymbol* symbol) {}

  void operator()(TemplateParametersSymbol* symbol) {}

  void operator()(BlockSymbol* symbol) {}

  void operator()(LambdaSymbol* symbol) {}

  void operator()(TypeParameterSymbol* symbol) {}

  void operator()(NonTypeParameterSymbol* symbol) {}

  void operator()(TemplateTypeParameterSymbol* symbol) {}

  void operator()(ConstraintTypeParameterSymbol* symbol) {}

  void operator()(OverloadSetSymbol* symbol) {}

  void operator()(BaseClassSymbol* symbol) {}

  void operator()(UsingDeclarationSymbol* symbol) {}
};

ExternalNameEncoder::ExternalNameEncoder() {}

void ExternalNameEncoder::out(std::string_view s) { externalName_.append(s); }

auto ExternalNameEncoder::encode(Symbol* symbol) -> std::string {
  std::string externalName;
  if (symbol) {
    std::swap(externalName, externalName_);
    visit(SymbolVisitor{*this}, symbol);
    std::swap(externalName, externalName_);
  }
  return externalName;
}

auto ExternalNameEncoder::encode(const Type* type) -> std::string {
  std::string externalName;
  if (type) {
    std::swap(externalName, externalName_);
    visit(TypeVisitor{*this}, type);
    std::swap(externalName, externalName_);
  }
  return externalName;
}

}  // namespace cxx