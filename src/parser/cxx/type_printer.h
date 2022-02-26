// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/type_visitor.h>

#include <iosfwd>
#include <string>
#include <tuple>

namespace cxx {

class TypePrinter final : TypeVisitor {
 public:
  void operator()(std::ostream& out, const QualifiedType& type,
                  std::string id = std::string());

  std::string toString(const QualifiedType& type,
                       std::string id = std::string());

  std::string toString(const QualifiedType& type, std::string id,
                       bool addFormals);

 private:
  void addQualifiers(std::string& out, Qualifiers qualifiers);

  void accept(const QualifiedType& type);

  void visit(const UndefinedType* type) override;
  void visit(const ErrorType* type) override;
  void visit(const AutoType* type) override;
  void visit(const DecltypeAutoType* type) override;
  void visit(const VoidType* type) override;
  void visit(const NullptrType* type) override;
  void visit(const BooleanType* type) override;
  void visit(const CharacterType* type) override;
  void visit(const IntegerType* type) override;
  void visit(const FloatingPointType* type) override;
  void visit(const EnumType* type) override;
  void visit(const ScopedEnumType* type) override;
  void visit(const PointerType* type) override;
  void visit(const PointerToMemberType* type) override;
  void visit(const ReferenceType* type) override;
  void visit(const RValueReferenceType* type) override;
  void visit(const ArrayType* type) override;
  void visit(const UnboundArrayType* type) override;
  void visit(const FunctionType* type) override;
  void visit(const MemberFunctionType* type) override;
  void visit(const NamespaceType* type) override;
  void visit(const ClassType* type) override;
  void visit(const TemplateType* type) override;
  void visit(const TemplateArgumentType* type) override;
  void visit(const ConceptType* type) override;

 private:
  std::string specifiers_;
  std::string ptrOps_;
  std::string declarator_;
  bool addFormals_ = false;
};

}  // namespace cxx
