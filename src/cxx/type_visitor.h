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

#include <cxx/types_fwd.h>

namespace cxx {

class TypeVisitor {
 public:
  virtual ~TypeVisitor();

  virtual void visit(const UndefinedType*) = 0;
  virtual void visit(const ErrorType*) = 0;
  virtual void visit(const UnresolvedType*) = 0;
  virtual void visit(const VoidType*) = 0;
  virtual void visit(const NullptrType*) = 0;
  virtual void visit(const BooleanType*) = 0;
  virtual void visit(const CharacterType*) = 0;
  virtual void visit(const IntegerType*) = 0;
  virtual void visit(const FloatingPointType*) = 0;
  virtual void visit(const EnumType*) = 0;
  virtual void visit(const ScopedEnumType*) = 0;
  virtual void visit(const PointerType*) = 0;
  virtual void visit(const PointerToMemberType*) = 0;
  virtual void visit(const ReferenceType*) = 0;
  virtual void visit(const RValueReferenceType*) = 0;
  virtual void visit(const ArrayType*) = 0;
  virtual void visit(const UnboundArrayType*) = 0;
  virtual void visit(const FunctionType*) = 0;
  virtual void visit(const MemberFunctionType*) = 0;
  virtual void visit(const NamespaceType*) = 0;
  virtual void visit(const ClassType*) = 0;
  virtual void visit(const TemplateType*) = 0;
  virtual void visit(const TemplateArgumentType*) = 0;
  virtual void visit(const ConceptType*) = 0;
};

}  // namespace cxx
