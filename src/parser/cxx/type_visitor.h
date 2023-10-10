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

#include <cxx/types_fwd.h>

namespace cxx {

class TypeVisitor {
 public:
  virtual ~TypeVisitor();

  virtual void visit(const InvalidType* type) = 0;
  virtual void visit(const NullptrType* type) = 0;
  virtual void visit(const DependentType* type) = 0;
  virtual void visit(const DecltypeAutoType* type) = 0;
  virtual void visit(const AutoType* type) = 0;
  virtual void visit(const VoidType* type) = 0;
  virtual void visit(const BoolType* type) = 0;
  virtual void visit(const CharType* type) = 0;
  virtual void visit(const SignedCharType* type) = 0;
  virtual void visit(const UnsignedCharType* type) = 0;
  virtual void visit(const Char8Type* type) = 0;
  virtual void visit(const Char16Type* type) = 0;
  virtual void visit(const Char32Type* type) = 0;
  virtual void visit(const WideCharType* type) = 0;
  virtual void visit(const ShortType* type) = 0;
  virtual void visit(const UnsignedShortType* type) = 0;
  virtual void visit(const IntType* type) = 0;
  virtual void visit(const UnsignedIntType* type) = 0;
  virtual void visit(const LongType* type) = 0;
  virtual void visit(const UnsignedLongType* type) = 0;
  virtual void visit(const FloatType* type) = 0;
  virtual void visit(const DoubleType* type) = 0;
  virtual void visit(const QualType* type) = 0;
  virtual void visit(const PointerType* type) = 0;
  virtual void visit(const LValueReferenceType* type) = 0;
  virtual void visit(const RValueReferenceType* type) = 0;
  virtual void visit(const ArrayType* type) = 0;
  virtual void visit(const FunctionType* type) = 0;
  virtual void visit(const ClassType* type) = 0;
  virtual void visit(const NamespaceType* type) = 0;
  virtual void visit(const MemberPointerType* type) = 0;
  virtual void visit(const ConceptType* type) = 0;
  virtual void visit(const EnumType* type) = 0;
  virtual void visit(const GenericType* type) = 0;
  virtual void visit(const PackType* type) = 0;
  virtual void visit(const ScopedEnumType* type) = 0;
};

}  // namespace cxx
