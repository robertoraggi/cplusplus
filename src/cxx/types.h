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

#include <cxx/type.h>

namespace cxx {

class UnresolvedType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class VoidType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class NullptrType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class BooleanType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class CharacterType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class IntegerType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class FloatingPointType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class EnumType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class EnumClassType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class PointerType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class PointerToMemberType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class ReferenceType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class RValueReferenceType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class ArrayType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class UnboundArrayType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class FunctionType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class MemberFunctionType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class NamespaceType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class ClassType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class TemplateType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class TemplateArgumentType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

}  // namespace cxx