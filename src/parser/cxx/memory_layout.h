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

#include <cxx/type_visitor.h>
#include <cxx/types.h>

#include <optional>
#include <stdexcept>

namespace cxx {

constexpr auto AlignTo(std::uint64_t n, std::uint64_t align) -> std::uint64_t {
  return (n + align - 1) / align * align;
}

class MemoryLayout final : TypeVisitor {
 public:
  static auto ofType(const QualifiedType& type)
      -> std::optional<std::tuple<std::uint64_t, std::uint64_t>>;

  auto operator()(const QualifiedType& type)
      -> std::optional<std::tuple<std::uint64_t, std::uint64_t>>;

 private:
  void visit(const UndefinedType*) override;
  void visit(const ErrorType*) override;
  void visit(const AutoType*) override;
  void visit(const DecltypeAutoType*) override;
  void visit(const VoidType*) override;
  void visit(const NullptrType*) override;
  void visit(const BooleanType*) override;
  void visit(const CharacterType*) override;
  void visit(const IntegerType*) override;
  void visit(const FloatingPointType*) override;
  void visit(const EnumType*) override;
  void visit(const ScopedEnumType*) override;
  void visit(const PointerType*) override;
  void visit(const PointerToMemberType*) override;
  void visit(const ReferenceType*) override;
  void visit(const RValueReferenceType*) override;
  void visit(const ArrayType*) override;
  void visit(const UnboundArrayType*) override;
  void visit(const FunctionType*) override;
  void visit(const MemberFunctionType*) override;
  void visit(const NamespaceType*) override;
  void visit(const ClassType*) override;
  void visit(const TemplateType*) override;
  void visit(const TemplateArgumentType*) override;
  void visit(const ConceptType*) override;

 private:
  std::uint64_t size_ = 0;
  std::uint64_t alignment_ = 0;
};

}  // namespace cxx
