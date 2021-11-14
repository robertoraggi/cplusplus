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

#include <cxx/memory_layout.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <fmt/format.h>

#include <stdexcept>

#if defined(_MSVC_LANG) && !defined(__PRETTY_FUNCTION__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace cxx {

std::tuple<std::uint64_t, std::uint64_t> MemoryLayout::ofType(
    const QualifiedType& type) {
  static MemoryLayout memoryLayout;
  return memoryLayout(type);
}

std::tuple<std::uint64_t, std::uint64_t> MemoryLayout::operator()(
    const QualifiedType& type) {
  std::uint64_t size = 0;
  std::uint64_t alignment = 0;
  std::swap(size_, size);
  std::swap(alignment_, alignment);
  type->accept(this);
  std::swap(size_, size);
  std::swap(alignment_, alignment);
  return std::tuple(size, alignment);
}

void MemoryLayout::visit(const UndefinedType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const ErrorType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const AutoType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const DecltypeAutoType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const VoidType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const NullptrType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const BooleanType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const CharacterType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const IntegerType* type) {
  switch (type->kind()) {
    case IntegerKind::kChar:
      size_ = alignment_ = 1;
      break;
    case IntegerKind::kShort:
      size_ = alignment_ = 2;
      break;
    case IntegerKind::kInt:
      size_ = alignment_ = 4;
      break;
    case IntegerKind::kLong:
      size_ = alignment_ = 8;
      break;
    case IntegerKind::kLongLong:
      size_ = alignment_ = 8;
      break;
    default:
      throw std::runtime_error("unrecognized integer kind");
  }  // switch
}

void MemoryLayout::visit(const FloatingPointType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const EnumType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const ScopedEnumType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const PointerType*) { size_ = alignment_ = 8; }

void MemoryLayout::MemoryLayout::visit(const PointerToMemberType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::MemoryLayout::visit(const ReferenceType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const RValueReferenceType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const ArrayType* type) {
  auto [arrayElementSize, arrayElementAlignment] = operator()(
      type->elementType());
  size_ = AlignTo(size_, arrayElementAlignment) +
          type->dimension() * arrayElementSize;
  alignment_ = arrayElementAlignment;
}

void MemoryLayout::visit(const UnboundArrayType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const FunctionType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const MemberFunctionType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const NamespaceType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const ClassType* type) {
  auto symbol = type->symbol();
  for (auto member : *symbol->scope()) {
    auto field = dynamic_cast<FieldSymbol*>(member);
    if (!field) continue;
    auto [memberSize, memberAlignment] = operator()(member->type());
    size_ = AlignTo(size_, memberAlignment) + memberSize;
    alignment_ = std::max(alignment_, memberAlignment);
  }
  if (size_)
    size_ = AlignTo(size_, alignment_);
  else
    size_ = alignment_ = 1;
}

void MemoryLayout::visit(const TemplateType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const TemplateArgumentType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

void MemoryLayout::visit(const ConceptType*) {
  throw std::runtime_error(fmt::format("TODO: {}", __PRETTY_FUNCTION__));
}

}  // namespace cxx
