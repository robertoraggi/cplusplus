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

std::optional<std::tuple<std::uint64_t, std::uint64_t>> MemoryLayout::ofType(
    const QualifiedType& type) {
  static MemoryLayout memoryLayout;
  return memoryLayout(type);
}

std::optional<std::tuple<std::uint64_t, std::uint64_t>>
MemoryLayout::operator()(const QualifiedType& type) {
  std::uint64_t size = 0;
  std::uint64_t alignment = 0;
  std::swap(size_, size);
  std::swap(alignment_, alignment);
  type->accept(this);
  std::swap(size_, size);
  std::swap(alignment_, alignment);
  if (!size) return std::nullopt;
  return std::tuple(size, alignment);
}

void MemoryLayout::visit(const UndefinedType*) {}

void MemoryLayout::visit(const ErrorType*) {}

void MemoryLayout::visit(const AutoType*) {}

void MemoryLayout::visit(const DecltypeAutoType*) {}

void MemoryLayout::visit(const VoidType*) {}

void MemoryLayout::visit(const NullptrType*) {}

void MemoryLayout::visit(const BooleanType*) {}

void MemoryLayout::visit(const CharacterType* ty) {
  switch (ty->kind()) {
    case CharacterKind::kChar8T:
      alignment_ = size_ = 1;
      break;
    case CharacterKind::kChar16T:
      alignment_ = size_ = 2;
      break;
    case CharacterKind::kChar32T:
      alignment_ = size_ = 4;
      break;
    case CharacterKind::kWCharT:
      alignment_ = size_ = 4;
      break;
    default:
      throw std::runtime_error(
          fmt::format("invalid character type: {}", __PRETTY_FUNCTION__));
  }  // switch
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

void MemoryLayout::visit(const FloatingPointType* ty) {
  switch (ty->kind()) {
    case FloatingPointKind::kFloat:
      size_ = alignment_ = 4;
      break;
    case FloatingPointKind::kFloat128:
      size_ = alignment_ = 16;
      break;
    case FloatingPointKind::kDouble:
      size_ = alignment_ = 8;
      break;
    case FloatingPointKind::kLongDouble:
      size_ = alignment_ = 16;
      break;
    default:
      throw std::runtime_error("unrecognized integer kind");
  }
}

void MemoryLayout::visit(const EnumType*) { size_ = alignment_ = 4; }

void MemoryLayout::visit(const ScopedEnumType*) {}

void MemoryLayout::visit(const PointerType*) { size_ = alignment_ = 8; }

void MemoryLayout::MemoryLayout::visit(const PointerToMemberType*) {}

void MemoryLayout::MemoryLayout::visit(const ReferenceType*) {}

void MemoryLayout::visit(const RValueReferenceType*) {}
void MemoryLayout::visit(const ArrayType* type) {
  auto element = operator()(type->elementType());

  if (!element) return;

  auto [arrayElementSize, arrayElementAlignment] = *element;

  size_ = AlignTo(size_, arrayElementAlignment) +
          type->dimension() * arrayElementSize;

  alignment_ = arrayElementAlignment;
}

void MemoryLayout::visit(const UnboundArrayType*) {}

void MemoryLayout::visit(const FunctionType*) {}

void MemoryLayout::visit(const MemberFunctionType*) {}

void MemoryLayout::visit(const NamespaceType*) {}

void MemoryLayout::visit(const ClassType* type) {
  auto symbol = type->symbol();

  std::size_t size = 0;
  std::size_t alignment = 0;

  for (auto member : *symbol->scope()) {
    auto field = dynamic_cast<FieldSymbol*>(member);
    if (!field) continue;

    auto layout = operator()(member->type());

    if (!layout) {
      // cannot compute the type of this class.
      return;
    }

    auto [memberSize, memberAlignment] = *layout;

    size = AlignTo(size, memberAlignment) + memberSize;
    alignment = std::max(alignment, memberAlignment);
  }

  if (size) {
    alignment_ = alignment;
    size_ = AlignTo(size, alignment_);
  } else {
    size_ = alignment_ = 1;
  }
}

void MemoryLayout::visit(const TemplateType*) {}

void MemoryLayout::visit(const TemplateArgumentType*) {}

void MemoryLayout::visit(const ConceptType*) {}

}  // namespace cxx
