// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cstdlib>
#include <optional>

namespace cxx {

namespace {
struct SizeOf {
  const MemoryLayout& memoryLayout;

  auto operator()(const BuiltinVaListType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const BuiltinMetaInfoType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const VoidType* type) const -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const NullptrType* type) const -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const DecltypeAutoType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const AutoType* type) const -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const BoolType* type) const -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const SignedCharType* type) const
      -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const ShortIntType* type) const
      -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const IntType* type) const -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const LongIntType* type) const -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLong();
  }

  auto operator()(const LongLongIntType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLongLong();
  }

  auto operator()(const Int128Type*) const -> std::optional<std::size_t> {
    return 16;
  }

  auto operator()(const UnsignedCharType* type) const
      -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const UnsignedShortIntType* type) const
      -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const UnsignedIntType* type) const
      -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const UnsignedLongIntType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLong();
  }

  auto operator()(const UnsignedLongLongIntType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLongLong();
  }

  auto operator()(const UnsignedInt128Type*) const
      -> std::optional<std::size_t> {
    return 16;
  }

  auto operator()(const CharType* type) const -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const Char8Type* type) const -> std::optional<std::size_t> {
    return 1;
  }

  auto operator()(const Char16Type* type) const -> std::optional<std::size_t> {
    return 2;
  }

  auto operator()(const Char32Type* type) const -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const WideCharType* type) const
      -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const FloatType* type) const -> std::optional<std::size_t> {
    return 4;
  }

  auto operator()(const DoubleType* type) const -> std::optional<std::size_t> {
    return 8;
  }

  auto operator()(const LongDoubleType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfLongDouble();
  }

  auto operator()(const QualType* type) const -> std::optional<std::size_t> {
    return visit(*this, type->elementType());
  }

  auto operator()(const BoundedArrayType* type) const
      -> std::optional<std::size_t> {
    auto elementSize = visit(*this, type->elementType());
    if (elementSize.has_value()) return *elementSize * type->size();
    return std::nullopt;
  }

  auto operator()(const UnboundedArrayType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const PointerType* type) const -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const LvalueReferenceType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const RvalueReferenceType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const FunctionType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const ClassType* type) const -> std::optional<std::size_t> {
    return type->symbol()->sizeInBytes();
  }

  auto operator()(const EnumType* type) const -> std::optional<std::size_t> {
    if (type->underlyingType()) {
      return visit(*this, type->underlyingType());
    }
    return 4;
  }

  auto operator()(const ScopedEnumType* type) const
      -> std::optional<std::size_t> {
    if (type->underlyingType()) {
      return visit(*this, type->underlyingType());
    }
    return 4;
  }

  auto operator()(const MemberObjectPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const MemberFunctionPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const NamespaceType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const TypeParameterType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const TemplateTypeParameterType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedNameType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const UnresolvedUnderlyingType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const OverloadSetType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }
};

struct AlignmentOf {
  const MemoryLayout& memoryLayout;

  auto operator()(const ClassType* type) const -> std::optional<std::size_t> {
    return type->symbol()->alignment();
  }

  auto operator()(const UnboundedArrayType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(auto type) const -> std::optional<std::size_t> {
    // ### TODO
    if (!type) return std::nullopt;
    return memoryLayout.sizeOf(type);
  }
};

}  // namespace

MemoryLayout::MemoryLayout(std::size_t bits) : bits_(bits) {
  sizeOfPointer_ = bits / 8;
  sizeOfLong_ = bits / 8;
  sizeOfLongLong_ = sizeOfLong_;
  sizeOfLongDouble_ = 8;
}

MemoryLayout::~MemoryLayout() = default;

auto MemoryLayout::bits() const -> std::size_t { return bits_; }

auto MemoryLayout::sizeOfSizeType() const -> std::size_t {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOfPointer() const -> std::size_t {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOfLong() const -> std::size_t { return sizeOfLong_; }

auto MemoryLayout::sizeOfLongLong() const -> std::size_t {
  return sizeOfLongLong_;
}

auto MemoryLayout::sizeOfLongDouble() const -> std::size_t {
  return sizeOfLongDouble_;
}

void MemoryLayout::setSizeOfPointer(std::size_t sizeOfPointer) {
  sizeOfPointer_ = sizeOfPointer;
}

void MemoryLayout::setSizeOfLong(std::size_t sizeOfLong) {
  sizeOfLong_ = sizeOfLong;
}

void MemoryLayout::setSizeOfLongLong(std::size_t sizeOfLongLong) {
  sizeOfLongLong_ = sizeOfLongLong;
}

void MemoryLayout::setSizeOfLongDouble(std::size_t sizeOfLongDouble) {
  sizeOfLongDouble_ = sizeOfLongDouble;
}

auto MemoryLayout::sizeOf(const Type* type) const
    -> std::optional<std::size_t> {
  if (!type) return std::nullopt;
  return visit(SizeOf{*this}, type);
}

auto MemoryLayout::alignmentOf(const Type* type) const
    -> std::optional<std::size_t> {
  if (!type) return std::nullopt;
  return visit(AlignmentOf{*this}, type);
}

auto MemoryLayout::triple() const -> const std::string& { return triple_; }

void MemoryLayout::setTriple(std::string triple) {
  triple_ = std::move(triple);
}

}  // namespace cxx