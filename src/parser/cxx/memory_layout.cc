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

#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cassert>
#include <cstdlib>

namespace cxx {

auto MemoryLayout::make(int bits) -> MemoryLayout* {
  auto* memoryLayout = new MemoryLayout();
  memoryLayout->bits_ = bits;
  memoryLayout->sizeOfPointer_ = bits / 8;
  memoryLayout->sizeOfLong_ = bits / 8;

  return memoryLayout;
}

auto MemoryLayout::bits() const -> int { return bits_; }

auto MemoryLayout::sizeOfPointer() const -> int { return sizeOfPointer_; }

auto MemoryLayout::sizeOfLong() const -> int { return sizeOfLong_; }

#define PROCESS_TYPE(kind) \
  case TypeKind::k##kind:  \
    return sizeOf((const kind##Type*)type);

auto MemoryLayout::sizeOf(const Type* type) const -> int {
  switch (type->kind()) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
    default:
      assert(!"unreachable");
      return 0;
  }  // switch
}

#undef PROCESS_TYPE

#define PROCESS_TYPE(kind) \
  case TypeKind::k##kind:  \
    return alignmentOf((const kind##Type*)type);

auto MemoryLayout::alignmentOf(const Type* type) const -> int {
  switch (type->kind()) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
    default:
      assert(!"unreachable");
      return 1;
  }  // switch
}

#undef PROCESS_TYPE

auto MemoryLayout::alignmentOf(const InvalidType* type) const -> int {
  assert(!"alignmentOf(InvalidType)");
  return 0;
}

auto MemoryLayout::alignmentOf(const DependentType* type) const -> int {
  assert(!"alignmentOf(DependentType)");
  return 0;
}

auto MemoryLayout::alignmentOf(const NullptrType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::alignmentOf(const AutoType* type) const -> int {
  assert(!"alignmentOf(AutoType)");
  return 0;
}

auto MemoryLayout::alignmentOf(const VoidType* type) const -> int {
  assert(!"alignmentOf(VoidType)");
  return 0;
}
auto MemoryLayout::alignmentOf(const BoolType* type) const -> int { return 1; }
auto MemoryLayout::alignmentOf(const CharType* type) const -> int { return 1; }
auto MemoryLayout::alignmentOf(const SignedCharType* type) const -> int {
  return 1;
}
auto MemoryLayout::alignmentOf(const UnsignedCharType* type) const -> int {
  return 1;
}
auto MemoryLayout::alignmentOf(const ShortType* type) const -> int { return 2; }
auto MemoryLayout::alignmentOf(const UnsignedShortType* type) const -> int {
  return 2;
}
auto MemoryLayout::alignmentOf(const IntType* type) const -> int { return 4; }
auto MemoryLayout::alignmentOf(const UnsignedIntType* type) const -> int {
  return 4;
}
auto MemoryLayout::alignmentOf(const LongType* type) const -> int {
  return sizeOfLong_;
}
auto MemoryLayout::alignmentOf(const UnsignedLongType* type) const -> int {
  return sizeOfLong_;
}
auto MemoryLayout::alignmentOf(const FloatType* type) const -> int { return 4; }
auto MemoryLayout::alignmentOf(const DoubleType* type) const -> int {
  return 8;
}
auto MemoryLayout::alignmentOf(const QualType* type) const -> int {
  return alignmentOf(type->elementType);
}
auto MemoryLayout::alignmentOf(const PointerType* type) const -> int {
  return sizeOfPointer_;
}
auto MemoryLayout::alignmentOf(const LValueReferenceType* type) const -> int {
  return sizeOfPointer_;
}
auto MemoryLayout::alignmentOf(const RValueReferenceType* type) const -> int {
  return sizeOfPointer_;
}
auto MemoryLayout::alignmentOf(const ArrayType* type) const -> int {
  return alignmentOf(type->elementType);
}
auto MemoryLayout::alignmentOf(const FunctionType* type) const -> int {
  return sizeOfPointer_;
}
auto MemoryLayout::alignmentOf(const ConceptType* type) const -> int {
  assert(!"alignmentOf(ConceptType)");
  return 0;
}
auto MemoryLayout::alignmentOf(const ClassType* type) const -> int {
  return type->symbol->alignment();
}
auto MemoryLayout::alignmentOf(const NamespaceType* type) const -> int {
  assert(!"alignmentOf(NamespaceType)");
  return 0;
}
auto MemoryLayout::alignmentOf(const MemberPointerType* type) const -> int {
  return sizeOfPointer_;
}
auto MemoryLayout::alignmentOf(const EnumType* type) const -> int { return 4; }
auto MemoryLayout::alignmentOf(const GenericType* type) const -> int {
  assert(!"alignmentOf(GenericType)");
  return 0;
}
auto MemoryLayout::alignmentOf(const PackType* type) const -> int {
  assert(!"alignmentOf(PackType)");
  return 0;
}

auto MemoryLayout::alignmentOf(const ScopedEnumType* type) const -> int {
  return alignmentOf(type->elementType);
}

// size of type

auto MemoryLayout::sizeOf(const InvalidType* type) const -> int {
  assert(!"todo");
  return 0;
}

auto MemoryLayout::sizeOf(const NullptrType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOf(const DependentType* type) const -> int {
  assert(!"sizeOfType of DependentType");
  return 0;
}

auto MemoryLayout::sizeOf(const AutoType* type) const -> int {
  assert(!"sizeOfType of placeholder type");
  return 0;
}

auto MemoryLayout::sizeOf(const VoidType* type) const -> int {
  assert(!"sizeOfType of incomplete type");
  return 0;
}

auto MemoryLayout::sizeOf(const BoolType* type) const -> int { return 1; }

auto MemoryLayout::sizeOf(const CharType* type) const -> int { return 1; }

auto MemoryLayout::sizeOf(const SignedCharType* type) const -> int { return 1; }

auto MemoryLayout::sizeOf(const UnsignedCharType* type) const -> int {
  return 1;
}

auto MemoryLayout::sizeOf(const ShortType* type) const -> int { return 2; }

auto MemoryLayout::sizeOf(const UnsignedShortType* type) const -> int {
  return 2;
}

auto MemoryLayout::sizeOf(const IntType* type) const -> int { return 4; }

auto MemoryLayout::sizeOf(const UnsignedIntType* type) const -> int {
  return 4;
}

auto MemoryLayout::sizeOf(const LongType* type) const -> int {
  return sizeOfLong_;
}

auto MemoryLayout::sizeOf(const UnsignedLongType* type) const -> int {
  return sizeOfLong_;
}

auto MemoryLayout::sizeOf(const FloatType* type) const -> int { return 4; }

auto MemoryLayout::sizeOf(const DoubleType* type) const -> int { return 8; }

auto MemoryLayout::sizeOf(const QualType* type) const -> int {
  return sizeOf(type->elementType);
}

auto MemoryLayout::sizeOf(const PointerType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOf(const LValueReferenceType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOf(const RValueReferenceType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOf(const ArrayType* type) const -> int {
  return sizeOf(type->elementType) * type->dim;
}

auto MemoryLayout::sizeOf(const FunctionType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOf(const ConceptType* type) const -> int {
  assert(!"sizeOfType0 of concept");
  return 0;
}

auto MemoryLayout::sizeOf(const ClassType* type) const -> int {
  return type->symbol->size();
}

auto MemoryLayout::sizeOf(const NamespaceType* type) const -> int {
  assert(!"sizeOfType of namespace");
  return 0;
}

auto MemoryLayout::sizeOf(const MemberPointerType* type) const -> int {
  return sizeOfPointer_;
}

auto MemoryLayout::sizeOf(const EnumType* type) const -> int { return 4; }

auto MemoryLayout::sizeOf(const GenericType* type) const -> int {
  assert(!"sizeOfType of generic type");
  return 0;
}

auto MemoryLayout::sizeOf(const PackType* type) const -> int {
  assert(!"sizeOfType of pack type");
  return 0;
}

auto MemoryLayout::sizeOf(const ScopedEnumType* type) const -> int {
  return sizeOf(type->elementType);
}

}  // namespace cxx