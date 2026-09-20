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

#include <bit>
#include <cstdlib>
#include <optional>

namespace cxx {
namespace {

[[nodiscard]] auto storageSizeInBytes(std::size_t numBits) -> std::size_t {
  auto bytes = (numBits + 7) / 8;
  std::size_t result = 1;
  while (result < bytes) result *= 2;
  return result;
}

[[nodiscard]] auto vectorWidthInBits(const MemoryLayout& memoryLayout,
                                     const VectorType* type)
    -> std::optional<std::size_t> {
  if (type->vectorKind() == VectorKind::kExt &&
      type->elementType()->kind() == TypeKind::kBool)
    return type->elementCount();

  auto elementSize = memoryLayout.sizeOf(type->elementType());
  if (!elementSize) return std::nullopt;

  return type->elementCount() * *elementSize * 8;
}

[[nodiscard]] auto atomicWidthInBits(const MemoryLayout& memoryLayout,
                                     const AtomicType* type)
    -> std::optional<std::size_t> {
  auto valueSize = memoryLayout.sizeOf(type->elementType());
  if (!valueSize) return std::nullopt;

  auto width = *valueSize * 8;
  if (!width) return std::size_t{8};
  if (width > memoryLayout.maxAtomicPromoteWidth()) return width;

  return std::bit_ceil(width);
}

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
    return memoryLayout.sizeOfWideChar();
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

  auto operator()(const Float16Type* type) const -> std::optional<std::size_t> {
    return 2;
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
    auto classSymbol = type->definition();
    if (!classSymbol->isComplete()) return std::nullopt;
    return classSymbol->sizeInBytes();
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
    return 2 * memoryLayout.sizeOfPointer();
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

  auto operator()(const UnresolvedBuiltinType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const OverloadSetType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const BitIntType* type) const -> std::optional<std::size_t> {
    return storageSizeInBytes(type->numBits());
  }

  auto operator()(const UnsignedBitIntType* type) const
      -> std::optional<std::size_t> {
    return storageSizeInBytes(type->numBits());
  }

  auto operator()(const UnresolvedBitIntType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const VectorType* type) const -> std::optional<std::size_t> {
    auto bits = vectorWidthInBits(memoryLayout, type);
    if (!bits) return std::nullopt;
    return storageSizeInBytes(*bits);
  }

  auto operator()(const UnresolvedVectorType* type) const
      -> std::optional<std::size_t> {
    return std::nullopt;
  }

  auto operator()(const ComplexType* type) const -> std::optional<std::size_t> {
    auto elementSize = memoryLayout.sizeOf(type->elementType());
    if (!elementSize) return std::nullopt;
    return *elementSize * 2;
  }

  auto operator()(const AtomicType* type) const -> std::optional<std::size_t> {
    auto width = atomicWidthInBits(memoryLayout, type);
    if (!width) return std::nullopt;
    return *width / 8;
  }
};

struct AlignmentOf {
  const MemoryLayout& memoryLayout;

  auto operator()(const QualType* type) const -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const EnumType* type) const -> std::optional<std::size_t> {
    if (type->underlyingType())
      return memoryLayout.alignmentOf(type->underlyingType());
    return 4;
  }

  auto operator()(const ScopedEnumType* type) const
      -> std::optional<std::size_t> {
    if (type->underlyingType())
      return memoryLayout.alignmentOf(type->underlyingType());
    return 4;
  }

  auto operator()(const ClassType* type) const -> std::optional<std::size_t> {
    auto classSymbol = type->definition();
    if (!classSymbol->isComplete()) return std::nullopt;
    return classSymbol->alignment();
  }

  auto operator()(const UnboundedArrayType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const BoundedArrayType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const MemberObjectPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const MemberFunctionPointerType* type) const
      -> std::optional<std::size_t> {
    return memoryLayout.sizeOfPointer();
  }

  auto operator()(const VectorType* type) const -> std::optional<std::size_t> {
    auto size = memoryLayout.sizeOf(type);
    if (!size) return std::nullopt;
    auto maximum = memoryLayout.maxVectorAlignment();
    if (maximum && maximum < *size) return maximum;
    return size;
  }

  auto operator()(const ComplexType* type) const -> std::optional<std::size_t> {
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(const AtomicType* type) const -> std::optional<std::size_t> {
    auto width = atomicWidthInBits(memoryLayout, type);
    if (!width) return std::nullopt;
    if (*width <= memoryLayout.maxAtomicPromoteWidth()) return *width / 8;
    return memoryLayout.alignmentOf(type->elementType());
  }

  auto operator()(auto type) const -> std::optional<std::size_t> {
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
  longDoubleMantissaDigits_ = 53;
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

auto MemoryLayout::longDoubleMantissaDigits() const -> std::size_t {
  return longDoubleMantissaDigits_;
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

void MemoryLayout::setSizeOfLongDouble(std::size_t sizeOfLongDouble,
                                       std::size_t mantissaDigits) {
  sizeOfLongDouble_ = sizeOfLongDouble;
  longDoubleMantissaDigits_ = mantissaDigits;
}

auto MemoryLayout::sizeOfWideChar() const -> std::size_t {
  return sizeOfWideChar_;
}

auto MemoryLayout::isWideCharSigned() const -> bool {
  return wideCharIsSigned_;
}

void MemoryLayout::setWideCharUnderlyingType(std::size_t size, bool isSigned) {
  sizeOfWideChar_ = size;
  wideCharIsSigned_ = isSigned;
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

auto MemoryLayout::arch() const -> std::string_view {
  return std::string_view{triple_}.substr(0, triple_.find('-'));
}

auto MemoryLayout::isWebAssembly() const -> bool {
  return arch().starts_with("wasm");
}

auto MemoryLayout::isDarwin() const -> bool {
  std::string_view triple{triple_};
  return triple.find("apple") != std::string_view::npos ||
         triple.find("darwin") != std::string_view::npos ||
         triple.find("macos") != std::string_view::npos;
}

auto MemoryLayout::usesArmMemberPointerAbi() const -> bool {
  const auto arch = this->arch();
  return arch.starts_with("arm") || arch.starts_with("aarch64") ||
         arch.starts_with("thumb") || isWebAssembly();
}

auto MemoryLayout::defaultNewAlignment() const -> std::size_t { return 16; }

auto MemoryLayout::maxAtomicInlineWidth() const -> std::size_t {
  const auto arch = this->arch();
  if (arch.starts_with("aarch64") || arch.starts_with("arm64")) return 128;
  return 64;
}

auto MemoryLayout::maxAtomicPromoteWidth() const -> std::size_t {
  const auto arch = this->arch();
  if (arch.starts_with("aarch64") || arch.starts_with("arm64")) return 128;
  if (arch.starts_with("x86_64")) return 128;
  return 64;
}

auto MemoryLayout::maxVectorAlignment() const -> std::size_t {
  const auto arch = this->arch();
  if (arch.starts_with("aarch64") || arch.starts_with("arm64")) return 16;
  return 0;
}

auto MemoryLayout::nullMemberObjectPointer() const -> std::int64_t {
  return -1;
}

auto MemoryLayout::classValueAbiKind() const -> ClassValueAbiKind {
  const auto arch = this->arch();
  if (isWebAssembly()) return ClassValueAbiKind::kSingleScalar;
  if (arch.starts_with("aarch64") || arch.starts_with("arm64"))
    return ClassValueAbiKind::kAArch64;
  if (arch.starts_with("x86_64") || arch.starts_with("amd64"))
    return ClassValueAbiKind::kX86_64;
  return ClassValueAbiKind::kDefault;
}

auto to_string(FramePointerKind kind) -> std::string_view {
  switch (kind) {
    case FramePointerKind::kNone:
      return "none";
    case FramePointerKind::kNonLeaf:
      return "non-leaf";
    case FramePointerKind::kAll:
      return "all";
  }
  return "none";
}

auto MemoryLayout::framePointerKind() const -> FramePointerKind {
  if (!isDarwin()) return FramePointerKind::kNone;
  const auto arch = this->arch();
  if (arch.starts_with("aarch64") || arch.starts_with("arm64"))
    return FramePointerKind::kNonLeaf;
  return FramePointerKind::kAll;
}

void MemoryLayout::setTriple(std::string triple) {
  triple_ = std::move(triple);
}
}  // namespace cxx
