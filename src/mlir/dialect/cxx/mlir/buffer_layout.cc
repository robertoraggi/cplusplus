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

#include <cxx/mlir/buffer_layout.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

namespace mlir::cxx {
namespace {

[[nodiscard]] auto isBufferScalarType(Type type) -> bool {
  return type.isF32() || type.isSignlessInteger(32);
}

[[nodiscard]] auto holdsNoBytes(Type type) -> bool {
  auto array = dyn_cast<ArrayType>(type);
  if (!array) return false;
  return array.getSize() == 0 || holdsNoBytes(array.getElementType());
}

[[nodiscard]] auto storagePositions(Type aggregate, bool includePadding)
    -> llvm::SmallVector<std::uint32_t> {
  auto record = dyn_cast<ClassType>(aggregate);
  llvm::SmallVector<std::uint32_t> positions;
  for (auto [position, field] :
       llvm::enumerate(getRecordElementTypes(aggregate))) {
    if (!includePadding && record && record.isPaddingElement(position))
      continue;
    if (holdsNoBytes(field)) continue;
    positions.push_back(static_cast<std::uint32_t>(position));
  }
  return positions;
}

}  // namespace

auto valueElementPositions(Type aggregate) -> llvm::SmallVector<std::uint32_t> {
  auto positions = storagePositions(aggregate, false);
  if (positions.empty()) return storagePositions(aggregate, true);
  return positions;
}

auto isValueElement(Type aggregate, std::int64_t position) -> bool {
  if (!isa<ClassType>(aggregate)) return true;
  return llvm::is_contained(valueElementPositions(aggregate), position);
}

auto valueElementIndex(Type aggregate, std::int64_t position)
    -> std::optional<std::uint32_t> {
  if (!getAggregateElementType(aggregate, position)) return std::nullopt;
  if (!isa<ClassType>(aggregate)) return static_cast<std::uint32_t>(position);
  auto positions = valueElementPositions(aggregate);
  auto found = llvm::find(positions, position);
  if (found == positions.end()) return std::nullopt;
  return static_cast<std::uint32_t>(found - positions.begin());
}

auto isBufferValueType(Type type) -> bool {
  if (isBufferScalarType(type)) return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && vector.getRank() == 1 && !vector.isScalable() &&
         vector.getNumElements() >= 2 && vector.getNumElements() <= 4 &&
         isBufferScalarType(vector.getElementType());
}

auto isBufferRecordType(Type type) -> bool {
  auto record = dyn_cast<ClassType>(type);
  if (!record) return false;
  auto positions = valueElementPositions(record);
  return !positions.empty() && llvm::all_of(positions, [&](auto position) {
    return isBufferLayoutType(record.getBody()[position]);
  });
}

auto isBufferLayoutType(Type type) -> bool {
  if (isBufferValueType(type)) return true;
  if (auto array = dyn_cast<ArrayType>(type))
    return array.getSize() > 0 && isBufferLayoutType(array.getElementType());
  return isBufferRecordType(type);
}

auto bufferLayoutSize(Type type) -> std::uint32_t {
  return static_cast<std::uint32_t>(DataLayout().getTypeSize(type));
}

auto bufferLayoutExtent(Type type) -> std::uint32_t {
  if (auto vector = dyn_cast<VectorType>(type))
    return 4 * static_cast<std::uint32_t>(vector.getNumElements());
  if (auto array = dyn_cast<ArrayType>(type)) {
    auto element = array.getElementType();
    return static_cast<std::uint32_t>(array.getSize() - 1) *
               bufferLayoutSize(element) +
           bufferLayoutExtent(element);
  }
  if (!isa<ClassType>(type)) return bufferLayoutSize(type);
  auto members = bufferLayoutMembers(type);
  return members.back().offset + bufferLayoutExtent(members.back().type);
}

auto bufferLayoutMembers(Type record) -> llvm::SmallVector<BufferLayoutMember> {
  auto type = cast<ClassType>(record);
  auto offsets = getClassElementOffsets(type, DataLayout());
  llvm::SmallVector<BufferLayoutMember> members;
  for (auto position : valueElementPositions(type))
    members.push_back({type.getBody()[position], position,
                       static_cast<std::uint32_t>(offsets[position]), 0});
  for (auto [member, next] : llvm::zip(members, llvm::drop_begin(members)))
    member.span = next.offset - member.offset;
  if (!members.empty())
    members.back().span = bufferLayoutSize(record) - members.back().offset;
  return members;
}

auto designatesNonValueElement(const StorageAddress& address) -> bool {
  auto memory = cast<MemRefType>(address.root.getType());
  Type object = memory.getElementType();
  auto path = ArrayRef(address.path);
  if (memory.getRank() == 1 && !path.empty()) path = path.drop_front();
  for (const auto& index : path) {
    if (auto member = std::get_if<StorageMember>(&index)) {
      if (!isValueElement(object, member->position)) return true;
      object = getAggregateElementType(object, member->position);
      continue;
    }
    auto array = dyn_cast<ArrayType>(object);
    if (!array) return false;
    object = array.getElementType();
  }
  return false;
}

}  // namespace mlir::cxx
