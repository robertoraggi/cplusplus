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
#include <cxx/mlir/spirv_buffer_layout.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVTypes.h>

namespace mlir::cxx {

auto spirvBufferLayoutType(Type type) -> Type {
  if (auto array = dyn_cast<ArrayType>(type)) {
    auto element = array.getElementType();
    return spirv::ArrayType::get(spirvBufferLayoutType(element),
                                 array.getSize(), bufferLayoutSize(element));
  }
  if (!isa<ClassType>(type)) return type;
  SmallVector<Type> fields;
  SmallVector<spirv::StructType::OffsetInfo> offsets;
  for (const auto& member : bufferLayoutMembers(type)) {
    fields.push_back(spirvBufferLayoutType(member.type));
    offsets.push_back(member.offset);
  }
  return spirv::StructType::get(fields, offsets);
}

auto hasSPIRVBufferLayout(Type actual, Type source) -> bool {
  auto expected = spirvBufferLayoutType(source);
  if (auto wanted = dyn_cast<spirv::ArrayType>(expected)) {
    auto array = dyn_cast<spirv::ArrayType>(actual);
    return array && array.getNumElements() == wanted.getNumElements() &&
           array.getArrayStride() == wanted.getArrayStride() &&
           hasSPIRVBufferLayout(array.getElementType(),
                                cast<ArrayType>(source).getElementType());
  }
  auto wanted = dyn_cast<spirv::StructType>(expected);
  if (!wanted) return actual == expected;
  auto record = dyn_cast<spirv::StructType>(actual);
  if (!record || !record.hasOffset()) return false;
  auto members = bufferLayoutMembers(source);
  if (record.getNumElements() != members.size()) return false;
  for (auto [index, member] : llvm::enumerate(members)) {
    if (record.getMemberOffset(index) != member.offset) return false;
    if (!hasSPIRVBufferLayout(record.getElementType(index), member.type))
      return false;
  }
  return true;
}

}  // namespace mlir::cxx
