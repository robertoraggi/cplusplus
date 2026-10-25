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

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <memory>
#include <variant>

namespace mlir {
class IRMapping;
}

namespace mlir::cxx {

class ReferenceRecordLayout {
 public:
  explicit ReferenceRecordLayout(Type record);
  [[nodiscard]] auto referenceFields() const -> ArrayRef<unsigned>;
  [[nodiscard]] auto referenceField(unsigned field) const
      -> std::optional<unsigned>;
  [[nodiscard]] auto stateField(unsigned field) const
      -> std::optional<unsigned>;
  [[nodiscard]] auto stateType() const -> Type;

 private:
  Type record_;
  SmallVector<unsigned> references_;
  SmallVector<unsigned> state_;
};

struct StorageMember {
  std::int32_t position;
  [[nodiscard]] auto operator==(const StorageMember&) const -> bool = default;
};

struct StorageElement {
  SmallVector<Value, 2> terms;
};

using StorageIndex = std::variant<StorageMember, StorageElement>;

struct StorageAddress {
  Value root;
  SmallVector<StorageIndex> path;
};

struct StorageReference {
  Value selector;
  SmallVector<StorageAddress> alternatives;
};

class StorageReferencePlan {
 public:
  StorageReferencePlan(std::unique_ptr<Block> operations,
                       StorageReference reference);

  [[nodiscard]] auto reference() const -> const StorageReference&;
  [[nodiscard]] auto materialize(OpBuilder& builder) -> StorageReference;

 private:
  std::unique_ptr<Block> operations_;
  StorageReference reference_;
};

class StorageReferenceLayout {
 public:
  void include(const StorageReference& reference);
  [[nodiscard]] auto covers(const StorageReference& reference) const -> bool;
  [[nodiscard]] auto empty() const -> bool;
  [[nodiscard]] auto type(MLIRContext* context) const -> TupleType;
  [[nodiscard]] auto pack(OpBuilder& builder, Location loc,
                          const StorageReference& reference) const
      -> FailureOr<Value>;
  [[nodiscard]] auto unpack(OpBuilder& builder, Location loc, Value descriptor,
                            const IRMapping* roots = nullptr) const
      -> StorageReference;

 private:
  struct Alternative {
    Value root;
    SmallVector<std::variant<StorageMember, std::monostate>> path;
    [[nodiscard]] auto operator==(const Alternative&) const -> bool = default;
  };
  [[nodiscard]] static auto shape(const StorageAddress& address) -> Alternative;
  SmallVector<Alternative> alternatives_;
  unsigned coordinates_ = 0;
};

[[nodiscard]] auto storageReferencePointer(OpBuilder& builder, Location loc,
                                           const StorageReference& reference,
                                           Type view = {}) -> FailureOr<Value>;

[[nodiscard]] auto isStorageRoot(Value storage) -> bool;

[[nodiscard]] auto verifyDeviceCallOperands(Operation* call) -> LogicalResult;

[[nodiscard]] auto getStorageElementType(const StorageAddress& address) -> Type;

[[nodiscard]] auto haveSameStoragePath(const StorageAddress& left,
                                       const StorageAddress& right) -> bool;

[[nodiscard]] auto resolveStorageAddress(Value pointer)
    -> FailureOr<StorageAddress>;

[[nodiscard]] auto storageIndex(OpBuilder& builder, Location loc,
                                const StorageElement& element) -> Value;

[[nodiscard]] auto storagePointer(OpBuilder& builder, Location loc,
                                  const StorageAddress& address, Type view = {})
    -> FailureOr<Value>;

[[nodiscard]] auto planStorageReference(MLIRContext* context, Value pointer)
    -> FailureOr<StorageReferencePlan>;

}  // namespace mlir::cxx
