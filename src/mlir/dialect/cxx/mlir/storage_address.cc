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

#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include <limits>

namespace mlir::cxx {

ReferenceRecordLayout::ReferenceRecordLayout(Type record) : record_(record) {
  if (!record) return;
  for (auto [index, field] : llvm::enumerate(getRecordElementTypes(record))) {
    if (isa<PointerType>(field))
      references_.push_back(index);
    else
      state_.push_back(index);
  }
}

auto ReferenceRecordLayout::referenceFields() const -> ArrayRef<unsigned> {
  return references_;
}

auto ReferenceRecordLayout::referenceField(unsigned field) const
    -> std::optional<unsigned> {
  auto found = llvm::find(references_, field);
  if (found == references_.end()) return std::nullopt;
  return unsigned(found - references_.begin());
}

auto ReferenceRecordLayout::stateField(unsigned field) const
    -> std::optional<unsigned> {
  auto found = llvm::find(state_, field);
  if (found == state_.end()) return std::nullopt;
  return unsigned(found - state_.begin());
}

auto ReferenceRecordLayout::stateType() const -> Type {
  if (state_.empty()) return {};
  auto record = dyn_cast<ClassType>(record_);
  if (!record) return {};
  SmallVector<Type> fields;
  SmallVector<unsigned> padding;
  for (auto index : state_) {
    if (record.isPaddingElement(index)) padding.push_back(fields.size());
    fields.push_back(record.getBody()[index]);
  }
  auto state = ClassType::getNamed(
      record.getContext(), ("device_state<" + record.getName() + ">").str());
  if (failed(state.setBody(fields, record.isPacked(), record.getAlignment(),
                           padding)))
    return {};
  return state;
}

namespace {

[[nodiscard]] auto objectType(Value pointer) -> Type {
  auto type = dyn_cast<PointerType>(pointer.getType());
  return type ? type.getElementType() : Type();
}

[[nodiscard]] auto isAddressDerivation(Operation* operation) -> bool {
  return isa<MemberOp, SubscriptOp, ArrayToPointerOp, PtrAddOp, BitcastOp,
             AddressSpaceCastOp>(operation);
}

[[nodiscard]] auto bridgedRoot(Value pointer) -> Value {
  if (auto reference = pointer.getDefiningOp<StorageRefOp>())
    return isStorageRoot(reference.getStorage()) ? reference.getStorage()
                                                 : Value();
  auto from = pointer.getDefiningOp<ptr::FromPtrOp>();
  if (!from || from.getMetadata()) return {};
  auto to = from.getPtr().getDefiningOp<ptr::ToPtrOp>();
  if (!to) return {};
  auto memory = dyn_cast<MemRefType>(to.getPtr().getType());
  if (!memory || memory.getRank() > 1 || !memory.getLayout().isIdentity())
    return {};
  if (!isStorageRoot(to.getPtr())) return {};
  if (objectType(pointer) != memory.getElementType()) return {};
  return to.getPtr();
}

[[nodiscard]] auto deriveAddress(Operation* operation, StorageAddress& address,
                                 Type& object) -> LogicalResult {
  if (auto member = dyn_cast<MemberOp>(operation)) {
    auto record = dyn_cast<ClassType>(object);
    if (!record || objectType(member.getBase()) != object) return failure();
    auto position = member.getMemberIndex();
    if (position < 0 || std::uint64_t(position) >= record.getBody().size())
      return failure();
    object = record.getBody()[position];
    address.path.push_back(StorageMember{position});
    return success();
  }
  if (auto subscript = dyn_cast<SubscriptOp>(operation)) {
    auto array = dyn_cast<ArrayType>(object);
    if (!array || objectType(subscript.getBase()) != object) return failure();
    object = array.getElementType();
    address.path.push_back(StorageElement{{subscript.getIndex()}});
    return success();
  }
  if (auto decay = dyn_cast<ArrayToPointerOp>(operation)) {
    auto array = dyn_cast<ArrayType>(object);
    if (!array || objectType(decay.getValue()) != object) return failure();
    object = array.getElementType();
    address.path.push_back(StorageElement{});
    return success();
  }
  if (auto addition = dyn_cast<PtrAddOp>(operation)) {
    if (!canBitcastStorageValue(object, objectType(addition.getBase())))
      return failure();
    if (auto array = dyn_cast<ArrayType>(object)) {
      if (canBitcastStorageValue(array.getElementType(),
                                 objectType(addition.getResult()))) {
        object = array.getElementType();
        address.path.push_back(StorageElement{{addition.getOffset()}});
        return success();
      }
    }
    if (!canBitcastStorageValue(object, objectType(addition.getResult())))
      return failure();
    if (!address.path.empty()) {
      if (auto element = std::get_if<StorageElement>(&address.path.back())) {
        element->terms.push_back(addition.getOffset());
        return success();
      }
    }
    return success(matchPattern(addition.getOffset(), m_Zero()));
  }
  return success(isa<BitcastOp, AddressSpaceCastOp>(operation));
}

class StorageReferenceBuilder {
 public:
  explicit StorageReferenceBuilder(OpBuilder& builder) : builder_(builder) {}

  [[nodiscard]] auto build(Value pointer) -> FailureOr<StorageReference> {
    SmallVector<std::pair<Value, bool>> worklist{{pointer, false}};
    while (!worklist.empty()) {
      auto [current, ready] = worklist.pop_back_val();
      if (references_.contains(current)) continue;
      if (!isa<PointerType>(current.getType())) return failure();
      if (auto address = resolveStorageAddress(current); succeeded(address)) {
        auto zero =
            arith::ConstantIntOp::create(builder_, current.getLoc(), 0, 32);
        references_[current] = StorageReference{zero, {std::move(*address)}};
        continue;
      }
      auto* operation = current.getDefiningOp();
      if (!operation) return failure();
      auto select = dyn_cast<arith::SelectOp>(operation);
      if (!select && !isAddressDerivation(operation)) return failure();
      SmallVector<Value, 2> children;
      if (select)
        children = {select.getTrueValue(), select.getFalseValue()};
      else
        children = {operation->getOperand(0)};
      if (!ready) {
        worklist.push_back({current, true});
        for (auto child : children) worklist.push_back({child, false});
        continue;
      }
      if (select) {
        auto result = combine(select, references_.find(children[0])->second,
                              references_.find(children[1])->second);
        if (failed(result)) return failure();
        references_[current] = std::move(*result);
        continue;
      }
      auto result = references_.find(children.front())->second;
      for (auto& address : result.alternatives) {
        auto object = getStorageElementType(address);
        if (!object || failed(deriveAddress(operation, address, object)))
          return failure();
      }
      references_[current] = std::move(result);
    }
    return references_.find(pointer)->second;
  }

 private:
  [[nodiscard]] auto reindex(Value selector, ArrayRef<unsigned> positions,
                             Location loc) -> Value {
    Value result =
        arith::ConstantIntOp::create(builder_, loc, positions.front(), 32);
    for (unsigned index = 1; index < positions.size(); ++index) {
      auto current = arith::ConstantIntOp::create(builder_, loc, index, 32);
      auto match = arith::CmpIOp::create(
          builder_, loc, arith::CmpIPredicate::eq, selector, current);
      auto position =
          arith::ConstantIntOp::create(builder_, loc, positions[index], 32);
      result = arith::SelectOp::create(builder_, loc, match, position, result);
    }
    return result;
  }

  [[nodiscard]] auto combine(arith::SelectOp select,
                             const StorageReference& lhs,
                             const StorageReference& rhs)
      -> FailureOr<StorageReference> {
    StorageReference result{{}, lhs.alternatives};
    SmallVector<unsigned> left, right;
    for (unsigned index = 0; index < lhs.alternatives.size(); ++index)
      left.push_back(index);
    for (const auto& address : rhs.alternatives) {
      auto found =
          llvm::find_if(result.alternatives, [&](const StorageAddress& other) {
            return haveSameStoragePath(address, other);
          });
      auto position = unsigned(found - result.alternatives.begin());
      right.push_back(position);
      if (found == result.alternatives.end()) {
        result.alternatives.push_back(address);
        continue;
      }
      for (auto [target, incoming] : llvm::zip(found->path, address.path)) {
        auto element = std::get_if<StorageElement>(&target);
        if (!element) continue;
        auto first = storageIndex(builder_, select.getLoc(), *element);
        auto second = storageIndex(builder_, select.getLoc(),
                                   std::get<StorageElement>(incoming));
        auto index = arith::SelectOp::create(
            builder_, select.getLoc(), select.getCondition(), first, second);
        element->terms.assign({index});
      }
    }
    if (result.alternatives.size() >
        std::uint64_t(std::numeric_limits<std::int32_t>::max()))
      return failure();
    auto first = reindex(lhs.selector, left, select.getLoc());
    auto second = reindex(rhs.selector, right, select.getLoc());
    result.selector = arith::SelectOp::create(
        builder_, select.getLoc(), select.getCondition(), first, second);
    return result;
  }

  OpBuilder& builder_;
  DenseMap<Value, StorageReference> references_;
};

[[nodiscard]] auto canMaterializeStoragePointer(const StorageAddress& address,
                                                Type view) -> bool {
  auto memory = dyn_cast<MemRefType>(address.root.getType());
  if (!memory || memory.getRank() > 1 || !memory.getLayout().isIdentity())
    return false;
  auto element = getStorageElementType(address);
  if (!element) return false;
  if (!view) return true;
  auto pointer = dyn_cast<PointerType>(view);
  if (!pointer) return false;
  return canBitcastStorageValue(element, pointer.getElementType());
}

}  // namespace

auto verifyDeviceCallOperands(Operation* call) -> LogicalResult {
  for (auto operand : call->getOperands()) {
    if (isa<PointerType>(operand.getType()))
      return call->emitError(
          "device call passes a reference whose storage root cannot be "
          "determined");
    if (isa<MemRefType>(operand.getType()) && !isStorageRoot(operand))
      return call->emitError("device calls must pass whole storage roots");
  }
  return success();
}

auto isStorageRoot(Value storage) -> bool {
  auto memory = dyn_cast<MemRefType>(storage.getType());
  if (!memory || memory.getRank() > 1 || !memory.getLayout().isIdentity())
    return false;
  if (!memory.hasStaticShape() && memory.getRank() != 1) return false;
  if (isa_and_nonnull<memref::AllocaOp, memref::AllocOp>(
          storage.getDefiningOp()))
    return true;
  auto argument = dyn_cast<BlockArgument>(storage);
  if (!argument) return false;
  auto function =
      dyn_cast<FunctionOpInterface>(argument.getOwner()->getParentOp());
  if (!function || function.getFunctionBody().empty()) return false;
  return argument.getOwner() == &function.getFunctionBody().front();
}

auto resolveStorageAddress(Value pointer) -> FailureOr<StorageAddress> {
  SmallVector<Operation*> derivations;
  Value current = pointer;
  Value root;
  while (!(root = bridgedRoot(current))) {
    auto* operation = current.getDefiningOp();
    if (!operation) return failure();
    if (!isAddressDerivation(operation)) return failure();
    if (!isa<PointerType>(current.getType())) return failure();
    derivations.push_back(operation);
    current = operation->getOperand(0);
  }

  StorageAddress address{root, {}};
  if (cast<MemRefType>(root.getType()).getRank() == 1)
    address.path.push_back(StorageElement{});
  auto object = cast<MemRefType>(root.getType()).getElementType();
  for (auto* operation : llvm::reverse(derivations))
    if (failed(deriveAddress(operation, address, object))) return failure();
  return address;
}

auto haveSameStoragePath(const StorageAddress& left,
                         const StorageAddress& right) -> bool {
  if (left.root != right.root || left.path.size() != right.path.size())
    return false;
  for (auto [first, second] : llvm::zip(left.path, right.path)) {
    if (first.index() != second.index()) return false;
    if (auto member = std::get_if<StorageMember>(&first))
      if (*member != std::get<StorageMember>(second)) return false;
  }
  return true;
}

auto getStorageElementType(const StorageAddress& address) -> Type {
  auto memory = dyn_cast<MemRefType>(address.root.getType());
  if (!memory || memory.getRank() > 1) return {};
  auto object = memory.getElementType();
  auto path = ArrayRef(address.path);
  if (memory.getRank() == 1 && !path.empty()) {
    if (!std::holds_alternative<StorageElement>(path.front())) return {};
    path = path.drop_front();
  }
  for (const auto& index : path) {
    if (auto member = std::get_if<StorageMember>(&index))
      object = getAggregateElementType(object, member->position);
    else if (auto array = dyn_cast_or_null<ArrayType>(object))
      object = array.getElementType();
    else
      return {};
    if (!object) return {};
  }
  return object;
}

auto storageIndex(OpBuilder& builder, Location loc,
                  const StorageElement& element) -> Value {
  Value result;
  for (auto term : element.terms) {
    if (!term.getType().isIndex())
      term = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                        term);
    result = result ? Value(arith::AddIOp::create(builder, loc, result, term))
                    : term;
  }
  return result ? result : arith::ConstantIndexOp::create(builder, loc, 0);
}

auto storagePointer(OpBuilder& builder, Location loc,
                    const StorageAddress& address, Type view)
    -> FailureOr<Value> {
  if (!canMaterializeStoragePointer(address, view)) return failure();
  auto memory = cast<MemRefType>(address.root.getType());
  auto object = memory.getElementType();
  auto viewType = dyn_cast_or_null<PointerType>(view);
  auto space = viewType ? viewType.getAddressSpace() : Attribute();
  auto pointerType = PointerType::get(builder.getContext(), object, space);
  Value pointer = StorageRefOp::create(builder, loc, pointerType, address.root);
  auto path = ArrayRef(address.path);
  if (memory.getRank() == 1 && !path.empty()) {
    auto element = std::get_if<StorageElement>(&path.front());
    if (!element) return failure();
    auto position = storageIndex(builder, loc, *element);
    pointer = PtrAddOp::create(builder, loc, pointerType, pointer, position);
    path = path.drop_front();
  }
  for (const auto& index : path) {
    if (auto member = std::get_if<StorageMember>(&index)) {
      auto record = dyn_cast<ClassType>(object);
      if (!record || member->position < 0 ||
          std::uint64_t(member->position) >= record.getBody().size())
        return failure();
      object = record.getBody()[member->position];
      pointer =
          MemberOp::create(builder, loc,
                           PointerType::get(builder.getContext(), object,
                                            pointerType.getAddressSpace()),
                           pointer, member->position);
      continue;
    }
    auto array = dyn_cast<ArrayType>(object);
    if (!array) return failure();
    object = array.getElementType();
    auto position = storageIndex(builder, loc, std::get<StorageElement>(index));
    pointer =
        SubscriptOp::create(builder, loc,
                            PointerType::get(builder.getContext(), object,
                                             pointerType.getAddressSpace()),
                            pointer, position);
  }
  if (view && view != pointer.getType()) {
    auto target = dyn_cast<PointerType>(view);
    if (!target || !haveSamePointerMemorySpace(
                       cast<PointerType>(pointer.getType()), target))
      return failure();
    pointer = BitcastOp::create(builder, loc, view, pointer);
  }
  return pointer;
}

auto StorageReferenceLayout::shape(const StorageAddress& address)
    -> Alternative {
  Alternative result{address.root, {}};
  for (const auto& index : address.path) {
    if (auto member = std::get_if<StorageMember>(&index))
      result.path.push_back(*member);
    else
      result.path.push_back(std::monostate{});
  }
  return result;
}

void StorageReferenceLayout::include(const StorageReference& reference) {
  for (const auto& address : reference.alternatives) {
    auto alternative = shape(address);
    unsigned coordinates =
        llvm::count_if(alternative.path, [](const auto& index) {
          return std::holds_alternative<std::monostate>(index);
        });
    coordinates_ = std::max(coordinates_, coordinates);
    if (!llvm::is_contained(alternatives_, alternative))
      alternatives_.push_back(std::move(alternative));
  }
}

auto StorageReferenceLayout::covers(const StorageReference& reference) const
    -> bool {
  return llvm::all_of(
      reference.alternatives, [&](const StorageAddress& address) {
        auto alternative = shape(address);
        auto coordinates =
            llvm::count_if(alternative.path, [](const auto& index) {
              return std::holds_alternative<std::monostate>(index);
            });
        return unsigned(coordinates) <= coordinates_ &&
               llvm::is_contained(alternatives_, alternative);
      });
}

auto StorageReferenceLayout::empty() const -> bool {
  return alternatives_.empty();
}

auto StorageReferenceLayout::type(MLIRContext* context) const -> TupleType {
  SmallVector<Type> fields{IntegerType::get(context, 32)};
  fields.append(coordinates_, IndexType::get(context));
  return TupleType::get(context, fields);
}

auto StorageReferenceLayout::pack(OpBuilder& builder, Location loc,
                                  const StorageReference& reference) const
    -> FailureOr<Value> {
  SmallVector<Value> selected;
  for (auto [index, address] : llvm::enumerate(reference.alternatives)) {
    auto alternative = shape(address);
    auto found = llvm::find(alternatives_, alternative);
    if (found == alternatives_.end()) return failure();
    SmallVector<Value> values{arith::ConstantIntOp::create(
        builder, loc, found - alternatives_.begin(), 32)};
    for (const auto& position : address.path)
      if (auto element = std::get_if<StorageElement>(&position))
        values.push_back(storageIndex(builder, loc, *element));
    while (values.size() < coordinates_ + 1)
      values.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
    if (!index) {
      selected = std::move(values);
      continue;
    }
    auto position = arith::ConstantIntOp::create(builder, loc, index, 32);
    auto condition = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, reference.selector, position);
    for (auto [target, incoming] : llvm::zip(selected, values))
      target =
          arith::SelectOp::create(builder, loc, condition, incoming, target);
  }
  if (selected.empty()) return failure();
  auto descriptor = type(builder.getContext());
  Value result = UndefOp::create(builder, loc, descriptor);
  for (auto [index, value] : llvm::enumerate(selected))
    result = InsertValueOp::create(builder, loc, descriptor, result, value,
                                   std::int64_t(index));
  return result;
}

auto StorageReferenceLayout::unpack(OpBuilder& builder, Location loc,
                                    Value descriptor,
                                    const IRMapping* roots) const
    -> StorageReference {
  StorageReference result;
  result.selector = ExtractValueOp::create(builder, loc, builder.getI32Type(),
                                           descriptor, std::int64_t(0));
  SmallVector<Value> coordinates;
  for (unsigned index = 0; index < coordinates_; ++index)
    coordinates.push_back(
        ExtractValueOp::create(builder, loc, builder.getIndexType(), descriptor,
                               std::int64_t(index + 1)));
  for (const auto& alternative : alternatives_) {
    auto root =
        roots ? roots->lookupOrDefault(alternative.root) : alternative.root;
    StorageAddress address{root, {}};
    unsigned coordinate = 0;
    for (const auto& position : alternative.path) {
      if (auto member = std::get_if<StorageMember>(&position))
        address.path.push_back(*member);
      else
        address.path.push_back(StorageElement{{coordinates[coordinate++]}});
    }
    result.alternatives.push_back(std::move(address));
  }
  return result;
}

auto storageReferencePointer(OpBuilder& builder, Location loc,
                             const StorageReference& reference, Type view)
    -> FailureOr<Value> {
  Value result;
  for (auto [index, address] : llvm::enumerate(reference.alternatives)) {
    auto pointer = storagePointer(builder, loc, address, view);
    if (failed(pointer)) return failure();
    if (!index) {
      result = *pointer;
      continue;
    }
    auto position = arith::ConstantIntOp::create(builder, loc, index, 32);
    auto condition = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, reference.selector, position);
    result = arith::SelectOp::create(builder, loc, condition, *pointer, result);
  }
  return result;
}

StorageReferencePlan::StorageReferencePlan(std::unique_ptr<Block> operations,
                                           StorageReference reference)
    : operations_(std::move(operations)), reference_(std::move(reference)) {}

auto StorageReferencePlan::reference() const -> const StorageReference& {
  return reference_;
}

auto StorageReferencePlan::materialize(OpBuilder& builder) -> StorageReference {
  for (auto& operation : llvm::make_early_inc_range(*operations_)) {
    operation.remove();
    builder.insert(&operation);
  }
  return reference_;
}

auto planStorageReference(MLIRContext* context, Value pointer)
    -> FailureOr<StorageReferencePlan> {
  auto operations = std::make_unique<Block>();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(operations.get());
  auto reference = StorageReferenceBuilder(builder).build(pointer);
  if (failed(reference)) return failure();
  for (const auto& address : reference->alternatives)
    if (!canMaterializeStoragePointer(address, pointer.getType()))
      return failure();
  return StorageReferencePlan(std::move(operations), std::move(*reference));
}

}  // namespace mlir::cxx
