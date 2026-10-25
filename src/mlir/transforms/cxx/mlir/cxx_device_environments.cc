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

#include <cxx/mlir/cxx_device_environments.h>
#include <cxx/mlir/storage_address.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir::cxx {

auto referenceEnvironmentUses(Value root)
    -> FailureOr<SmallVector<OpOperand*>> {
  SmallVector<Value> worklist{root};
  SmallVector<OpOperand*> uses;
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (!visited.insert(value).second) continue;
    for (auto& use : value.getUses()) {
      auto* operation = use.getOwner();
      if (isa<ptr::ToPtrOp, ptr::FromPtrOp, StorageRefOp>(operation)) {
        if (auto from = dyn_cast<ptr::FromPtrOp>(operation))
          if (from.getMetadata()) return failure();
        worklist.push_back(operation->getResult(0));
      } else if (auto member = dyn_cast<MemberOp>(operation)) {
        auto record =
            cast<PointerType>(member.getBase().getType()).getElementType();
        auto field = getAggregateElementType(record, member.getMemberIndex());
        if (isa_and_nonnull<PointerType>(field))
          if (cast<PointerType>(member.getType()).getElementType() != field)
            return failure();
        for (auto& field : member.getResult().getUses()) uses.push_back(&field);
      } else if (isa<func::CallOp>(operation)) {
        uses.push_back(&use);
      } else {
        return failure();
      }
    }
  }
  return uses;
}

namespace {

[[nodiscard]] auto targetParameter(OpOperand& use) -> Value {
  auto call = dyn_cast<func::CallOp>(use.getOwner());
  if (!call) return {};
  auto target = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!target || target.isExternal()) return {};
  if (use.getOperandNumber() >= target.getNumArguments()) return {};
  return target.getArgument(use.getOperandNumber());
}

[[nodiscard]] auto environmentType(Value root) -> ClassType {
  if (auto pointer = dyn_cast<PointerType>(root.getType()))
    return dyn_cast<ClassType>(pointer.getElementType());
  auto memory = dyn_cast<MemRefType>(root.getType());
  if (!memory || memory.getRank()) return {};
  return dyn_cast<ClassType>(memory.getElementType());
}

[[nodiscard]] auto fieldIndex(Value address) -> std::optional<unsigned> {
  auto member = address.getDefiningOp<MemberOp>();
  if (!member || member.getMemberIndex() < 0) return std::nullopt;
  return member.getMemberIndex();
}

[[nodiscard]] auto referenceFieldUse(OpOperand& use,
                                     const ReferenceRecordLayout& layout)
    -> std::optional<unsigned> {
  auto field = fieldIndex(use.get());
  return field ? layout.referenceField(*field) : std::nullopt;
}

}  // namespace

auto isReadOnlyReferenceEnvironment(Value parameter) -> bool {
  SmallVector<Value> worklist{parameter};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (!visited.insert(value).second) continue;
    auto uses = referenceEnvironmentUses(value);
    if (failed(uses)) return false;
    ReferenceRecordLayout layout(environmentType(value));
    for (auto* use : *uses) {
      if (fieldIndex(use->get()) && !referenceFieldUse(*use, layout)) continue;
      if (auto load = dyn_cast<LoadOp>(use->getOwner())) {
        if (use->getOperandNumber() != 0 || !isa<PointerType>(load.getType()))
          return false;
        if (!load->getDiscardableAttrDictionary().empty()) return false;
        continue;
      }
      auto target = targetParameter(*use);
      if (!target) return false;
      worklist.push_back(target);
    }
  }
  return true;
}

auto referenceEnvironmentInitializer(Value parameter)
    -> FailureOr<ReferenceEnvironmentInitializer> {
  auto record = environmentType(parameter);
  if (!record || record.getBody().empty()) return failure();
  auto uses = referenceEnvironmentUses(parameter);
  if (failed(uses)) return failure();
  ReferenceEnvironmentInitializer result;
  ReferenceRecordLayout layout(record);
  if (layout.referenceFields().empty()) return failure();
  result.parameters.resize(layout.referenceFields().size());
  SmallVector<bool> initialized(layout.referenceFields().size(), false);
  for (auto* use : *uses) {
    if (fieldIndex(use->get()) && !referenceFieldUse(*use, layout)) continue;
    auto store = dyn_cast<StoreOp>(use->getOwner());
    if (!store || use->getOperandNumber() != 1) return failure();
    if (!store->getDiscardableAttrDictionary().empty()) return failure();
    auto index = referenceFieldUse(*use, layout);
    if (!index || *index >= initialized.size() || initialized[*index])
      return failure();
    auto source = dyn_cast<BlockArgument>(store.getValue());
    auto argument = dyn_cast<BlockArgument>(parameter);
    if (!source || !argument || source.getOwner() != argument.getOwner())
      return failure();
    if (!isa<PointerType>(source.getType()) || source == argument)
      return failure();
    initialized[*index] = true;
    result.parameters[*index] = source.getArgNumber();
    result.stores.push_back(store);
  }
  if (!llvm::all_of(initialized, [](bool value) { return value; }))
    return failure();
  auto argument = dyn_cast<BlockArgument>(parameter);
  if (!argument) return failure();
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function) return failure();
  DominanceInfo dominance(function);
  auto returns = function.walk([&](func::ReturnOp returned) {
    for (auto store : result.stores)
      if (!dominance.dominates(store, returned)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (returns.wasInterrupted()) return failure();
  return result;
}

auto bindReferenceEnvironment(Value root, Value state, ValueRange fields,
                              RewriterBase& rewriter) -> LogicalResult {
  auto uses = referenceEnvironmentUses(root);
  if (failed(uses)) return failure();
  ReferenceRecordLayout layout(environmentType(root));
  if (fields.size() != layout.referenceFields().size()) return failure();
  SmallVector<MemberOp> members;
  for (auto* use : *uses) {
    auto member = use->get().getDefiningOp<MemberOp>();
    if (!member) {
      if (!isa<func::CallOp>(use->getOwner())) return failure();
      continue;
    }
    auto field = layout.referenceField(member.getMemberIndex());
    if (field) {
      auto load = dyn_cast<LoadOp>(use->getOwner());
      if (!load || load.getType() != fields[*field].getType()) return failure();
    }
    if (!llvm::is_contained(members, member)) members.push_back(member);
  }

  for (auto* use : *uses) {
    if (use->get().getDefiningOp<MemberOp>()) continue;
    rewriter.setInsertionPoint(use->getOwner());
    auto binding =
        BindReferencesOp::create(rewriter, use->getOwner()->getLoc(),
                                 use->get().getType(), state, fields);
    rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(binding); });
  }
  for (auto member : members) {
    auto field = layout.referenceField(member.getMemberIndex());
    if (field) {
      for (auto* user : llvm::make_early_inc_range(member->getUsers()))
        rewriter.replaceOp(cast<LoadOp>(user), fields[*field]);
      continue;
    }
    auto position = layout.stateField(member.getMemberIndex());
    if (!state || !position) return failure();
    rewriter.setInsertionPoint(member);
    auto pointer = storagePointer(
        rewriter, member.getLoc(),
        {state, {StorageMember{std::int32_t(*position)}}}, member.getType());
    if (failed(pointer)) return failure();
    rewriter.replaceOp(member, *pointer);
  }
  SmallVector<Operation*> operations;
  SmallPtrSet<Operation*, 16> collected;
  SmallVector<Value> worklist{root};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (!visited.insert(value).second) continue;
    for (auto* operation : value.getUsers()) {
      if (!isa<ptr::ToPtrOp, ptr::FromPtrOp, StorageRefOp, MemberOp>(operation))
        continue;
      if (!collected.insert(operation).second) continue;
      operations.push_back(operation);
      for (auto result : operation->getResults()) worklist.push_back(result);
    }
  }
  for (auto* operation : llvm::reverse(operations))
    if (operation->use_empty()) rewriter.eraseOp(operation);
  return success();
}

namespace {

struct FieldBinding {
  Value value;
  Operation* initializer = nullptr;
};

class EnvironmentBinding {
 public:
  explicit EnvironmentBinding(Value root)
      : root_(root), record_(environmentType(root)), layout_(record_) {}

  [[nodiscard]] auto analyze() -> LogicalResult {
    if (!record_ || record_.getBody().empty()) return failure();
    if (!root_.getDefiningOp()->getDiscardableAttrDictionary().empty())
      return failure();
    if (layout_.referenceFields().empty()) return failure();
    bindings_.resize(layout_.referenceFields().size());
    auto uses = referenceEnvironmentUses(root_);
    if (failed(uses)) return failure();
    uses_ = std::move(*uses);
    auto scope = root_.getDefiningOp()
                     ->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    DominanceInfo dominance(scope);
    for (auto* use : uses_) {
      if (fieldIndex(use->get()) && !referenceFieldUse(*use, layout_)) continue;
      if (auto store = dyn_cast<StoreOp>(use->getOwner())) {
        if (use->getOperandNumber() != 1 ||
            !store->getDiscardableAttrDictionary().empty())
          return failure();
        auto index = referenceFieldUse(*use, layout_);
        if (!index || *index >= bindings_.size()) return failure();
        if (failed(updateBinding(*index, {store.getValue(), store}, dominance)))
          return failure();
        stores_.push_back(store);
        continue;
      }
      if (auto load = dyn_cast<LoadOp>(use->getOwner())) {
        if (use->getOperandNumber() != 0 ||
            !load->getDiscardableAttrDictionary().empty())
          return failure();
        continue;
      }
      auto parameter = targetParameter(*use);
      if (!parameter) return failure();
      if (isReadOnlyReferenceEnvironment(parameter)) continue;
      auto initializer = referenceEnvironmentInitializer(parameter);
      if (failed(initializer)) return failure();
      auto call = cast<func::CallOp>(use->getOwner());
      for (auto [index, source] : llvm::enumerate(initializer->parameters)) {
        if (source >= call.getNumOperands()) return failure();
        if (failed(updateBinding(index, {call.getOperand(source), call},
                                 dominance)))
          return failure();
      }
    }
    for (auto [field, binding] :
         llvm::zip(layout_.referenceFields(), bindings_)) {
      if (!binding.value || binding.value.getType() != record_.getBody()[field])
        return failure();
      auto plan = planStorageReference(root_.getContext(), binding.value);
      if (failed(plan)) return failure();
      for (const auto& address : plan->reference().alternatives)
        if (address.root == root_) return failure();
    }
    for (auto* use : uses_) {
      if (fieldIndex(use->get()) && !referenceFieldUse(*use, layout_)) continue;
      if (isa<StoreOp>(use->getOwner())) continue;
      for (const auto& binding : bindings_)
        if (!dominance.dominates(binding.initializer, use->getOwner()))
          return failure();
      for (const auto& binding : bindings_)
        if (!dominance.dominates(binding.value, use->getOwner()))
          return failure();
      auto call = dyn_cast<func::CallOp>(use->getOwner());
      if (!call) continue;
      for (auto [index, value] : llvm::enumerate(call.getOperands())) {
        if (index == use->getOperandNumber() ||
            !isa<PointerType>(value.getType()))
          continue;
        if (value == use->get()) continue;
        if (call.getCallee().empty()) return failure();
        auto plan = planStorageReference(root_.getContext(), value);
        if (failed(plan)) return failure();
      }
    }
    return success();
  }

  [[nodiscard]] auto apply(RewriterBase& rewriter) -> LogicalResult {
    Value state;
    if (auto type = layout_.stateType()) {
      rewriter.setInsertionPoint(root_.getDefiningOp());
      auto memory =
          MemRefType::get({}, type, MemRefLayoutAttrInterface(),
                          ptr::GenericSpaceAttr::get(root_.getContext()));
      state = memref::AllocaOp::create(rewriter, root_.getLoc(), memory);
    }
    SmallVector<Value> fields;
    for (const auto& binding : bindings_) fields.push_back(binding.value);
    for (auto store : stores_) rewriter.eraseOp(store);
    if (failed(bindReferenceEnvironment(root_, state, fields, rewriter)))
      return failure();
    if (root_.use_empty()) rewriter.eraseOp(root_.getDefiningOp());
    return success();
  }

 private:
  [[nodiscard]] auto updateBinding(unsigned index, FieldBinding incoming,
                                   const DominanceInfo& dominance)
      -> LogicalResult {
    auto& binding = bindings_[index];
    if (!binding.value ||
        dominance.dominates(binding.initializer, incoming.initializer)) {
      binding = incoming;
      return success();
    }
    return success(
        dominance.dominates(incoming.initializer, binding.initializer));
  }

  Value root_;
  ClassType record_;
  ReferenceRecordLayout layout_;
  SmallVector<FieldBinding> bindings_;
  SmallVector<OpOperand*> uses_;
  SmallVector<StoreOp> stores_;
};

}  // namespace

auto bindReferenceEnvironments(Operation* operation) -> LogicalResult {
  SmallVector<Value> roots;
  operation->walk([&](Operation* allocation) {
    if (isa<AllocaOp, memref::AllocaOp>(allocation))
      roots.push_back(allocation->getResult(0));
  });
  IRRewriter rewriter(operation->getContext());
  for (auto root : roots) {
    EnvironmentBinding binding(root);
    if (failed(binding.analyze())) continue;
    if (failed(binding.apply(rewriter))) return failure();
  }
  return success();
}

}  // namespace mlir::cxx
