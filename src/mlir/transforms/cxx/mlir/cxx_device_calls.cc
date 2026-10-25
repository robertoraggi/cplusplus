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
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/passes.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir {
namespace {

[[nodiscard]] auto isReferenceEnvironment(Type type) -> bool {
  auto fields = cxx::getRecordElementTypes(type);
  return llvm::any_of(fields,
                      [](Type field) { return isa<cxx::PointerType>(field); });
}

class ObjectStorageToMemRef final : public OpRewritePattern<cxx::AllocaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::AllocaOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto pointer = cast<cxx::PointerType>(op.getType());
    auto object = pointer.getElementType();
    if (isReferenceEnvironment(object)) return failure();
    if (pointer.getAddressSpace() || isa<cxx::PointerType>(object))
      return failure();
    if (!MemRefType::isValidElementType(object)) return failure();
    if (!op->getDiscardableAttrDictionary().empty()) return failure();
    if (op.getAlignment() < 0) return failure();
    if (op.getAlignment() && !llvm::isPowerOf2_64(op.getAlignment()))
      return failure();
    auto scope = op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (!isa_and_nonnull<FunctionOpInterface>(scope)) return failure();

    auto type = MemRefType::get({}, object, MemRefLayoutAttrInterface(),
                                ptr::GenericSpaceAttr::get(op.getContext()));
    auto alignment = op.getAlignment()
                         ? rewriter.getI64IntegerAttr(op.getAlignment())
                         : IntegerAttr();
    auto allocation =
        memref::AllocaOp::create(rewriter, op.getLoc(), type, alignment);
    auto address = cxx::storagePointer(rewriter, op.getLoc(), {allocation, {}});
    if (failed(address)) return failure();
    rewriter.replaceOp(op, *address);
    return success();
  }
};

template <class Op>
class AccessThroughReference final : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  auto matchAndRewrite(Op op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (succeeded(cxx::resolveStorageAddress(op.getAddr()))) return failure();
    auto plan = cxx::planStorageReference(op->getContext(), op.getAddr());
    if (failed(plan)) return failure();

    auto reference = plan->materialize(rewriter);
    if (reference.alternatives.size() == 1) {
      auto address = cxx::storagePointer(rewriter, op.getLoc(),
                                         reference.alternatives.front(),
                                         op.getAddr().getType());
      if (failed(address)) return failure();
      rewriter.modifyOpInPlace(op,
                               [&] { op.getAddrMutable().assign(*address); });
      return success();
    }
    SmallVector<scf::IfOp> branches;
    for (unsigned index = 1; index < reference.alternatives.size(); ++index) {
      auto alternative =
          arith::ConstantIntOp::create(rewriter, op.getLoc(), index, 32);
      auto match =
          arith::CmpIOp::create(rewriter, op.getLoc(), arith::CmpIPredicate::eq,
                                reference.selector, alternative);
      auto branch = scf::IfOp::create(rewriter, op.getLoc(),
                                      op->getResultTypes(), match, true);
      branches.push_back(branch);
      rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
      if (failed(emitAccess(op, reference.alternatives[index], rewriter)))
        return failure();
      rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    }
    if (failed(emitAccess(op, reference.alternatives.front(), rewriter)))
      return failure();
    if (op->getNumResults()) {
      for (auto branch : llvm::reverse(ArrayRef(branches).drop_front())) {
        rewriter.setInsertionPointAfter(branch);
        scf::YieldOp::create(rewriter, op.getLoc(), branch.getResults());
      }
    }
    rewriter.replaceOp(op, branches.front().getResults());
    return success();
  }

 private:
  [[nodiscard]] static auto emitAccess(Op op,
                                       const cxx::StorageAddress& address,
                                       PatternRewriter& rewriter)
      -> LogicalResult {
    auto pointer = cxx::storagePointer(rewriter, op.getLoc(), address,
                                       op.getAddr().getType());
    if (failed(pointer)) return failure();
    IRMapping mapping;
    mapping.map(op.getAddr(), *pointer);
    auto* access = rewriter.clone(*op, mapping);
    if (op->getNumResults())
      scf::YieldOp::create(rewriter, op.getLoc(), access->getResults());
    return success();
  }
};

[[nodiscard]] auto planReferenceEnvironment(Value pointer)
    -> FailureOr<SmallVector<cxx::StorageReferencePlan>> {
  auto environment = pointer.getDefiningOp<cxx::BindReferencesOp>();
  if (!environment) return failure();
  SmallVector<cxx::StorageReferencePlan> plans;
  for (auto value : environment.getFields()) {
    auto plan = cxx::planStorageReference(pointer.getContext(), value);
    if (failed(plan)) return failure();
    plans.push_back(std::move(*plan));
  }
  return plans;
}

struct ReferenceParameter {
  unsigned root;
  SmallVector<std::variant<cxx::StorageMember, unsigned>> path;
};

struct ReferenceSelection {
  unsigned selector;
  SmallVector<ReferenceParameter> alternatives;
};

using ReferenceArgument = std::variant<ReferenceParameter, ReferenceSelection>;

struct ReferenceEnvironment {
  SmallVector<ReferenceArgument> fields;
  std::optional<unsigned> state;
  bool initializer = false;
};

struct UnusedParameter {};

using ArgumentParameter =
    std::variant<unsigned, ReferenceParameter, ReferenceSelection,
                 ReferenceEnvironment, UnusedParameter>;

class CallSignature {
 public:
  [[nodiscard]] auto build(func::CallOp call, func::FuncOp callee,
                           OpBuilder& builder) -> LogicalResult {
    SmallVector<bool> initializers(call.getNumOperands(), false);
    SmallPtrSet<Operation*, 16> initialized;
    for (auto [index, value] : llvm::enumerate(call.getOperands())) {
      auto environment = value.getDefiningOp<cxx::BindReferencesOp>();
      if (!environment) continue;
      auto initializer =
          cxx::referenceEnvironmentInitializer(callee.getArgument(index));
      if (failed(initializer)) continue;
      bool matches = llvm::all_of(
          llvm::enumerate(initializer->parameters), [&](auto field) {
            return field.value() < call.getNumOperands() &&
                   environment.getFields()[field.index()] ==
                       call.getOperand(field.value());
          });
      if (!matches) continue;
      initializers[index] = true;
      for (auto store : initializer->stores) initialized.insert(store);
    }
    SmallVector<bool> unused;
    for (auto argument : callee.getArguments())
      unused.push_back(llvm::all_of(argument.getUses(), [&](OpOperand& use) {
        return initialized.contains(use.getOwner());
      }));
    SmallVector<std::optional<cxx::StorageReferencePlan>> plans;
    SmallVector<std::optional<SmallVector<cxx::StorageReferencePlan>>>
        environments;
    for (auto [position, value] : llvm::enumerate(call.getOperands())) {
      environments.emplace_back();
      if (unused[position]) {
        plans.emplace_back();
        continue;
      }
      if (!isa<cxx::PointerType>(value.getType())) {
        plans.emplace_back();
        continue;
      }
      auto environment = planReferenceEnvironment(value);
      if (succeeded(environment)) {
        if (!initializers[position] &&
            !cxx::isReadOnlyReferenceEnvironment(callee.getArgument(position)))
          return failure();
        environments.back().emplace(std::move(*environment));
        plans.emplace_back();
        continue;
      }
      auto plan = cxx::planStorageReference(builder.getContext(), value);
      if (failed(plan)) return failure();
      plans.emplace_back(std::move(*plan));
    }

    for (auto [position, input] :
         llvm::enumerate(llvm::zip(call.getOperands(), plans, environments))) {
      auto& [value, plan, environment] = input;
      if (unused[position]) {
        parameters_.push_back(UnusedParameter{});
        continue;
      }
      if (environment) {
        ReferenceEnvironment parameter;
        parameter.initializer = initializers[position];
        auto binding = value.getDefiningOp<cxx::BindReferencesOp>();
        if (auto state = binding.getState()) {
          auto found = llvm::find(operands_, state);
          parameter.state = unsigned(found - operands_.begin());
          if (found == operands_.end()) operands_.push_back(state);
        }
        for (auto& field : *environment)
          parameter.fields.push_back(
              addReference(field.materialize(builder), builder, call.getLoc()));
        parameters_.push_back(std::move(parameter));
        continue;
      }
      if (!plan) {
        parameters_.push_back(unsigned(operands_.size()));
        operands_.push_back(value);
        continue;
      }
      auto reference =
          addReference(plan->materialize(builder), builder, call.getLoc());
      std::visit(
          [&](auto parameter) { parameters_.push_back(std::move(parameter)); },
          std::move(reference));
    }
    return success();
  }

  [[nodiscard]] auto key(OpBuilder& builder) const -> ArrayAttr {
    SmallVector<Attribute> key;
    for (const auto& parameter : parameters_) {
      if (auto value = std::get_if<unsigned>(&parameter)) {
        key.push_back(TypeAttr::get(operands_[*value].getType()));
      } else if (std::holds_alternative<UnusedParameter>(parameter)) {
        key.push_back(builder.getUnitAttr());
      } else if (auto reference = std::get_if<ReferenceParameter>(&parameter)) {
        key.push_back(referenceKey(builder, *reference));
      } else if (auto environment =
                     std::get_if<ReferenceEnvironment>(&parameter)) {
        SmallVector<Attribute> fields;
        for (const auto& field : environment->fields)
          fields.push_back(referenceArgumentKey(builder, field));
        SmallVector<Attribute> shape{
            builder.getStringAttr("environment"),
            builder.getBoolAttr(environment->initializer),
            builder.getArrayAttr(fields)};
        if (environment->state) {
          shape.push_back(builder.getI32IntegerAttr(*environment->state));
          shape.push_back(
              TypeAttr::get(operands_[*environment->state].getType()));
        }
        key.push_back(builder.getArrayAttr(shape));
      } else {
        key.push_back(referenceArgumentKey(
            builder, std::get<ReferenceSelection>(parameter)));
      }
    }
    return builder.getArrayAttr(key);
  }

  [[nodiscard]] auto operands() const -> ValueRange { return operands_; }

  [[nodiscard]] auto apply(func::FuncOp function) const -> LogicalResult {
    auto& entry = function.getBody().front();
    auto originals = llvm::to_vector(entry.getArguments());
    SmallVector<Value> arguments;
    SmallVector<Type> types;
    for (auto operand : operands_) {
      types.push_back(operand.getType());
      arguments.push_back(
          entry.addArgument(operand.getType(), function.getLoc()));
    }
    IRRewriter builder(function.getContext());
    builder.setInsertionPointToStart(&entry);
    for (auto [original, parameter] : llvm::zip(originals, parameters_)) {
      builder.setInsertionPointToStart(&entry);
      if (auto value = std::get_if<unsigned>(&parameter)) {
        original.replaceAllUsesWith(arguments[*value]);
        continue;
      }
      if (std::holds_alternative<UnusedParameter>(parameter)) continue;
      if (auto environment = std::get_if<ReferenceEnvironment>(&parameter)) {
        if (environment->initializer) {
          auto initializer = cxx::referenceEnvironmentInitializer(original);
          if (failed(initializer))
            return function.emitError(
                "invalid reference environment initializer");
          for (auto store : initializer->stores) {
            auto member = store.getAddr().getDefiningOp<cxx::MemberOp>();
            store.erase();
            if (member && member->use_empty()) member.erase();
          }
          builder.setInsertionPointToStart(&entry);
        }
        auto record = cast<cxx::ClassType>(
            cast<cxx::PointerType>(original.getType()).getElementType());
        cxx::ReferenceRecordLayout layout(record);
        Value state =
            environment->state ? arguments[*environment->state] : Value();
        SmallVector<Value> fields;
        for (auto [index, field] : llvm::enumerate(environment->fields)) {
          auto value =
              referenceValue(builder, function.getLoc(), field, arguments,
                             record.getBody()[layout.referenceFields()[index]]);
          if (failed(value)) return failure();
          fields.push_back(*value);
        }
        if (failed(cxx::bindReferenceEnvironment(original, state, fields,
                                                 builder)))
          return function.emitError("invalid reference environment uses");
        continue;
      }
      auto reference =
          std::holds_alternative<ReferenceParameter>(parameter)
              ? ReferenceArgument(std::get<ReferenceParameter>(parameter))
              : ReferenceArgument(std::get<ReferenceSelection>(parameter));
      auto pointer = referenceValue(builder, function.getLoc(), reference,
                                    arguments, original.getType());
      if (failed(pointer))
        return function.emitError("reference parameter has an invalid path");
      original.replaceAllUsesWith(*pointer);
    }
    llvm::BitVector erased(entry.getNumArguments());
    erased.set(0, originals.size());
    entry.eraseArguments(erased);
    function.setArgAttrsAttr({});
    function.setFunctionType(
        builder.getFunctionType(types, function.getResultTypes()));
    for (auto [index, operand] : llvm::enumerate(operands_)) {
      auto binding = cxx::deviceResourceBinding(operand);
      if (!binding) continue;
      function.setArgAttr(index, cxx::kResourceBindingAttrName, binding);
      if (auto resource = cxx::deviceResourceIndex(operand))
        function.setArgAttr(index, cxx::kResourceIndexAttrName, resource);
      function.setArgAttr(
          index, cxx::kResourceReadOnlyAttrName,
          builder.getBoolAttr(cxx::isReadOnlyDeviceResource(operand)));
      function.setArgAttr(
          index, cxx::kResourceUnsignedAttrName,
          builder.getBoolAttr(cxx::hasUnsignedDeviceElements(operand)));
    }

    return success();
  }

 private:
  [[nodiscard]] auto addReference(const cxx::StorageReference& reference,
                                  OpBuilder& builder, Location loc)
      -> ReferenceArgument {
    if (reference.alternatives.size() == 1)
      return addReference(reference.alternatives.front(), builder, loc);
    ReferenceSelection selection{unsigned(operands_.size()), {}};
    operands_.push_back(reference.selector);
    for (const auto& address : reference.alternatives)
      selection.alternatives.push_back(addReference(address, builder, loc));
    return selection;
  }

  [[nodiscard]] auto referenceArgumentKey(
      OpBuilder& builder, const ReferenceArgument& argument) const
      -> ArrayAttr {
    if (auto reference = std::get_if<ReferenceParameter>(&argument))
      return referenceKey(builder, *reference);
    const auto& selection = std::get<ReferenceSelection>(argument);
    SmallVector<Attribute> alternatives;
    for (const auto& reference : selection.alternatives)
      alternatives.push_back(referenceKey(builder, reference));
    return builder.getArrayAttr({builder.getI32IntegerAttr(selection.selector),
                                 builder.getArrayAttr(alternatives)});
  }

  [[nodiscard]] static auto referenceValue(OpBuilder& builder, Location loc,
                                           const ReferenceArgument& argument,
                                           ValueRange arguments, Type view)
      -> FailureOr<Value> {
    if (auto reference = std::get_if<ReferenceParameter>(&argument))
      return referenceValue(builder, loc, *reference, arguments, view);
    const auto& selection = std::get<ReferenceSelection>(argument);
    Value result;
    for (auto [index, reference] : llvm::enumerate(selection.alternatives)) {
      auto value = referenceValue(builder, loc, reference, arguments, view);
      if (failed(value)) return failure();
      if (!index) {
        result = *value;
        continue;
      }
      auto position = arith::ConstantIntOp::create(builder, loc, index, 32);
      auto condition =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                arguments[selection.selector], position);
      result = arith::SelectOp::create(builder, loc, condition, *value, result);
    }
    return result;
  }

  [[nodiscard]] auto addReference(const cxx::StorageAddress& address,
                                  OpBuilder& builder, Location loc)
      -> ReferenceParameter {
    auto found = llvm::find(operands_, address.root);
    auto root = unsigned(found - operands_.begin());
    if (found == operands_.end()) operands_.push_back(address.root);
    ReferenceParameter reference{root, {}};
    for (const auto& index : address.path) {
      if (auto member = std::get_if<cxx::StorageMember>(&index)) {
        reference.path.push_back(*member);
      } else {
        reference.path.push_back(unsigned(operands_.size()));
        operands_.push_back(cxx::storageIndex(
            builder, loc, std::get<cxx::StorageElement>(index)));
      }
    }
    return reference;
  }

  [[nodiscard]] auto referenceKey(OpBuilder& builder,
                                  const ReferenceParameter& reference) const
      -> ArrayAttr {
    SmallVector<Attribute> path;
    for (const auto& index : reference.path) {
      if (auto member = std::get_if<cxx::StorageMember>(&index))
        path.push_back(builder.getI32IntegerAttr(member->position));
      else
        path.push_back(builder.getUnitAttr());
    }
    SmallVector<Attribute> key{
        builder.getI32IntegerAttr(reference.root),
        TypeAttr::get(operands_[reference.root].getType()),
        builder.getArrayAttr(path)};
    if (auto binding = cxx::deviceResourceBinding(operands_[reference.root])) {
      key.push_back(binding);
      if (auto resource = cxx::deviceResourceIndex(operands_[reference.root]))
        key.push_back(resource);
      key.push_back(builder.getBoolAttr(
          cxx::isReadOnlyDeviceResource(operands_[reference.root])));
      key.push_back(builder.getBoolAttr(
          cxx::hasUnsignedDeviceElements(operands_[reference.root])));
    }
    return builder.getArrayAttr(key);
  }

  [[nodiscard]] static auto referenceValue(OpBuilder& builder, Location loc,
                                           const ReferenceParameter& reference,
                                           ValueRange arguments, Type view)
      -> FailureOr<Value> {
    cxx::StorageAddress address{arguments[reference.root], {}};
    for (const auto& index : reference.path) {
      if (auto member = std::get_if<cxx::StorageMember>(&index))
        address.path.push_back(*member);
      else
        address.path.push_back(
            cxx::StorageElement{{arguments[std::get<unsigned>(index)]}});
    }
    return cxx::storagePointer(builder, loc, address, view);
  }

  SmallVector<ArgumentParameter> parameters_;
  SmallVector<Value> operands_;
};

class ReferenceResults {
 public:
  [[nodiscard]] auto apply(func::FuncOp function) -> LogicalResult {
    struct ReturnSite {
      func::ReturnOp operation;
      cxx::StorageReferencePlan plan;
    };
    SmallVector<ReturnSite> sites;
    auto result = function.walk([&](func::ReturnOp ret) {
      auto plan =
          cxx::planStorageReference(function.getContext(), ret.getOperand(0));
      if (failed(plan)) {
        ret.emitError("device reference result has no storage provenance");
        return WalkResult::interrupt();
      }
      for (const auto& address : plan->reference().alternatives) {
        auto argument = dyn_cast<BlockArgument>(address.root);
        if (!argument || argument.getOwner() != &function.getBody().front()) {
          ret.emitError(
              "device reference result leaves function-local storage");
          return WalkResult::interrupt();
        }
      }
      layout_.include(plan->reference());
      sites.push_back({ret, std::move(*plan)});
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
    if (sites.empty())
      return function.emitError(
          "reference-returning device function has no return");
    for (auto& site : sites) {
      OpBuilder builder(site.operation);
      auto value = layout_.pack(builder, site.operation.getLoc(),
                                site.plan.materialize(builder));
      if (failed(value))
        return site.operation.emitError("invalid reference result layout");
      site.operation.getOperandsMutable().assign(ValueRange{*value});
    }
    function_ = function;
    function.setFunctionType(
        FunctionType::get(function.getContext(), function.getArgumentTypes(),
                          {layout_.type(function.getContext())}));
    return success();
  }

  [[nodiscard]] auto rebuild(OpBuilder& builder, Location loc, Value result,
                             ValueRange arguments, Type view) const
      -> FailureOr<Value> {
    IRMapping roots;
    auto function = function_;
    for (auto [parameter, argument] :
         llvm::zip(function.getArguments(), arguments))
      roots.map(parameter, argument);
    return cxx::storageReferencePointer(
        builder, loc, layout_.unpack(builder, loc, result, &roots), view);
  }

 private:
  func::FuncOp function_;
  cxx::StorageReferenceLayout layout_;
};

class ForwardIfReferences final : public OpRewritePattern<scf::IfOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(scf::IfOp branch, PatternRewriter& rewriter) const
      -> LogicalResult override {
    struct ReferenceResult {
      unsigned position;
      Type pointer;
      cxx::StorageReferenceLayout layout;
      SmallVector<cxx::StorageReferencePlan> plans;
    };
    SmallVector<ReferenceResult> references;
    DominanceInfo dominance(
        branch->getParentWithTrait<OpTrait::AutomaticAllocationScope>());
    for (auto [index, result] : llvm::enumerate(branch.getResults())) {
      if (!isa<cxx::PointerType>(result.getType())) continue;
      ReferenceResult reference{unsigned(index), result.getType(), {}, {}};
      for (auto& region : branch->getRegions()) {
        if (region.empty()) return failure();
        auto yield = cast<scf::YieldOp>(region.front().getTerminator());
        auto plan = cxx::planStorageReference(branch.getContext(),
                                              yield.getOperand(index));
        if (failed(plan)) return failure();
        for (const auto& address : plan->reference().alternatives)
          if (!dominance.dominates(address.root, branch.getOperation()))
            return failure();
        reference.layout.include(plan->reference());
        reference.plans.push_back(std::move(*plan));
      }
      references.push_back(std::move(reference));
    }
    if (references.empty()) return failure();

    for (auto& reference : references) {
      for (auto [region, plan] :
           llvm::zip(branch->getRegions(), reference.plans)) {
        auto yield = cast<scf::YieldOp>(region.front().getTerminator());
        rewriter.setInsertionPoint(yield);
        auto descriptor = reference.layout.pack(rewriter, yield.getLoc(),
                                                plan.materialize(rewriter));
        if (failed(descriptor)) return failure();
        rewriter.modifyOpInPlace(
            yield, [&] { yield->setOperand(reference.position, *descriptor); });
      }
      auto result = branch.getResult(reference.position);
      auto uses = llvm::to_vector(llvm::map_range(
          result.getUses(), [](OpOperand& use) { return &use; }));
      rewriter.modifyOpInPlace(branch, [&] {
        result.setType(reference.layout.type(branch.getContext()));
      });
      rewriter.setInsertionPointAfter(branch);
      auto pointer = cxx::storageReferencePointer(
          rewriter, branch.getLoc(),
          reference.layout.unpack(rewriter, branch.getLoc(), result),
          reference.pointer);
      if (failed(pointer)) return failure();
      for (auto* use : uses) use->set(*pointer);
    }
    return success();
  }
};

class ForwardWhileReferences final : public OpRewritePattern<scf::WhileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(scf::WhileOp loop, PatternRewriter& rewriter) const
      -> LogicalResult override {
    struct Forwarded {
      unsigned position;
      Type pointer;
      cxx::StorageReferencePlan plan;
      cxx::StorageReferenceLayout layout;
    };
    auto condition =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    SmallVector<Forwarded> references;
    DominanceInfo dominance(
        loop->getParentWithTrait<OpTrait::AutomaticAllocationScope>());
    for (auto [index, value] : llvm::enumerate(condition.getArgs())) {
      if (!isa<cxx::PointerType>(value.getType())) continue;
      auto plan = cxx::planStorageReference(loop.getContext(), value);
      if (failed(plan)) return failure();
      for (const auto& address : plan->reference().alternatives)
        if (!dominance.dominates(address.root, loop.getOperation()))
          return failure();
      cxx::StorageReferenceLayout layout;
      layout.include(plan->reference());
      references.push_back({unsigned(index), value.getType(), std::move(*plan),
                            std::move(layout)});
    }
    if (references.empty()) return failure();
    for (auto& reference : references) {
      rewriter.setInsertionPoint(condition);
      auto descriptor = reference.layout.pack(
          rewriter, loop.getLoc(), reference.plan.materialize(rewriter));
      if (failed(descriptor)) return failure();
      condition.getArgsMutable()
          .slice(reference.position, 1)
          .assign(*descriptor);
      auto argument = loop.getAfter().front().getArgument(reference.position);
      auto uses = llvm::to_vector(llvm::map_range(
          argument.getUses(), [](OpOperand& use) { return &use; }));
      argument.setType(descriptor->getType());
      rewriter.setInsertionPointToStart(&loop.getAfter().front());
      auto pointer = cxx::storageReferencePointer(
          rewriter, loop.getLoc(),
          reference.layout.unpack(rewriter, loop.getLoc(), argument),
          reference.pointer);
      if (failed(pointer)) return failure();
      for (auto* use : uses) use->set(*pointer);

      auto result = loop.getResult(reference.position);
      uses = llvm::to_vector(llvm::map_range(
          result.getUses(), [](OpOperand& use) { return &use; }));
      result.setType(descriptor->getType());
      if (uses.empty()) continue;
      rewriter.setInsertionPointAfter(loop);
      pointer = cxx::storageReferencePointer(
          rewriter, loop.getLoc(),
          reference.layout.unpack(rewriter, loop.getLoc(), result),
          reference.pointer);
      if (failed(pointer)) return failure();
      for (auto* use : uses) use->set(*pointer);
    }
    return success();
  }
};

[[nodiscard]] auto legalizeForwardedReferences(Operation* operation)
    -> LogicalResult {
  RewritePatternSet patterns(operation->getContext());
  patterns.add<ForwardIfReferences, ForwardWhileReferences>(
      operation->getContext());
  if (failed(applyPatternsGreedily(operation, std::move(patterns))))
    return failure();
  return cxx::bindReferenceEnvironments(operation);
}

struct IncomingEdge {
  BranchOpInterface branch;
  unsigned successor;
};

class PointerBlockArguments {
  static constexpr unsigned kLayoutAttempts = 8;

  struct Candidate {
    BlockArgument argument;
    SmallVector<IncomingEdge> edges;
    cxx::StorageReferenceLayout layout;
  };

  struct Speculation {
    BlockArgument descriptor;
    Operation* first;
    Value pointer;
  };

 public:
  explicit PointerBlockArguments(Operation* scope) : scope_(scope) {}

  [[nodiscard]] auto run() -> LogicalResult {
    for (;;) {
      auto pending = candidates();
      if (pending.empty()) return success();
      auto legalized = legalizeOne(pending);
      if (failed(legalized)) return failure();
      if (*legalized) continue;
      legalized = legalize(pending);
      if (failed(legalized)) return failure();
      if (!*legalized) return success();
    }
  }

 private:
  [[nodiscard]] auto candidates() const -> SmallVector<Candidate> {
    SmallVector<Candidate> result;
    scope_->walk([&](Block* block) {
      if (block->isEntryBlock()) return;
      for (auto argument : block->getArguments()) {
        if (!isa<cxx::PointerType>(argument.getType())) continue;
        auto edges = incomingEdges(block);
        if (failed(edges)) continue;
        Candidate candidate{argument, std::move(*edges), {}};
        for (auto edge : candidate.edges)
          if (auto incomingPlan = plan(edge, argument); succeeded(incomingPlan))
            candidate.layout.include(incomingPlan->reference());
        if (!candidate.layout.empty()) result.push_back(std::move(candidate));
      }
    });
    return result;
  }

  [[nodiscard]] auto legalizeOne(MutableArrayRef<Candidate> pending)
      -> FailureOr<bool> {
    for (auto& candidate : pending) {
      auto legalized = legalize(MutableArrayRef<Candidate>(candidate));
      if (failed(legalized) || *legalized) return legalized;
    }
    return false;
  }

  [[nodiscard]] static auto incomingEdges(Block* block)
      -> FailureOr<SmallVector<IncomingEdge>> {
    SmallVector<IncomingEdge> edges;
    for (auto it = block->pred_begin(); it != block->pred_end(); ++it) {
      auto branch = dyn_cast<BranchOpInterface>((*it)->getTerminator());
      if (!branch) return failure();
      if (branch.getSuccessorOperands(it.getSuccessorIndex())
              .getProducedOperandCount())
        return failure();
      edges.push_back({branch, it.getSuccessorIndex()});
    }
    return edges;
  }

  [[nodiscard]] static auto incoming(IncomingEdge edge, BlockArgument argument)
      -> Value {
    return edge.branch.getSuccessorOperands(
        edge.successor)[argument.getArgNumber()];
  }

  [[nodiscard]] static auto plan(IncomingEdge edge, BlockArgument argument)
      -> FailureOr<cxx::StorageReferencePlan> {
    return cxx::planStorageReference(argument.getContext(),
                                     incoming(edge, argument));
  }

  [[nodiscard]] static auto speculate(const Candidate& candidate)
      -> FailureOr<Speculation> {
    auto argument = candidate.argument;
    auto* block = argument.getOwner();
    Speculation speculation{
        block->addArgument(candidate.layout.type(argument.getContext()),
                           argument.getLoc()),
        &block->front(), Value()};
    OpBuilder builder(block, block->begin());
    auto pointer = cxx::storageReferencePointer(
        builder, argument.getLoc(),
        candidate.layout.unpack(builder, argument.getLoc(),
                                speculation.descriptor),
        argument.getType());
    if (failed(pointer)) {
      revert(candidate, speculation);
      return failure();
    }
    speculation.pointer = *pointer;
    argument.replaceAllUsesWith(*pointer);
    return speculation;
  }

  static void revert(const Candidate& candidate,
                     const Speculation& speculation) {
    auto* block = candidate.argument.getOwner();
    if (auto pointer = speculation.pointer)
      pointer.replaceAllUsesWith(candidate.argument);
    SmallVector<Operation*> materialized;
    for (auto& operation : *block) {
      if (&operation == speculation.first) break;
      materialized.push_back(&operation);
    }
    for (auto* operation : llvm::reverse(materialized)) operation->erase();
    block->eraseArgument(speculation.descriptor.getArgNumber());
  }

  [[nodiscard]] static auto legalize(MutableArrayRef<Candidate> group)
      -> FailureOr<bool> {
    for (unsigned attempt = 0; attempt < kLayoutAttempts; ++attempt) {
      SmallVector<Speculation> speculations;
      for (const auto& candidate : group) {
        auto speculation = speculate(candidate);
        if (failed(speculation)) {
          revertAll(group, speculations);
          return false;
        }
        speculations.push_back(*speculation);
      }
      SmallVector<SmallVector<cxx::StorageReferencePlan>> plans;
      bool resolved = true;
      bool covered = true;
      for (auto& candidate : group) {
        auto& incomingPlans = plans.emplace_back();
        for (auto edge : candidate.edges) {
          auto incomingPlan = plan(edge, candidate.argument);
          if (failed(incomingPlan)) {
            resolved = false;
            break;
          }
          if (!candidate.layout.covers(incomingPlan->reference())) {
            covered = false;
            candidate.layout.include(incomingPlan->reference());
          }
          incomingPlans.push_back(std::move(*incomingPlan));
        }
        if (!resolved) break;
      }
      if (resolved && covered) return connect(group, plans);
      revertAll(group, speculations);
      if (!resolved) return false;
    }
    return false;
  }

  static void revertAll(ArrayRef<Candidate> group,
                        ArrayRef<Speculation> speculations) {
    for (auto [candidate, speculation] : llvm::reverse(
             llvm::zip(group.take_front(speculations.size()), speculations)))
      revert(candidate, speculation);
  }

  [[nodiscard]] static auto connect(
      ArrayRef<Candidate> group,
      MutableArrayRef<SmallVector<cxx::StorageReferencePlan>> plans)
      -> FailureOr<bool> {
    for (auto [candidate, incomingPlans] : llvm::zip(group, plans)) {
      for (auto [edge, incomingPlan] :
           llvm::zip(candidate.edges, incomingPlans)) {
        auto branch = edge.branch;
        OpBuilder builder(branch);
        auto packed = candidate.layout.pack(builder, branch.getLoc(),
                                            incomingPlan.materialize(builder));
        if (failed(packed))
          return branch->emitError(
              "device code cannot represent this pointer in a loop");
        branch.getSuccessorOperands(edge.successor).append(*packed);
      }
    }
    SmallVector<BlockArgument> originals;
    for (const auto& candidate : group) originals.push_back(candidate.argument);
    llvm::sort(originals, [](BlockArgument left, BlockArgument right) {
      return left.getArgNumber() > right.getArgNumber();
    });
    for (auto original : originals) {
      auto* block = original.getOwner();
      for (auto it = block->pred_begin(); it != block->pred_end(); ++it)
        cast<BranchOpInterface>((*it)->getTerminator())
            .getSuccessorOperands(it.getSuccessorIndex())
            .erase(original.getArgNumber());
      block->eraseArgument(original.getArgNumber());
    }
    return true;
  }

  Operation* scope_;
};

class LocalReferenceCalls {
  enum class State { Preparing, Ready };

  struct Specialization {
    func::FuncOp function;
    State state = State::Preparing;
    std::optional<ReferenceResults> results;
  };

  struct PendingCall {
    func::CallOp operation;
    CallSignature signature;
    Specialization* target;
  };

  using WorkItem = std::variant<func::CallOp, PendingCall, Specialization*>;

 public:
  explicit LocalReferenceCalls(Operation* scope)
      : scope_(scope), symbols_(scope) {}

  [[nodiscard]] auto run() -> LogicalResult {
    SmallVector<WorkItem> worklist;
    enqueueCalls(scope_, worklist);
    while (!worklist.empty()) {
      auto item = worklist.pop_back_val();
      if (auto call = std::get_if<func::CallOp>(&item)) {
        if (failed(prepare(*call, worklist))) return failure();
      } else if (auto call = std::get_if<PendingCall>(&item)) {
        if (failed(complete(*call))) return failure();
      } else {
        auto target = std::get<Specialization*>(item);
        auto results = target->function.getResultTypes();
        if (results.size() == 1 && isa<cxx::PointerType>(results.front())) {
          target->results.emplace();
          if (failed(target->results->apply(target->function)))
            return failure();
        }
        target->state = State::Ready;
      }
    }
    return success();
  }

 private:
  void enqueueCalls(Operation* operation, SmallVectorImpl<WorkItem>& worklist) {
    SmallVector<func::CallOp> calls;
    operation->walk([&](func::CallOp call) {
      if (call->getParentWithTrait<OpTrait::SymbolTable>() == scope_)
        calls.push_back(call);
    });
    for (auto call : llvm::reverse(calls)) worklist.emplace_back(call);
  }

  [[nodiscard]] auto prepare(func::CallOp call,
                             SmallVectorImpl<WorkItem>& worklist)
      -> LogicalResult {
    bool pointerArguments = llvm::any_of(call.getOperandTypes(), [](Type type) {
      return isa<cxx::PointerType>(type);
    });
    bool pointerResult = call.getNumResults() == 1 &&
                         isa<cxx::PointerType>(call.getResultTypes().front());
    if (!pointerArguments && !pointerResult) return success();
    auto allocationScope =
        call->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (allocationScope &&
        failed(cxx::bindReferenceEnvironments(allocationScope)))
      return failure();
    auto callee = symbols_.lookup<func::FuncOp>(call.getCallee());
    if (!callee || callee.isExternal()) return success();
    OpBuilder builder(call);
    CallSignature signature;
    if (failed(signature.build(call, callee, builder))) return success();

    auto key = std::make_pair(callee.getOperation(), signature.key(builder));
    auto found = specializations_.find(key);
    if (found != specializations_.end()) {
      auto target = found->second.get();
      if (target->state == State::Preparing)
        return call.emitError("recursive device calls are not supported");
      worklist.emplace_back(PendingCall{call, std::move(signature), target});
      return success();
    }

    auto specialization = std::make_unique<Specialization>();
    specialization->function = cast<func::FuncOp>(callee->clone());
    specialization->function.setPrivate();
    symbols_.insert(specialization->function);
    if (failed(signature.apply(specialization->function))) return failure();
    if (failed(legalizeForwardedReferences(specialization->function)))
      return failure();
    auto target = specialization.get();
    specializations_.try_emplace(key, std::move(specialization));
    worklist.emplace_back(PendingCall{call, std::move(signature), target});
    worklist.emplace_back(target);
    enqueueCalls(target->function, worklist);
    return success();
  }

  [[nodiscard]] auto complete(PendingCall& pending) -> LogicalResult {
    auto call = pending.operation;
    auto target = pending.target;
    OpBuilder builder(call);
    auto replacement = func::CallOp::create(
        builder, call.getLoc(), target->function.getResultTypes(),
        target->function.getSymName(), pending.signature.operands());
    if (target->results) {
      auto pointer = target->results->rebuild(
          builder, call.getLoc(), replacement.getResult(0),
          pending.signature.operands(), call.getResult(0).getType());
      if (failed(pointer))
        return call.emitError("device reference result has an invalid path");
      call.getResult(0).replaceAllUsesWith(*pointer);
    } else {
      call.replaceAllUsesWith(replacement.getResults());
    }
    call.erase();
    return success();
  }

  Operation* scope_;
  SymbolTable symbols_;
  DenseMap<std::pair<Operation*, ArrayAttr>, std::unique_ptr<Specialization>>
      specializations_;
};

class LegalizeCxxDeviceCallsPass final
    : public PassWrapper<LegalizeCxxDeviceCallsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeCxxDeviceCallsPass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "legalize-cxx-device-calls";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Pass local references as whole storage roots and subobject indices";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, func::FuncDialect, memref::MemRefDialect,
                    ptr::PtrDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet roots(&getContext());
    roots.add<ObjectStorageToMemRef>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(roots))))
      return signalPassFailure();

    if (failed(PointerBlockArguments(getOperation()).run()))
      return signalPassFailure();
    if (failed(legalizeForwardedReferences(getOperation())))
      return signalPassFailure();
    auto result = getOperation().walk([&](Operation* scope) {
      if (!scope->hasTrait<OpTrait::SymbolTable>())
        return WalkResult::advance();
      return succeeded(LocalReferenceCalls(scope).run())
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (result.wasInterrupted()) return signalPassFailure();

    RewritePatternSet accesses(&getContext());
    accesses.add<AccessThroughReference<cxx::LoadOp>,
                 AccessThroughReference<cxx::StoreOp>,
                 AccessThroughReference<cxx::AtomicLoadOp>,
                 AccessThroughReference<cxx::AtomicStoreOp>,
                 AccessThroughReference<cxx::AtomicRMWOp>,
                 AccessThroughReference<cxx::AtomicCompareExchangeOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(accesses))))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createLegalizeCxxDeviceCallsPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::LegalizeCxxDeviceCallsPass>();
}

void cxx::registerLegalizeCxxDeviceCallsPass() {
  mlir::registerPass(createLegalizeCxxDeviceCallsPass);
}
