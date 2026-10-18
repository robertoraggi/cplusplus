#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/mlir_source_region.h>
#include <cxx/mlir/passes.h>
#include <cxx/time_trace.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

namespace cxx::ir {
namespace {
class SourceRegionAnalysis {
 public:
  SourceRegionAnalysis(llvm::ArrayRef<mlir::Operation*> operations,
                       llvm::ArrayRef<mlir::Block*> continuations,
                       mlir::DominanceInfo& dominance,
                       mlir::Operation* iterationEntry = nullptr)
      : operations_(operations),
        continuations_(continuations),
        continuationSet_(continuations.begin(), continuations.end()),
        dominance_(dominance),
        iterationEntry_(iterationEntry) {}

  [[nodiscard]] auto run() -> SourceRegionPreflight {
    if (operations_.empty()) return std::string("empty");
    if (continuations_.empty()) return std::string("missing-continuation");
    for (auto* block : continuations_) {
      if (!block || block->empty()) return std::string("missing-continuation");
    }
    region_ = continuations_.front()->getParent();
    for (auto* block : continuations_) {
      if (block->getParent() != region_) return std::string("escaping-edge");
    }
    if (!mlir::isa<mlir::cxx::FuncOp>(region_->getParentOp()))
      return std::string("not-function-body");
    for (auto* op : operations_) {
      if (op->getParentRegion() != region_) return std::string("nested-region");
      if (!members_.insert(op).second)
        return std::string("duplicate-operation");
      std::string reason;
      op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* nested) {
        if (nested != op &&
            mlir::isa<mlir::RegionBranchTerminatorOpInterface>(nested) &&
            mlir::isa<mlir::RegionBranchOpInterface>(nested->getParentOp()))
          return mlir::WalkResult::advance();
        reason = checkOperation(nested);
        return reason.empty() ? mlir::WalkResult::advance()
                              : mlir::WalkResult::interrupt();
      });
      if (!reason.empty()) return reason;
    }
    if (auto reason = collectBoundaries(); !reason.empty()) return reason;
    if (!plan_.entry) return std::string("missing-entry");
    if (plan_.exits.empty()) return std::string("no-normal-exit");
    for (auto range : plan_.ranges) {
      if (range.first != plan_.entry && range.first->getPrevNode())
        return std::string("partial-interior-block");
      if (!dominance_.dominates(plan_.entry, range.first))
        return std::string("entry-dominance");
    }
    for (auto* op : operations_) {
      if (!iterationEntry_ &&
          llvm::is_contained(op->getSuccessors(), plan_.entry->getBlock()))
        return std::string("entry-backedge");
    }
    if (auto reason = collectValues(); !reason.empty()) return reason;
    return std::move(plan_);
  }

 private:
  [[nodiscard]] auto checkOperation(mlir::Operation* op) const -> std::string {
    if (op->getNumRegions()) {
      if (!mlir::isa<mlir::RegionBranchOpInterface>(op))
        return "unsupported-region";
    }
    if (mlir::isa<mlir::cxx::AllocaOp, mlir::cxx::DynAllocaOp>(op))
      return "storage-movement";
    if (mlir::isa<mlir::cxx::InvokeOp, mlir::cxx::LandingPadOp,
                  mlir::cxx::ResumeOp>(op))
      return "unwind";
    if (mlir::isa<mlir::cxx::GotoOp, mlir::cxx::LabelOp,
                  mlir::cxx::LabelAddressOp, mlir::cxx::IndirectGotoOp,
                  mlir::cxx::AsmGotoOp, mlir::cxx::CleanupBranchOp>(op))
      return "nonlocal-transfer";
    if (op->hasTrait<mlir::OpTrait::IsTerminator>() || op->getNumSuccessors()) {
      if (!mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp,
                     mlir::cf::SwitchOp>(op))
        return "unsupported-terminator";
    }
    return {};
  }

  [[nodiscard]] auto addEntry(mlir::Operation* op) -> bool {
    if (plan_.entry && plan_.entry != op) return false;
    plan_.entry = op;
    return true;
  }

  [[nodiscard]] auto addExit(mlir::Operation* from, mlir::Operation* to,
                             std::optional<unsigned> successor) -> std::string {
    if (!continuationSet_.contains(to->getBlock())) return "escaping-edge";
    auto [target, inserted] = exitTargets_.try_emplace(to->getBlock(), to);
    if (!inserted && target->second != to) return "multiple-continuations";
    plan_.exits.push_back({from, successor, to});
    return {};
  }

  [[nodiscard]] auto collectBoundaries() -> std::string {
    llvm::DenseSet<mlir::Block*> blocks;
    for (auto* op : operations_) {
      auto* block = op->getBlock();
      if (!dominance_.isReachableFromEntry(block)) return "unreachable-block";
      auto* previous = op->getPrevNode();
      if (!members_.contains(previous)) {
        if (!blocks.insert(block).second) return "discontiguous-block";
        auto* last = op;
        while (members_.contains(last->getNextNode()))
          last = last->getNextNode();
        plan_.ranges.push_back({op, last});
        if (previous || block->isEntryBlock()) {
          if (!addEntry(op)) return "multiple-entry";
        }
        if (!previous) {
          for (auto* predecessor : block->getPredecessors()) {
            if (!members_.contains(predecessor->getTerminator()) &&
                !addEntry(op))
              return "multiple-entry";
          }
        }
      }
      if (auto* next = op->getNextNode()) {
        if (!members_.contains(next)) {
          auto reason = addExit(op, next, std::nullopt);
          if (!reason.empty()) return reason;
        }
      }
      for (unsigned index = 0; index < op->getNumSuccessors(); ++index) {
        auto* successor = op->getSuccessor(index);
        if (successor->getParent() != region_ || successor->empty())
          return "escaping-edge";
        if (members_.contains(&successor->front()) &&
            &successor->front() != iterationEntry_)
          continue;
        auto reason = addExit(op, &successor->front(), index);
        if (!reason.empty()) return reason;
        auto branch = mlir::cast<mlir::BranchOpInterface>(op);
        auto operands = branch.getSuccessorOperands(index);
        if (operands.getProducedOperandCount())
          return "produced-successor-value";
        if (operands.size() != successor->getNumArguments())
          return "exit-argument-count";
        for (auto [operand, argument] : llvm::zip(
                 operands.getForwardedOperands(), successor->getArguments())) {
          if (operand.getType() != argument.getType())
            return "exit-argument-type";
        }
      }
    }
    return {};
  }

  [[nodiscard]] auto isInternal(mlir::Value value) const -> bool {
    if (auto* definition = value.getDefiningOp())
      return members_.contains(definition);
    auto argument = mlir::cast<mlir::BlockArgument>(value);
    auto* block = argument.getOwner();
    return block != plan_.entry->getBlock() && !block->empty() &&
           members_.contains(&block->front());
  }

  [[nodiscard]] auto escapes(mlir::Value value) const -> bool {
    return llvm::any_of(value.getUsers(), [&](mlir::Operation* user) {
      return !members_.contains(region_->findAncestorOpInRegion(*user));
    });
  }

  [[nodiscard]] auto collectValues() -> std::string {
    llvm::SetVector<mlir::Value> captures;
    llvm::SetVector<mlir::Value> liveOuts;
    for (auto* op : operations_) {
      llvm::SetVector<mlir::Value> operands;
      operands.insert(op->getOperands().begin(), op->getOperands().end());
      mlir::getUsedValuesDefinedAbove(op->getRegions(), operands);
      for (auto operand : operands) {
        if (!isInternal(operand)) captures.insert(operand);
      }
      for (auto result : op->getResults()) {
        if (escapes(result)) liveOuts.insert(result);
      }
    }
    for (auto range : plan_.ranges) {
      for (auto argument : range.first->getBlock()->getArguments()) {
        if (isInternal(argument) && escapes(argument))
          liveOuts.insert(argument);
      }
    }
    for (auto capture : captures) {
      if (!dominance_.properlyDominates(capture, plan_.entry))
        return "capture-dominance";
    }
    for (auto value : liveOuts) {
      for (auto exit : plan_.exits) {
        if (value.getDefiningOp() == exit.operation) continue;
        if (!dominance_.properlyDominates(value, exit.operation))
          return "live-out-dominance";
      }
      for (auto* user : value.getUsers()) {
        auto* ancestor = region_->findAncestorOpInRegion(*user);
        if (members_.contains(ancestor)) continue;
        if (!ancestor || !dominance_.dominates(plan_.entry, ancestor))
          return "external-use-dominance";
      }
    }
    plan_.captures.assign(captures.begin(), captures.end());
    plan_.liveOuts.assign(liveOuts.begin(), liveOuts.end());
    return {};
  }

  llvm::ArrayRef<mlir::Operation*> operations_;
  llvm::ArrayRef<mlir::Block*> continuations_;
  llvm::DenseSet<mlir::Block*> continuationSet_;
  llvm::DenseMap<mlir::Block*, mlir::Operation*> exitTargets_;
  mlir::DominanceInfo& dominance_;
  mlir::Operation* iterationEntry_;
  mlir::Region* region_ = nullptr;
  llvm::DenseSet<mlir::Operation*> members_;
  SourceRegionPlan plan_{};
};
[[nodiscard]] auto outlineSourceRegion(
    mlir::IRRewriter& rewriter, const SourceRegionPlan& plan,
    mlir::TypeRange resultTypes,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(
        mlir::IRRewriter&, const SourceRegionExit&, mlir::ValueRange)>
        yieldValues) -> mlir::scf::ExecuteRegionOp {
  auto* context = plan.entry->getContext();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  for (auto exit : plan.exits) {
    auto* continuation = exit.target->getBlock();
    if (exit.target != &continuation->front())
      rewriter.splitBlock(continuation, exit.target->getIterator());
  }
  auto* prefix = plan.entry->getBlock();
  auto* entry = rewriter.splitBlock(prefix, plan.entry->getIterator());
  rewriter.setInsertionPointToEnd(prefix);
  auto scope = mlir::scf::ExecuteRegionOp::create(
      rewriter, plan.entry->getLoc(), resultTypes, false);
  rewriter.moveBlockBefore(entry, &scope.getRegion(), scope.getRegion().end());
  for (auto range : plan.ranges) {
    auto* block = range.first->getBlock();
    if (block == entry) continue;
    rewriter.moveBlockBefore(block, &scope.getRegion(),
                             scope.getRegion().end());
  }
  for (auto exit : plan.exits) {
    llvm::SmallVector<mlir::Value> arguments;
    if (exit.successor) {
      auto branch = mlir::cast<mlir::BranchOpInterface>(exit.operation);
      auto operands = branch.getSuccessorOperands(*exit.successor);
      if (exit.operation->getNumSuccessors() == 1) {
        rewriter.setInsertionPoint(exit.operation);
        auto values =
            yieldValues(rewriter, exit, operands.getForwardedOperands());
        mlir::scf::YieldOp::create(rewriter, exit.operation->getLoc(), values);
        rewriter.eraseOp(exit.operation);
        continue;
      }
      auto* yieldBlock = rewriter.createBlock(&scope.getRegion());
      for (auto value : operands.getForwardedOperands())
        arguments.push_back(
            yieldBlock->addArgument(value.getType(), exit.operation->getLoc()));
      rewriter.modifyOpInPlace(exit.operation, [&] {
        exit.operation->setSuccessor(yieldBlock, *exit.successor);
      });
      rewriter.setInsertionPointToEnd(yieldBlock);
    } else {
      rewriter.setInsertionPointAfter(exit.operation);
    }
    auto values = yieldValues(rewriter, exit, arguments);
    mlir::scf::YieldOp::create(rewriter, exit.operation->getLoc(), values);
  }
  return scope;
}

void normalizeUnreachableExits(mlir::Region& region) {
  mlir::Operation* prototype = nullptr;
  llvm::SmallVector<mlir::Operation*> unreachable;
  for (auto& block : region) {
    auto* terminator = block.getTerminator();
    if (mlir::isa<mlir::cxx::UnreachableOp>(terminator)) {
      unreachable.push_back(terminator);
      continue;
    }
    if (terminator->getNumSuccessors() || terminator->getNumRegions() ||
        terminator->getNumResults())
      continue;
    if (prototype && prototype->getName() != terminator->getName()) return;
    prototype = terminator;
  }
  if (!prototype) return;
  mlir::IRRewriter rewriter(region.getContext());
  for (auto* op : unreachable) {
    rewriter.setInsertionPoint(op);
    auto value =
        mlir::arith::ConstantIntOp::create(rewriter, op->getLoc(), 0, 1);
    mlir::cxx::BuiltinCallOp::create(rewriter, op->getLoc(), mlir::TypeRange{},
                                     "__builtin_assume",
                                     mlir::ValueRange{value});
    llvm::SmallVector<mlir::Value> operands;
    for (auto operand : prototype->getOperands())
      operands.push_back(mlir::cxx::UndefOp::create(rewriter, op->getLoc(),
                                                    operand.getType()));
    auto* exit = rewriter.clone(*prototype);
    exit->setLoc(op->getLoc());
    exit->setOperands(operands);
    rewriter.eraseOp(op);
  }
}

class SourceCompletions {
 public:
  explicit SourceCompletions(mlir::Type selectorType,
                             bool requireSelector = false)
      : requireSelector_(requireSelector) {
    types_.push_back(selectorType);
  }

  void addTarget(mlir::Operation* target) {
    if (!targetIndices_.try_emplace(target, targets_.size()).second) return;
    targets_.push_back(target);
    offsets_.push_back(types_.size());
    if (target == &target->getBlock()->front())
      llvm::append_range(types_, target->getBlock()->getArgumentTypes());
  }

  void add(const SourceRegionPlan& plan) {
    for (auto exit : plan.exits) addTarget(exit.target);
    for (auto value : plan.liveOuts) {
      if (!valueIndices_.try_emplace(value, values_.size()).second) continue;
      values_.push_back(value);
    }
  }

  void seal() {
    hasSelector_ = requireSelector_ || targets_.size() != 1;
    if (!hasSelector_) {
      types_.erase(types_.begin());
      for (auto& offset : offsets_) --offset;
    }
    valueOffset_ = types_.size();
    for (auto value : values_) types_.push_back(value.getType());
  }

  [[nodiscard]] auto types() const -> mlir::TypeRange { return types_; }

  [[nodiscard]] auto targetIndex(mlir::Operation* target) const -> unsigned {
    return targetIndices_.find(target)->second;
  }

  [[nodiscard]] auto offset(mlir::Operation* target) const -> unsigned {
    return offsets_[targetIndex(target)];
  }

  [[nodiscard]] auto initial(mlir::IRRewriter& rewriter, mlir::Location loc,
                             unsigned index) const
      -> llvm::SmallVector<mlir::Value> {
    llvm::SmallVector<mlir::Value> result;
    if (hasSelector_)
      result.push_back(
          mlir::arith::ConstantIntOp::create(rewriter, loc, index, 32));
    for (auto type : llvm::drop_begin(types_, result.size()))
      result.push_back(mlir::cxx::UndefOp::create(rewriter, loc, type));
    return result;
  }

  [[nodiscard]] auto outline(mlir::IRRewriter& rewriter,
                             const SourceRegionPlan& plan) const
      -> mlir::scf::ExecuteRegionOp {
    return outlineSourceRegion(
        rewriter, plan, types_,
        [&](mlir::IRRewriter& builder, const SourceRegionExit& exit,
            mlir::ValueRange arguments) {
          auto result = initial(builder, exit.operation->getLoc(),
                                targetIndex(exit.target));
          for (auto [index, value] : llvm::enumerate(arguments))
            result[offset(exit.target) + index] = value;
          for (auto value : plan.liveOuts) {
            auto index = valueIndices_.find(value)->second;
            result[valueOffset_ + index] = value;
          }
          return result;
        });
  }

  void replaceLiveOuts(mlir::IRRewriter& rewriter,
                       mlir::Operation* container) const {
    for (auto [index, value] : llvm::enumerate(values_)) {
      rewriter.replaceUsesWithIf(
          value, container->getResult(valueOffset_ + index),
          [&](mlir::OpOperand& use) {
            return !container->isAncestor(use.getOwner());
          });
    }
  }

  void dispatch(mlir::IRRewriter& rewriter, mlir::Location loc,
                mlir::ValueRange results,
                mlir::Operation* excluded = nullptr) const {
    llvm::SmallVector<mlir::Block*> blocks;
    llvm::SmallVector<mlir::ValueRange> operands;
    llvm::SmallVector<int32_t> selectors;
    for (auto [index, target] : llvm::enumerate(targets_)) {
      if (target == excluded) continue;
      blocks.push_back(target->getBlock());
      operands.push_back(results.slice(offsets_[index],
                                       target->getBlock()->getNumArguments()));
      selectors.push_back(index);
    }
    if (blocks.empty()) {
      mlir::cxx::UnreachableOp::create(rewriter, loc);
      return;
    }
    if (blocks.size() == 1) {
      mlir::cf::BranchOp::create(rewriter, loc, blocks.front(),
                                 operands.front());
      return;
    }
    if (blocks.size() == 2) {
      auto selector = mlir::arith::ConstantIntOp::create(rewriter, loc,
                                                         selectors.front(), 32);
      auto chosen = mlir::arith::CmpIOp::create(rewriter, loc,
                                                mlir::arith::CmpIPredicate::eq,
                                                results.front(), selector);
      mlir::cf::CondBranchOp::create(rewriter, loc, chosen, blocks[0],
                                     operands[0], blocks[1], operands[1]);
      return;
    }
    mlir::cf::SwitchOp::create(rewriter, loc, results.front(), blocks.front(),
                               operands.front(),
                               llvm::ArrayRef(selectors).drop_front(),
                               mlir::BlockRange(blocks).drop_front(),
                               llvm::ArrayRef(operands).drop_front());
  }

 private:
  llvm::SmallVector<mlir::Operation*> targets_;
  llvm::DenseMap<mlir::Operation*, unsigned> targetIndices_;
  llvm::SmallVector<unsigned> offsets_;
  llvm::SmallVector<mlir::Type> types_;
  llvm::SmallVector<mlir::Value> values_;
  llvm::DenseMap<mlir::Value, unsigned> valueIndices_;
  unsigned valueOffset_ = 0;
  bool requireSelector_ = false;
  bool hasSelector_ = true;
};

struct SourceIfCondition {
  mlir::Value value;
  mlir::Operation* anchor;
  bool eraseAnchor = false;
  bool inverted = false;
  mlir::Block* prefixes[2] = {};
};

struct SourceIfDestination {
  mlir::Block* block;
  llvm::SmallVector<mlir::Value> arguments;
  mlir::Block* prefix = nullptr;
};

[[nodiscard]] auto sourceIfDestination(const SourceRegionPlan& condition,
                                       mlir::Block* block,
                                       mlir::ValueRange arguments,
                                       mlir::Block* thenBlock,
                                       mlir::Block* elseBlock)
    -> std::optional<SourceIfDestination> {
  if (block == thenBlock || block == elseBlock)
    return SourceIfDestination{block, {arguments.begin(), arguments.end()}};
  if (block->getNumArguments() || !llvm::hasSingleElement(*block))
    return std::nullopt;
  auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(block->getTerminator());
  if (!branch) return std::nullopt;
  if (branch.getDest() != thenBlock && branch.getDest() != elseBlock)
    return std::nullopt;
  const auto member =
      llvm::any_of(condition.ranges, [&](SourceRegionRange range) {
        return range.first == branch && range.last == branch;
      });
  if (!member) return std::nullopt;
  return SourceIfDestination{
      branch.getDest(),
      {branch.getDestOperands().begin(), branch.getDestOperands().end()},
      block};
}

[[nodiscard]] auto materializeSourceIfCondition(mlir::IRRewriter& rewriter,
                                                const SourceIfPlan& plan)
    -> SourceIfCondition {
  auto* thenBlock = plan.thenRegion.entry->getBlock();
  auto* elseBlock = plan.elseRegion.entry->getBlock();
  for (auto range : plan.condition.ranges) {
    if (range.first != plan.condition.entry) continue;
    auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(range.last);
    if (!branch) break;
    auto trueDest =
        sourceIfDestination(plan.condition, branch.getTrueDest(),
                            branch.getTrueDestOperands(), thenBlock, elseBlock);
    auto falseDest = sourceIfDestination(plan.condition, branch.getFalseDest(),
                                         branch.getFalseDestOperands(),
                                         thenBlock, elseBlock);
    if (!trueDest || !falseDest || trueDest->block == falseDest->block) break;
    const auto direct =
        llvm::all_of(plan.condition.ranges, [&](SourceRegionRange part) {
          if (part.first == plan.condition.entry) return true;
          return part.first->getBlock() == trueDest->prefix ||
                 part.first->getBlock() == falseDest->prefix;
        });
    if (!direct) break;
    const auto inverted = trueDest->block == elseBlock;
    auto& thenDest = inverted ? *falseDest : *trueDest;
    auto& elseDest = inverted ? *trueDest : *falseDest;
    for (auto [argument, value] :
         llvm::zip(thenBlock->getArguments(), thenDest.arguments))
      rewriter.replaceAllUsesWith(argument, value);
    for (auto [argument, value] :
         llvm::zip(elseBlock->getArguments(), elseDest.arguments))
      rewriter.replaceAllUsesWith(argument, value);
    thenBlock->eraseArguments([](mlir::BlockArgument) { return true; });
    elseBlock->eraseArguments([](mlir::BlockArgument) { return true; });
    return {branch.getCondition(),
            branch,
            true,
            inverted,
            {thenDest.prefix, elseDest.prefix}};
  }
  llvm::SmallVector<mlir::Type> conditionTypes{rewriter.getI1Type()};
  for (auto value : plan.condition.liveOuts)
    conditionTypes.push_back(value.getType());
  const auto thenOffset = conditionTypes.size();
  llvm::append_range(conditionTypes, thenBlock->getArgumentTypes());
  const auto elseOffset = conditionTypes.size();
  llvm::append_range(conditionTypes, elseBlock->getArgumentTypes());
  auto condition = outlineSourceRegion(
      rewriter, plan.condition, conditionTypes,
      [thenBlock, &plan, &conditionTypes, thenOffset, elseOffset](
          mlir::IRRewriter& builder, const SourceRegionExit& exit,
          mlir::ValueRange arguments) -> llvm::SmallVector<mlir::Value> {
        auto value = mlir::arith::ConstantIntOp::create(
            builder, exit.operation->getLoc(),
            exit.target->getBlock() == thenBlock, 1);
        llvm::SmallVector<mlir::Value> values{value};
        values.append(plan.condition.liveOuts.begin(),
                      plan.condition.liveOuts.end());
        for (auto type : llvm::drop_begin(conditionTypes, values.size()))
          values.push_back(mlir::cxx::UndefOp::create(
              builder, exit.operation->getLoc(), type));
        auto offset =
            exit.target->getBlock() == thenBlock ? thenOffset : elseOffset;
        for (auto [index, argument] : llvm::enumerate(arguments))
          values[offset + index] = argument;
        return values;
      });
  for (auto [value, result] : llvm::zip(
           plan.condition.liveOuts,
           condition.getResults().slice(1, plan.condition.liveOuts.size()))) {
    rewriter.replaceUsesWithIf(value, result, [&](mlir::OpOperand& use) {
      return !condition->isAncestor(use.getOwner());
    });
  }
  for (auto [block, offset] :
       {std::pair(thenBlock, thenOffset), std::pair(elseBlock, elseOffset)}) {
    for (auto [index, argument] : llvm::enumerate(block->getArguments()))
      rewriter.replaceAllUsesWith(argument,
                                  condition.getResult(offset + index));
    block->eraseArguments([](mlir::BlockArgument) { return true; });
  }
  return {condition.getResult(0), condition, false};
}

}  // namespace

auto SourceRegionPlan::continuation() const -> mlir::Operation* {
  if (exits.empty()) return nullptr;
  auto* target = exits.front().target;
  for (auto exit : exits) {
    if (exit.target != target) return nullptr;
  }
  return target;
}

auto SourceRegionPlan::continuationArguments() const
    -> mlir::Block::BlockArgListType {
  auto* target = continuation();
  if (!target || target != &target->getBlock()->front()) return {};
  return target->getBlock()->getArguments();
}

auto preflightSourceRegion(llvm::ArrayRef<mlir::Operation*> operations,
                           llvm::ArrayRef<mlir::Block*> continuations,
                           mlir::DominanceInfo& dominance,
                           mlir::Operation* iterationEntry)
    -> SourceRegionPreflight {
  return SourceRegionAnalysis(operations, continuations, dominance,
                              iterationEntry)
      .run();
}

auto preflightSourceRegion(llvm::ArrayRef<mlir::Operation*> operations,
                           mlir::DominanceInfo& dominance)
    -> SourceRegionPreflight {
  llvm::DenseSet<mlir::Operation*> members(operations.begin(),
                                           operations.end());
  llvm::SetVector<mlir::Block*> continuations;
  for (auto* op : operations) {
    if (auto* next = op->getNextNode(); next && !members.contains(next))
      continuations.insert(next->getBlock());
    for (auto* block : op->getSuccessors()) {
      if (block->empty() || !members.contains(&block->front()))
        continuations.insert(block);
    }
  }
  if (continuations.empty()) return std::string("no-normal-exit");
  return preflightSourceRegion(operations, continuations.getArrayRef(),
                               dominance);
}

auto preflightSourceIf(const SourceRegionPlan& region,
                       llvm::ArrayRef<mlir::Operation*> condition,
                       mlir::Block* thenBlock, mlir::Block* elseBlock,
                       mlir::Block* continuation,
                       mlir::DominanceInfo& dominance) -> SourceIfPreflight {
  if (!thenBlock || !elseBlock) return std::string("missing-if-role");
  if (thenBlock == elseBlock || thenBlock == continuation ||
      elseBlock == continuation)
    return std::string("shared-if-role");
  llvm::DenseSet<mlir::Operation*> conditionMembers(condition.begin(),
                                                    condition.end());
  llvm::SmallVector<mlir::Operation*> arms[2];
  llvm::DenseSet<mlir::Operation*> members;
  for (auto range : region.ranges) {
    for (auto* op = range.first;; op = op->getNextNode()) {
      members.insert(op);
      if (!conditionMembers.contains(op)) {
        if (dominance.dominates(thenBlock, op->getBlock()))
          arms[0].push_back(op);
        else if (dominance.dominates(elseBlock, op->getBlock()))
          arms[1].push_back(op);
      }
      if (op == range.last) break;
    }
  }
  for (auto* op : condition) {
    if (!members.contains(op)) return std::string("if-condition-ownership");
  }
  for (auto* arm : {thenBlock, elseBlock}) {
    for (auto* predecessor : arm->getPredecessors()) {
      if (!conditionMembers.contains(predecessor->getTerminator()))
        return std::string("if-arm-incoming");
    }
  }
  SourceIfPlan result;
  auto conditionPlan =
      preflightSourceRegion(condition, {thenBlock, elseBlock}, dominance);
  if (auto* reason = std::get_if<std::string>(&conditionPlan))
    return "if-condition-" + *reason;
  result.condition = std::move(std::get<SourceRegionPlan>(conditionPlan));
  SourceRegionPlan* plans[] = {&result.thenRegion, &result.elseRegion};
  for (unsigned index = 0; index < 2; ++index) {
    auto checked = preflightSourceRegion(arms[index], dominance);
    if (auto* reason = std::get_if<std::string>(&checked))
      return "if-arm-" + *reason;
    *plans[index] = std::move(std::get<SourceRegionPlan>(checked));
  }
  if (result.thenRegion.entry != &thenBlock->front() ||
      result.elseRegion.entry != &elseBlock->front())
    return std::string("if-arm-entry");
  for (auto* plan : {&result.thenRegion, &result.elseRegion}) {
    for (auto exit : plan->exits) {
      if (exit.target != &exit.target->getBlock()->front())
        return std::string("if-phase-continuation");
    }
  }
  return result;
}

auto preflightSourceLoop(llvm::ArrayRef<mlir::Operation*> operations,
                         mlir::Block* header, mlir::Block* continuation,
                         mlir::DominanceInfo& dominance)
    -> SourceLoopPreflight {
  if (!header || header->empty()) return std::string("missing-loop-header");
  llvm::SmallVector<mlir::Operation*> iteration;
  for (auto* op : operations) {
    if (!dominance.dominates(header, op->getBlock())) continue;
    if (continuation && dominance.dominates(continuation, op->getBlock()))
      continue;
    iteration.push_back(op);
  }
  llvm::DenseSet<mlir::Operation*> members(iteration.begin(), iteration.end());
  llvm::SetVector<mlir::Block*> exits;
  exits.insert(header);
  for (auto* op : iteration) {
    if (auto* next = op->getNextNode(); next && !members.contains(next))
      exits.insert(next->getBlock());
    for (auto* successor : op->getSuccessors()) {
      if (successor->empty() || !members.contains(&successor->front()))
        exits.insert(successor);
    }
  }
  auto checked = preflightSourceRegion(iteration, exits.getArrayRef(),
                                       dominance, &header->front());
  if (auto* reason = std::get_if<std::string>(&checked)) return *reason;
  auto plan = std::move(std::get<SourceRegionPlan>(checked));
  if (plan.entry != &header->front()) return std::string("loop-entry");
  return SourceLoopPlan{std::move(plan)};
}

void finalizeSourceScope(const SourceRegionPlan& plan,
                         mlir::RewriterBase::Listener* listener) {
  mlir::IRRewriter rewriter(plan.entry->getContext());
  rewriter.setListener(listener);
  SourceCompletions completions(rewriter.getI32Type());
  completions.add(plan);
  completions.seal();
  auto scope = completions.outline(rewriter, plan);
  completions.replaceLiveOuts(rewriter, scope);
  rewriter.setInsertionPointAfter(scope);
  completions.dispatch(rewriter, scope.getLoc(), scope.getResults());
}

void finalizeSourceLoop(const SourceLoopPlan& plan,
                        mlir::RewriterBase::Listener* listener) {
  auto* entry = plan.iteration.entry;
  auto loc = entry->getLoc();
  auto* prefix = entry->getBlock();
  mlir::IRRewriter rewriter(entry->getContext());
  rewriter.setListener(listener);
  SourceCompletions completions(rewriter.getI32Type(), true);
  completions.addTarget(entry);
  completions.add(plan.iteration);
  completions.seal();
  auto scope = completions.outline(rewriter, plan.iteration);
  rewriter.setInsertionPoint(scope);
  auto initial =
      completions.initial(rewriter, loc, completions.targetIndex(entry));
  if (prefix->getNumArguments()) {
    auto start = completions.offset(entry);
    for (auto [index, value] : llvm::enumerate(prefix->getArguments()))
      initial[start + index] = value;
  }
  auto loop =
      mlir::scf::WhileOp::create(rewriter, loc, completions.types(), initial);
  auto* before = rewriter.createBlock(&loop.getBefore());
  auto* after = rewriter.createBlock(&loop.getAfter());
  for (auto type : completions.types()) {
    before->addArgument(type, loc);
    after->addArgument(type, loc);
  }
  rewriter.moveOpBefore(scope, before, before->end());
  for (auto [index, value] : llvm::enumerate(prefix->getArguments())) {
    rewriter.replaceUsesWithIf(
        value, before->getArgument(completions.offset(entry) + index),
        [&](mlir::OpOperand& use) {
          return scope->isAncestor(use.getOwner());
        });
  }
  rewriter.setInsertionPointToEnd(before);
  auto repeat = mlir::arith::ConstantIntOp::create(
      rewriter, loc, completions.targetIndex(entry), 32);
  auto condition =
      mlir::arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::eq,
                                  scope.getResult(0), repeat);
  mlir::scf::ConditionOp::create(rewriter, loc, condition, scope.getResults());
  rewriter.setInsertionPointToEnd(after);
  mlir::scf::YieldOp::create(rewriter, loc, after->getArguments());
  completions.replaceLiveOuts(rewriter, loop);
  rewriter.setInsertionPointAfter(loop);
  completions.dispatch(rewriter, loc, loop.getResults(), entry);
}

auto finalizeSourceRegions(mlir::ModuleOp module, TimeTrace* timeTrace)
    -> mlir::LogicalResult {
  module.getContext()->getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::RewritePatternSet patterns(module.getContext());
  mlir::cf::BranchOp::getCanonicalizationPatterns(patterns,
                                                  module.getContext());
  mlir::cf::CondBranchOp::getCanonicalizationPatterns(patterns,
                                                      module.getContext());
  mlir::cf::SwitchOp::getCanonicalizationPatterns(patterns,
                                                  module.getContext());
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Normal);
  auto result = module.walk([&](mlir::Region* body) {
    auto* op = body->getParentOp();
    if (auto function = mlir::dyn_cast<mlir::cxx::FuncOp>(op);
        function && function->removeAttr("cxx.source_scf_fallback")) {
      return mlir::WalkResult::advance();
    }
    auto& region = *body;
    if (region.getBlocks().size() < 2) return mlir::WalkResult::advance();
    const auto hasDecision = llvm::any_of(region, [](mlir::Block& block) {
      return block.getTerminator()->getNumSuccessors() > 1;
    });
    const auto hasExit = llvm::any_of(region, [](mlir::Block& block) {
      return block.getTerminator()->getNumSuccessors() == 0;
    });
    if (!hasDecision && hasExit) return mlir::WalkResult::advance();
    normalizeUnreachableExits(region);
    {
      TimeTrace::Scope trace(timeTrace, "SCF simplify");
      if (mlir::failed(mlir::applyPatternsGreedily(region, frozen, config)))
        return mlir::WalkResult::interrupt();
    }
    TimeTrace::Scope trace(timeTrace, "SCF lift");
    mlir::DominanceInfo dominance(op);
    if (mlir::failed(cxx::liftCxxControlFlowRegion(region, dominance))) {
      op->emitError("could not finalize source control-flow region");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

void finalizeSourceIf(const SourceIfPlan& plan,
                      mlir::RewriterBase::Listener* listener) {
  auto* context = plan.condition.entry->getContext();
  mlir::IRRewriter rewriter(context);
  rewriter.setListener(listener);
  auto condition = materializeSourceIfCondition(rewriter, plan);
  SourceCompletions completions(rewriter.getI32Type());
  completions.add(plan.thenRegion);
  completions.add(plan.elseRegion);
  completions.seal();
  auto thenScope = completions.outline(rewriter, plan.thenRegion);
  auto elseScope = completions.outline(rewriter, plan.elseRegion);
  if (condition.eraseAnchor)
    rewriter.setInsertionPoint(condition.anchor);
  else
    rewriter.setInsertionPointAfter(condition.anchor);
  auto ifOp = mlir::scf::IfOp::create(rewriter, condition.anchor->getLoc(),
                                      completions.types(), condition.value,
                                      false, false);
  if (condition.eraseAnchor) rewriter.eraseOp(condition.anchor);
  for (auto [scope, prefix] : {std::pair(thenScope, condition.prefixes[0]),
                               std::pair(elseScope, condition.prefixes[1])}) {
    if (!prefix) continue;
    auto* branch = prefix->getTerminator();
    rewriter.modifyOpInPlace(branch, [&] {
      branch->setSuccessor(&scope.getRegion().front(), 0);
      branch->setOperands(mlir::ValueRange{});
    });
    rewriter.moveBlockBefore(prefix, &scope.getRegion(),
                             scope.getRegion().begin());
  }
  if (condition.inverted) std::swap(thenScope, elseScope);
  auto* thenBody = thenScope->getBlock();
  auto* elseBody = elseScope->getBlock();
  rewriter.moveBlockBefore(thenBody, &ifOp.getThenRegion(),
                           ifOp.getThenRegion().end());
  rewriter.moveBlockBefore(elseBody, &ifOp.getElseRegion(),
                           ifOp.getElseRegion().end());
  for (auto* body : {thenBody, elseBody}) {
    rewriter.setInsertionPointToEnd(body);
    mlir::scf::YieldOp::create(
        rewriter, ifOp.getLoc(),
        (&ifOp.getThenRegion() == body->getParent() ? thenScope : elseScope)
            .getResults());
  }
  rewriter.setInsertionPointAfter(ifOp);
  completions.replaceLiveOuts(rewriter, ifOp);
  completions.dispatch(rewriter, ifOp.getLoc(), ifOp.getResults());
}
}  // namespace cxx::ir
