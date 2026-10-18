#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/PatternMatch.h>

#include <optional>
#include <string>
#include <variant>

namespace cxx {
class TimeTrace;
}

namespace cxx::ir {
struct SourceRegionRange {
  mlir::Operation* first;
  mlir::Operation* last;
};

struct SourceRegionExit {
  mlir::Operation* operation;
  std::optional<unsigned> successor;
  mlir::Operation* target;
};

struct SourceRegionPlan {
  mlir::Operation* entry;
  llvm::SmallVector<SourceRegionRange> ranges;
  llvm::SmallVector<SourceRegionExit> exits;
  llvm::SmallVector<mlir::Value> captures;
  llvm::SmallVector<mlir::Value> liveOuts;
  [[nodiscard]] auto continuation() const -> mlir::Operation*;
  [[nodiscard]] auto continuationArguments() const
      -> mlir::Block::BlockArgListType;
};

using SourceRegionPreflight = std::variant<std::string, SourceRegionPlan>;

[[nodiscard]] auto preflightSourceRegion(
    llvm::ArrayRef<mlir::Operation*> operations,
    llvm::ArrayRef<mlir::Block*> continuations, mlir::DominanceInfo& dominance,
    mlir::Operation* iterationEntry = nullptr) -> SourceRegionPreflight;
[[nodiscard]] auto preflightSourceRegion(
    llvm::ArrayRef<mlir::Operation*> operations, mlir::DominanceInfo& dominance)
    -> SourceRegionPreflight;

struct SourceIfPlan {
  SourceRegionPlan condition;
  SourceRegionPlan thenRegion;
  SourceRegionPlan elseRegion;
};

using SourceIfPreflight = std::variant<std::string, SourceIfPlan>;

[[nodiscard]] auto preflightSourceIf(
    const SourceRegionPlan& region, llvm::ArrayRef<mlir::Operation*> condition,
    mlir::Block* thenBlock, mlir::Block* elseBlock, mlir::Block* continuation,
    mlir::DominanceInfo& dominance) -> SourceIfPreflight;

struct SourceLoopPlan {
  SourceRegionPlan iteration;
};

using SourceLoopPreflight = std::variant<std::string, SourceLoopPlan>;

[[nodiscard]] auto preflightSourceLoop(
    llvm::ArrayRef<mlir::Operation*> operations, mlir::Block* header,
    mlir::Block* continuation, mlir::DominanceInfo& dominance)
    -> SourceLoopPreflight;

void finalizeSourceScope(const SourceRegionPlan& plan,
                         mlir::RewriterBase::Listener* listener = nullptr);

void finalizeSourceLoop(const SourceLoopPlan& plan,
                        mlir::RewriterBase::Listener* listener = nullptr);

[[nodiscard]] auto finalizeSourceRegions(mlir::ModuleOp module,
                                         TimeTrace* timeTrace = nullptr)
    -> mlir::LogicalResult;

void finalizeSourceIf(const SourceIfPlan& plan,
                      mlir::RewriterBase::Listener* listener = nullptr);
}  // namespace cxx::ir
