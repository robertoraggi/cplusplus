#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/passes.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/Config/llvm-config.h>
#include <mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace mlir {
namespace {

class CxxControlFlowToSCFTransformation
    : public ControlFlowToSCFTransformation {
 public:
  auto getUndefValue(Location loc, OpBuilder& builder, Type type)
      -> Value override {
    return cxx::UndefOp::create(builder, loc, type);
  }

  auto createUnreachableTerminator(Location loc, OpBuilder& builder,
                                   Region& region)
      -> FailureOr<Operation*> override {
    return cxx::UnreachableOp::create(builder, loc).getOperation();
  }
};

template <typename Transformation>
[[nodiscard]] auto canLiftBranch(Operation* operation,
                                 Transformation& transformation) -> bool {
  if (!operation->getNumSuccessors()) return true;
  auto branch = dyn_cast<BranchOpInterface>(operation);
  if (!branch) return false;
  if (!isMemoryEffectFree(operation)) return false;
#if LLVM_VERSION_MAJOR < 23
  if (auto switchOp = dyn_cast<cf::SwitchOp>(operation)) {
    if (cast<IntegerType>(switchOp.getFlag().getType()).getWidth() > 32)
      return false;
  }
#endif
  for (unsigned index = 0; index < operation->getNumSuccessors(); ++index) {
    if (branch.getSuccessorOperands(index).getProducedOperandCount())
      return false;
  }
  if (operation->getNumSuccessors() == 1) return true;
  if constexpr (requires {
                  transformation.canConvertMultiSuccessorBranchOp(operation);
                }) {
    return transformation.canConvertMultiSuccessorBranchOp(operation);
  } else {
    return isa<cf::CondBranchOp, cf::SwitchOp>(operation);
  }
}

[[nodiscard]] auto canLiftRegion(Region& region,
                                 CFGToSCFInterface& transformation) -> bool {
  llvm::df_iterator_default_set<Block*, 16> reachable;
  for (auto block : llvm::depth_first_ext(&region.front(), reachable))
    (void)block;
  for (auto& block : region) {
    if (!reachable.contains(&block)) return false;
  }
  auto result = region.walk([&](Operation* operation) {
    if (!canLiftBranch(operation, transformation))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

class LiftCxxControlFlowToSCFPass
    : public PassWrapper<LiftCxxControlFlowToSCFPass,
                         OperationPass<cxx::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCxxControlFlowToSCFPass)

  LiftCxxControlFlowToSCFPass() = default;
  LiftCxxControlFlowToSCFPass(const LiftCxxControlFlowToSCFPass& other)
      : PassWrapper(other) {}

  Option<unsigned> maxBlocks{
      *this, "max-blocks",
      llvm::cl::desc("Maximum CFG blocks to lift (0 is unlimited)"),
      llvm::cl::init(0)};

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "cxx-lift-cf-to-scf";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Lift control flow in Cxx function bodies to structured control "
           "flow";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto& body = getOperation().getBody();
    if (body.empty()) {
      markAllAnalysesPreserved();
      return;
    }

    auto blockCount = body.getBlocks().size();
    if (maxBlocks && blockCount > maxBlocks) {
      emitRemark(getOperation().getLoc())
          << "leaving @" << getOperation().getSymName()
          << " in CF: " << blockCount
          << " blocks exceeds max-blocks=" << maxBlocks.getValue();
      markAllAnalysesPreserved();
      return;
    }

    auto changed =
        ::cxx::liftCxxControlFlowRegion(body, getAnalysis<DominanceInfo>());
    if (failed(changed)) {
      signalPassFailure();
      return;
    }
    if (!*changed) markAllAnalysesPreserved();
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createLiftCxxControlFlowToSCFPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::LiftCxxControlFlowToSCFPass>();
}

void cxx::registerLiftCxxControlFlowToSCFPass() {
  mlir::registerPass(createLiftCxxControlFlowToSCFPass);
}

auto cxx::liftCxxControlFlowRegion(mlir::Region& region,
                                   mlir::DominanceInfo& dominance)
    -> mlir::FailureOr<bool> {
  if (region.empty()) return false;
  mlir::CxxControlFlowToSCFTransformation transformation;
  if (!mlir::canLiftRegion(region, transformation)) return false;
  return mlir::transformCFGToSCF(region, transformation, dominance);
}
