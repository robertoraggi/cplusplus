#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/passes.h>
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

class LiftCxxControlFlowToSCFPass
    : public PassWrapper<LiftCxxControlFlowToSCFPass,
                         OperationPass<cxx::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCxxControlFlowToSCFPass)

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

    CxxControlFlowToSCFTransformation transformation;
    auto changed =
        transformCFGToSCF(body, transformation, getAnalysis<DominanceInfo>());
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
