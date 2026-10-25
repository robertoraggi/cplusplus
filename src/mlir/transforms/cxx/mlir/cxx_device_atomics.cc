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
#include <cxx/mlir/passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir {
namespace {

class ExpandNandToCompareExchange final
    : public OpRewritePattern<cxx::AtomicRMWOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::AtomicRMWOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (op.getKind() != cxx::AtomicRMWKind::Nand) return failure();
    auto loc = op.getLoc();
    auto type = op.getType();
    auto guess = arith::ConstantIntOp::create(rewriter, loc, type, 0);
    auto loop =
        scf::WhileOp::create(rewriter, loc, TypeRange{type}, ValueRange{guess});
    auto* before = rewriter.createBlock(&loop.getBefore(), {}, {type}, {loc});
    auto expected = before->getArgument(0);
    auto both = arith::AndIOp::create(rewriter, loc, expected, op.getValue());
    auto ones = arith::ConstantIntOp::create(rewriter, loc, type, -1);
    auto desired = arith::XOrIOp::create(rewriter, loc, both, ones);
    auto exchange = cxx::AtomicCompareExchangeOp::create(
        rewriter, loc, op.getAddr(), expected, desired, op.getOrdering(),
        cxx::MemoryOrder::Relaxed, op.getScope(), true, op.getVolatile_());
    auto retry = arith::XOrIOp::create(
        rewriter, loc, exchange.getExchanged(),
        arith::ConstantIntOp::create(rewriter, loc, 1, 1));
    scf::ConditionOp::create(rewriter, loc, retry,
                             ValueRange{exchange.getOld()});
    auto* after = rewriter.createBlock(&loop.getAfter(), {}, {type}, {loc});
    scf::YieldOp::create(rewriter, loc, after->getArguments());
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

class LegalizeCxxDeviceAtomicsPass final
    : public PassWrapper<LegalizeCxxDeviceAtomicsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeCxxDeviceAtomicsPass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "legalize-cxx-device-atomics";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Express Cxx atomic operations without a device instruction as "
           "compare-exchange loops";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandNandToCompareExchange>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createLegalizeCxxDeviceAtomicsPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::LegalizeCxxDeviceAtomicsPass>();
}

void cxx::registerLegalizeCxxDeviceAtomicsPass() {
  mlir::registerPass(createLegalizeCxxDeviceAtomicsPass);
}
