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
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace {

[[nodiscard]] auto hasInternalLinkage(cxx::FuncOp op) -> bool {
  auto linkage = op.getLinkageKind();
  return linkage && *linkage == cxx::LinkageKind::Internal;
}

[[nodiscard]] auto isNoInline(cxx::FuncOp op) -> bool {
  auto inlineKind = op.getInlineKind();
  return inlineKind && *inlineKind == cxx::InlineKind::NoInline;
}

class FuncToFunc : public OpConversionPattern<cxx::FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FuncOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto type = op.getFunctionType();
    auto function = func::FuncOp::create(
        rewriter, op.getLoc(), op.getSymName(),
        rewriter.getFunctionType(type.getInputs(), type.getResults()),
        llvm::to_vector(op->getDiscardableAttrs()));
    if (auto argAttrs = op.getArgAttrsAttr())
      function.setArgAttrsAttr(argAttrs);
    if (auto resAttrs = op.getResAttrsAttr())
      function.setResAttrsAttr(resAttrs);
    if (isNoInline(op)) function.setNoInline(true);
    if (hasInternalLinkage(op) || op.getBody().empty()) function.setPrivate();
    rewriter.inlineRegionBefore(op.getBody(), function.getBody(),
                                function.getBody().end());
    rewriter.eraseOp(op);
    return success();
  }
};

[[nodiscard]] auto convertsToFuncCall(cxx::CallOp op,
                                      SymbolTableCollection& symbols) -> bool {
  if (!op.getCalleeAttr()) return false;
  auto callee =
      symbols.lookupNearestSymbolFrom<cxx::FuncOp>(op, op.getCalleeAttr());
  return !callee || !callee.getFunctionType().getVariadic();
}

class CallToFunc : public OpConversionPattern<cxx::CallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, op.getResultTypes(), op.getCalleeAttr(), adaptor.getInputs());
    return success();
  }
};

class ReturnToFunc : public OpConversionPattern<cxx::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ReturnOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getInput());
    return success();
  }
};

class ConvertCxxToFuncPass
    : public PassWrapper<ConvertCxxToFuncPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxToFuncPass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "convert-cxx-to-func";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Lower cxx functions, direct calls and returns to the func dialect";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect>();
    func::registerInlinerExtension(registry);
  }

  void runOnOperation() override {
    SymbolTableCollection symbols;
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addDynamicallyLegalOp<cxx::FuncOp>(
        [](cxx::FuncOp op) { return op.getFunctionType().getVariadic(); });
    target.addDynamicallyLegalOp<cxx::ReturnOp>(
        [](cxx::ReturnOp op) { return isa<cxx::FuncOp>(op->getParentOp()); });
    target.addDynamicallyLegalOp<cxx::CallOp>(
        [&](cxx::CallOp op) { return !convertsToFuncCall(op, symbols); });
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    RewritePatternSet patterns(&getContext());
    patterns.add<FuncToFunc, CallToFunc, ReturnToFunc>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxToFuncPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxToFuncPass>();
}

void cxx::registerConvertCxxToFuncPass() {
  mlir::registerPass(createConvertCxxToFuncPass);
}
