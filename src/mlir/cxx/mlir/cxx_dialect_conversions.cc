// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/mlir/cxx_dialect_conversions.h>

// cxx
#include <cxx/mlir/cxx_dialect.h>

// mlir
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

namespace {

class FuncOpLowering : public OpConversionPattern<cxx::FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FuncOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    auto funcType = op.getFunctionType();

    SmallVector<Type> argumentTypes;
    for (auto argType : funcType.getInputs()) {
      auto convertedType = typeConverter->convertType(argType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert function argument type");
      }
      argumentTypes.push_back(convertedType);
    }

    SmallVector<Type> resultTypes;
    for (auto resultType : funcType.getResults()) {
      auto convertedType = typeConverter->convertType(resultType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert function result type");
      }

      resultTypes.push_back(convertedType);
    }

    const auto returnType = resultTypes.empty()
                                ? LLVM::LLVMVoidType::get(getContext())
                                : resultTypes.front();

    const auto isVarArg = false;

    auto llvmFuncType =
        LLVM::LLVMFunctionType::get(returnType, argumentTypes, isVarArg);

    auto func = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), op.getSymName(),
                                                  llvmFuncType);

    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());

    rewriter.eraseOp(op);

    return success();
  }
};

class ReturnOpLowering : public OpConversionPattern<cxx::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ReturnOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class AllocaOpLowering : public OpConversionPattern<cxx::AllocaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto ptrTy = dyn_cast<cxx::PointerType>(op.getType());
    if (!ptrTy) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to be a pointer type");
    }

    auto resultType = LLVM::LLVMPointerType::get(context);
    auto elementType = typeConverter->convertType(ptrTy.getElementType());

    auto size = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));

    auto x = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, resultType,
                                                         elementType, size);
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<cxx::LoadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType,
                                              adaptor.getAddr());

    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<cxx::StoreOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto valueType = typeConverter->convertType(op.getValue().getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert store value type");
    }

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                               adaptor.getAddr());

    return success();
  }
};

class IntConstantOpLowering : public OpConversionPattern<cxx::IntConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IntConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert integer constant type");
    }

    auto valueAttr = adaptor.getValueAttr();
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, valueAttr);

    return success();
  }
};

class CxxToLLVMLoweringPass
    : public PassWrapper<CxxToLLVMLoweringPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxxToLLVMLoweringPass)

  auto getArgument() const -> StringRef override { return "cxx-to-llvm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final;
};

}  // namespace

void CxxToLLVMLoweringPass::runOnOperation() {
  auto context = &getContext();
  auto module = getOperation();

  // set up the data layout
  mlir::DataLayout dataLayout(module);

  // set up the type converter
  LLVMTypeConverter typeConverter{context};
  typeConverter.addConversion([](cxx::IntegerType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  });

  typeConverter.addConversion([](cxx::PointerType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // set up the conversion patterns
  ConversionTarget target(*context);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<cxx::CxxDialect>();

  RewritePatternSet patterns(context);
  patterns.insert<FuncOpLowering, ReturnOpLowering, AllocaOpLowering,
                  LoadOpLowering, StoreOpLowering, IntConstantOpLowering>(
      typeConverter, context);

  populateFunctionOpInterfaceTypeConversionPattern<cxx::FuncOp>(patterns,
                                                                typeConverter);

  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir

auto cxx::createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::CxxToLLVMLoweringPass>();
}

auto cxx::lowerToMLIR(mlir::ModuleOp module) -> mlir::LogicalResult {
  mlir::PassManager pm(module->getName());

  // debug dialect conversions
#if false
  module->getContext()->disableMultithreading();
  pm.enableIRPrinting();
#endif

  pm.addPass(cxx::createLowerToLLVMPass());

  if (failed(pm.run(module))) {
    return mlir::failure();
  }

  return mlir::success();
}
