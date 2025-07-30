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

    if (!elementType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert element type of alloca");
    }

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

class BoolConstantOpLowering : public OpConversionPattern<cxx::BoolConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::BoolConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert boolean constant type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType,
                                                  adaptor.getValue());

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

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType,
                                                  adaptor.getValue());

    return success();
  }
};

class FloatConstantOpLowering
    : public OpConversionPattern<cxx::FloatConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FloatConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert float constant type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType,
                                                  adaptor.getValue());

    return success();
  }
};

class IntegralCastOpLowering : public OpConversionPattern<cxx::IntegralCastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IntegralCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert integral cast type");
    }

    const auto sourceType = dyn_cast<cxx::IntegerType>(op.getValue().getType());
    const auto targetType = dyn_cast<cxx::IntegerType>(op.getType());
    const auto isSigned = targetType.getIsSigned();

    if (sourceType.getWidth() == targetType.getWidth()) {
      // no conversion needed, just replace the op with the value
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    if (targetType.getWidth() < sourceType.getWidth()) {
      // truncation
      if (isSigned) {
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultType,
                                                   adaptor.getValue());
      } else {
        rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, resultType,
                                                  adaptor.getValue());
      }
      return success();
    }

    // extension

    if (isSigned) {
      rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, resultType,
                                                adaptor.getValue());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, resultType,
                                                adaptor.getValue());
    }

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

  typeConverter.addConversion([](cxx::BoolType type) {
    // todo: i8/i32 for data and i1 for control flow
    return IntegerType::get(type.getContext(), 8);
  });

  typeConverter.addConversion([](cxx::IntegerType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  });

  typeConverter.addConversion([](cxx::FloatType type) -> Type {
    auto width = type.getWidth();
    switch (width) {
      case 16:
        return Float16Type::get(type.getContext());
      case 32:
        return Float32Type::get(type.getContext());
      case 64:
        return Float64Type::get(type.getContext());
      default:
        return {};
    }  // switch
  });

  typeConverter.addConversion([](cxx::PointerType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  DenseMap<cxx::ClassType, Type> convertedClassTypes;
  typeConverter.addConversion([&](cxx::ClassType type) -> Type {
    if (auto it = convertedClassTypes.find(type);
        it != convertedClassTypes.end()) {
      return it->second;
    }

    auto structType =
        LLVM::LLVMStructType::getIdentified(type.getContext(), type.getName());

    convertedClassTypes[type] = structType;

    SmallVector<Type> fieldTypes;
    bool isPacked = false;

    for (auto field : type.getBody()) {
      auto convertedFieldType = typeConverter.convertType(field);
      // todo: check if the field type was converted successfully
      fieldTypes.push_back(convertedFieldType);
    }

    structType.setBody(fieldTypes, isPacked);

    return structType;
  });

  // set up the conversion patterns
  ConversionTarget target(*context);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<cxx::CxxDialect>();

  RewritePatternSet patterns(context);
  patterns.insert<FuncOpLowering, ReturnOpLowering, AllocaOpLowering,
                  LoadOpLowering, StoreOpLowering, BoolConstantOpLowering,
                  IntConstantOpLowering, FloatConstantOpLowering,
                  IntegralCastOpLowering>(typeConverter, context);

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
    module.print(llvm::errs());
    return mlir::failure();
  }

  return mlir::success();
}
