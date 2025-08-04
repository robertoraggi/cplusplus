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
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

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

    if (op.getBody().empty()) {
      func.setLinkage(LLVM::linkage::Linkage::External);
    }

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

class CallOpLowering : public OpConversionPattern<cxx::CallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    SmallVector<Type> argumentTypes;
    for (auto argType : op.getOperandTypes()) {
      auto convertedType = typeConverter->convertType(argType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert call argument type");
      }
      argumentTypes.push_back(convertedType);
    }

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert call result types");
    }

    auto llvmCallOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), resultTypes, adaptor.getCallee(), adaptor.getInputs());

    rewriter.replaceOp(op, llvmCallOp);
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
        op.getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));

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

class IntToBoolOpLowering : public OpConversionPattern<cxx::IntToBoolOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IntToBoolOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert int to bool type");
    }

    auto c0 = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), adaptor.getValue().getType(), 0);

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, LLVM::ICmpPredicate::ne, adaptor.getValue(), c0);

    return success();
  }
};

class BoolToIntOpLowering : public OpConversionPattern<cxx::BoolToIntOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::BoolToIntOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert bool to int type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, resultType,
                                              adaptor.getValue());

    return success();
  }
};

class NotOpLowering : public OpConversionPattern<cxx::NotOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::NotOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert not operation");
    }

    auto c1 = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), adaptor.getValue().getType(), -1);

    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, resultType, adaptor.getValue(),
                                             c1);

    return success();
  }
};

class AddIOpLowering : public OpConversionPattern<cxx::AddIOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AddIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert addi operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());

    return success();
  }
};

class SubIOpLowering : public OpConversionPattern<cxx::SubIOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::SubIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert subi operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());

    return success();
  }
};

class MulIOpLowering : public OpConversionPattern<cxx::MulIOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::MulIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert muli operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());

    return success();
  }
};

class DivIOpLowering : public OpConversionPattern<cxx::DivIOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::DivIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert divi operation type");
    }

    bool isSigned = true;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getType())) {
      isSigned = intType.getIsSigned();
    }

    if (isSigned) {
      rewriter.replaceOpWithNewOp<LLVM::SDivOp>(
          op, resultType, adaptor.getLhs(), adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::UDivOp>(
          op, resultType, adaptor.getLhs(), adaptor.getRhs());
    }

    return success();
  }
};

class ModIOpLowering : public OpConversionPattern<cxx::ModIOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ModIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert modi operation type");
    }

    bool isSigned = true;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getType())) {
      isSigned = intType.getIsSigned();
    }

    if (isSigned) {
      rewriter.replaceOpWithNewOp<LLVM::SRemOp>(
          op, resultType, adaptor.getLhs(), adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::URemOp>(
          op, resultType, adaptor.getLhs(), adaptor.getRhs());
    }

    return success();
  }
};

class ShiftLeftOpLowering : public OpConversionPattern<cxx::ShiftLeftOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ShiftLeftOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert shift left operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ShlOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());

    return success();
  }
};

class ShiftRightOpLowering : public OpConversionPattern<cxx::ShiftRightOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ShiftRightOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert shift right operation type");
    }

    bool isSigned = true;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getType())) {
      isSigned = intType.getIsSigned();
    }

    if (isSigned) {
      rewriter.replaceOpWithNewOp<LLVM::AShrOp>(
          op, resultType, adaptor.getLhs(), adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::LShrOp>(
          op, resultType, adaptor.getLhs(), adaptor.getRhs());
    }

    return success();
  }
};

class EqualOpLowering : public OpConversionPattern<cxx::EqualOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::EqualOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert equal operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, LLVM::ICmpPredicate::eq, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class NotEquaOpLowering : public OpConversionPattern<cxx::NotEqualOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::NotEqualOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert not equal operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, LLVM::ICmpPredicate::ne, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class LessThanOpLowering : public OpConversionPattern<cxx::LessThanOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LessThanOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert less than operation type");
    }

    auto predicate = LLVM::ICmpPredicate::slt;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getLhs().getType())) {
      if (intType.getIsSigned()) {
        predicate = LLVM::ICmpPredicate::slt;
      } else {
        predicate = LLVM::ICmpPredicate::ult;
      }
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, predicate, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

class LessEqualOpLowering : public OpConversionPattern<cxx::LessEqualOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LessEqualOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert less equal operation type");
    }

    auto predicate = LLVM::ICmpPredicate::sle;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getLhs().getType())) {
      if (intType.getIsSigned()) {
        predicate = LLVM::ICmpPredicate::sle;
      } else {
        predicate = LLVM::ICmpPredicate::ule;
      }
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, predicate, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

class GreaterThanOpLowering : public OpConversionPattern<cxx::GreaterThanOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::GreaterThanOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert greater than operation type");
    }

    auto predicate = LLVM::ICmpPredicate::sgt;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getLhs().getType())) {
      if (intType.getIsSigned()) {
        predicate = LLVM::ICmpPredicate::sgt;
      } else {
        predicate = LLVM::ICmpPredicate::ugt;
      }
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, predicate, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

class GreaterEqualOpLowering : public OpConversionPattern<cxx::GreaterEqualOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::GreaterEqualOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert greater equal operation type");
    }

    auto predicate = LLVM::ICmpPredicate::sge;

    if (auto intType = dyn_cast<cxx::IntegerType>(op.getLhs().getType())) {
      if (intType.getIsSigned()) {
        predicate = LLVM::ICmpPredicate::sge;
      } else {
        predicate = LLVM::ICmpPredicate::uge;
      }
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, predicate, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

//
// floating point operations
//

class AddFOpLowering : public OpConversionPattern<cxx::AddFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AddFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert addf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FAddOp>(op, resultType, adaptor.getLhs(),
                                              adaptor.getRhs());

    return success();
  }
};

class SubFOpLowering : public OpConversionPattern<cxx::SubFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::SubFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert subf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, resultType, adaptor.getLhs(),
                                              adaptor.getRhs());

    return success();
  }
};

class MulFOpLowering : public OpConversionPattern<cxx::MulFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::MulFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert mulf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FMulOp>(op, resultType, adaptor.getLhs(),
                                              adaptor.getRhs());

    return success();
  }
};

class DivFOpLowering : public OpConversionPattern<cxx::DivFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::DivFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert divf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FDivOp>(op, resultType, adaptor.getLhs(),
                                              adaptor.getRhs());

    return success();
  }
};

class FloatingPointCastOpLowering
    : public OpConversionPattern<cxx::FloatingPointCastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FloatingPointCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert floating point cast type");
    }

    const auto sourceType = dyn_cast<cxx::FloatType>(op.getValue().getType());
    const auto targetType = dyn_cast<cxx::FloatType>(op.getType());

    if (sourceType.getWidth() == targetType.getWidth()) {
      // no conversion needed, just replace the op with the value
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    if (sourceType.getWidth() < targetType.getWidth()) {
      // extension
      rewriter.replaceOpWithNewOp<LLVM::FPExtOp>(op, resultType,
                                                 adaptor.getValue());
      return success();
    }

    // truncation
    rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, resultType,
                                                 adaptor.getValue());

    return success();
  }
};

class IntToFloatOpLowering : public OpConversionPattern<cxx::IntToFloatOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IntToFloatOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert int to float type");
    }

    rewriter.replaceOpWithNewOp<LLVM::SIToFPOp>(op, resultType,
                                                adaptor.getValue());

    return success();
  }
};

class FloatToIntOpLowering : public OpConversionPattern<cxx::FloatToIntOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FloatToIntOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert float to int type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FPToSIOp>(op, resultType,
                                                adaptor.getValue());

    return success();
  }
};

//
// control flow operations
//

class CondBranchOpLowering : public OpConversionPattern<cxx::CondBranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::CondBranchOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());

    return success();
  }
};

struct LabelConverter {
  DenseMap<StringRef, Block *> labels;
};

class GotoOpLowering : public OpConversionPattern<cxx::GotoOp> {
 public:
  GotoOpLowering(const TypeConverter &typeConverter,
                 const LabelConverter &labelConverter, MLIRContext *context,
                 PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        labelConverter_(labelConverter) {}

  auto matchAndRewrite(cxx::GotoOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto context = getContext();

    auto targetBlock = labelConverter_.labels.lookup(op.getLabel());

    if (auto nextOp = ++op->getIterator(); isa<cf::BranchOp>(&*nextOp)) {
      rewriter.eraseOp(&*nextOp);
    }

    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, targetBlock);

    return success();
  }

 private:
  const LabelConverter &labelConverter_;
};

class LabelOpLowering : public OpConversionPattern<cxx::LabelOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LabelOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override {
    auto context = getContext();

    rewriter.eraseOp(op);

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
  DataLayout dataLayout(module);

  // set up the type converter
  LLVMTypeConverter typeConverter{context};

  typeConverter.addConversion([](cxx::VoidType type) {
    return LLVM::LLVMVoidType::get(type.getContext());
  });

  typeConverter.addConversion([](cxx::BoolType type) {
    // todo: i8/i32 for data and i1 for control flow
    return IntegerType::get(type.getContext(), 1);
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

  LabelConverter labelConverter;

  module.walk([&](Operation *op) {
    if (auto labelOp = dyn_cast<cxx::LabelOp>(op)) {
      labelConverter.labels[labelOp.getName()] = labelOp->getBlock();
    }
  });

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<cxx::CxxDialect>();

  RewritePatternSet patterns(context);

  // function operations
  patterns.insert<FuncOpLowering, ReturnOpLowering, CallOpLowering>(
      typeConverter, context);

  // memory operations
  patterns.insert<AllocaOpLowering, LoadOpLowering, StoreOpLowering>(
      typeConverter, context);

  // cast operations
  patterns
      .insert<IntToBoolOpLowering, BoolToIntOpLowering, IntegralCastOpLowering>(
          typeConverter, context);

  // constant operations
  patterns.insert<BoolConstantOpLowering, IntConstantOpLowering,
                  FloatConstantOpLowering>(typeConverter, context);

  // unary operations
  patterns.insert<NotOpLowering>(typeConverter, context);

  // binary operations
  patterns
      .insert<AddIOpLowering, SubIOpLowering, MulIOpLowering, DivIOpLowering,
              ModIOpLowering, ShiftLeftOpLowering, ShiftRightOpLowering>(
          typeConverter, context);

  // comparison operations
  patterns.insert<EqualOpLowering, NotEquaOpLowering, LessThanOpLowering,
                  LessEqualOpLowering, GreaterThanOpLowering,
                  GreaterEqualOpLowering>(typeConverter, context);

  // floating point operations
  patterns
      .insert<AddFOpLowering, SubFOpLowering, MulFOpLowering, DivFOpLowering>(
          typeConverter, context);

  // floating point cast operations
  patterns.insert<FloatingPointCastOpLowering, IntToFloatOpLowering,
                  FloatToIntOpLowering>(typeConverter, context);

  // control flow operations
  patterns.insert<CondBranchOpLowering>(typeConverter, context);
  patterns.insert<LabelOpLowering>(typeConverter, context);
  patterns.insert<GotoOpLowering>(typeConverter, labelConverter, context);

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
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(module))) {
    module.print(llvm::errs());
    return mlir::failure();
  }

  return mlir::success();
}
