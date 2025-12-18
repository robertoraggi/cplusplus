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
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace mlir {

namespace {

class FuncOpLowering : public OpConversionPattern<cxx::FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FuncOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    if (failed(convertFunctionTyype(op, rewriter))) {
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    }

    auto funcType = op.getFunctionType();
    auto llvmFuncType = typeConverter->convertType(funcType);

    auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), op.getSymName(),
                                         llvmFuncType);

    if (op.getBody().empty()) {
      func.setLinkage(LLVM::linkage::Linkage::External);
    }

    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());

    rewriter.eraseOp(op);

    return success();
  }

  auto convertFunctionTyype(cxx::FuncOp funcOp,
                            ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    auto type = funcOp.getFunctionType();
    const auto& typeConverter = *getTypeConverter();

    TypeConverter::SignatureConversion result(type.getInputs().size());
    SmallVector<Type, 1> newResults;
    if (failed(typeConverter.convertSignatureArgs(type.getInputs(), result)) ||
        failed(typeConverter.convertTypes(type.getResults(), newResults)) ||
        failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                           typeConverter, &result)))
      return failure();

    auto newType = cxx::FunctionType::get(rewriter.getContext(),
                                          result.getConvertedTypes(),
                                          newResults, type.getVariadic());

    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });

    return success();
  }
};

class GlobalOpLowering : public OpConversionPattern<cxx::GlobalOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::GlobalOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    auto elementType = getTypeConverter()->convertType(op.getGlobalType());

    rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        op, elementType, op.getConstant(), LLVM::linkage::Linkage::Private,
        op.getSymName(), adaptor.getValue().value());

    return success();
  }
};

class ReturnOpLowering : public OpConversionPattern<cxx::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ReturnOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class CallOpLowering : public OpConversionPattern<cxx::CallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::CallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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

    auto llvmCallOp =
        LLVM::CallOp::create(rewriter, op.getLoc(), resultTypes,
                             adaptor.getCallee(), adaptor.getInputs());

    if (op.getVarCalleeType().has_value()) {
      auto varCalleeType =
          typeConverter->convertType(op.getVarCalleeType().value());
      llvmCallOp.setVarCalleeType(cast<LLVM::LLVMFunctionType>(varCalleeType));
    }

    rewriter.replaceOp(op, llvmCallOp);
    return success();
  }
};

class AddressOfOpLowering : public OpConversionPattern<cxx::AddressOfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AddressOfOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert address of type");
    }

    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, resultType,
                                                   adaptor.getSymName());

    return success();
  }
};

class AllocaOpLowering : public OpConversionPattern<cxx::AllocaOp> {
 public:
  AllocaOpLowering(const TypeConverter& typeConverter,
                   const DataLayout& dataLayout, MLIRContext* context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::AllocaOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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

    auto size = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

    auto x = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, resultType,
                                                         elementType, size);
    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class LoadOpLowering : public OpConversionPattern<cxx::LoadOp> {
 public:
  LoadOpLowering(const TypeConverter& typeConverter,
                 const DataLayout& dataLayout, MLIRContext* context,
                 PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::LoadOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType,
                                              adaptor.getAddr());

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class StoreOpLowering : public OpConversionPattern<cxx::StoreOp> {
 public:
  StoreOpLowering(const TypeConverter& typeConverter,
                  const DataLayout& dataLayout, MLIRContext* context,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::StoreOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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

 private:
  const DataLayout& dataLayout_;
};

class SubscriptOpLowering : public OpConversionPattern<cxx::SubscriptOp> {
 public:
  SubscriptOpLowering(const TypeConverter& typeConverter,
                      const DataLayout& dataLayout, MLIRContext* context,
                      PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::SubscriptOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::SubscriptOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto ptrType = dyn_cast<cxx::PointerType>(op.getBase().getType());

    if (!ptrType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert subscript operation type");
    }

    if (!llvm::isa<cxx::ArrayType>(ptrType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "expected base type of subscript to be an array type");
    }

    SmallVector<LLVM::GEPArg> indices;
    indices.push_back(0);
    indices.push_back(adaptor.getIndex());

    auto resultType = LLVM::LLVMPointerType::get(context);
    auto elementType = typeConverter->convertType(ptrType.getElementType());

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resultType, elementType,
                                             adaptor.getBase(), indices);

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class PtrAddOpLowering : public OpConversionPattern<cxx::PtrAddOp> {
 public:
  PtrAddOpLowering(const TypeConverter& typeConverter,
                   const DataLayout& dataLayout, MLIRContext* context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::PtrAddOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::PtrAddOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert pointer addition result type");
    }

    auto elementType = typeConverter->convertType(
        dyn_cast<cxx::PointerType>(op.getBase().getType()).getElementType());

    SmallVector<LLVM::GEPArg> indices;
    indices.push_back(adaptor.getOffset());

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resultType, elementType,
                                             adaptor.getBase(), indices);

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class MemberOpLowering : public OpConversionPattern<cxx::MemberOp> {
 public:
  MemberOpLowering(const TypeConverter& typeConverter,
                   const DataLayout& dataLayout, MLIRContext* context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::MemberOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::MemberOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto pointerType = cast<cxx::PointerType>(op.getBase().getType());
    auto classType = dyn_cast<cxx::ClassType>(pointerType.getElementType());

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert member result type");
    }

    auto elementType = typeConverter->convertType(classType);

    SmallVector<LLVM::GEPArg> indices;
    indices.push_back(0);
    indices.push_back(adaptor.getMemberIndex());

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resultType, elementType,
                                             adaptor.getBase(), indices);

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class BoolConstantOpLowering : public OpConversionPattern<cxx::BoolConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::BoolConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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

class NullPtrConstantOpLowering
    : public OpConversionPattern<cxx::NullPtrConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::NullPtrConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert nullptr constant type");
    }

    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);

    return success();
  }
};

class IntegralCastOpLowering : public OpConversionPattern<cxx::IntegralCastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IntegralCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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
    const auto isSigned = sourceType.getIsSigned();

    if (sourceType.getWidth() == targetType.getWidth()) {
      // no conversion needed, just replace the op with the value
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    if (targetType.getWidth() < sourceType.getWidth()) {
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultType,
                                                 adaptor.getValue());
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
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert int to bool type");
    }

    auto c0 = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                       adaptor.getValue().getType(), 0);

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, LLVM::ICmpPredicate::ne, adaptor.getValue(), c0);

    return success();
  }
};

class FloatToBoolOpLowering : public OpConversionPattern<cxx::FloatToBoolOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FloatToBoolOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert float to bool type");
    }

    auto c0 = LLVM::ZeroOp::create(rewriter, op.getLoc(),
                                   adaptor.getValue().getType());

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::une, adaptor.getValue(), c0);

    return success();
  }
};

class PtrToBoolOpLowering : public OpConversionPattern<cxx::PtrToBoolOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::PtrToBoolOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert pointer to bool type");
    }

    auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(),
                                     adaptor.getValue().getType());

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, LLVM::ICmpPredicate::ne, adaptor.getValue(), zero);

    return success();
  }
};

class ArrayToPointerOpLowering
    : public OpConversionPattern<cxx::ArrayToPointerOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ArrayToPointerOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto ptrType = dyn_cast<cxx::PointerType>(op.getValue().getType());

    if (!ptrType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert subscript operation type");
    }

    auto arrayType = dyn_cast<cxx::ArrayType>(ptrType.getElementType());
    if (!arrayType) {
      return rewriter.notifyMatchFailure(
          op, "expected base type of subscript to be an array type");
    }

    SmallVector<LLVM::GEPArg> indices;

    indices.push_back(0);
    indices.push_back(0);

    auto resultType = LLVM::LLVMPointerType::get(context);
    auto elementType = typeConverter->convertType(ptrType.getElementType());

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resultType, elementType,
                                             adaptor.getValue(), indices);

    return success();
  }
};

class BoolToIntOpLowering : public OpConversionPattern<cxx::BoolToIntOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::BoolToIntOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert not operation");
    }

    auto c1 = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                       adaptor.getValue().getType(), -1);

    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, resultType, adaptor.getValue(),
                                             c1);

    return success();
  }
};

class AddIOpLowering : public OpConversionPattern<cxx::AddIOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AddIOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
    } else if (isa<cxx::PointerType>(op.getLhs().getType())) {
      predicate = LLVM::ICmpPredicate::ult;
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
                       ConversionPatternRewriter& rewriter) const
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
    } else if (isa<cxx::PointerType>(op.getLhs().getType())) {
      predicate = LLVM::ICmpPredicate::ule;
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
                       ConversionPatternRewriter& rewriter) const
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
    } else if (isa<cxx::PointerType>(op.getLhs().getType())) {
      predicate = LLVM::ICmpPredicate::ugt;
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
                       ConversionPatternRewriter& rewriter) const
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
    } else if (isa<cxx::PointerType>(op.getLhs().getType())) {
      predicate = LLVM::ICmpPredicate::uge;
    }

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, resultType, predicate, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

//
// bitwise operations
//

class AndOpLowering : public OpConversionPattern<cxx::AndOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AndOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert and operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());

    return success();
  }
};

class OrOpLowering : public OpConversionPattern<cxx::OrOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::OrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert or operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, resultType, adaptor.getLhs(),
                                            adaptor.getRhs());

    return success();
  }
};

class XorOpLowering : public OpConversionPattern<cxx::XorOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::XorOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert xor operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());

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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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

class LessThanFOpLowering : public OpConversionPattern<cxx::LessThanFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LessThanFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert less thanf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::olt, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class LessEqualFOpLowering : public OpConversionPattern<cxx::LessEqualFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LessEqualFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert less equalf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::ole, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class GreaterThanFOpLowering : public OpConversionPattern<cxx::GreaterThanFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::GreaterThanFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert greater thanf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::ogt, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class GreaterEqualFOpLowering
    : public OpConversionPattern<cxx::GreaterEqualFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::GreaterEqualFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert greater equalf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::oge, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class EqualFOpLowering : public OpConversionPattern<cxx::EqualFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::EqualFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert equalf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::oeq, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class NotEqualFOpLowering : public OpConversionPattern<cxx::NotEqualFOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::NotEqualFOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert not equalf operation type");
    }

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, resultType, LLVM::FCmpPredicate::one, adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

class FloatingPointCastOpLowering
    : public OpConversionPattern<cxx::FloatingPointCastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::FloatingPointCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
                       ConversionPatternRewriter& rewriter) const
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
  DenseMap<StringRef, Block*> labels;
};

class GotoOpLowering : public OpConversionPattern<cxx::GotoOp> {
 public:
  GotoOpLowering(const TypeConverter& typeConverter,
                 const LabelConverter& labelConverter, MLIRContext* context,
                 PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        labelConverter_(labelConverter) {}

  auto matchAndRewrite(cxx::GotoOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
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
  const LabelConverter& labelConverter_;
};

class LabelOpLowering : public OpConversionPattern<cxx::LabelOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LabelOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto context = getContext();

    rewriter.eraseOp(op);

    return success();
  }
};

class SwitchOpLowering : public OpConversionPattern<cxx::SwitchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::SwitchOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto context = getContext();

    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, adaptor.getValue(), op.getDefaultDestination(),
        adaptor.getDefaultOperands(), *adaptor.getCaseValues(),
        op.getCaseDestinations(), adaptor.getCaseOperands());

    return success();
  }
};

class CxxToLLVMLoweringPass
    : public PassWrapper<CxxToLLVMLoweringPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxxToLLVMLoweringPass)

  auto getArgument() const -> StringRef override { return "cxx-to-llvm"; }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<DLTIDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final;
};

}  // namespace

void CxxToLLVMLoweringPass::runOnOperation() {
  auto context = &getContext();
  auto module = getOperation();

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

  typeConverter.addConversion([&](cxx::ArrayType type) -> Type {
    auto elementType = typeConverter.convertType(type.getElementType());
    auto size = type.getSize();

    return LLVM::LLVMArrayType::get(elementType, size);
  });

  typeConverter.addConversion([&](cxx::FunctionType type) -> Type {
    SmallVector<Type> inputs;
    for (auto argType : type.getInputs()) {
      auto convertedType = typeConverter.convertType(argType);
      inputs.push_back(convertedType);
    }
    SmallVector<Type> results;
    for (auto resultType : type.getResults()) {
      auto convertedType = typeConverter.convertType(resultType);
      results.push_back(convertedType);
    }
    if (results.size() > 1) {
      return {};
    }
    if (results.empty()) {
      results.push_back(LLVM::LLVMVoidType::get(type.getContext()));
    }
    auto context = type.getContext();
    return LLVM::LLVMFunctionType::get(context, results.front(), inputs,
                                       type.getVariadic());
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

  module.walk([&](Operation* op) {
    if (auto labelOp = dyn_cast<cxx::LabelOp>(op)) {
      labelConverter.labels[labelOp.getName()] = labelOp->getBlock();
    }
  });

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<cxx::CxxDialect>();

  RewritePatternSet patterns(context);

  // function operations
  patterns.insert<FuncOpLowering, GlobalOpLowering, ReturnOpLowering,
                  CallOpLowering, AddressOfOpLowering>(typeConverter, context);

  // memory operations
  DataLayout dataLayout{module};

  patterns.insert<AllocaOpLowering, LoadOpLowering, StoreOpLowering,
                  SubscriptOpLowering, MemberOpLowering>(typeConverter,
                                                         dataLayout, context);

  // cast operations
  patterns.insert<IntToBoolOpLowering, FloatToBoolOpLowering,
                  PtrToBoolOpLowering, BoolToIntOpLowering,
                  IntegralCastOpLowering, ArrayToPointerOpLowering>(
      typeConverter, context);

  // constant operations
  patterns.insert<BoolConstantOpLowering, IntConstantOpLowering,
                  FloatConstantOpLowering, NullPtrConstantOpLowering>(
      typeConverter, context);

  // unary operations
  patterns.insert<NotOpLowering>(typeConverter, context);

  // binary operations
  patterns
      .insert<AddIOpLowering, SubIOpLowering, MulIOpLowering, DivIOpLowering,
              ModIOpLowering, ShiftLeftOpLowering, ShiftRightOpLowering>(
          typeConverter, context);

  // pointer arithmetic operations
  patterns.insert<PtrAddOpLowering>(typeConverter, dataLayout, context);

  // comparison operations
  patterns.insert<EqualOpLowering, NotEquaOpLowering, LessThanOpLowering,
                  LessEqualOpLowering, GreaterThanOpLowering,
                  GreaterEqualOpLowering>(typeConverter, context);

  // bitwise operations
  patterns.insert<AndOpLowering, OrOpLowering, XorOpLowering>(typeConverter,
                                                              context);

  // floating point operations
  patterns
      .insert<AddFOpLowering, SubFOpLowering, MulFOpLowering, DivFOpLowering>(
          typeConverter, context);

  // floating point comparison operations
  patterns
      .insert<LessThanFOpLowering, LessEqualFOpLowering, GreaterThanFOpLowering,
              GreaterEqualFOpLowering, EqualFOpLowering, NotEqualFOpLowering>(
          typeConverter, context);

  // floating point cast operations
  patterns.insert<FloatingPointCastOpLowering, IntToFloatOpLowering,
                  FloatToIntOpLowering>(typeConverter, context);

  // control flow operations
  patterns.insert<CondBranchOpLowering, LabelOpLowering, SwitchOpLowering>(
      typeConverter, context);
  patterns.insert<GotoOpLowering>(typeConverter, labelConverter, context);

  populateFunctionOpInterfaceTypeConversionPattern<cxx::FuncOp>(patterns,
                                                                typeConverter);

  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  auto targetTriple =
      mlir::cast<mlir::StringAttr>(module->getAttr("cxx.triple"));

  module->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                  mlir::StringAttr::get(context, targetTriple.str()));

  auto dataLayoutDescr =
      mlir::cast<mlir::StringAttr>(module->getAttr("cxx.data-layout"));

  module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                  mlir::StringAttr::get(context, dataLayoutDescr.str()));
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
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(module))) {
    module.print(llvm::errs());
    return mlir::failure();
  }

  return mlir::success();
}

auto cxx::exportToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext& context)
    -> std::unique_ptr<llvm::Module> {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(module, context);
  module->getContext()->loadDialect<mlir::LLVM::LLVMDialect>();

  return llvmModule;
}
