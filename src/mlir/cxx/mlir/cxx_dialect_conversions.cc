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

#include <cxx/mlir/cxx_dialect_conversions.h>

// cxx
#include <cxx/cxx_fwd.h>
#include <cxx/mlir/cxx_dialect.h>

// mlir
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
#include <mlir/Transforms/RegionUtils.h>

#include <format>

namespace mlir {

namespace {

static auto getBoolMemoryType(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 8);
}

static auto isBoolElementType(cxx::PointerType ptrTy) -> bool {
  return ptrTy.getElementType().isInteger(1);
}

static auto convertLinkage(mlir::cxx::LinkageKind kind)
    -> LLVM::linkage::Linkage {
  switch (kind) {
    case mlir::cxx::LinkageKind::External:
      return LLVM::linkage::Linkage::External;
    case mlir::cxx::LinkageKind::Internal:
      return LLVM::linkage::Linkage::Internal;
    case mlir::cxx::LinkageKind::LinkOnceODR:
      return LLVM::linkage::Linkage::LinkonceODR;
    case mlir::cxx::LinkageKind::WeakODR:
      return LLVM::linkage::Linkage::WeakODR;
    case mlir::cxx::LinkageKind::AvailableExternally:
      return LLVM::linkage::Linkage::AvailableExternally;
    case mlir::cxx::LinkageKind::Appending:
      return LLVM::linkage::Linkage::Appending;
    default:
      return LLVM::linkage::Linkage::External;
  }
}

static auto targetNeedsComdat(ModuleOp module) -> bool {
  auto tripleAttr = module->getAttrOfType<mlir::StringAttr>("cxx.triple");
  if (!tripleAttr) return false;
  llvm::Triple triple(tripleAttr.getValue());
  return triple.isOSBinFormatELF() || triple.isOSBinFormatCOFF();
}

static auto getOrCreateComdat(OpBuilder& rewriter, ModuleOp module,
                              StringRef symbolName) -> SymbolRefAttr {
  auto comdatOp = module.lookupSymbol<LLVM::ComdatOp>("__comdat");
  if (!comdatOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    comdatOp = LLVM::ComdatOp::create(rewriter, module.getLoc(), "__comdat");
    comdatOp.getBody().emplaceBlock();
  }

  auto& comdatBlock = comdatOp.getBody().front();
  for (auto& op : comdatBlock) {
    if (auto sel = dyn_cast<LLVM::ComdatSelectorOp>(op)) {
      if (sel.getSymName() == symbolName) {
        return SymbolRefAttr::get(
            rewriter.getContext(), "__comdat",
            {FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)});
      }
    }
  }

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&comdatBlock);
    LLVM::ComdatSelectorOp::create(rewriter, module.getLoc(), symbolName,
                                   LLVM::comdat::Comdat::Any);
  }

  return SymbolRefAttr::get(
      rewriter.getContext(), "__comdat",
      {FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)});
}

static auto linkageNeedsComdat(LLVM::linkage::Linkage linkage) -> bool {
  return linkage == LLVM::linkage::Linkage::LinkonceODR ||
         linkage == LLVM::linkage::Linkage::WeakODR;
}

class FuncOpLowering : public OpConversionPattern<cxx::FuncOp> {
 public:
  FuncOpLowering(const TypeConverter& typeConverter, bool needsComdat,
                 MLIRContext* context, PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::FuncOp>(typeConverter, context, benefit),
        needsComdat_(needsComdat) {}

  auto matchAndRewrite(cxx::FuncOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    if (failed(convertFunctionTyype(op, rewriter))) {
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    }

    auto funcType = op.getFunctionType();
    auto llvmFuncType = typeConverter->convertType(funcType);

    auto linkage = convertLinkage(
        op.getLinkageKind().value_or(cxx::LinkageKind::External));

    auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), op.getSymName(),
                                         llvmFuncType, linkage);

    if (op.getInlineKind() != cxx::InlineKind::InlineHint) {
      func.setNoInline(true);
    }

    if (op.getBody().empty()) {
      func.setLinkage(LLVM::linkage::Linkage::External);
    } else if (needsComdat_ && linkageNeedsComdat(linkage)) {
      auto module = op->getParentOfType<ModuleOp>();
      auto comdatRef = getOrCreateComdat(rewriter, module, op.getSymName());
      func.setComdatAttr(comdatRef);
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

 private:
  bool needsComdat_;
};

static void emitAggregateInit(ConversionPatternRewriter& rewriter, Location loc,
                              Value& result, Type elementType,
                              ArrayAttr arrAttr);

static Value emitAttrAsValue(ConversionPatternRewriter& rewriter, Location loc,
                             Type type, Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType) intType = IntegerType::get(type.getContext(), 64);
    auto adjusted = rewriter.getIntegerAttr(intType, intAttr.getInt());
    return LLVM::ConstantOp::create(rewriter, loc, intType, adjusted);
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    auto floatType = dyn_cast<FloatType>(type);
    if (!floatType) floatType = Float64Type::get(type.getContext());
    auto adjusted = FloatAttr::get(floatType, floatAttr.getValueAsDouble());
    return LLVM::ConstantOp::create(rewriter, loc, floatType, adjusted);
  }

  if (isa<UnitAttr>(attr) && isa<LLVM::LLVMPointerType>(type)) {
    return LLVM::ZeroOp::create(rewriter, loc, type);
  }

  if (auto arrAttr = dyn_cast<ArrayAttr>(attr)) {
    if (isa<LLVM::LLVMStructType>(type) || isa<LLVM::LLVMArrayType>(type)) {
      Value agg = LLVM::UndefOp::create(rewriter, loc, type);
      emitAggregateInit(rewriter, loc, agg, type, arrAttr);
      return agg;
    }
  }

  return LLVM::ZeroOp::create(rewriter, loc, type);
}

static void emitAggregateInit(ConversionPatternRewriter& rewriter, Location loc,
                              Value& result, Type elementType,
                              ArrayAttr arrAttr) {
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(elementType)) {
    auto body = structType.getBody();
    for (unsigned i = 0; i < arrAttr.size() && i < body.size(); ++i) {
      auto fieldType = body[i];
      auto fieldAttr = arrAttr[i];
      Value fieldVal = emitAttrAsValue(rewriter, loc, fieldType, fieldAttr);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, fieldVal, i);
    }
  } else if (auto arrType = dyn_cast<LLVM::LLVMArrayType>(elementType)) {
    auto elemType = arrType.getElementType();
    for (unsigned i = 0; i < arrAttr.size(); ++i) {
      auto elemAttrVal = arrAttr[i];
      Value elemVal = emitAttrAsValue(rewriter, loc, elemType, elemAttrVal);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, elemVal, i);
    }
  }
}

class GlobalOpLowering : public OpConversionPattern<cxx::GlobalOp> {
 public:
  GlobalOpLowering(const TypeConverter& typeConverter, bool needsComdat,
                   MLIRContext* context, PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::GlobalOp>(typeConverter, context, benefit),
        needsComdat_(needsComdat) {}

  auto matchAndRewrite(cxx::GlobalOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    auto elementType =
        op.getGlobalType().isInteger(1)
            ? getBoolMemoryType(op.getContext())
            : getTypeConverter()->convertType(op.getGlobalType());

    auto linkage = convertLinkage(
        op.getLinkageKind().value_or(cxx::LinkageKind::External));

    Attribute value = adaptor.getValueAttr();
    if (!value && linkage != LLVM::linkage::Linkage::External) {
      value = rewriter.getZeroAttr(elementType);
    }

    bool needsZeroPtrInit = isa_and_nonnull<UnitAttr>(value);

    bool needsWideStringInit = false;
    unsigned wideElementWidth = 0;
    if (auto strAttr = dyn_cast_or_null<StringAttr>(value)) {
      if (auto arrType = dyn_cast<LLVM::LLVMArrayType>(elementType)) {
        if (auto intElType = dyn_cast<IntegerType>(arrType.getElementType())) {
          if (intElType.getWidth() > 8) {
            needsWideStringInit = true;
            wideElementWidth = intElType.getWidth();
          }
        }
      }
    }

    bool needsAggregateInit = isa_and_nonnull<ArrayAttr>(value) &&
                              (isa<LLVM::LLVMStructType>(elementType) ||
                               isa<LLVM::LLVMArrayType>(elementType));

    auto globalOp = LLVM::GlobalOp::create(
        rewriter, op.getLoc(), elementType, op.getConstant(), linkage,
        op.getSymName(),
        (needsZeroPtrInit || needsWideStringInit || needsAggregateInit)
            ? Attribute{}
            : value);

    if (needsZeroPtrInit) {
      auto& region = globalOp.getInitializerRegion();
      auto* block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);
      auto zero = LLVM::ZeroOp::create(rewriter, op.getLoc(), elementType);
      LLVM::ReturnOp::create(rewriter, op.getLoc(), zero.getResult());
    } else if (needsWideStringInit) {
      auto strAttr = cast<StringAttr>(value);
      auto rawBytes = strAttr.getValue();
      auto arrType = cast<LLVM::LLVMArrayType>(elementType);
      auto intElType = cast<IntegerType>(arrType.getElementType());
      unsigned bytesPerElement = wideElementWidth / 8;
      unsigned numElements = arrType.getNumElements();

      auto& region = globalOp.getInitializerRegion();
      auto* block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);

      Value arr = LLVM::UndefOp::create(rewriter, op.getLoc(), elementType);

      for (unsigned i = 0; i < numElements; ++i) {
        uint64_t val = 0;
        for (unsigned b = 0;
             b < bytesPerElement && i * bytesPerElement + b < rawBytes.size();
             ++b) {
          val |= static_cast<uint64_t>(
                     static_cast<uint8_t>(rawBytes[i * bytesPerElement + b]))
                 << (b * 8);
        }
        auto constVal =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), intElType,
                                     rewriter.getIntegerAttr(intElType, val));
        arr = LLVM::InsertValueOp::create(rewriter, op.getLoc(), arr, constVal,
                                          i);
      }

      LLVM::ReturnOp::create(rewriter, op.getLoc(), arr);
    } else if (needsAggregateInit) {
      auto arrAttr = cast<ArrayAttr>(value);
      auto& region = globalOp.getInitializerRegion();
      auto* block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);

      Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), elementType);

      emitAggregateInit(rewriter, op.getLoc(), result, elementType, arrAttr);

      LLVM::ReturnOp::create(rewriter, op.getLoc(), result);
    }

    rewriter.eraseOp(op);

    if (needsComdat_ && linkageNeedsComdat(linkage)) {
      auto module = globalOp->getParentOfType<ModuleOp>();
      auto comdatRef = getOrCreateComdat(rewriter, module, op.getSymName());
      globalOp.setComdatAttr(comdatRef);
    }

    return success();
  }

 private:
  bool needsComdat_;
};

class VTableOpLowering : public OpConversionPattern<cxx::VTableOp> {
 public:
  VTableOpLowering(const TypeConverter& typeConverter, bool needsComdat,
                   MLIRContext* context, PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::VTableOp>(typeConverter, context, benefit),
        needsComdat_(needsComdat) {}

  auto matchAndRewrite(cxx::VTableOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto entries = op.getEntries();
    auto numEntries = entries.size();

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto arrayType = LLVM::LLVMArrayType::get(ptrType, numEntries);

    auto linkage = convertLinkage(
        op.getLinkageKind().value_or(cxx::LinkageKind::External));

    auto globalOp = LLVM::GlobalOp::create(
        rewriter, op.getLoc(), arrayType, /*isConstant=*/true, linkage,
        op.getSymName(), /*value=*/Attribute{});

    if (needsComdat_ && linkageNeedsComdat(linkage)) {
      auto module = op->getParentOfType<ModuleOp>();
      auto comdatRef = getOrCreateComdat(rewriter, module, op.getSymName());
      globalOp.setComdatAttr(comdatRef);
    }

    // Build the initializer region
    auto& region = globalOp.getInitializerRegion();
    auto* block = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(block);

    Value arr = LLVM::UndefOp::create(rewriter, op.getLoc(), arrayType);

    for (auto [i, entry] : llvm::enumerate(entries)) {
      Value element;
      if (auto symRef = mlir::dyn_cast<FlatSymbolRefAttr>(entry)) {
        element =
            LLVM::AddressOfOp::create(rewriter, op.getLoc(), ptrType, symRef);
      } else {
        element = LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
      }
      arr = LLVM::InsertValueOp::create(rewriter, op.getLoc(), arr, element, i);
    }

    LLVM::ReturnOp::create(rewriter, op.getLoc(), arr);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  bool needsComdat_;
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

    LLVM::CallOp llvmCallOp;
    if (!op.getCalleeAttr()) {
      // create an indirect call
      auto inputs = adaptor.getInputs();
      auto callee = inputs[0];
      auto callArgs = inputs.drop_front();

      SmallVector<Type> argTypes;
      for (auto arg : callArgs) {
        argTypes.push_back(arg.getType());
      }

      auto llvmFuncType = LLVM::LLVMFunctionType::get(
          rewriter.getContext(),
          resultTypes.empty() ? rewriter.getType<LLVM::LLVMVoidType>()
                              : resultTypes.front(),
          argTypes, /*isVarArg=*/false);

      llvmCallOp =
          LLVM::CallOp::create(rewriter, op.getLoc(), llvmFuncType, inputs);
    } else {
      llvmCallOp =
          LLVM::CallOp::create(rewriter, op.getLoc(), resultTypes,
                               adaptor.getCallee(), adaptor.getInputs());
    }

    if (op.getVarCalleeType().has_value()) {
      auto varCalleeType =
          typeConverter->convertType(op.getVarCalleeType().value());
      llvmCallOp.setVarCalleeType(cast<LLVM::LLVMFunctionType>(varCalleeType));
    }

    rewriter.replaceOp(op, llvmCallOp);
    return success();
  }
};

class BuiltinCallOpLowering : public OpConversionPattern<cxx::BuiltinCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto loc = op.getLoc();
    auto name = op.getBuiltinName();

    if (name == "__builtin_va_start") {
      if (adaptor.getInputs().size() < 1) {
        return rewriter.notifyMatchFailure(
            op, "va_start expects at least 1 argument");
      }
      LLVM::VaStartOp::create(rewriter, loc, adaptor.getInputs()[0]);
      rewriter.eraseOp(op);
      return success();
    }

    if (name == "__builtin_va_end") {
      if (adaptor.getInputs().size() != 1) {
        return rewriter.notifyMatchFailure(op, "va_end expects 1 argument");
      }
      LLVM::VaEndOp::create(rewriter, loc, adaptor.getInputs()[0]);
      rewriter.eraseOp(op);
      return success();
    }

    if (name == "__builtin_va_copy") {
      if (adaptor.getInputs().size() != 2) {
        return rewriter.notifyMatchFailure(op, "va_copy expects 2 arguments");
      }
      LLVM::VaCopyOp::create(rewriter, loc, adaptor.getInputs()[0],
                             adaptor.getInputs()[1]);
      rewriter.eraseOp(op);
      return success();
    }

    if (name == "__builtin_va_arg") {
      if (adaptor.getInputs().size() != 1) {
        return rewriter.notifyMatchFailure(op, "va_arg expects 1 argument");
      }
      SmallVector<Type> resultTypes;
      if (failed(
              typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert va_arg result type");
      }
      if (resultTypes.empty()) {
        return rewriter.notifyMatchFailure(op, "va_arg must have a result");
      }
      auto vaArgOp = LLVM::VaArgOp::create(rewriter, loc, resultTypes.front(),
                                           adaptor.getInputs()[0]);
      rewriter.replaceOp(op, vaArgOp);
      return success();
    }

    llvm::StringRef funcName = name;
    if (funcName.starts_with("__builtin_")) {
      funcName = funcName.drop_front(std::strlen("__builtin_"));
    }

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert builtin call result types");
    }

    SmallVector<Type> argTypes;
    for (auto arg : adaptor.getInputs()) {
      argTypes.push_back(arg.getType());
    }

    auto module = op->getParentOfType<ModuleOp>();
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto funcType = LLVM::LLVMFunctionType::get(
          rewriter.getContext(),
          resultTypes.empty() ? rewriter.getType<LLVM::LLVMVoidType>()
                              : resultTypes.front(),
          argTypes, /*isVarArg=*/false);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
    }

    auto callOp = LLVM::CallOp::create(rewriter, loc, resultTypes, funcName,
                                       adaptor.getInputs());
    rewriter.replaceOp(op, callOp);
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

    auto elementType = isBoolElementType(ptrTy)
                           ? getBoolMemoryType(context)
                           : typeConverter->convertType(ptrTy.getElementType());

    if (!elementType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert element type of alloca");
    }

    auto size = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

    auto x = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        op, resultType, elementType, size, op.getAlignment());

    if (auto diLocal =
            op->getAttrOfType<LLVM::DILocalVariableAttr>("cxx.di_local")) {
      auto expr = LLVM::DIExpressionAttr::get(context, {});
      LLVM::DbgDeclareOp::create(rewriter, op.getLoc(), x, diLocal, expr);
    }

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

    auto ptrTy = dyn_cast<cxx::PointerType>(op.getAddr().getType());
    if (ptrTy && isBoolElementType(ptrTy)) {
      auto i8Type = getBoolMemoryType(context);
      auto loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), i8Type,
                                         adaptor.getAddr(), op.getAlignment());
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultType, loaded);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
          op, resultType, adaptor.getAddr(), op.getAlignment());
    }

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

    auto ptrTy = dyn_cast<cxx::PointerType>(op.getAddr().getType());
    if (ptrTy && isBoolElementType(ptrTy)) {
      auto i8Type = getBoolMemoryType(context);
      auto extended = LLVM::ZExtOp::create(rewriter, op.getLoc(), i8Type,
                                           adaptor.getValue());
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
          op, extended, adaptor.getAddr(), op.getAlignment());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
          op, adaptor.getValue(), adaptor.getAddr(), op.getAlignment());
    }

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class MemSetZeroOpLowering : public OpConversionPattern<cxx::MemSetZeroOp> {
 public:
  MemSetZeroOpLowering(const TypeConverter& typeConverter,
                       const DataLayout& dataLayout, MLIRContext* context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::MemSetZeroOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::MemSetZeroOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto context = getContext();
    auto loc = op.getLoc();

    auto i8Ty = rewriter.getI8Type();

    auto zeroVal = LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                            rewriter.getI8IntegerAttr(0));

    auto sizeVal =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                 rewriter.getI64IntegerAttr(op.getSize()));

    rewriter.replaceOpWithNewOp<LLVM::MemsetOp>(op, adaptor.getAddr(), zeroVal,
                                                sizeVal,
                                                /*isVolatile=*/false);

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

    if (isa<cxx::ArrayType>(dyn_cast<cxx::PointerType>(op.getBase().getType())
                                .getElementType())) {
      indices.push_back(0);  // dereference the array pointer
    }

    indices.push_back(adaptor.getOffset());

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resultType, elementType,
                                             adaptor.getBase(), indices);

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class PtrDiffOpLowering : public OpConversionPattern<cxx::PtrDiffOp> {
 public:
  PtrDiffOpLowering(const TypeConverter& typeConverter,
                    const DataLayout& dataLayout, MLIRContext* context,
                    PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::PtrDiffOp>(typeConverter, context, benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::PtrDiffOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto context = getContext();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert pointer difference result type");
    }

    auto lhsType = typeConverter->convertType(adaptor.getLhs().getType());
    if (!lhsType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert pointer difference left-hand side type");
    }

    auto rhsType = typeConverter->convertType(adaptor.getRhs().getType());
    if (!rhsType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert pointer difference right-hand side type");
    }

    auto loc = op->getLoc();

    auto lhs =
        LLVM::PtrToIntOp::create(rewriter, loc, resultType, adaptor.getLhs());

    auto rhs =
        LLVM::PtrToIntOp::create(rewriter, loc, resultType, adaptor.getRhs());

    rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, resultType, lhs, rhs);

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

    if (!classType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected class type for member base");
    }

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert member result type");
    }

    auto elementType = typeConverter->convertType(classType);

    auto memberIndex = adaptor.getMemberIndex();

    SmallVector<LLVM::GEPArg> indices;
    indices.push_back(0);
    indices.push_back(memberIndex);

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resultType, elementType,
                                             adaptor.getBase(), indices);

    return success();
  }

 private:
  const DataLayout& dataLayout_;
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

class PtrToIntOpLowering : public OpConversionPattern<cxx::PtrToIntOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::PtrToIntOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert ptr to int type");
    }

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultType,
                                                  adaptor.getValue());

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
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() final;
};

}  // namespace

void CxxToLLVMLoweringPass::runOnOperation() {
  auto context = &getContext();
  auto module = getOperation();

  // set up the type converter
  LLVMTypeConverter typeConverter{context};

  typeConverter.addConversion([](cxx::ExprType type) -> Type { return type; });

  typeConverter.addConversion([](cxx::VoidType type) {
    return LLVM::LLVMVoidType::get(type.getContext());
  });

  typeConverter.addConversion([](cxx::PointerType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  typeConverter.addConversion([&](cxx::ArrayType type) -> Type {
    auto elementType = type.getElementType().isInteger(1)
                           ? getBoolMemoryType(type.getContext())
                           : typeConverter.convertType(type.getElementType());
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
      auto convertedFieldType = field.isInteger(1)
                                    ? getBoolMemoryType(type.getContext())
                                    : typeConverter.convertType(field);
      // todo: check if the field type was converted successfully
      fieldTypes.push_back(convertedFieldType);
    }

    if (fieldTypes.empty()) {
      fieldTypes.push_back(IntegerType::get(type.getContext(), 8));
    }

    if (!fieldTypes.empty()) {
      structType.setBody(fieldTypes, isPacked);
    }

    return structType;
  });

  // set up the conversion patterns
  ConversionTarget target(*context);

  bool needsComdat = targetNeedsComdat(module);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<cxx::CxxDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  // Keep todo ops legal - they signal unresolved AST/type checker issues
  // and must not be lowered to object code.
  target.addLegalOp<cxx::TodoExprOp, cxx::TodoStmtOp>();

  RewritePatternSet patterns(context);

  // globals
  patterns.insert<FuncOpLowering>(typeConverter, needsComdat, context);
  patterns.insert<GlobalOpLowering>(typeConverter, needsComdat, context);
  patterns.insert<VTableOpLowering>(typeConverter, needsComdat, context);
  patterns.insert<ReturnOpLowering, CallOpLowering, BuiltinCallOpLowering,
                  AddressOfOpLowering>(typeConverter, context);

  // memory operations
  DataLayout dataLayout{module};

  patterns.insert<AllocaOpLowering, LoadOpLowering, StoreOpLowering,
                  MemSetZeroOpLowering, SubscriptOpLowering, MemberOpLowering>(
      typeConverter, dataLayout, context);

  // cast operations
  patterns.insert<ArrayToPointerOpLowering, PtrToIntOpLowering>(typeConverter,
                                                                context);

  // constant operations
  patterns.insert<NullPtrConstantOpLowering>(typeConverter, context);

  // pointer arithmetic operations
  patterns.insert<PtrAddOpLowering, PtrDiffOpLowering>(typeConverter,
                                                       dataLayout, context);

  populateFunctionOpInterfaceTypeConversionPattern<cxx::FuncOp>(patterns,
                                                                typeConverter);

  // arith dialect lowering
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

#if false
  {
    // remove unreachable code
    IRRewriter rewriter(context);
    module.walk([&](cxx::FuncOp funcOp) {
      for (auto& region : funcOp->getRegions()) {
        (void)eraseUnreachableBlocks(rewriter, region);
      }
    });
  }
#endif

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

#if false
  pm.addPass(mlir::createCSEPass());
#endif

  if (failed(pm.run(module))) {
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
