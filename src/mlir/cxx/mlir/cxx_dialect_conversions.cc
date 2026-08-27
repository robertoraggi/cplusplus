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

#include <cxx/cxx_fwd.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/token.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Intrinsics.h>
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
constexpr std::uint32_t kDefaultGlobalCtorPriority = 65535;
}
namespace {
static auto getBoolMemoryType(MLIRContext* context) -> IntegerType {
  return IntegerType::get(context, 8);
}

static auto isBoolElementType(cxx::PointerType ptrTy) -> bool {
  return ptrTy.getElementType().isInteger(1);
}

static auto foldConstantInt(Value value) -> std::optional<std::int64_t> {
  if (auto constOp = value.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

static auto atomicOrderingFromValue(Value value) -> LLVM::AtomicOrdering {
  auto folded = foldConstantInt(value);
  if (!folded) return LLVM::AtomicOrdering::seq_cst;

  switch (*folded) {
    case 0:
      return LLVM::AtomicOrdering::monotonic;
    case 1:
    case 2:
      return LLVM::AtomicOrdering::acquire;
    case 3:
      return LLVM::AtomicOrdering::release;
    case 4:
      return LLVM::AtomicOrdering::acq_rel;
    case 5:
      return LLVM::AtomicOrdering::seq_cst;
    default:
      return LLVM::AtomicOrdering::seq_cst;
  }
}

static auto atomicIntegerType(Type llvmElementType, MLIRContext* context,
                              const DataLayout& dataLayout)
    -> std::optional<IntegerType> {
  if (!llvmElementType) return std::nullopt;

  switch (dataLayout.getTypeSize(llvmElementType)) {
    case 1:
    case 2:
    case 4:
    case 8:
      return IntegerType::get(context,
                              dataLayout.getTypeSizeInBits(llvmElementType));
    default:
      return std::nullopt;
  }
}

using ::cxx::BuiltinFunctionKind;
using ::cxx::Token;

constexpr StringRef kVaArgKeywordSyntaxBuiltinName = "__builtin_va_arg";

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

static auto convertVisibility(mlir::cxx::Visibility visibility)
    -> LLVM::Visibility {
  switch (visibility) {
    case mlir::cxx::Visibility::Hidden:
      return LLVM::Visibility::Hidden;
    case mlir::cxx::Visibility::Protected:
      return LLVM::Visibility::Protected;
    case mlir::cxx::Visibility::Default:
      return LLVM::Visibility::Default;
  }
  return LLVM::Visibility::Default;
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

    if (auto visibility = op.getVisibility_()) {
      func.setVisibility_(convertVisibility(*visibility));
    }

    if (auto aliasName = op.getAliasName()) {
      emitAlias(rewriter, op.getLoc(), func, llvmFuncType, *aliasName);
    }

    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());

    rewriter.eraseOp(op);

    return success();
  }

  static void emitAlias(ConversionPatternRewriter& rewriter, Location loc,
                        LLVM::LLVMFuncOp func, Type aliasType,
                        StringRef aliasName) {
    auto module = func->getParentOfType<ModuleOp>();
    if (!module || module.lookupSymbol(aliasName)) return;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    auto alias = LLVM::AliasOp::create(rewriter, loc, aliasType,
                                       func.getLinkage(), aliasName);
    alias.setVisibility_(LLVM::Visibility::Hidden);

    auto block = rewriter.createBlock(&alias.getInitializerRegion());
    rewriter.setInsertionPointToStart(block);
    auto ptrType = LLVM::LLVMPointerType::get(func.getContext());
    auto addr = LLVM::AddressOfOp::create(
        rewriter, loc, ptrType,
        FlatSymbolRefAttr::get(func.getContext(), func.getSymName()));
    LLVM::ReturnOp::create(rewriter, loc, addr.getResult());
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
                              ArrayAttr arrAttr, ModuleOp module = nullptr);

static std::string createStringGlobal(ConversionPatternRewriter& rewriter,
                                      Location loc, ModuleOp module,
                                      StringRef content) {
  unsigned idx = 0;
  std::string name;
  do {
    name = std::format(".cxx.str.{}", idx++);
  } while (mlir::SymbolTable::lookupSymbolIn(module, name));
  auto context = module.getContext();
  auto i8Type = IntegerType::get(context, 8);
  auto strType = LLVM::LLVMArrayType::get(i8Type, content.size());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  LLVM::GlobalOp::create(rewriter, loc, strType, /*isConstant=*/true,
                         LLVM::linkage::Linkage::Internal, name,
                         rewriter.getStringAttr(content));
  return name;
}

static Value emitAttrAsValue(ConversionPatternRewriter& rewriter, Location loc,
                             Type type, Attribute attr,
                             ModuleOp module = nullptr) {
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

  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    if (isa<LLVM::LLVMPointerType>(type) && module) {
      auto ptrType = cast<LLVM::LLVMPointerType>(type);
      auto strName =
          createStringGlobal(rewriter, loc, module, strAttr.getValue());
      return LLVM::AddressOfOp::create(
          rewriter, loc, ptrType,
          FlatSymbolRefAttr::get(loc.getContext(), strName));
    }
  }

  if (auto arrAttr = dyn_cast<ArrayAttr>(attr)) {
    if (isa<LLVM::LLVMStructType>(type) || isa<LLVM::LLVMArrayType>(type)) {
      Value agg = LLVM::ZeroOp::create(rewriter, loc, type);
      emitAggregateInit(rewriter, loc, agg, type, arrAttr, module);
      return agg;
    }
  }

  return LLVM::ZeroOp::create(rewriter, loc, type);
}

static void emitAggregateInit(ConversionPatternRewriter& rewriter, Location loc,
                              Value& result, Type elementType,
                              ArrayAttr arrAttr, ModuleOp module) {
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(elementType)) {
    auto body = structType.getBody();
    for (unsigned i = 0; i < arrAttr.size() && i < body.size(); ++i) {
      auto fieldType = body[i];
      auto fieldAttr = arrAttr[i];
      Value fieldVal =
          emitAttrAsValue(rewriter, loc, fieldType, fieldAttr, module);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, fieldVal, i);
    }
  } else if (auto arrType = dyn_cast<LLVM::LLVMArrayType>(elementType)) {
    auto elemType = arrType.getElementType();
    for (unsigned i = 0; i < arrAttr.size(); ++i) {
      auto elemAttrVal = arrAttr[i];
      Value elemVal =
          emitAttrAsValue(rewriter, loc, elemType, elemAttrVal, module);
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

    const auto hasRegionInit = !op.getInitializer().empty();

    Attribute value = adaptor.getValueAttr();
    if (!value && !hasRegionInit &&
        linkage != LLVM::linkage::Linkage::External) {
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

    bool needsStringPtrInit = isa_and_nonnull<StringAttr>(value) &&
                              isa<LLVM::LLVMPointerType>(elementType);

    auto globalOp = LLVM::GlobalOp::create(
        rewriter, op.getLoc(), elementType, op.getConstant(), linkage,
        op.getSymName(),
        (needsZeroPtrInit || needsWideStringInit || needsAggregateInit ||
         needsStringPtrInit || hasRegionInit)
            ? Attribute{}
            : value);

    if (auto alignment = op.getAlignment()) {
      globalOp.setAlignment(*alignment);
    }

    if (hasRegionInit) {
      auto& llvmRegion = globalOp.getInitializerRegion();
      rewriter.inlineRegionBefore(op.getInitializer(), llvmRegion,
                                  llvmRegion.end());
      if (failed(rewriter.convertRegionTypes(&llvmRegion, *typeConverter))) {
        return failure();
      }
    } else if (needsZeroPtrInit) {
      auto& region = globalOp.getInitializerRegion();
      auto block = rewriter.createBlock(&region);
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
      auto block = rewriter.createBlock(&region);
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
    } else if (needsStringPtrInit) {
      auto strAttr = cast<StringAttr>(value);
      auto module = op->getParentOfType<ModuleOp>();
      auto strName =
          createStringGlobal(rewriter, op.getLoc(), module, strAttr.getValue());
      auto& region = globalOp.getInitializerRegion();
      auto block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);
      auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
      auto addr = LLVM::AddressOfOp::create(
          rewriter, op.getLoc(), ptrType,
          FlatSymbolRefAttr::get(op.getContext(), strName));
      LLVM::ReturnOp::create(rewriter, op.getLoc(), addr.getResult());
    } else if (needsAggregateInit) {
      auto arrAttr = cast<ArrayAttr>(value);
      auto module = op->getParentOfType<ModuleOp>();
      auto& region = globalOp.getInitializerRegion();
      auto block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);

      Value result = LLVM::ZeroOp::create(rewriter, op.getLoc(), elementType);

      emitAggregateInit(rewriter, op.getLoc(), result, elementType, arrAttr,
                        module);

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
    auto vbaseOffsets = op.getVbaseOffsets();
    auto vcallOffsets = op.getVcallOffsets();
    auto slots = op.getSlots();

    auto numEntries =
        vbaseOffsets.size() + vcallOffsets.size() + 2 + slots.size();

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

    auto& region = globalOp.getInitializerRegion();
    auto block = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(block);

    auto nullPtr = [&]() -> Value {
      return LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
    };

    auto offsetWord = [&](std::int64_t offset) -> Value {
      if (offset == 0) return nullPtr();
      auto offsetConst =
          LLVM::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI64Type(),
                                   rewriter.getI64IntegerAttr(offset));
      return LLVM::IntToPtrOp::create(rewriter, op.getLoc(), ptrType,
                                      offsetConst.getResult());
    };

    Value arr = LLVM::UndefOp::create(rewriter, op.getLoc(), arrayType);
    std::int64_t index = 0;

    auto append = [&](Value element) {
      arr = LLVM::InsertValueOp::create(rewriter, op.getLoc(), arr, element,
                                        index++);
    };

    for (auto entry : vbaseOffsets) {
      append(offsetWord(mlir::cast<IntegerAttr>(entry).getInt()));
    }

    for (auto entry : vcallOffsets) {
      append(offsetWord(mlir::cast<IntegerAttr>(entry).getInt()));
    }

    append(offsetWord(op.getOffsetToTop()));

    if (auto typeInfo = op.getTypeInfo()) {
      append(LLVM::AddressOfOp::create(
          rewriter, op.getLoc(), ptrType,
          FlatSymbolRefAttr::get(rewriter.getContext(), *typeInfo)));
    } else {
      append(nullPtr());
    }

    for (auto entry : slots) {
      if (auto symRef = mlir::dyn_cast<FlatSymbolRefAttr>(entry)) {
        append(
            LLVM::AddressOfOp::create(rewriter, op.getLoc(), ptrType, symRef));
      } else {
        append(nullPtr());
      }
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
    rewriter.replaceOp(op, LLVM::ReturnOp::create(rewriter, op.getLoc(),
                                                  adaptor.getOperands()));
    return success();
  }
};

class UnreachableOpLowering : public OpConversionPattern<cxx::UnreachableOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::UnreachableOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
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

    for (auto argType : op.getOperandTypes()) {
      if (!typeConverter->convertType(argType)) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert call argument type");
      }
    }

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert call result types");
    }

    LLVM::CallOp llvmCallOp;
    if (auto calleeOperand = adaptor.getCalleeOperand()) {
      SmallVector<Type> argTypes;
      for (auto arg : adaptor.getInputs()) {
        argTypes.push_back(arg.getType());
      }

      auto llvmFuncType = LLVM::LLVMFunctionType::get(
          rewriter.getContext(),
          resultTypes.empty() ? LLVM::LLVMVoidType::get(rewriter.getContext())
                              : resultTypes.front(),
          argTypes, /*isVarArg=*/false);

      SmallVector<Value> operands{calleeOperand};
      operands.append(adaptor.getInputs().begin(), adaptor.getInputs().end());

      llvmCallOp =
          LLVM::CallOp::create(rewriter, op.getLoc(), llvmFuncType, operands);
    } else {
      llvmCallOp = LLVM::CallOp::create(rewriter, op.getLoc(), resultTypes,
                                        op.getCalleeAttr().getAttr(),
                                        adaptor.getInputs());
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
  BuiltinCallOpLowering(const TypeConverter& typeConverter,
                        const DataLayout& dataLayout, MLIRContext* context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::BuiltinCallOp>(typeConverter, context,
                                                benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (op.getBuiltinName() == kVaArgKeywordSyntaxBuiltinName) {
      return lowerVaArg(op, adaptor, rewriter);
    }

    auto kind = Token::builtinFunctionKind(op.getBuiltinName());

    switch (kind) {
      case BuiltinFunctionKind::T___BUILTIN_VA_START:
      case BuiltinFunctionKind::T___BUILTIN_C23_VA_START:
        return lowerVaStart(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___BUILTIN_VA_END:
        return lowerVaEnd(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___BUILTIN_VA_COPY:
        return lowerVaCopy(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___BUILTIN_ASSUME_ALIGNED:
        return lowerAssumeAligned(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___BUILTIN_BSWAP32:
      case BuiltinFunctionKind::T___BUILTIN_BSWAP64:
        return lowerSimpleIntrinsic(op, adaptor, rewriter, "llvm.bswap");

      case BuiltinFunctionKind::T___BUILTIN_MEMCPY:
        return lowerMemIntrinsic(op, adaptor, rewriter, "llvm.memcpy");

      case BuiltinFunctionKind::T___BUILTIN_MEMMOVE:
        return lowerMemIntrinsic(op, adaptor, rewriter, "llvm.memmove");

      case BuiltinFunctionKind::T___BUILTIN_MEMSET:
        return lowerMemIntrinsic(op, adaptor, rewriter, "llvm.memset");

      case BuiltinFunctionKind::T___BUILTIN_CTZ:
      case BuiltinFunctionKind::T___BUILTIN_CTZL:
      case BuiltinFunctionKind::T___BUILTIN_CTZLL:
      case BuiltinFunctionKind::T___BUILTIN_CTZG:
        return lowerSimpleIntrinsic(op, adaptor, rewriter, "llvm.cttz");

      case BuiltinFunctionKind::T___BUILTIN_CLZG:
        return lowerSimpleIntrinsic(op, adaptor, rewriter, "llvm.ctlz");

      case BuiltinFunctionKind::T___ATOMIC_LOAD_N:
      case BuiltinFunctionKind::T___C11_ATOMIC_LOAD:
        return lowerAtomicLoad(op, adaptor, rewriter, /*hasOutParam=*/false);
      case BuiltinFunctionKind::T___ATOMIC_LOAD:
        return lowerAtomicLoad(op, adaptor, rewriter, /*hasOutParam=*/true);

      case BuiltinFunctionKind::T___ATOMIC_STORE_N:
      case BuiltinFunctionKind::T___C11_ATOMIC_STORE:
        return lowerAtomicStore(op, adaptor, rewriter, /*hasOutParam=*/false);
      case BuiltinFunctionKind::T___ATOMIC_STORE:
        return lowerAtomicStore(op, adaptor, rewriter, /*hasOutParam=*/true);

      case BuiltinFunctionKind::T___C11_ATOMIC_INIT:
        return lowerAtomicInit(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___ATOMIC_EXCHANGE_N:
      case BuiltinFunctionKind::T___C11_ATOMIC_EXCHANGE:
        return lowerAtomicExchange(op, adaptor, rewriter,
                                   /*hasOutParam=*/false);
      case BuiltinFunctionKind::T___ATOMIC_EXCHANGE:
        return lowerAtomicExchange(op, adaptor, rewriter,
                                   /*hasOutParam=*/true);

      case BuiltinFunctionKind::T___ATOMIC_COMPARE_EXCHANGE_N:
        return lowerAtomicCompareExchange(op, adaptor, rewriter,
                                          /*hasOutParam=*/false,
                                          /*fixedWeak=*/std::nullopt);
      case BuiltinFunctionKind::T___ATOMIC_COMPARE_EXCHANGE:
        return lowerAtomicCompareExchange(op, adaptor, rewriter,
                                          /*hasOutParam=*/true,
                                          /*fixedWeak=*/std::nullopt);
      case BuiltinFunctionKind::T___C11_ATOMIC_COMPARE_EXCHANGE_STRONG:
        return lowerAtomicCompareExchange(op, adaptor, rewriter,
                                          /*hasOutParam=*/false,
                                          /*fixedWeak=*/false);
      case BuiltinFunctionKind::T___C11_ATOMIC_COMPARE_EXCHANGE_WEAK:
        return lowerAtomicCompareExchange(op, adaptor, rewriter,
                                          /*hasOutParam=*/false,
                                          /*fixedWeak=*/true);

      case BuiltinFunctionKind::T___ATOMIC_ADD_FETCH:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::add,
                              /*returnsPostOp=*/true);
      case BuiltinFunctionKind::T___ATOMIC_SUB_FETCH:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::sub,
                              /*returnsPostOp=*/true);
      case BuiltinFunctionKind::T___ATOMIC_AND_FETCH:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::_and,
                              /*returnsPostOp=*/true);
      case BuiltinFunctionKind::T___ATOMIC_OR_FETCH:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::_or,
                              /*returnsPostOp=*/true);
      case BuiltinFunctionKind::T___ATOMIC_XOR_FETCH:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::_xor,
                              /*returnsPostOp=*/true);
      case BuiltinFunctionKind::T___ATOMIC_NAND_FETCH:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::nand,
                              /*returnsPostOp=*/true);

      case BuiltinFunctionKind::T___ATOMIC_FETCH_ADD:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::add,
                              /*returnsPostOp=*/false);
      case BuiltinFunctionKind::T___C11_ATOMIC_FETCH_ADD:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::add,
                              /*returnsPostOp=*/false,
                              /*scalePointerAddend=*/true);
      case BuiltinFunctionKind::T___ATOMIC_FETCH_SUB:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::sub,
                              /*returnsPostOp=*/false);
      case BuiltinFunctionKind::T___C11_ATOMIC_FETCH_SUB:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::sub,
                              /*returnsPostOp=*/false,
                              /*scalePointerAddend=*/true);
      case BuiltinFunctionKind::T___ATOMIC_FETCH_AND:
      case BuiltinFunctionKind::T___C11_ATOMIC_FETCH_AND:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::_and,
                              /*returnsPostOp=*/false);
      case BuiltinFunctionKind::T___ATOMIC_FETCH_OR:
      case BuiltinFunctionKind::T___C11_ATOMIC_FETCH_OR:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::_or,
                              /*returnsPostOp=*/false);
      case BuiltinFunctionKind::T___ATOMIC_FETCH_XOR:
      case BuiltinFunctionKind::T___C11_ATOMIC_FETCH_XOR:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::_xor,
                              /*returnsPostOp=*/false);
      case BuiltinFunctionKind::T___ATOMIC_FETCH_NAND:
      case BuiltinFunctionKind::T___C11_ATOMIC_FETCH_NAND:
        return lowerAtomicRmw(op, adaptor, rewriter, LLVM::AtomicBinOp::nand,
                              /*returnsPostOp=*/false);

      case BuiltinFunctionKind::T___ATOMIC_TEST_AND_SET:
        return lowerAtomicTestAndSet(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___ATOMIC_CLEAR:
        return lowerAtomicClear(op, adaptor, rewriter);

      case BuiltinFunctionKind::T___ATOMIC_THREAD_FENCE:
      case BuiltinFunctionKind::T___C11_ATOMIC_THREAD_FENCE:
        return lowerAtomicFence(op, adaptor, rewriter,
                                /*singleThread=*/false);
      case BuiltinFunctionKind::T___ATOMIC_SIGNAL_FENCE:
      case BuiltinFunctionKind::T___C11_ATOMIC_SIGNAL_FENCE:
        return lowerAtomicFence(op, adaptor, rewriter, /*singleThread=*/true);

      case BuiltinFunctionKind::T___ATOMIC_ALWAYS_LOCK_FREE:
      case BuiltinFunctionKind::T___ATOMIC_IS_LOCK_FREE:
      case BuiltinFunctionKind::T___C11_ATOMIC_IS_LOCK_FREE:
        return lowerAtomicIsLockFree(op, adaptor, rewriter);

      default:
        return rewriter.notifyMatchFailure(op, "unknown builtin");
    }
  }

 private:
  auto lowerVaStart(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    if (adaptor.getInputs().empty()) {
      return rewriter.notifyMatchFailure(
          op, "va_start expects at least 1 argument");
    }
    LLVM::VaStartOp::create(rewriter, op.getLoc(), adaptor.getInputs()[0]);
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerVaEnd(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const -> LogicalResult {
    if (adaptor.getInputs().size() != 1) {
      return rewriter.notifyMatchFailure(op, "va_end expects 1 argument");
    }
    LLVM::VaEndOp::create(rewriter, op.getLoc(), adaptor.getInputs()[0]);
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerVaCopy(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const -> LogicalResult {
    if (adaptor.getInputs().size() != 2) {
      return rewriter.notifyMatchFailure(op, "va_copy expects 2 arguments");
    }
    LLVM::VaCopyOp::create(rewriter, op.getLoc(), adaptor.getInputs()[0],
                           adaptor.getInputs()[1]);
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerAssumeAligned(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    if (adaptor.getInputs().size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "assume_aligned expects at least 2 arguments");
    }
    auto loc = op.getLoc();
    auto trueBit = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                            rewriter.getBoolAttr(true));
    LLVM::AssumeOp::create(rewriter, loc, trueBit, "align",
                           adaptor.getInputs());
    rewriter.replaceOp(op, adaptor.getInputs()[0]);
    return success();
  }

  auto lowerVaArg(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const -> LogicalResult {
    if (adaptor.getInputs().size() != 1) {
      return rewriter.notifyMatchFailure(op, "va_arg expects 1 argument");
    }
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert va_arg result type");
    }
    if (resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "va_arg must have a result");
    }
    auto vaArgOp = LLVM::VaArgOp::create(
        rewriter, op.getLoc(), resultTypes.front(), adaptor.getInputs()[0]);
    rewriter.replaceOp(op, vaArgOp);
    return success();
  }

  auto lowerSimpleIntrinsic(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter& rewriter,
                            StringRef intrinsicName) const -> LogicalResult {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert builtin call result types");
    }

    Type resultType;
    if (!resultTypes.empty()) resultType = resultTypes.front();

    auto intrinsicOp = LLVM::CallIntrinsicOp::create(
        rewriter, op.getLoc(), resultType,
        rewriter.getStringAttr(intrinsicName), adaptor.getInputs());

    rewriter.replaceOp(op, intrinsicOp);
    return success();
  }

  auto lowerMemIntrinsic(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter& rewriter,
                         StringRef intrinsicName) const -> LogicalResult {
    auto loc = op.getLoc();

    std::vector<Value> inputs;
    for (auto input : adaptor.getInputs()) inputs.push_back(input);

    inputs.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0)));

    LLVM::CallIntrinsicOp::create(rewriter, loc,
                                  rewriter.getStringAttr(intrinsicName),
                                  mlir::ValueRange{inputs});

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, inputs[0]);
    }
    return success();
  }

  auto atomicPointeeIntegerType(cxx::BuiltinCallOp op, unsigned argIndex) const
      -> std::optional<IntegerType> {
    if (argIndex >= op.getInputs().size()) return std::nullopt;
    auto ptrTy = dyn_cast<cxx::PointerType>(op.getInputs()[argIndex].getType());
    if (!ptrTy) return std::nullopt;
    Type llvmElementType =
        isBoolElementType(ptrTy)
            ? getBoolMemoryType(getContext())
            : getTypeConverter()->convertType(ptrTy.getElementType());
    return atomicIntegerType(llvmElementType, getContext(), dataLayout_);
  }

  auto atomicPointerElementStrideBytes(cxx::BuiltinCallOp op,
                                       unsigned argIndex) const
      -> std::optional<std::int64_t> {
    if (argIndex >= op.getInputs().size()) return std::nullopt;
    auto outerPtrTy =
        dyn_cast<cxx::PointerType>(op.getInputs()[argIndex].getType());
    if (!outerPtrTy) return std::nullopt;
    auto innerPtrTy = dyn_cast<cxx::PointerType>(outerPtrTy.getElementType());
    if (!innerPtrTy) return std::nullopt;
    auto pointeeType =
        getTypeConverter()->convertType(innerPtrTy.getElementType());
    if (!pointeeType ||
        isa<LLVM::LLVMVoidType, LLVM::LLVMFunctionType>(pointeeType)) {
      return std::nullopt;
    }
    return dataLayout_.getTypeSize(pointeeType);
  }

  auto lowerAtomicLoad(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter,
                       bool hasOutParam) const -> LogicalResult {
    auto loc = op.getLoc();

    if (hasOutParam) {
      if (adaptor.getInputs().size() < 3) {
        return rewriter.notifyMatchFailure(op,
                                           "atomic_load expects 3 arguments");
      }
      auto elementType = atomicPointeeIntegerType(op, 0);
      if (!elementType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported atomic_load element type");
      }
      auto width = static_cast<unsigned>(dataLayout_.getTypeSize(*elementType));
      auto order = atomicOrderingFromValue(adaptor.getInputs()[2]);
      auto loaded = LLVM::LoadOp::create(rewriter, loc, *elementType,
                                         adaptor.getInputs()[0], width, false,
                                         false, false, false, order);
      LLVM::StoreOp::create(rewriter, loc, loaded, adaptor.getInputs()[1],
                            width, false, false, false,
                            LLVM::AtomicOrdering::not_atomic);
      rewriter.eraseOp(op);
      return success();
    }

    if (adaptor.getInputs().size() < 2) {
      return rewriter.notifyMatchFailure(op, "atomic_load expects 2 arguments");
    }
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes)) ||
        resultTypes.empty()) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert atomic_load result type");
    }
    auto width =
        static_cast<unsigned>(dataLayout_.getTypeSize(resultTypes.front()));
    auto order = atomicOrderingFromValue(adaptor.getInputs()[1]);
    auto loaded = LLVM::LoadOp::create(rewriter, loc, resultTypes.front(),
                                       adaptor.getInputs()[0], width, false,
                                       false, false, false, order);
    rewriter.replaceOp(op, loaded);
    return success();
  }

  auto lowerAtomicStore(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter& rewriter,
                        bool hasOutParam) const -> LogicalResult {
    auto loc = op.getLoc();

    if (hasOutParam) {
      if (adaptor.getInputs().size() < 3) {
        return rewriter.notifyMatchFailure(op,
                                           "atomic_store expects 3 arguments");
      }
      auto elementType = atomicPointeeIntegerType(op, 0);
      if (!elementType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported atomic_store element type");
      }
      auto width = static_cast<unsigned>(dataLayout_.getTypeSize(*elementType));
      auto value = LLVM::LoadOp::create(
          rewriter, loc, *elementType, adaptor.getInputs()[1], width, false,
          false, false, false, LLVM::AtomicOrdering::not_atomic);
      auto order = atomicOrderingFromValue(adaptor.getInputs()[2]);
      LLVM::StoreOp::create(rewriter, loc, value, adaptor.getInputs()[0], width,
                            false, false, false, order);
      rewriter.eraseOp(op);
      return success();
    }

    if (adaptor.getInputs().size() < 3) {
      return rewriter.notifyMatchFailure(op,
                                         "atomic_store expects 3 arguments");
    }
    auto value = adaptor.getInputs()[1];
    auto width =
        static_cast<unsigned>(dataLayout_.getTypeSize(value.getType()));
    auto order = atomicOrderingFromValue(adaptor.getInputs()[2]);
    LLVM::StoreOp::create(rewriter, loc, value, adaptor.getInputs()[0], width,
                          false, false, false, order);
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerAtomicInit(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    if (adaptor.getInputs().size() < 2) {
      return rewriter.notifyMatchFailure(op, "atomic_init expects 2 arguments");
    }
    auto value = adaptor.getInputs()[1];
    auto width =
        static_cast<unsigned>(dataLayout_.getTypeSize(value.getType()));
    LLVM::StoreOp::create(rewriter, op.getLoc(), value, adaptor.getInputs()[0],
                          width, false, false, false,
                          LLVM::AtomicOrdering::not_atomic);
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerAtomicExchange(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter& rewriter,
                           bool hasOutParam) const -> LogicalResult {
    auto loc = op.getLoc();

    if (hasOutParam) {
      if (adaptor.getInputs().size() < 4) {
        return rewriter.notifyMatchFailure(
            op, "atomic_exchange expects 4 arguments");
      }
      auto elementType = atomicPointeeIntegerType(op, 0);
      if (!elementType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported atomic_exchange element type");
      }
      auto width = static_cast<unsigned>(dataLayout_.getTypeSize(*elementType));
      auto newValue = LLVM::LoadOp::create(
          rewriter, loc, *elementType, adaptor.getInputs()[1], width, false,
          false, false, false, LLVM::AtomicOrdering::not_atomic);
      auto order = atomicOrderingFromValue(adaptor.getInputs()[3]);
      auto old =
          LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg,
                                    adaptor.getInputs()[0], newValue, order);
      LLVM::StoreOp::create(rewriter, loc, old, adaptor.getInputs()[2], width,
                            false, false, false,
                            LLVM::AtomicOrdering::not_atomic);
      rewriter.eraseOp(op);
      return success();
    }

    if (adaptor.getInputs().size() < 3) {
      return rewriter.notifyMatchFailure(op,
                                         "atomic_exchange expects 3 arguments");
    }
    auto newValue = adaptor.getInputs()[1];
    auto order = atomicOrderingFromValue(adaptor.getInputs()[2]);
    auto old =
        LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg,
                                  adaptor.getInputs()[0], newValue, order);
    rewriter.replaceOp(op, old);
    return success();
  }

  auto reinterpretAsInteger(Value value, IntegerType intType, Location loc,
                            ConversionPatternRewriter& rewriter) const
      -> Value {
    if (value.getType() == intType) return value;
    if (isa<LLVM::LLVMPointerType>(value.getType())) {
      return LLVM::PtrToIntOp::create(rewriter, loc, intType, value);
    }
    return LLVM::BitcastOp::create(rewriter, loc, intType, value);
  }

  auto lowerAtomicCompareExchange(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter,
                                  bool hasOutParam,
                                  std::optional<bool> fixedWeak) const
      -> LogicalResult {
    auto loc = op.getLoc();

    auto resolved = atomicPointeeIntegerType(op, 0);
    if (!resolved) {
      return rewriter.notifyMatchFailure(
          op, "unsupported compare_exchange element type");
    }
    IntegerType elementType = *resolved;
    auto width = static_cast<unsigned>(dataLayout_.getTypeSize(elementType));

    Value expectedPtr;
    Value desiredValue;
    Value weakValue;
    Value successOrderValue;
    Value failureOrderValue;

    if (hasOutParam) {
      if (adaptor.getInputs().size() < 6) {
        return rewriter.notifyMatchFailure(
            op, "atomic_compare_exchange expects 6 arguments");
      }
      expectedPtr = adaptor.getInputs()[1];
      desiredValue = LLVM::LoadOp::create(
          rewriter, loc, elementType, adaptor.getInputs()[2], width, false,
          false, false, false, LLVM::AtomicOrdering::not_atomic);
      weakValue = adaptor.getInputs()[3];
      successOrderValue = adaptor.getInputs()[4];
      failureOrderValue = adaptor.getInputs()[5];
    } else {
      const std::size_t minArgs = fixedWeak.has_value() ? 5 : 6;
      if (adaptor.getInputs().size() < minArgs) {
        return rewriter.notifyMatchFailure(
            op, "compare_exchange expects more arguments");
      }
      expectedPtr = adaptor.getInputs()[1];
      desiredValue = reinterpretAsInteger(adaptor.getInputs()[2], elementType,
                                          loc, rewriter);
      if (fixedWeak.has_value()) {
        successOrderValue = adaptor.getInputs()[3];
        failureOrderValue = adaptor.getInputs()[4];
      } else {
        weakValue = adaptor.getInputs()[3];
        successOrderValue = adaptor.getInputs()[4];
        failureOrderValue = adaptor.getInputs()[5];
      }
    }

    auto expected = LLVM::LoadOp::create(
        rewriter, loc, elementType, expectedPtr, width, false, false, false,
        false, LLVM::AtomicOrdering::not_atomic);

    bool isWeak = fixedWeak.value_or(false);
    if (!fixedWeak.has_value() && weakValue) {
      if (auto folded = foldConstantInt(weakValue)) isWeak = *folded != 0;
    }

    auto successOrder = atomicOrderingFromValue(successOrderValue);
    auto failureOrder = atomicOrderingFromValue(failureOrderValue);

    auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, adaptor.getInputs()[0], expected, desiredValue,
        successOrder, failureOrder, /*syncscope=*/StringRef(), width, isWeak,
        /*isVolatile=*/false);

    auto oldValue = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg,
                                                 ArrayRef<std::int64_t>{0});
    auto succeeded = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg,
                                                  ArrayRef<std::int64_t>{1});

    LLVM::StoreOp::create(rewriter, loc, oldValue, expectedPtr, width, false,
                          false, false, LLVM::AtomicOrdering::not_atomic);

    rewriter.replaceOp(op, succeeded);
    return success();
  }

  auto lowerAtomicRmw(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter& rewriter,
                      LLVM::AtomicBinOp binOp, bool returnsPostOp,
                      bool scalePointerAddend = false) const -> LogicalResult {
    if (adaptor.getInputs().size() < 3) {
      return rewriter.notifyMatchFailure(op, "atomic RMW expects 3 arguments");
    }
    auto loc = op.getLoc();
    auto ptr = adaptor.getInputs()[0];
    auto val = adaptor.getInputs()[1];
    auto order = atomicOrderingFromValue(adaptor.getInputs()[2]);

    if (scalePointerAddend) {
      if (auto stride = atomicPointerElementStrideBytes(op, 0);
          stride && *stride > 1) {
        auto strideConst = LLVM::ConstantOp::create(
            rewriter, loc, val.getType(),
            rewriter.getIntegerAttr(val.getType(), *stride));
        val = LLVM::MulOp::create(rewriter, loc, val, strideConst);
      }
    }

    const bool isFloat = isa<FloatType>(val.getType());
    if (isFloat && binOp == LLVM::AtomicBinOp::add) {
      binOp = LLVM::AtomicBinOp::fadd;
    } else if (isFloat && binOp == LLVM::AtomicBinOp::sub) {
      binOp = LLVM::AtomicBinOp::fsub;
    }

    auto rmw = LLVM::AtomicRMWOp::create(rewriter, loc, binOp, ptr, val, order);

    auto reinterpretResult = [&](Value value) -> Value {
      if (!isa<cxx::PointerType>(op.getResult().getType())) return value;
      return LLVM::IntToPtrOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(getContext()), value);
    };

    if (!returnsPostOp) {
      rewriter.replaceOp(op, reinterpretResult(rmw));
      return success();
    }

    Value postOp;
    switch (binOp) {
      case LLVM::AtomicBinOp::add:
        postOp = LLVM::AddOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::sub:
        postOp = LLVM::SubOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::fadd:
        postOp = LLVM::FAddOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::fsub:
        postOp = LLVM::FSubOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::_and:
        postOp = LLVM::AndOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::_or:
        postOp = LLVM::OrOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::_xor:
        postOp = LLVM::XOrOp::create(rewriter, loc, rmw, val);
        break;
      case LLVM::AtomicBinOp::nand: {
        auto anded = LLVM::AndOp::create(rewriter, loc, rmw, val);
        auto allOnes = LLVM::ConstantOp::create(
            rewriter, loc, anded.getType(),
            rewriter.getIntegerAttr(anded.getType(), -1));
        postOp = LLVM::XOrOp::create(rewriter, loc, anded, allOnes);
        break;
      }
      default:
        postOp = rmw;
    }
    rewriter.replaceOp(op, reinterpretResult(postOp));
    return success();
  }

  auto lowerAtomicTestAndSet(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    if (adaptor.getInputs().size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "atomic_test_and_set expects 2 arguments");
    }
    auto loc = op.getLoc();
    auto i8Type = IntegerType::get(getContext(), 8);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i8Type,
                                        rewriter.getIntegerAttr(i8Type, 1));
    auto order = atomicOrderingFromValue(adaptor.getInputs()[1]);
    auto old = LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg,
                                         adaptor.getInputs()[0], one, order);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i8Type,
                                         rewriter.getIntegerAttr(i8Type, 0));
    auto wasSet =
        LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ne, old, zero);
    rewriter.replaceOp(op, wasSet);
    return success();
  }

  auto lowerAtomicClear(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    if (adaptor.getInputs().size() < 2) {
      return rewriter.notifyMatchFailure(op,
                                         "atomic_clear expects 2 arguments");
    }
    auto loc = op.getLoc();
    auto i8Type = IntegerType::get(getContext(), 8);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i8Type,
                                         rewriter.getIntegerAttr(i8Type, 0));
    auto order = atomicOrderingFromValue(adaptor.getInputs()[1]);
    LLVM::StoreOp::create(rewriter, loc, zero, adaptor.getInputs()[0], 1, false,
                          false, false, order);
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerAtomicFence(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter& rewriter,
                        bool singleThread) const -> LogicalResult {
    auto order = adaptor.getInputs().empty()
                     ? LLVM::AtomicOrdering::seq_cst
                     : atomicOrderingFromValue(adaptor.getInputs()[0]);
    LLVM::FenceOp::create(rewriter, op.getLoc(), order,
                          singleThread ? "singlethread" : "");
    rewriter.eraseOp(op);
    return success();
  }

  auto lowerAtomicIsLockFree(cxx::BuiltinCallOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter& rewriter) const
      -> LogicalResult {
    if (adaptor.getInputs().empty()) {
      return rewriter.notifyMatchFailure(
          op, "lock_free query expects a size argument");
    }
    auto loc = op.getLoc();
    auto i1Type = rewriter.getI1Type();
    auto size = foldConstantInt(adaptor.getInputs()[0]);
    const bool lockFree = size.has_value() && (*size == 1 || *size == 2 ||
                                               *size == 4 || *size == 8);
    auto result = LLVM::ConstantOp::create(
        rewriter, loc, i1Type,
        rewriter.getIntegerAttr(i1Type, lockFree ? 1 : 0));
    rewriter.replaceOp(op, result);
    return success();
  }

  const DataLayout& dataLayout_;
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

    rewriter.replaceOp(
        op, LLVM::AddressOfOp::create(rewriter, op.getLoc(), resultType,
                                      adaptor.getSymName()));

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

    auto x = LLVM::AllocaOp::create(rewriter, op.getLoc(), resultType,
                                    elementType, size, op.getAlignment());

    rewriter.replaceOp(op, x);

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

class DynAllocaOpLowering : public OpConversionPattern<cxx::DynAllocaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::DynAllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto i8Type = rewriter.getIntegerType(8);
    auto resultType = LLVM::LLVMPointerType::get(getContext());
    auto x = LLVM::AllocaOp::create(rewriter, op.getLoc(), resultType, i8Type,
                                    adaptor.getSize(), op.getAlignment());
    rewriter.replaceOp(op, x);
    return success();
  }
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
      rewriter.replaceOp(
          op, LLVM::TruncOp::create(rewriter, op.getLoc(), resultType, loaded));
    } else {
      rewriter.replaceOp(
          op, LLVM::LoadOp::create(rewriter, op.getLoc(), resultType,
                                   adaptor.getAddr(), op.getAlignment()));
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

    if (isa<LLVM::LLVMStructType>(valueType) ||
        isa<LLVM::LLVMArrayType>(valueType)) {
      if (auto loadOp = adaptor.getValue().getDefiningOp<LLVM::LoadOp>()) {
        auto size = dataLayout_.getTypeSize(valueType);
        auto i64Ty = rewriter.getI64Type();
        auto sizeVal = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(size));
        rewriter.replaceOp(
            op, LLVM::MemcpyOp::create(rewriter, op.getLoc(), adaptor.getAddr(),
                                       loadOp.getAddr(), sizeVal,
                                       /*isVolatile=*/false));
        return success();
      }
    }

    auto ptrTy = dyn_cast<cxx::PointerType>(op.getAddr().getType());
    if (ptrTy && isBoolElementType(ptrTy)) {
      auto i8Type = getBoolMemoryType(context);
      auto extended = LLVM::ZExtOp::create(rewriter, op.getLoc(), i8Type,
                                           adaptor.getValue());
      rewriter.replaceOp(
          op, LLVM::StoreOp::create(rewriter, op.getLoc(), extended,
                                    adaptor.getAddr(), op.getAlignment()));
    } else {
      rewriter.replaceOp(
          op, LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                                    adaptor.getAddr(), op.getAlignment()));
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

    rewriter.replaceOp(
        op, LLVM::MemsetOp::create(rewriter, op.getLoc(), adaptor.getAddr(),
                                   zeroVal, sizeVal,
                                   /*isVolatile=*/false));

    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class MemCpyOpLowering : public OpConversionPattern<cxx::MemCpyOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::MemCpyOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto loc = op.getLoc();
    auto sizeVal =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                 rewriter.getI64IntegerAttr(op.getSize()));
    rewriter.replaceOp(op,
                       LLVM::MemcpyOp::create(rewriter, loc, adaptor.getDest(),
                                              adaptor.getSrc(), sizeVal,
                                              /*isVolatile=*/false));
    return success();
  }
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

    rewriter.replaceOp(
        op, LLVM::GEPOp::create(rewriter, op.getLoc(), resultType, elementType,
                                adaptor.getBase(), indices));

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

    auto ptrType = dyn_cast<cxx::PointerType>(op.getType());
    if (!ptrType) {
      return rewriter.notifyMatchFailure(
          op, "expected pointer result type for ptradd");
    }

    auto cxxElementType = ptrType.getElementType();

    mlir::Type elementType;
    if (cxxElementType && !isa<cxx::VoidType>(cxxElementType)) {
      elementType = typeConverter->convertType(cxxElementType);
    } else {
      elementType = IntegerType::get(context, 8);
    }

    if (!elementType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert element type for ptradd");
    }

    SmallVector<LLVM::GEPArg> indices;

    indices.push_back(adaptor.getOffset());

    rewriter.replaceOp(
        op, LLVM::GEPOp::create(rewriter, op.getLoc(), resultType, elementType,
                                adaptor.getBase(), indices));

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

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert pointer difference result type");
    }

    auto loc = op->getLoc();

    auto lhs =
        LLVM::PtrToIntOp::create(rewriter, loc, resultType, adaptor.getLhs());

    auto rhs =
        LLVM::PtrToIntOp::create(rewriter, loc, resultType, adaptor.getRhs());

    mlir::Value diff = LLVM::SubOp::create(rewriter, loc, resultType, lhs, rhs);

    auto ptrType = dyn_cast<cxx::PointerType>(op.getLhs().getType());
    if (ptrType) {
      auto cxxElementType = ptrType.getElementType();
      if (cxxElementType && !isa<cxx::VoidType>(cxxElementType)) {
        if (auto elementType = typeConverter->convertType(cxxElementType)) {
          auto elementSize = dataLayout_.getTypeSize(elementType);
          if (elementSize > 1) {
            auto sizeConst = LLVM::ConstantOp::create(
                rewriter, loc, resultType,
                rewriter.getIntegerAttr(resultType,
                                        static_cast<int64_t>(elementSize)));
            diff = LLVM::SDivOp::create(rewriter, loc, resultType, diff,
                                        sizeConst);
          }
        }
      }
    }

    rewriter.replaceOp(op, diff);
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

    rewriter.replaceOp(
        op, LLVM::GEPOp::create(rewriter, op.getLoc(), resultType, elementType,
                                adaptor.getBase(), indices));

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

    rewriter.replaceOp(op,
                       LLVM::ZeroOp::create(rewriter, op.getLoc(), resultType));

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

    rewriter.replaceOp(
        op, LLVM::GEPOp::create(rewriter, op.getLoc(), resultType, elementType,
                                adaptor.getValue(), indices));

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

    rewriter.replaceOp(
        op, LLVM::PtrToIntOp::create(rewriter, op.getLoc(), resultType,
                                     adaptor.getValue()));

    return success();
  }
};

class IntToPtrOpLowering : public OpConversionPattern<cxx::IntToPtrOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IntToPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert int to ptr type");
    }

    rewriter.replaceOp(
        op, LLVM::IntToPtrOp::create(rewriter, op.getLoc(), resultType,
                                     adaptor.getValue()));

    return success();
  }
};

class BitcastOpLowering : public OpConversionPattern<cxx::BitcastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::BitcastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert bitcast type");
    }

    auto inputType = adaptor.getValue().getType();

    if (isa<LLVM::LLVMPointerType>(inputType) &&
        isa<LLVM::LLVMPointerType>(resultType)) {
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    if (isa<LLVM::LLVMStructType>(resultType) &&
        !isa<LLVM::LLVMStructType>(inputType)) {
      auto structType = mlir::cast<LLVM::LLVMStructType>(resultType);
      if (!structType.getBody().empty()) {
        auto firstFieldType = structType.getBody()[0];
        mlir::Value fieldVal = adaptor.getValue();
        if (inputType != firstFieldType) {
          if (isa<LLVM::LLVMPointerType>(inputType) &&
              isa<mlir::IntegerType>(firstFieldType)) {
            fieldVal = LLVM::PtrToIntOp::create(rewriter, op.getLoc(),
                                                firstFieldType, fieldVal);
          } else if (isa<mlir::IntegerType>(inputType) &&
                     isa<mlir::IntegerType>(firstFieldType)) {
            auto srcWidth = mlir::cast<mlir::IntegerType>(inputType).getWidth();
            auto dstWidth =
                mlir::cast<mlir::IntegerType>(firstFieldType).getWidth();
            if (srcWidth < dstWidth) {
              fieldVal = LLVM::ZExtOp::create(rewriter, op.getLoc(),
                                              firstFieldType, fieldVal);
            }
          }
        }
        if (fieldVal.getType() == firstFieldType) {
          auto undef = LLVM::UndefOp::create(rewriter, op.getLoc(), resultType);
          rewriter.replaceOp(
              op, LLVM::InsertValueOp::create(rewriter, op.getLoc(), undef,
                                              fieldVal, ArrayRef<int64_t>{0}));
          return success();
        }
      }

      auto one =
          LLVM::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(1));

      auto alloca = LLVM::AllocaOp::create(
          rewriter, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
          resultType, one, /*alignment=*/0);

      LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), alloca);

      auto load =
          LLVM::LoadOp::create(rewriter, op.getLoc(), resultType, alloca);

      rewriter.replaceOp(op, load.getResult());

      return success();
    }

    if (inputType == resultType) {
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    if (isa<LLVM::LLVMStructType>(inputType) ||
        isa<LLVM::LLVMArrayType>(inputType)) {
      auto one =
          LLVM::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(1));
      auto alloca = LLVM::AllocaOp::create(
          rewriter, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
          inputType, one, /*alignment=*/0);
      LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), alloca);
      auto load =
          LLVM::LoadOp::create(rewriter, op.getLoc(), resultType, alloca);
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    rewriter.replaceOp(
        op, LLVM::BitcastOp::create(rewriter, op.getLoc(), resultType,
                                    adaptor.getValue()));
    return success();
  }
};

class LabelAddressOpLowering : public OpConversionPattern<cxx::LabelAddressOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::LabelAddressOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!op.getTagIdAttr())
      return rewriter.notifyMatchFailure(op, "label_address not resolved");

    auto funcNameAttr = op.getFuncNameAttr();

    if (!funcNameAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "label_address missing function name");
    }

    auto funcName = funcNameAttr.getValue();

    auto tagId = static_cast<unsigned>(op.getTagId().value());
    auto ctx = op.getContext();
    auto blockAddrAttr = LLVM::BlockAddressAttr::get(
        ctx, mlir::FlatSymbolRefAttr::get(ctx, funcName),
        LLVM::BlockTagAttr::get(ctx, tagId));
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    rewriter.replaceOpWithNewOp<LLVM::BlockAddressOp>(op, ptrType,
                                                      blockAddrAttr);
    return success();
  }
};

class IndirectGotoOpLowering : public OpConversionPattern<cxx::IndirectGotoOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::IndirectGotoOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto destinations = op.getDestinations();
    llvm::SmallVector<mlir::Block*> targets(destinations.begin(),
                                            destinations.end());
    llvm::SmallVector<mlir::ValueRange> succOperands(targets.size());
    rewriter.replaceOpWithNewOp<LLVM::IndirectBrOp>(op, adaptor.getTarget(),
                                                    succOperands, targets);
    return success();
  }
};

class ZeroOpLowering : public OpConversionPattern<cxx::ZeroOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ZeroOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert zero type");
    }

    rewriter.replaceOp(op,
                       LLVM::ZeroOp::create(rewriter, op.getLoc(), resultType));
    return success();
  }
};

class UndefOpLowering : public OpConversionPattern<cxx::UndefOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::UndefOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert undef type");
    }

    rewriter.replaceOp(
        op, LLVM::UndefOp::create(rewriter, op.getLoc(), resultType));
    return success();
  }
};

class ReshapeOpLowering : public OpConversionPattern<cxx::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ReshapeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto dstType = typeConverter->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    auto dstArr = dyn_cast<LLVM::LLVMArrayType>(dstType);
    if (!dstArr)
      return rewriter.notifyMatchFailure(op, "result is not an LLVM array");

    Value src = adaptor.getValue();
    auto srcArr = dyn_cast<LLVM::LLVMArrayType>(src.getType());
    if (!srcArr)
      return rewriter.notifyMatchFailure(op, "value is not an LLVM array");

    auto loc = op.getLoc();
    auto elemType = dstArr.getElementType();
    Value expanded = LLVM::UndefOp::create(rewriter, loc, dstArr);
    for (unsigned i = 0; i < dstArr.getNumElements(); ++i) {
      Value elem;
      if (i < srcArr.getNumElements()) {
        elem = LLVM::ExtractValueOp::create(rewriter, loc, src, i);
      } else {
        elem = LLVM::ConstantOp::create(rewriter, loc, elemType,
                                        rewriter.getIntegerAttr(elemType, 0));
      }
      expanded = LLVM::InsertValueOp::create(rewriter, loc, expanded, elem, i);
    }
    rewriter.replaceOp(op, expanded);
    return success();
  }
};

class InsertValueOpLowering : public OpConversionPattern<cxx::InsertValueOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::InsertValueOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto typeConverter = getTypeConverter();
    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert insertvalue type");
    }

    mlir::Value insertedValue = adaptor.getValue();

    if (auto structType =
            mlir::dyn_cast<mlir::LLVM::LLVMStructType>(resultType)) {
      auto fieldTypes = structType.getBody();
      uint64_t pos = op.getPosition();
      if (pos < fieldTypes.size()) {
        auto fieldTy = fieldTypes[pos];
        auto valTy = insertedValue.getType();
        if (valTy != fieldTy) {
          if (auto srcInt = mlir::dyn_cast<mlir::IntegerType>(valTy)) {
            if (auto dstInt = mlir::dyn_cast<mlir::IntegerType>(fieldTy)) {
              if (srcInt.getWidth() < dstInt.getWidth())
                insertedValue = mlir::arith::ExtUIOp::create(
                    rewriter, op.getLoc(), dstInt, insertedValue);
              else if (srcInt.getWidth() > dstInt.getWidth())
                insertedValue = mlir::arith::TruncIOp::create(
                    rewriter, op.getLoc(), dstInt, insertedValue);
            }
          }
        }
      }
    }

    rewriter.replaceOp(op, LLVM::InsertValueOp::create(
                               rewriter, op.getLoc(), adaptor.getContainer(),
                               insertedValue, op.getPosition()));
    return success();
  }
};

class BitfieldLoadOpLowering : public OpConversionPattern<cxx::BitfieldLoadOp> {
 public:
  BitfieldLoadOpLowering(const TypeConverter& typeConverter,
                         const DataLayout& dataLayout, MLIRContext* context,
                         PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::BitfieldLoadOp>(typeConverter, context,
                                                 benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::BitfieldLoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto context = getContext();
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert bitfield load result type");
    }

    auto storageSizeBytes = static_cast<unsigned>(op.getAlignment());
    auto storageSizeBits = storageSizeBytes * 8;
    auto bitOffset = static_cast<unsigned>(op.getBitOffset());
    auto bitWidth = static_cast<unsigned>(op.getBitWidth());
    bool isSigned = op.getIsSigned();

    auto storageType = IntegerType::get(context, storageSizeBits);

    auto loaded = LLVM::LoadOp::create(rewriter, loc, storageType,
                                       adaptor.getAddr(), storageSizeBytes);

    mlir::Value extracted;
    if (isSigned) {
      auto shlAmount = storageSizeBits - bitOffset - bitWidth;
      auto shrAmount = storageSizeBits - bitWidth;

      auto shlConst = LLVM::ConstantOp::create(
          rewriter, loc, storageType,
          rewriter.getIntegerAttr(storageType, shlAmount));
      auto shifted = LLVM::ShlOp::create(rewriter, loc, loaded, shlConst);

      auto shrConst = LLVM::ConstantOp::create(
          rewriter, loc, storageType,
          rewriter.getIntegerAttr(storageType, shrAmount));
      extracted = LLVM::AShrOp::create(rewriter, loc, shifted, shrConst);
    } else {
      auto shrConst = LLVM::ConstantOp::create(
          rewriter, loc, storageType,
          rewriter.getIntegerAttr(storageType, bitOffset));
      auto shifted = LLVM::LShrOp::create(rewriter, loc, loaded, shrConst);

      auto mask = (1ULL << bitWidth) - 1;
      auto maskConst =
          LLVM::ConstantOp::create(rewriter, loc, storageType,
                                   rewriter.getIntegerAttr(storageType, mask));
      extracted = LLVM::AndOp::create(rewriter, loc, shifted, maskConst);
    }

    auto resultBits = resultType.getIntOrFloatBitWidth();
    if (resultBits < storageSizeBits) {
      extracted = LLVM::TruncOp::create(rewriter, loc, resultType, extracted);
    } else if (resultBits > storageSizeBits) {
      if (isSigned) {
        extracted = LLVM::SExtOp::create(rewriter, loc, resultType, extracted);
      } else {
        extracted = LLVM::ZExtOp::create(rewriter, loc, resultType, extracted);
      }
    }

    rewriter.replaceOp(op, extracted);
    return success();
  }

 private:
  const DataLayout& dataLayout_;
};

class BitfieldStoreOpLowering
    : public OpConversionPattern<cxx::BitfieldStoreOp> {
 public:
  BitfieldStoreOpLowering(const TypeConverter& typeConverter,
                          const DataLayout& dataLayout, MLIRContext* context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<cxx::BitfieldStoreOp>(typeConverter, context,
                                                  benefit),
        dataLayout_(dataLayout) {}

  auto matchAndRewrite(cxx::BitfieldStoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto context = getContext();
    auto loc = op.getLoc();

    auto storageSizeBytes = static_cast<unsigned>(op.getAlignment());
    auto storageSizeBits = storageSizeBytes * 8;
    auto bitOffset = static_cast<unsigned>(op.getBitOffset());
    auto bitWidth = static_cast<unsigned>(op.getBitWidth());

    auto storageType = IntegerType::get(context, storageSizeBits);

    auto loaded = LLVM::LoadOp::create(rewriter, loc, storageType,
                                       adaptor.getAddr(), storageSizeBytes);

    auto value = adaptor.getValue();
    auto valueBits = value.getType().getIntOrFloatBitWidth();
    if (valueBits > storageSizeBits) {
      value = LLVM::TruncOp::create(rewriter, loc, storageType, value);
    } else if (valueBits < storageSizeBits) {
      value = LLVM::ZExtOp::create(rewriter, loc, storageType, value);
    }

    auto fieldMask = (1ULL << bitWidth) - 1;
    auto fieldMaskConst = LLVM::ConstantOp::create(
        rewriter, loc, storageType,
        rewriter.getIntegerAttr(storageType, fieldMask));
    auto maskedValue =
        LLVM::AndOp::create(rewriter, loc, value, fieldMaskConst);

    auto shiftConst = LLVM::ConstantOp::create(
        rewriter, loc, storageType,
        rewriter.getIntegerAttr(storageType, bitOffset));
    auto shiftedValue =
        LLVM::ShlOp::create(rewriter, loc, maskedValue, shiftConst);

    auto clearMask = ~(fieldMask << bitOffset);
    auto clearMaskConst = LLVM::ConstantOp::create(
        rewriter, loc, storageType,
        rewriter.getIntegerAttr(storageType, clearMask));
    auto clearedLoaded =
        LLVM::AndOp::create(rewriter, loc, loaded, clearMaskConst);

    auto combined =
        LLVM::OrOp::create(rewriter, loc, clearedLoaded, shiftedValue);

    rewriter.replaceOp(
        op, LLVM::StoreOp::create(rewriter, loc, combined, adaptor.getAddr(),
                                  storageSizeBytes));
    return success();
  }

 private:
  const DataLayout& dataLayout_;
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

  ConversionTarget target(*context);

  bool needsComdat = targetNeedsComdat(module);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<cxx::CxxDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  target.addLegalOp<cxx::TodoExprOp, cxx::TodoStmtOp>();

  RewritePatternSet patterns(context);

  patterns.insert<FuncOpLowering>(typeConverter, needsComdat, context);
  patterns.insert<GlobalOpLowering>(typeConverter, needsComdat, context);
  patterns.insert<VTableOpLowering>(typeConverter, needsComdat, context);
  patterns.insert<ReturnOpLowering, UnreachableOpLowering, CallOpLowering,
                  AddressOfOpLowering, DynAllocaOpLowering>(typeConverter,
                                                            context);

  DataLayout dataLayout{module};

  patterns.insert<AllocaOpLowering, LoadOpLowering, StoreOpLowering,
                  MemSetZeroOpLowering, SubscriptOpLowering, MemberOpLowering,
                  BitfieldLoadOpLowering, BitfieldStoreOpLowering,
                  BuiltinCallOpLowering>(typeConverter, dataLayout, context);

  patterns.insert<ArrayToPointerOpLowering, PtrToIntOpLowering,
                  IntToPtrOpLowering, BitcastOpLowering, MemCpyOpLowering>(
      typeConverter, context);

  patterns.insert<LabelAddressOpLowering, IndirectGotoOpLowering>(typeConverter,
                                                                  context);

  patterns.insert<NullPtrConstantOpLowering, ZeroOpLowering, UndefOpLowering,
                  ReshapeOpLowering, InsertValueOpLowering>(typeConverter,
                                                            context);

  patterns.insert<PtrAddOpLowering, PtrDiffOpLowering>(typeConverter,
                                                       dataLayout, context);

  populateFunctionOpInterfaceTypeConversionPattern<cxx::FuncOp>(patterns,
                                                                typeConverter);

  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  SmallVector<Attribute> globalCtors;
  module.walk([&](cxx::GlobalCtorOp ctorOp) {
    globalCtors.push_back(ctorOp.getCtorAttr());
    ctorOp.erase();
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  if (!globalCtors.empty()) {
    auto loc = module.getLoc();
    OpBuilder ctorBuilder(context);
    ctorBuilder.setInsertionPointToEnd(module.getBody());

    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto subFnTy = LLVM::LLVMFunctionType::get(voidTy, {});
    auto subFn =
        LLVM::LLVMFuncOp::create(ctorBuilder, loc, "_GLOBAL__sub_I_main",
                                 subFnTy, LLVM::Linkage::Internal);
    auto* entry = subFn.addEntryBlock(ctorBuilder);
    ctorBuilder.setInsertionPointToStart(entry);
    for (auto ctor : globalCtors) {
      LLVM::CallOp::create(ctorBuilder, loc, TypeRange{},
                           cast<FlatSymbolRefAttr>(ctor), ValueRange{});
    }
    LLVM::ReturnOp::create(ctorBuilder, loc, ValueRange{});

    ctorBuilder.setInsertionPointToEnd(module.getBody());
    LLVM::GlobalCtorsOp::create(
        ctorBuilder, loc,
        ctorBuilder.getArrayAttr(
            {FlatSymbolRefAttr::get(subFn.getSymNameAttr())}),
        ctorBuilder.getArrayAttr(
            {ctorBuilder.getI32IntegerAttr(kDefaultGlobalCtorPriority)}),
        ctorBuilder.getArrayAttr({LLVM::ZeroAttr::get(context)}));
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

  if (llvmModule) {
    llvmModule->addModuleFlag(llvm::Module::Max, "Dwarf Version", 5);
  }

  return llvmModule;
}
