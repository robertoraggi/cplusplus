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
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir {
namespace {

[[nodiscard]] auto toPtr(Value value, PatternRewriter& rewriter) -> Value {
  auto pointer = cast<PtrLikeTypeInterface>(value.getType());
  auto space = cast<ptr::MemorySpaceAttrInterface>(pointer.getMemorySpace());
  auto type = ptr::PtrType::get(space);
  return ptr::ToPtrOp::create(rewriter, value.getLoc(), type, value);
}

[[nodiscard]] auto fromPtr(Value value, cxx::PointerType resultType,
                           PatternRewriter& rewriter) -> Value {
  auto space = cast<ptr::PtrType>(value.getType()).getMemorySpace();
  auto type = resultType;
  if (type.getMemorySpace() != space) {
    auto clone = resultType.clonePtrWith(space, std::nullopt);
    type = cast<cxx::PointerType>(*clone);
  }
  Value result =
      ptr::FromPtrOp::create(rewriter, value.getLoc(), type, value, Value());
  if (type != resultType)
    result =
        cxx::BitcastOp::create(rewriter, value.getLoc(), resultType, result);
  return result;
}

class LoadToPtr : public OpRewritePattern<cxx::LoadOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::LoadOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto type = op.getType();
    if (!LLVM::isCompatibleType(type)) return failure();
    auto pointer = toPtr(op.getAddr(), rewriter);
    bool isBool = type.isInteger(1);
    if (isBool) type = rewriter.getI8Type();
    Value result = ptr::LoadOp::create(rewriter, op.getLoc(), type, pointer,
                                       op.getAlignment());
    if (isBool)
      result =
          arith::TruncIOp::create(rewriter, op.getLoc(), op.getType(), result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class StoreToPtr : public OpRewritePattern<cxx::StoreOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::StoreOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto value = op.getValue();
    if (!LLVM::isCompatibleType(value.getType())) return failure();
    if (value.getType().isInteger(1))
      value = arith::ExtUIOp::create(rewriter, op.getLoc(),
                                     rewriter.getI8Type(), value);
    auto pointer = toPtr(op.getAddr(), rewriter);
    ptr::StoreOp::create(rewriter, op.getLoc(), value, pointer,
                         op.getAlignment());
    rewriter.eraseOp(op);
    return success();
  }
};

class PtrAddToPtr : public OpRewritePattern<cxx::PtrAddOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::PtrAddOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto resultType = cast<cxx::PointerType>(op.getType());
    auto element = resultType.getElementType();
    if (isa<cxx::VoidType>(element) || element.isInteger(1))
      element = rewriter.getI8Type();
    auto offset = arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), op.getOffset());
    auto stride = ptr::TypeOffsetOp::create(rewriter, op.getLoc(),
                                            offset.getType(), element);
    auto bytes = arith::MulIOp::create(rewriter, op.getLoc(), offset, stride);
    auto pointer = toPtr(op.getBase(), rewriter);
    auto result =
        ptr::PtrAddOp::create(rewriter, op.getLoc(), pointer.getType(), pointer,
                              bytes, ptr::PtrAddFlags::inbounds);
    rewriter.replaceOp(op, fromPtr(result, resultType, rewriter));
    return success();
  }
};

class NullToPtr : public OpRewritePattern<cxx::NullPtrConstantOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::NullPtrConstantOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto resultType = cast<cxx::PointerType>(op.getType());
    auto space =
        cast<ptr::MemorySpaceAttrInterface>(resultType.getMemorySpace());
    auto type = ptr::PtrType::get(space);
    auto null = ptr::NullAttr::get(type);
    auto result = ptr::ConstantOp::create(rewriter, op.getLoc(), type, null);
    rewriter.replaceOp(op, fromPtr(result, resultType, rewriter));
    return success();
  }
};

class ConvertCxxToPtrPass
    : public PassWrapper<ConvertCxxToPtrPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxToPtrPass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "convert-cxx-to-ptr";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Convert Cxx scalar memory access and pointer arithmetic to Ptr";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, arith::ArithDialect, ptr::PtrDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LoadToPtr, StoreToPtr, PtrAddToPtr, NullToPtr>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxToPtrPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxToPtrPass>();
}

void cxx::registerConvertCxxToPtrPass() {
  mlir::registerPass(createConvertCxxToPtrPass);
}
