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

#include <cxx/device_profile_limits.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/passes.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Pass/PassRegistry.h>

namespace mlir {
namespace {

struct KernelStorage {
  cxx::FuncOp kernel;
  std::uint64_t bytes = 0;
};

class WorkgroupBudget {
 public:
  WorkgroupBudget(ModuleOp module, std::uint64_t limit)
      : layout_(module), limit_(limit) {}

  [[nodiscard]] auto reserve(cxx::WorkgroupAllocOp allocation)
      -> LogicalResult {
    auto kernel = allocation->getParentOfType<cxx::FuncOp>();
    if (!kernel || !cxx::isKernelEntry(kernel))
      return allocation.emitError(
          "threadgroup allocation requires a kernel entry");
    auto object = cast<cxx::PointerType>(allocation.getType()).getElementType();
    auto alignment = layout_.getTypeABIAlignment(object);
    if (allocation.getAlignment() > static_cast<std::int64_t>(alignment))
      return allocation.emitError(
          "over-aligned threadgroup storage is unsupported by the device "
          "profile");
    auto& storage = storageOf(kernel);
    auto offset = llvm::alignTo(storage.bytes, alignment);
    auto size = layout_.getTypeSize(object).getFixedValue();
    if (!fits(offset, size))
      return allocation.emitError("threadgroup storage exceeds the ")
             << limit_ << "-byte device limit";
    storage.bytes = offset + size;
    return success();
  }

 private:
  [[nodiscard]] auto fits(std::uint64_t offset, std::uint64_t size) const
      -> bool {
    return offset <= limit_ && size <= limit_ - offset;
  }

  [[nodiscard]] auto storageOf(cxx::FuncOp kernel) -> KernelStorage& {
    auto found = llvm::find_if(kernels_, [&](const KernelStorage& storage) {
      return storage.kernel == kernel;
    });
    if (found != kernels_.end()) return *found;
    return kernels_.emplace_back(KernelStorage{kernel});
  }

  DataLayout layout_;
  std::uint64_t limit_;
  SmallVector<KernelStorage> kernels_;
};

[[nodiscard]] auto workgroupStorage(OpBuilder& builder,
                                    cxx::WorkgroupAllocOp allocation)
    -> FailureOr<Value> {
  auto pointer = cast<cxx::PointerType>(allocation.getType());
  auto type =
      MemRefType::get({}, pointer.getElementType(), MemRefLayoutAttrInterface(),
                      gpu::AddressSpaceAttr::get(builder.getContext(),
                                                 gpu::AddressSpace::Workgroup));
  auto alignment = allocation.getAlignment()
                       ? builder.getI64IntegerAttr(allocation.getAlignment())
                       : IntegerAttr();
  auto storage =
      memref::AllocOp::create(builder, allocation.getLoc(), type, alignment);
  return cxx::storagePointer(builder, allocation.getLoc(), {storage, {}},
                             pointer);
}

class ConvertCxxWorkgroupStoragePass final
    : public PassWrapper<ConvertCxxWorkgroupStoragePass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxWorkgroupStoragePass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "convert-cxx-workgroup-storage";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Preserve kernel-local AS3 allocations as workgroup MemRef storage";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, gpu::GPUDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<cxx::WorkgroupAllocOp> allocations;
    module.walk([&](cxx::WorkgroupAllocOp op) { allocations.push_back(op); });
    if (allocations.empty()) return;
    auto limits = cxx::deviceProfileLimitsOf(module);
    if (!limits) {
      module.emitError("threadgroup storage requires a device profile");
      return signalPassFailure();
    }
    WorkgroupBudget budget(module, limits->maxWorkgroupStorageBytes);
    for (auto allocation : allocations) {
      if (failed(budget.reserve(allocation))) return signalPassFailure();
      OpBuilder builder(allocation);
      auto address = workgroupStorage(builder, allocation);
      if (failed(address)) {
        allocation.emitError("cannot represent threadgroup storage address");
        return signalPassFailure();
      }
      allocation.getResult().replaceAllUsesWith(*address);
      allocation.erase();
    }
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxWorkgroupStoragePass()
    -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxWorkgroupStoragePass>();
}

void cxx::registerConvertCxxWorkgroupStoragePass() {
  mlir::registerPass(createConvertCxxWorkgroupStoragePass);
}
