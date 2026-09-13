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

#include <cxx/mlir/mlir_emitter.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/RegionUtils.h>

namespace cxx::ir {
namespace {
struct SerializedCleanups {
  mlir::SmallVector<mlir::Value> addresses;
  mlir::SmallVector<mlir::Value> activeFlags;
  mlir::SmallVector<mlir::Attribute> destructors;
  mlir::SmallVector<mlir::Attribute> depths;
  mlir::SmallVector<std::int32_t> activeFlagIndices;

  SerializedCleanups(MlirEmitter& emitter,
                     std::span<const CleanupAction> cleanups) {
    for (const auto& cleanup : cleanups) {
      addresses.push_back(emitter.value(cleanup.address));
      destructors.push_back(mlir::FlatSymbolRefAttr::get(
          emitter.function(cleanup.destructor).getSymNameAttr()));
      depths.push_back(emitter.builder().getI64IntegerAttr(cleanup.depth));
      if (cleanup.activeFlag) {
        activeFlagIndices.push_back(
            static_cast<std::int32_t>(activeFlags.size()));
        activeFlags.push_back(emitter.value(cleanup.activeFlag));
      } else {
        activeFlagIndices.push_back(-1);
      }
    }
  }
};
}  // namespace

auto MlirEmitter::beginCleanupRegion() -> CleanupRegionRef {
  CleanupRegion region;
  region.startBlock = builder_.getInsertionBlock();
  if (region.startBlock) {
    auto insertionPoint = builder_.getInsertionPoint();
    if (insertionPoint != region.startBlock->begin())
      region.startAnchor = &*std::prev(insertionPoint);
  }
  const auto id = nextCleanupRegionId_++;
  cleanupRegions_.try_emplace(id, region);
  return HandleAccess::make<CleanupRegionTag>(id);
}

void MlirEmitter::endCleanupRegion(CleanupRegionRef region) {
  cleanupRegions_.erase(HandleAccess::id(region));
}

auto MlirEmitter::activateConditionalCleanup(ValueRef address, BlockRef entry,
                                             CleanupRegionRef regionRef)
    -> ValueRef {
  auto allocaOp = value(address).getDefiningOp<mlir::cxx::AllocaOp>();
  auto entryBlock = block(entry);
  if (allocaOp && entryBlock && allocaOp->getBlock() != entryBlock) {
    // A captured position follows its original block, not a hoisted alloca.
    for (auto& [id, region] : cleanupRegions_) {
      if (region.startAnchor != allocaOp.getOperation()) continue;
      auto position = allocaOp->getIterator();
      region.startAnchor = position == allocaOp->getBlock()->begin()
                               ? nullptr
                               : &*std::prev(position);
    }
    allocaOp->moveBefore(entryBlock, entryBlock->begin());
  }

  auto loc = value(address).getLoc();
  auto boolType = integerType(1);
  ValueRef flag;
  {
    mlir::OpBuilder::InsertionGuard guard(builder_);
    setInsertionBlockStart(entry);
    flag = allocate(loc, pointerType(boolType), 1);
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder_);
    const auto& region = cleanupRegions_.at(HandleAccess::id(regionRef));
    auto flagOp = value(flag).getDefiningOp();
    if (region.startAnchor) {
      builder_.setInsertionPointAfter(region.startAnchor);
    } else if (region.startBlock && region.startBlock != flagOp->getBlock()) {
      builder_.setInsertionPointToStart(region.startBlock);
    } else {
      builder_.setInsertionPointAfter(flagOp);
    }
    auto inactive = constantInt(loc, boolType, 0);
    store(loc, inactive, flag, 1);
  }

  auto active = constantInt(loc, boolType, 1);
  store(loc, active, flag, 1);
  return flag;
}

void MlirEmitter::defineLabel(SourceLocation loc, std::string_view name,
                              std::int64_t cleanupDepth) {
  mlir::cxx::LabelOp::create(builder_, getLocation(loc), name, cleanupDepth);
}

void MlirEmitter::branchWithCleanups(SourceLocation loc, CleanupTarget target,
                                     std::span<const CleanupAction> cleanups) {
  SerializedCleanups snapshot(*this, cleanups);

  if (target.block) {
    mlir::cxx::CleanupBranchOp::create(
        builder_, getLocation(loc), snapshot.addresses, snapshot.activeFlags,
        builder_.getArrayAttr(snapshot.destructors),
        builder_.getDenseI32ArrayAttr(snapshot.activeFlagIndices),
        block(target.block));
    return;
  }

  mlir::cxx::GotoOp::create(
      builder_, getLocation(loc), snapshot.addresses, snapshot.activeFlags,
      builder_.getArrayAttr(snapshot.destructors),
      builder_.getArrayAttr(snapshot.depths),
      builder_.getDenseI32ArrayAttr(snapshot.activeFlagIndices), target.label);
}

void MlirEmitter::indirectGoto(SourceLocation loc, ValueRef target) {
  mlir::cxx::IndirectGotoOp::create(builder_, getLocation(loc), value(target),
                                    mlir::BlockRange{});
}

auto MlirEmitter::labelAddress(SourceLocation loc, TypeRef resultType,
                               std::string_view name, FunctionRef function)
    -> ValueRef {
  auto functionNameAttr = function
                              ? builder_.getStringAttr(functionName(function))
                              : mlir::StringAttr{};
  return wrap(mlir::cxx::LabelAddressOp::create(
      builder_, getLocation(loc),
      mlir::cast<mlir::cxx::PointerType>(type(resultType)), name,
      mlir::IntegerAttr{}, functionNameAttr));
}

void MlirEmitter::resolveFunctionControlFlow(FunctionRef funcOp) {
  mlir::IRRewriter rewriter(function(funcOp).getContext());

  llvm::DenseMap<llvm::StringRef, mlir::Block*> labels;
  llvm::DenseMap<llvm::StringRef, std::int64_t> labelCleanupDepths;

  for (auto& block : function(funcOp).getBody()) {
    for (auto& op : block) {
      if (auto labelOp = mlir::dyn_cast<mlir::cxx::LabelOp>(&op)) {
        labels[labelOp.getName()] = labelOp->getBlock();
        labelCleanupDepths[labelOp.getName()] =
            static_cast<std::int64_t>(labelOp.getCleanupDepth());
      }
    }
  }

  llvm::SmallVector<mlir::cxx::GotoOp> gotoOps;
  llvm::SmallVector<mlir::cxx::LabelOp> labelOps;
  llvm::SmallVector<mlir::cxx::CleanupBranchOp> cleanupBranchOps;
  llvm::SmallVector<mlir::cxx::LabelAddressOp> labelAddressOps;
  llvm::SmallVector<mlir::cxx::IndirectGotoOp> indirectGotoOps;

  for (auto& block : function(funcOp).getBody()) {
    for (auto& op : block) {
      if (auto gotoOp = mlir::dyn_cast<mlir::cxx::GotoOp>(&op))
        gotoOps.push_back(gotoOp);
      else if (auto labelOp = mlir::dyn_cast<mlir::cxx::LabelOp>(&op))
        labelOps.push_back(labelOp);
      else if (auto cbOp = mlir::dyn_cast<mlir::cxx::CleanupBranchOp>(&op))
        cleanupBranchOps.push_back(cbOp);
      else if (auto laOp = mlir::dyn_cast<mlir::cxx::LabelAddressOp>(&op))
        labelAddressOps.push_back(laOp);
      else if (auto igOp = mlir::dyn_cast<mlir::cxx::IndirectGotoOp>(&op))
        indirectGotoOps.push_back(igOp);
    }
  }

  if (auto module = this->module()) {
    for (auto& moduleOp : *module.getBody()) {
      if (auto globalOp = mlir::dyn_cast<mlir::cxx::GlobalOp>(&moduleOp)) {
        globalOp.walk([&](mlir::cxx::LabelAddressOp laOp) {
          if (auto fnAttr = laOp.getFuncNameAttr())
            if (fnAttr.getValue() == mlir::StringRef{functionName(funcOp)})
              labelAddressOps.push_back(laOp);
        });
      }
    }
  }

  auto emitCleanupCalls = [&](mlir::Location loc, mlir::ValueRange addresses,
                              mlir::ArrayAttr destructors,
                              mlir::ValueRange activeFlags,
                              llvm::ArrayRef<std::int32_t> activeFlagIndices,
                              llvm::function_ref<bool(unsigned)> selects) {
    for (unsigned i = 0; i < addresses.size(); ++i) {
      if (selects && !selects(i)) continue;

      auto dtorRef = mlir::cast<mlir::FlatSymbolRefAttr>(destructors[i]);
      mlir::SmallVector<mlir::Type> resultTypes;
      if (auto dtorFunc = findFunction(dtorRef.getValue())) {
        auto results = function(dtorFunc).getFunctionType().getResults();
        resultTypes.append(results.begin(), results.end());
      }

      const auto flagIndex =
          i < activeFlagIndices.size() ? activeFlagIndices[i] : -1;

      if (flagIndex < 0) {
        mlir::cxx::CallOp::create(rewriter, loc, resultTypes, dtorRef,
                                  mlir::ValueRange{addresses[i]});
        continue;
      }

      auto* currentBlock = rewriter.getInsertionBlock();
      auto* continueBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto* destroyBlock = rewriter.createBlock(continueBlock);

      rewriter.setInsertionPointToEnd(destroyBlock);
      mlir::cxx::CallOp::create(rewriter, loc, resultTypes, dtorRef,
                                mlir::ValueRange{addresses[i]});
      mlir::cf::BranchOp::create(rewriter, loc, continueBlock);

      rewriter.setInsertionPointToEnd(currentBlock);
      auto flag = mlir::cxx::LoadOp::create(rewriter, loc, rewriter.getI1Type(),
                                            activeFlags[flagIndex], 1);
      mlir::cf::CondBranchOp::create(rewriter, loc, flag, destroyBlock,
                                     continueBlock);

      rewriter.setInsertionPointToStart(continueBlock);
    }
  };

  for (auto gotoOp : gotoOps) {
    auto targetBlock = labels.lookup(gotoOp.getLabel());
    if (!targetBlock) continue;

    auto labelDepth = labelCleanupDepths.lookup(gotoOp.getLabel());
    auto depths = gotoOp.getDepths();

    rewriter.setInsertionPoint(gotoOp);

    if (auto nextOp = ++gotoOp->getIterator();
        mlir::isa<mlir::cf::BranchOp>(&*nextOp)) {
      rewriter.eraseOp(&*nextOp);
    }

    emitCleanupCalls(gotoOp.getLoc(), gotoOp.getAddresses(),
                     gotoOp.getDestructors(), gotoOp.getActiveFlags(),
                     gotoOp.getActiveFlagIndices(), [&](unsigned i) {
                       auto depthAttr =
                           mlir::cast<mlir::IntegerAttr>(depths[i]);
                       return depthAttr.getValue().getSExtValue() >= labelDepth;
                     });

    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(gotoOp, targetBlock);
  }

  if (!labelAddressOps.empty()) {
    auto ctx = function(funcOp).getContext();
    llvm::DenseMap<llvm::StringRef, unsigned> labelToTagId;
    llvm::SmallVector<mlir::Block*> labelTargets;
    unsigned nextTagId = 0;

    for (auto labelAddrOp : labelAddressOps) {
      auto name = labelAddrOp.getLabelName();
      unsigned tagId;
      auto it = labelToTagId.find(name);
      if (it == labelToTagId.end()) {
        tagId = nextTagId++;
        labelToTagId[name] = tagId;
        auto targetBlock = labels.lookup(name);
        if (targetBlock) {
          auto tagAttr = mlir::LLVM::BlockTagAttr::get(ctx, tagId);
          rewriter.setInsertionPointToStart(targetBlock);
          mlir::LLVM::BlockTagOp::create(rewriter, labelAddrOp.getLoc(),
                                         tagAttr);
          labelTargets.push_back(targetBlock);
        }
      } else {
        tagId = it->second;
      }
      rewriter.modifyOpInPlace(labelAddrOp, [&] {
        labelAddrOp.setTagIdAttr(
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), tagId));
      });
    }

    for (auto igOp : indirectGotoOps) {
      rewriter.setInsertionPoint(igOp);
      rewriter.replaceOpWithNewOp<mlir::cxx::IndirectGotoOp>(
          igOp, igOp.getTarget(), mlir::BlockRange{labelTargets});
    }
  }

  for (auto labelOp : labelOps) {
    rewriter.eraseOp(labelOp);
  }

  for (auto cbOp : cleanupBranchOps) {
    rewriter.setInsertionPoint(cbOp);

    emitCleanupCalls(cbOp.getLoc(), cbOp.getAddresses(), cbOp.getDestructors(),
                     cbOp.getActiveFlags(), cbOp.getActiveFlagIndices(),
                     nullptr);

    rewriter.setInsertionPoint(cbOp);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(cbOp, cbOp.getDest());
  }

  for (auto& region : function(funcOp)->getRegions()) {
    eraseUnreachableBlocks(rewriter, region);
  }
}

}  // namespace cxx::ir
