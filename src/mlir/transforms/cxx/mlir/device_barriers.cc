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
#include <cxx/mlir/device_barriers.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir {
namespace {

struct CallSites {
  func::FuncOp callee;
  SmallVector<func::CallOp> calls;
};

[[nodiscard]] auto isInvariantStorage(Value root) -> bool {
  if (cxx::isReadOnlyDeviceResource(root)) return true;
  auto memory = dyn_cast<MemRefType>(root.getType());
  if (!memory) return false;
  auto space =
      dyn_cast_or_null<spirv::StorageClassAttr>(memory.getMemorySpace());
  return space && space.getValue() == spirv::StorageClass::Uniform;
}

[[nodiscard]] auto isUniformSource(Operation* operation) -> bool {
  return isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp, cxx::UndefOp,
             ub::PoisonOp>(operation);
}

[[nodiscard]] auto yieldedValues(Operation* branch, unsigned index)
    -> SmallVector<Value> {
  SmallVector<Value> values;
  for (auto& region : branch->getRegions())
    values.push_back(region.front().getTerminator()->getOperand(index));
  return values;
}

class DeviceBarrierUniformity {
 public:
  explicit DeviceBarrierUniformity(gpu::GPUModuleOp module) : module_(module) {
    module_.walk([&](func::CallOp call) {
      auto callee = module_.lookupSymbol<func::FuncOp>(call.getCallee());
      if (callee) callSitesOf(callee).push_back(call);
    });
  }

  [[nodiscard]] auto check() -> LogicalResult {
    llvm::SetVector<Operation*> participants;
    module_.walk(
        [&](cxx::ControlBarrierOp barrier) { participants.insert(barrier); });
    for (unsigned index = 0; index < participants.size(); ++index) {
      auto operation = participants[index];
      if (failed(checkControl(operation))) return failure();
      auto function = operation->getParentOfType<func::FuncOp>();
      if (!function) continue;
      for (auto call : callSitesOf(function)) participants.insert(call);
    }
    return success();
  }

 private:
  [[nodiscard]] auto callSitesOf(func::FuncOp callee)
      -> SmallVectorImpl<func::CallOp>& {
    auto found = llvm::find_if(callSites_, [&](const CallSites& sites) {
      return sites.callee == callee;
    });
    if (found != callSites_.end()) return found->calls;
    return callSites_.emplace_back(CallSites{callee, {}}).calls;
  }

  [[nodiscard]] static auto controlConditions(Operation* parent)
      -> FailureOr<SmallVector<Value>> {
    if (auto branch = dyn_cast<scf::IfOp>(parent))
      return SmallVector<Value>{branch.getCondition()};
    if (auto branch = dyn_cast<scf::IndexSwitchOp>(parent))
      return SmallVector<Value>{branch.getArg()};
    if (auto loop = dyn_cast<scf::ForOp>(parent))
      return SmallVector<Value>{loop.getLowerBound(), loop.getUpperBound(),
                                loop.getStep()};
    if (auto loop = dyn_cast<scf::WhileOp>(parent))
      return SmallVector<Value>{loop.getConditionOp().getCondition()};
    return failure();
  }

  [[nodiscard]] auto checkControl(Operation* operation) -> LogicalResult {
    for (auto current = operation; current->getParentOp() != module_;
         current = current->getParentOp()) {
      auto parent = current->getParentOp();
      if (!llvm::hasSingleElement(*current->getParentRegion()))
        return parent->emitError(
            "cannot prove workgroup-uniform control flow for a collective "
            "barrier");
      if (isa<func::FuncOp, gpu::GPUFuncOp>(parent)) return success();
      auto conditions = controlConditions(parent);
      if (failed(conditions))
        return parent->emitError(
            "cannot prove workgroup-uniform control flow for a collective "
            "barrier");
      for (auto condition : *conditions)
        if (!isUniform(condition))
          return parent->emitError(
              "collective barrier requires workgroup-uniform control flow");
    }
    return success();
  }

  [[nodiscard]] auto isUniform(Value value) -> bool {
    llvm::SetVector<Value> pending;
    pending.insert(value);
    for (unsigned index = 0; index < pending.size(); ++index) {
      SmallVector<Value> inputs;
      if (!dependencies(pending[index], inputs)) return false;
      pending.insert(inputs.begin(), inputs.end());
    }
    return true;
  }

  [[nodiscard]] auto argumentInputs(BlockArgument argument,
                                    SmallVectorImpl<Value>& inputs) -> bool {
    auto owner = argument.getOwner()->getParentOp();
    auto index = argument.getArgNumber();
    if (auto function = dyn_cast<func::FuncOp>(owner)) {
      auto& calls = callSitesOf(function);
      for (auto call : calls) inputs.push_back(call.getOperand(index));
      return !calls.empty();
    }
    if (auto loop = dyn_cast<scf::ForOp>(owner)) {
      inputs.append(
          {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
      if (index) {
        inputs.push_back(loop.getInitArgs()[index - 1]);
        inputs.push_back(
            loop.getBody()->getTerminator()->getOperand(index - 1));
      }
      return true;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
      inputs.push_back(loop.getConditionOp().getCondition());
      if (argument.getOwner() == &loop.getBefore().front()) {
        inputs.push_back(loop.getInits()[index]);
        inputs.push_back(loop.getYieldOp().getOperand(index));
      } else {
        inputs.push_back(loop.getConditionOp().getArgs()[index]);
      }
      return true;
    }
    return false;
  }

  [[nodiscard]] static auto invariantLoad(Value address,
                                          SmallVectorImpl<Value>& inputs)
      -> bool {
    auto storage = cxx::resolveStorageAddress(address);
    if (failed(storage) || !isInvariantStorage(storage->root)) return false;
    for (const auto& step : storage->path)
      if (auto element = std::get_if<cxx::StorageElement>(&step))
        inputs.append(element->terms.begin(), element->terms.end());
    return true;
  }

  [[nodiscard]] auto resultInputs(OpResult result,
                                  SmallVectorImpl<Value>& inputs) -> bool {
    auto operation = result.getOwner();
    auto index = result.getResultNumber();
    if (auto branch = dyn_cast<scf::IfOp>(operation)) {
      inputs.push_back(branch.getCondition());
      inputs.append(yieldedValues(branch, index));
      return true;
    }
    if (auto branch = dyn_cast<scf::IndexSwitchOp>(operation)) {
      inputs.push_back(branch.getArg());
      inputs.append(yieldedValues(branch, index));
      return true;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      inputs.append({loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
                     loop.getInitArgs()[index]});
      inputs.push_back(loop.getBody()->getTerminator()->getOperand(index));
      return true;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      inputs.push_back(loop.getConditionOp().getCondition());
      inputs.push_back(loop.getConditionOp().getArgs()[index]);
      return true;
    }
    return false;
  }

  [[nodiscard]] auto dependencies(Value value, SmallVectorImpl<Value>& inputs)
      -> bool {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return argumentInputs(argument, inputs);
    auto operation = value.getDefiningOp();
    if (matchPattern(value, m_Constant()) || isUniformSource(operation))
      return true;
    if (auto load = dyn_cast<cxx::LoadOp>(operation))
      return invariantLoad(load.getAddr(), inputs);
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (!isInvariantStorage(load.getMemRef())) return false;
      inputs.append(load.getIndices().begin(), load.getIndices().end());
      return true;
    }
    if (operation->getNumRegions())
      return resultInputs(cast<OpResult>(value), inputs);
    if (!isMemoryEffectFree(operation)) return false;
    if (!isa<arith::ArithDialect>(operation->getDialect())) return false;
    inputs.append(operation->getOperands().begin(),
                  operation->getOperands().end());
    return true;
  }

  gpu::GPUModuleOp module_;
  SmallVector<CallSites> callSites_;
};

}  // namespace
}  // namespace mlir

auto cxx::checkDeviceBarrierUniformity(mlir::gpu::GPUModuleOp module)
    -> mlir::LogicalResult {
  return mlir::DeviceBarrierUniformity(module).check();
}
