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

#include <cxx/mlir/symbol_binding.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace cxx {

namespace {

using mlir::LLVM::Visibility;
using mlir::LLVM::linkage::Linkage;

struct BoundSymbol {
  llvm::StringRef name;
  Linkage linkage = Linkage::External;
  Visibility visibility = Visibility::Default;
  bool isDefinition = false;
  bool isFunction = false;
  bool isThreadLocal = false;
};

[[nodiscard]] auto hasLocalLinkage(Linkage linkage) -> bool {
  return linkage == Linkage::Internal || linkage == Linkage::Private;
}

[[nodiscard]] auto isObjectSymbol(const BoundSymbol& symbol) -> bool {
  return !symbol.name.starts_with("llvm.");
}

class SymbolBinding {
 public:
  explicit SymbolBinding(mlir::ModuleOp module) {
    if (auto triple = module->getAttrOfType<mlir::StringAttr>("cxx.triple"))
      triple_ = llvm::Triple(triple.getValue());
    if (auto model =
            module->getAttrOfType<mlir::StringAttr>("cxx.relocation-model"))
      relocationModel_ = model.getValue();
    pie_ = module->hasAttr("cxx.pie-level");
  }

  [[nodiscard]] auto bindsLocally(const BoundSymbol& symbol) const -> bool {
    if (!isObjectSymbol(symbol)) return false;
    if (hasLocalLinkage(symbol.linkage)) return true;
    const bool externWeak = symbol.linkage == Linkage::ExternWeak;
    if (symbol.visibility != Visibility::Default && !externWeak) return true;
    if (triple_.isOSBinFormatCOFF()) return !externWeak;
    if (!triple_.isOSBinFormatELF()) return false;
    if (!producesExecutable()) return false;
    if (symbol.isDefinition) return true;
    if (!isStaticExecutable()) return false;
    if (symbol.isThreadLocal) return false;
    if (symbol.isFunction) return !externWeak;
    return true;
  }

 private:
  [[nodiscard]] auto isStaticExecutable() const -> bool {
    return relocationModel_ == "static";
  }

  [[nodiscard]] auto producesExecutable() const -> bool {
    return isStaticExecutable() || pie_;
  }

  llvm::Triple triple_;
  llvm::StringRef relocationModel_;
  bool pie_ = false;
};

[[nodiscard]] auto definesForLinker(bool hasBody, Linkage linkage) -> bool {
  return hasBody && linkage != Linkage::AvailableExternally;
}

[[nodiscard]] auto hasInitializer(mlir::LLVM::GlobalOp global) -> bool {
  return global.getValueOrNull() || !global.getInitializerRegion().empty();
}

[[nodiscard]] auto boundSymbolOf(mlir::LLVM::LLVMFuncOp function)
    -> BoundSymbol {
  return {.name = function.getSymName(),
          .linkage = function.getLinkage(),
          .visibility = function.getVisibility_(),
          .isDefinition =
              definesForLinker(!function.isExternal(), function.getLinkage()),
          .isFunction = true};
}

[[nodiscard]] auto boundSymbolOf(mlir::LLVM::GlobalOp global) -> BoundSymbol {
  return {.name = global.getSymName(),
          .linkage = global.getLinkage(),
          .visibility = global.getVisibility_(),
          .isDefinition =
              definesForLinker(hasInitializer(global), global.getLinkage()),
          .isThreadLocal = global.getThreadLocal_()};
}

[[nodiscard]] auto boundSymbolOf(mlir::LLVM::AliasOp alias) -> BoundSymbol {
  return {.name = alias.getSymName(),
          .linkage = alias.getLinkage(),
          .visibility = alias.getVisibility_(),
          .isDefinition = true,
          .isThreadLocal = alias.getThreadLocal_()};
}

template <typename Op>
void markDSOLocal(mlir::ModuleOp module, const SymbolBinding& binding) {
  for (auto op : module.getOps<Op>()) {
    if (binding.bindsLocally(boundSymbolOf(op))) op.setDsoLocal(true);
  }
}

}  // namespace

void markDSOLocalSymbols(mlir::ModuleOp module) {
  const SymbolBinding binding{module};
  markDSOLocal<mlir::LLVM::LLVMFuncOp>(module, binding);
  markDSOLocal<mlir::LLVM::GlobalOp>(module, binding);
  markDSOLocal<mlir::LLVM::AliasOp>(module, binding);
}

}  // namespace cxx
