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
#include <cxx/mlir/mlir_emitter.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Target/LLVMIR/Import.h>

#include <format>
#include <memory>

namespace cxx::ir {

auto MlirEmitter::beginModule(const ModuleInfo& info) -> ModuleRef {
  mlir::Location loc = builder_.getUnknownLoc();
  if (!info.sourceFile.empty())
    loc = mlir::FileLineColLoc::get(context(), info.sourceFile, 0, 0);
  auto module = mlir::ModuleOp::create(builder_, loc, info.name);
  builder_.setInsertionPointToStart(module.getBody());

  if (!info.targetTriple.empty()) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();

    auto triple = llvm::Triple{llvm::StringRef{info.targetTriple}};
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
      cxx_runtime_error(std::format("failed to find target for triple '{}': {}",
                                    triple.getTriple(), error));
    }
    llvm::TargetOptions options;
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(triple, "generic", "", options,
                                    std::optional<llvm::Reloc::Model>()));
    auto layout = targetMachine->createDataLayout();
    module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                    mlir::translateDataLayout(layout, context()));
    module->setAttr("cxx.triple", builder_.getStringAttr(info.targetTriple));
    module->setAttr("cxx.data-layout",
                    builder_.getStringAttr(layout.getStringRepresentation()));
  }

  if (!info.framePointer.empty() && info.framePointer != "none") {
    module->setAttr("cxx.frame-pointer",
                    builder_.getStringAttr(mlir::StringRef{
                        info.framePointer.data(), info.framePointer.size()}));
  }

  if (!info.debugCompilationDirectory.empty()) {
    module->setAttr("cxx.debug-compilation-dir",
                    builder_.getStringAttr(mlir::StringRef{
                        info.debugCompilationDirectory.data(),
                        info.debugCompilationDirectory.size()}));
  }

  const auto id = static_cast<std::uint32_t>(modules_.size());
  modules_.push_back(module);
  moduleStack_.push_back(module_);
  module_ = module;
  return HandleAccess::make<ModuleTag>(id);
}

void MlirEmitter::endModule() {
  module_ = moduleStack_.back();
  moduleStack_.pop_back();
}

auto MlirEmitter::module(ModuleRef ref) -> mlir::ModuleOp {
  const auto id = HandleAccess::id(ref);
  if (id >= modules_.size()) return {};
  return modules_[id];
}

}  // namespace cxx::ir
