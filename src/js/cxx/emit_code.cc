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

#include "emit_code.h"

#ifdef CXX_WITH_MLIR
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/mlir/codegen.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/translation_unit.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/MLIRContext.h>

#include <format>
#include <sstream>
#endif

namespace cxx::js {

#ifndef CXX_WITH_MLIR

auto hasCodeGenerator() -> bool { return false; }

auto generateCode(TranslationUnit*, std::string_view, bool)
    -> std::optional<GeneratedCode> {
  return std::nullopt;
}

#else

auto hasCodeGenerator() -> bool { return true; }

auto generateCode(TranslationUnit* unit, std::string_view format,
                  bool debugInfo) -> std::optional<GeneratedCode> {
  const auto objectFile = format == "obj";

  mlir::MLIRContext context{mlir::MLIRContext::Threading::DISABLED};

  context.loadDialect<mlir::cxx::CxxDialect>();

  Codegen codegen(context, unit, debugInfo);

  auto ir = codegen(unit->ast());

  std::ostringstream out;
  llvm::raw_os_ostream os(out);

  auto textOutput = [&] {
    os.flush();
    return GeneratedCode{.text = out.str()};
  };

  auto printingFlags = [&] {
    mlir::OpPrintingFlags flags;
    if (debugInfo) flags.enableDebugInfo(true);
    return flags;
  };

  if (format == "cxxir") {
    ir.module->print(os, printingFlags());
    return textOutput();
  }

  if (failed(lowerToMLIR(ir.module))) {
    if (objectFile) return GeneratedCode{};
    return GeneratedCode{.text = std::format("<error lowering to {}>", format)};
  }

  if (format == "mlir") {
    ir.module->print(os, printingFlags());
    return textOutput();
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = exportToLLVMIR(ir.module, llvmContext);
  llvmModule->setSourceFileName(unit->fileName());

  if (format == "llvm") {
    llvmModule->print(os, nullptr);
    return textOutput();
  }

  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();
  LLVMInitializeWebAssemblyAsmPrinter();

  llvm::TargetOptions opt;

  auto RM = std::optional<llvm::Reloc::Model>();

  auto triple = llvm::Triple{codegen.control()->memoryLayout()->triple()};

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);

  auto targetMachine =
      target->createTargetMachine(llvm::Triple{triple}, "generic", "", opt, RM);

  llvm::legacy::PassManager pm;

  llvm::SmallString<0> outputBuffer;
  llvm::raw_svector_ostream outBytes(outputBuffer);

  llvm::CodeGenFileType fileType = objectFile
                                       ? llvm::CodeGenFileType::ObjectFile
                                       : llvm::CodeGenFileType::AssemblyFile;

  if (targetMachine->addPassesToEmitFile(pm, outBytes, nullptr, fileType)) {
    return GeneratedCode{};
  }

  pm.run(*llvmModule);

  if (!objectFile) {
    return GeneratedCode{
        .text = std::string(outputBuffer.begin(), outputBuffer.size())};
  }

  return GeneratedCode{.objectCode = std::vector<std::uint8_t>(
                           outputBuffer.begin(), outputBuffer.end())};
}

#endif

}  // namespace cxx::js
