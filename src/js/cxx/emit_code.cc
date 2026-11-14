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

#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/cxxir/cxxir_emitter.h>
#include <cxx/memory_layout.h>
#include <cxx/private/path.h>
#include <cxx/translation_unit.h>

#include <format>

#ifdef CXX_WITH_MLIR
#include <cxx/mlir/backend.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/mlir/mlir_emitter.h>
#include <cxx/mlir/passes.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#endif

namespace cxx::js {

auto targetsDevice(TranslationUnit* unit) -> bool {
  return unit->control()->memoryLayout()->deviceProfile() !=
         DeviceProfile::kNone;
}

auto hostTargetRequired(std::string_view output) -> std::string {
  return std::format(
      "{} needs a host target; the device profile emits cxxir, mlir, asm, "
      "obj, wgsl and spirv",
      output);
}

#ifndef CXX_WITH_MLIR

auto hasCodeGenerator() -> bool { return false; }

auto generateCode(TranslationUnit* unit, std::string_view format,
                  bool debugInfo, int, std::string_view mlirPipeline,
                  bool locations) -> std::optional<GeneratedCode> {
  if (format != "cxxir" || !mlirPipeline.empty()) return std::nullopt;
  ir::CxxIREmitter emitter{unit};
  Codegen codegen{
      emitter,
      unit,
      {.debugInfo = debugInfo,
       .debugCompilationDirectory = fs::working_directory().string()}};
  auto generated = codegen(unit->ast());
  if (!generated.module) return GeneratedCode{};
  return GeneratedCode{
      .text = std::format(
          "{}", emitter.module(generated.module, locations || debugInfo))};
}

#else

namespace {

[[nodiscard]] auto backendOutputOf(std::string_view format)
    -> std::optional<BackendOutput> {
  if (format == "cxxir") return BackendOutput::kCxxIR;
  if (format == "mlir") return BackendOutput::kMLIR;
  if (format == "llvm") return BackendOutput::kLLVMIR;
  if (format == "asm") return BackendOutput::kAssembly;
  if (format == "obj") return BackendOutput::kObject;
  if (format == "spirv") return BackendOutput::kSPIRV;
  if (format == "wgsl") return BackendOutput::kWGSL;
  return std::nullopt;
}

[[nodiscard]] auto isKernelFormat(std::string_view format) -> bool {
  return format == "wgsl" || format == "spirv";
}

class DiagnosticMessages {
 public:
  explicit DiagnosticMessages(llvm::raw_ostream& out) : out_(out) {}

  auto record(mlir::Diagnostic& diagnostic) -> bool {
    print(diagnostic);
    for (auto& note : diagnostic.getNotes()) print(note);
    return true;
  }

 private:
  void print(mlir::Diagnostic& diagnostic) {
    if (auto location =
            diagnostic.getLocation()->findInstanceOf<mlir::FileLineColLoc>())
      out_ << location.getFilename().getValue() << ":" << location.getLine()
           << ":" << location.getColumn() << ": ";
    switch (diagnostic.getSeverity()) {
      case mlir::DiagnosticSeverity::Error:
        out_ << "error: ";
        break;
      case mlir::DiagnosticSeverity::Warning:
        out_ << "warning: ";
        break;
      case mlir::DiagnosticSeverity::Note:
        out_ << "note: ";
        break;
      case mlir::DiagnosticSeverity::Remark:
        out_ << "remark: ";
        break;
    }
    diagnostic.print(out_);
    out_ << "\n";
  }

  llvm::raw_ostream& out_;
};

}  // namespace

auto hasCodeGenerator() -> bool { return true; }

auto generateCode(TranslationUnit* unit, std::string_view format,
                  bool debugInfo, int optimizationLevel,
                  std::string_view mlirPipeline, bool locations)
    -> std::optional<GeneratedCode> {
  const auto output = backendOutputOf(format);
  if (!output) return std::nullopt;
  if (targetsDevice(unit) && format == "llvm")
    return GeneratedCode{.error = hostTargetRequired(format)};

  GeneratedCode generated;
  llvm::raw_string_ostream errors(generated.error);
  DiagnosticMessages diagnostics{errors};
  registerPipelinePasses();
  mlir::MLIRContext context{mlir::MLIRContext::Threading::DISABLED};
  mlir::ScopedDiagnosticHandler handler(&context,
                                        [&](mlir::Diagnostic& diagnostic) {
                                          (void)diagnostics.record(diagnostic);
                                          return mlir::success();
                                        });
  context.loadDialect<mlir::cxx::CxxDialect>();
  ir::MlirEmitter emitter{context, unit};
  Codegen codegen{
      emitter,
      unit,
      {.debugInfo = debugInfo,
       .debugCompilationDirectory = fs::working_directory().string(),
       .runsAlwaysInliner = llvmPipelineRunsAlwaysInliner()}};
  auto result = codegen(unit->ast());
  if (!result.module) return generated;
  mlir::OwningOpRef<mlir::ModuleOp> module{emitter.module(result.module)};

  if (isKernelFormat(format)) mlirPipeline = {};
  if (targetsDevice(unit) && format != "cxxir") mlirPipeline = {};

  auto artifact = compileModule(*module,
                                {.output = *output,
                                 .optimizationLevel = optimizationLevel,
                                 .debugInfo = debugInfo || locations,
                                 .passPipeline = mlirPipeline,
                                 .diagnosticSink =
                                     [&](mlir::Diagnostic& diagnostic) {
                                       return diagnostics.record(diagnostic);
                                     }},
                                errors);
  if (!artifact) {
    generated.error += artifact.error();
    if (format == "obj" || isKernelFormat(format)) return generated;
    if (artifact.error() == "failed to run MLIR pass pipeline") {
      generated.text = "<error running MLIR pipeline>";
      return generated;
    }
    generated.text = std::format("<error lowering to {}>", format);
    return generated;
  }

  if (isKernelFormat(format)) {
    generated.kernels.push_back(
        {std::move(artifact->manifest),
         std::vector<std::uint8_t>(artifact->code.begin(),
                                   artifact->code.end())});
    return generated;
  }
  if (format == "obj") {
    generated.objectCode.assign(artifact->code.begin(), artifact->code.end());
    return generated;
  }
  generated.text = std::move(artifact->code);
  return generated;
}

#endif

}  // namespace cxx::js
