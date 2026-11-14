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

#include <cxx/mlir/passes.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

namespace cxx {

namespace {

void registerTargetIndependentPasses() {
  registerConvertCxxToPtrPass();
  registerConvertCxxToMemRefPass();
  registerConvertCxxToFuncPass();
  registerConvertCxxToMathPass();
  registerLiftCxxControlFlowToSCFPass();
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerCanonicalizerPass();
  mlir::registerMem2RegPass();
  mlir::registerSROAPass();
  mlir::memref::registerExpandStridedMetadataPass();
  mlir::registerPass([] { return mlir::createLowerAffinePass(); });
}

}  // namespace

void registerHostPipelinePasses() {
  registerTargetIndependentPasses();
  registerCxxToLLVMPass();
}

#ifdef CXX_WITH_SPIRV
void registerDevicePipelinePasses() {
  registerTargetIndependentPasses();
  registerConvertCxxKernelToGpuPass();
  registerLegalizeCxxDeviceCallsPass();
  registerConvertCxxDeviceToSPIRVPass();
}
#endif

void registerPipelinePasses() {
  registerHostPipelinePasses();
#ifdef CXX_WITH_SPIRV
  registerDevicePipelinePasses();
#endif
}

}  // namespace cxx
