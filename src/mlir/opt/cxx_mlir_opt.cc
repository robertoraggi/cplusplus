#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/passes.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Ptr/IR/PtrDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#ifdef CXX_WITH_SPIRV
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#endif

int main(int argc, char** argv) {
  cxx::registerCxxToLLVMPass();
  cxx::registerConvertCxxToPtrPass();
  cxx::registerConvertCxxToMemRefPass();
  cxx::registerConvertCxxToFuncPass();
  cxx::registerConvertCxxToMathPass();
  cxx::registerLiftCxxControlFlowToSCFPass();
#ifdef CXX_WITH_SPIRV
  cxx::registerLegalizeCxxDeviceAtomicsPass();
  cxx::registerConvertCxxWorkgroupStoragePass();
  cxx::registerConvertCxxKernelToGpuPass();
  cxx::registerLegalizeCxxDeviceCallsPass();
  cxx::registerConvertCxxDeviceToSPIRVPass();
  mlir::spirv::registerSPIRVLowerABIAttributesPass();
  mlir::spirv::registerSPIRVUpdateVCEPass();
#endif
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerCanonicalizerPass();
  mlir::registerMem2RegPass();
  mlir::registerSROAPass();
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::registerSymbolDCEPass();
  mlir::memref::registerExpandStridedMetadataPass();
  mlir::registerPass([] { return mlir::createLowerAffinePass(); });
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  registry.insert<mlir::math::MathDialect>();
#ifdef CXX_WITH_SPIRV
  registry.insert<mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect>();
#endif
  registry.insert<mlir::cxx::CxxDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::ptr::PtrDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Cxx MLIR optimizer", registry));
}
