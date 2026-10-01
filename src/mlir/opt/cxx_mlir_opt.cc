#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/passes.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

int main(int argc, char** argv) {
  cxx::registerCxxToLLVMPass();
  cxx::registerLiftCxxControlFlowToSCFPass();
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerCanonicalizerPass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::cxx::CxxDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Cxx MLIR optimizer", registry));
}
