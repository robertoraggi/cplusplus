add_library(cxx-mlir-transforms cxx_to_llvm.cc cxx_to_scf.cc cxx_to_ptr.cc cxx_to_memref.cc)

llvm_update_compile_flags(cxx-mlir-transforms)

target_include_directories(cxx-mlir-transforms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..
    PRIVATE ${PROJECT_SOURCE_DIR}/src/parser
)

target_link_libraries(cxx-mlir-transforms PUBLIC cxx-mlir-dialect)

mlir_target_link_libraries(cxx-mlir-transforms PUBLIC
    MLIRAnalysis
    MLIRAffineToStandard
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRControlFlowToSCF
    MLIRLLVMCommonConversion
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRMemRefToLLVM
    MLIRPass
    MLIRPtrToLLVM
    MLIRSCFToControlFlow
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRUBToLLVM
    MLIRVectorToLLVM
)
