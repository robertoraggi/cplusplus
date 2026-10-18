add_library(cxx-mlir-transforms cxx_to_llvm.cc cxx_to_scf.cc cxx_to_ptr.cc cxx_to_memref.cc
    cxx_to_func.cc cxx_to_math.cc)

llvm_update_compile_flags(cxx-mlir-transforms)

target_include_directories(cxx-mlir-transforms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..
    PRIVATE ${PROJECT_SOURCE_DIR}/src/parser
)

target_link_libraries(cxx-mlir-transforms PUBLIC cxx-mlir-dialect)

mlir_target_link_libraries(cxx-mlir-transforms PUBLIC
    MLIRAnalysis
    MLIRAffineToStandard
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRControlFlowToSCF
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLLVMCommonConversion
    MLIRMathDialect
    MLIRMathToLLVM
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

if (TARGET MLIRGPUDialect AND TARGET MLIRSPIRVDialect
    AND TARGET MLIRGPUToSPIRV AND TARGET MLIRSPIRVTransforms
    AND TARGET MLIRSPIRVSerialization AND TARGET MLIRSPIRVDeserialization)
    target_sources(cxx-mlir-transforms PRIVATE cxx_to_gpu.cc)
    target_compile_definitions(cxx-mlir-transforms PUBLIC CXX_WITH_SPIRV)
    set_property(TARGET cxx-mlir-transforms PROPERTY CXX_WITH_SPIRV ON)
    mlir_target_link_libraries(cxx-mlir-transforms PUBLIC
        MLIRGPUDialect MLIRSPIRVDialect MLIRUBDialect
    )
endif()
