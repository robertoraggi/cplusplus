
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS}) # for tablegen
add_mlir_dialect(CxxOps cxx)
set(LLVM_TARGET_DEFINITIONS CxxOps.td)

mlir_tablegen(CxxOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CxxOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CxxOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CxxOpsAttributes.cpp.inc -gen-attrdef-defs)

target_sources(MLIRCxxOpsIncGen PRIVATE
    CxxOpsEnums.h.inc
    CxxOpsEnums.cpp.inc
    CxxOpsAttributes.h.inc
    CxxOpsAttributes.cpp.inc
)

add_library(cxx-mlir-dialect cxx_dialect.cc)

target_compile_features(cxx-mlir-dialect PUBLIC cxx_std_23)

llvm_update_compile_flags(cxx-mlir-dialect)

target_include_directories(cxx-mlir-dialect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

mlir_target_link_libraries(cxx-mlir-dialect PUBLIC
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRDLTIDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRVectorDialect
)

if (LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(cxx-mlir-dialect PUBLIC LLVM)
endif()

add_dependencies(cxx-mlir-dialect MLIRCxxOpsIncGen)
