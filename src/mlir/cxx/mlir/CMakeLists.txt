
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS}) # for tablegen
add_mlir_dialect(CxxOps cxx)

aux_source_directory(. SOURCES)
add_library(cxx-mlir ${SOURCES})

llvm_update_compile_flags(cxx-mlir)

target_include_directories(cxx-mlir
    PUBLIC ${CXX_MLIR_SOURCE_DIR} ${CXX_MLIR_BINARY_DIR}
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(cxx-mlir PUBLIC
    cxx-parser
    MLIRIR
    MLIRFuncDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
)
add_dependencies(cxx-mlir MLIRCxxOpsIncGen)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(cxx-mlir PRIVATE
        -Wno-unused-variable
        -Wno-covered-switch-default
        -Wno-unused-but-set-variable
    )
endif()
