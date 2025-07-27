
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS}) # for tablegen
add_mlir_dialect(CxxOps cxx)

set(SOURCES
    codegen.cc
    codegen_declarations.cc
    codegen_declarators.cc
    codegen_expressions.cc
    codegen_names.cc
    codegen_specifiers.cc
    codegen_statements.cc
    codegen_units.cc
    convert_type.cc
    cxx_dialect.cc
    cxx_dialect_conversions.cc
)

add_library(cxx-mlir ${SOURCES})

llvm_update_compile_flags(cxx-mlir)

target_include_directories(cxx-mlir
    PUBLIC ${CXX_MLIR_SOURCE_DIR} ${CXX_MLIR_BINARY_DIR}
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(cxx-mlir PUBLIC
    cxx-parser
    MLIRControlFlowDialect
    MLIRControlFlowToLLVM
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRPass
    MLIRTransforms
)

target_compile_definitions(cxx-mlir PUBLIC CXX_WITH_MLIR)

add_dependencies(cxx-mlir MLIRCxxOpsIncGen)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")

    target_compile_options(cxx-mlir PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-all -Wno-extra>
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(cxx-mlir PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-covered-switch-default>
        )
    endif()

endif()
