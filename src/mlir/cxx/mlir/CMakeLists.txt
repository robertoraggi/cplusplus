
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

set(SOURCES
    mlir_debug_emitter.cc
    mlir_emitter.cc
    mlir_emitter_module.cc
    mlir_emitter_control_flow.cc
    convert_debug_type.cc
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
    cxx-codegen
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRBuiltinToLLVMIRTranslation
    MLIRControlFlowDialect
    MLIRControlFlowToLLVM
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRPass
    MLIRTargetLLVMIRExport
    MLIRTargetLLVMIRImport
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorToLLVM
)

if (CXX_ENABLE_LLVM_OPTIMIZER)
    if (NOT LLVM_LINK_LLVM_DYLIB)
        llvm_map_components_to_libnames(cxx_mlir_llvm_libs passes)
        target_link_libraries(cxx-mlir PUBLIC ${cxx_mlir_llvm_libs})
    endif()
    target_compile_definitions(cxx-mlir PUBLIC CXX_WITH_LLVM_OPTIMIZER)
else()
    message(STATUS "cxx: the LLVM optimization pipeline is disabled")
endif()

if (EMSCRIPTEN)
    target_link_libraries(cxx-mlir PUBLIC LLVMWebAssemblyCodeGen)
endif()

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
