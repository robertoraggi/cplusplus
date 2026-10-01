set(SOURCES
    mlir_debug_emitter.cc
    mlir_emitter.cc
    mlir_emitter_module.cc
    mlir_emitter_control_flow.cc
    cxx_dialect_conversions.cc
)

add_library(cxx-mlir ${SOURCES})

llvm_update_compile_flags(cxx-mlir)

target_include_directories(cxx-mlir
    PUBLIC ${CXX_MLIR_SOURCE_DIR}
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(cxx-mlir PUBLIC
    cxx-codegen
    cxx-mlir-transforms
)

mlir_target_link_libraries(cxx-mlir PUBLIC
    MLIRArithDialect
    MLIRBuiltinToLLVMIRTranslation
    MLIRControlFlowDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRPass
    MLIRTargetLLVMIRExport
    MLIRTargetLLVMIRImport
    MLIRTransforms
    MLIRVectorDialect
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
