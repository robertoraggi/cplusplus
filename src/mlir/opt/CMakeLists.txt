add_executable(cxx-mlir-opt cxx_mlir_opt.cc)

llvm_update_compile_flags(cxx-mlir-opt)

target_link_libraries(cxx-mlir-opt PRIVATE
    cxx-mlir-transforms
    MLIRMlirOptMain
)

mlir_target_link_libraries(cxx-mlir-opt PRIVATE MLIRFuncDialect)

if (CXX_INSTALL_TOOLS)
    install(TARGETS cxx-mlir-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
