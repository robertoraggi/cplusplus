
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()

find_package(MLIR REQUIRED CONFIG)

LIST(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")

include(AddMLIR)
include(AddLLVM)
include(TableGen)
include(HandleLLVMOptions)

SET(CXX_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(CXX_MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(cxx/mlir)
