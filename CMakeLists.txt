# Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.json" PACKAGE_JSON)

string(JSON version GET ${PACKAGE_JSON} version)

project(cplusplus VERSION ${version})

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if (POLICY CMP0219)
    cmake_policy(SET CMP0219 NEW)
endif()

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(ExternalProject)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.json" package_json)

# option to install tools
option(CXX_INSTALL_TOOLS "Install tools" ON)
option(CXX_INSTALL_WASI_SYSROOT "Install wasi sysroot" OFF)
option(CXX_ENABLE_MLIR "Enable MLIR" OFF)
option(CXX_LIBCXX_WITH_CLANG "Link with libc++" OFF)
option(CXX_BUILD_TESTS "Build tests" ON)
option(CXX_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization" OFF)

if (EMSCRIPTEN)
    set(cxx_llvm_optimizer_default OFF)
else()
    set(cxx_llvm_optimizer_default ON)
endif()
option(CXX_ENABLE_LLVM_OPTIMIZER "Link the LLVM optimization pipeline"
       ${cxx_llvm_optimizer_default})

if(NOT CMAKE_SYSTEM_NAME STREQUAL "WASI")
    find_package(Threads)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CXX_LIBCXX_WITH_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# if CMAKE_SYSTEM_NAME is WASI disable the exceptions, and the tests
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    set(CXX_BUILD_TESTS OFF)

    # set the executable suffix to .wasm
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
endif()

#
# google test
#
if(CXX_BUILD_TESTS)

    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        URL         https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH    SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    )

    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

endif()

#
# simdjson
#
FetchContent_Declare(
    simdjson
    URL             https://github.com/simdjson/simdjson/releases/download/v4.6.8/singleheader.zip
    URL_HASH        SHA256=dd2b0790507f03321d3a359e2ea94129257319be081fbddbae079adfe7561011
)

FetchContent_MakeAvailable(simdjson)

add_library(simdjson "${simdjson_SOURCE_DIR}/simdjson.cpp")
set_target_properties(simdjson PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${simdjson_SOURCE_DIR}")

if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    target_compile_definitions(simdjson PUBLIC _WASI_EMULATED_MMAN)
    target_link_libraries(simdjson PUBLIC wasi-emulated-mman)
endif()

FetchContent_Declare(
    wasi_sysroot
    URL         https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-32/wasi-sysroot-32.0.tar.gz
    URL_HASH    SHA256=f2537f6e5804f7f24e32dd140e9371e9670a9fd2646b6a813ddb896b29954b12
)

FetchContent_MakeAvailable(wasi_sysroot)
FetchContent_GetProperties(wasi_sysroot)

FetchContent_Declare(
    wasi_builtins
    URL         https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-32/libclang_rt-32.0.tar.gz
    URL_HASH    SHA256=e42ba168766a4a1161d44b740d22c336b39424190ca7a672acbd8b71b8d4372a
)

FetchContent_MakeAvailable(wasi_builtins)
FetchContent_GetProperties(wasi_builtins)

if (CXX_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(src)

if (CXX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cxxConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cxx
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cxxConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

install(
    EXPORT cxxTargets
    FILE cxxTargets.cmake
    NAMESPACE cxx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cxx
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cxxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cxxConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cxx
)

set(CLANGD_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/clangd.in")

if (EMSCRIPTEN)
    set(CLANGD_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/clangd_emscripten.in")
endif()

configure_file(
    ${CLANGD_CONFIG_FILE}
    ${CMAKE_CURRENT_SOURCE_DIR}/.clangd
    @ONLY
)
