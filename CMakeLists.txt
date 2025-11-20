# Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(ExternalProject)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.json" package_json)

# option to install tools
option(CXX_INSTALL_TOOLS "Install tools" ON)
option(CXX_INSTALL_WASI_SYSROOT "Install wasi sysroot" OFF)
option(CXX_ENABLE_FLATBUFFERS "Enable flatbuffers" OFF)
option(CXX_ENABLE_MLIR "Enable MLIR" OFF)
option(CXX_LIBCXX_WITH_CLANG "Link with libc++" OFF)
option(CXX_BUILD_TESTS "Build tests" ON)
option(CXX_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization" OFF)

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
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    )

    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

endif()

#
# utfcpp
#
FetchContent_Declare(
    utfcpp
    URL                 https://github.com/nemtrif/utfcpp/archive/refs/tags/v4.0.5.tar.gz
    PATCH_COMMAND       patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/patches/utfcpp.patch
)

FetchContent_MakeAvailable(utfcpp)
FetchContent_GetProperties(utfcpp)

#
# nlohmann_json
#
FetchContent_Declare(
    nlohmann_json
    URL             https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip
    URL_HASH        SHA256=b8cb0ef2dd7f57f18933997c9934bb1fa962594f701cd5a8d3c2c80541559372
)

FetchContent_MakeAvailable(nlohmann_json)

add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_SOURCE_DIR}/single_include")

FetchContent_Declare(
    wasi_sysroot
    URL         https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-29/wasi-sysroot-29.0.tar.gz
    URL_HASH    SHA256=d99d5c4b277a725b7b56bf9d591609972ecac3207520a66408927280f191f6c7
)

FetchContent_MakeAvailable(wasi_sysroot)
FetchContent_GetProperties(wasi_sysroot)

if (CXX_ENABLE_FLATBUFFERS)
    set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

    if (FLATBUFFERS_FLATC_EXECUTABLE)
        set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_Declare(
        flatbuffers
        URL         https://github.com/google/flatbuffers/archive/refs/tags/v25.1.24.tar.gz
        URL_HASH    SHA256=ffc668a310e77607d393f3c18b32715f223da1eac4c4d6e0579a11df8e6b59cf
    )

    FetchContent_MakeAvailable(flatbuffers)
    FetchContent_GetProperties(flatbuffers)

    add_library(flatbuffers::header-only INTERFACE IMPORTED)
    set_target_properties(flatbuffers::header-only PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${flatbuffers_SOURCE_DIR}/include")
endif()


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
