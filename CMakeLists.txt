# Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

project(cplusplus)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" version)

# option to install tools
option(CXX_INSTALL_TOOLS "Install tools" ON)
option(CXX_INSTALL_WASI_SYSROOT "Install wasi sysroot" OFF)
option(CXX_LIBCXX_WITH_CLANG "Link with libc++" OFF)
OPTION(CXX_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CXX_LIBCXX_WITH_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# if CMAKE_SYSTEM_NAME is WASI disable the exceptions
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")

    set(FMT_OS OFF CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    fmt
    GIT_REPOSITORY  https://github.com/fmtlib/fmt
    GIT_TAG         10.1.0
    GIT_SHALLOW     1
)

# set FMT_INSTALL to OFF to avoid installing fmt
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
    utfcpp
    GIT_REPOSITORY  https://github.com/nemtrif/utfcpp
    GIT_TAG         v3.2.4
    GIT_SHALLOW     1
)

FetchContent_MakeAvailable(utfcpp)

FetchContent_Declare(
    nlohmann_json
    URL             https://github.com/nlohmann/json/releases/download/v3.11.2/include.zip
    URL_HASH        SHA256=e5c7a9f49a16814be27e4ed0ee900ecd0092bfb7dbfca65b5a421b774dccaaed
)

FetchContent_MakeAvailable(nlohmann_json)

add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_SOURCE_DIR}/single_include")

FetchContent_Declare(
    wasi_sysroot
    URL https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-20/wasi-sysroot-20.0.tar.gz
    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/patches/wasi-sysroot.patch
)

FetchContent_MakeAvailable(wasi_sysroot)
FetchContent_GetProperties(wasi_sysroot)


set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

if (FLATBUFFERS_FLATC_EXECUTABLE)
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    flatbuffers
    URL https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz
)

FetchContent_MakeAvailable(flatbuffers)
FetchContent_GetProperties(flatbuffers)

add_library(flatbuffers::header-only INTERFACE IMPORTED)
set_target_properties(flatbuffers::header-only PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${flatbuffers_SOURCE_DIR}/include")

enable_testing()

add_subdirectory(tools/kwgen)
add_subdirectory(src)
add_subdirectory(tests)

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
