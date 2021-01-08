# Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

include(FetchContent)

find_package(fmt 6.1.2 CONFIG QUIET)

set(CMAKE_CXX_STANDARD 20)

if(NOT fmt_FOUND)
    message(STATUS "fetching fmt")

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        7.1.3
    )

    FetchContent_MakeAvailable(fmt)
endif()

FetchContent_Declare(
    utfcpp
    GIT_REPOSITORY https://github.com/nemtrif/utfcpp
    GIT_TAG        v3.1.2
)

FetchContent_MakeAvailable(utfcpp)

add_subdirectory(tools/kwgen)

aux_source_directory(src/cxx SOURCES)

add_library(CPlusPlus ${SOURCES}
    # generated files
    keywords-priv.h
)

target_include_directories(CPlusPlus PUBLIC src)
target_include_directories(CPlusPlus PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(CPlusPlus fmt::fmt-header-only utf8::cpp)

add_custom_command(OUTPUT keywords-priv.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/src/cxx/keywords.kwgen > keywords-priv.h
    DEPENDS kwgen src/cxx/keywords.kwgen
    COMMENT "Generate keywords-priv.h")

add_executable(cxx-frontend src/frontend/frontend.cc)
target_link_libraries(cxx-frontend CPlusPlus)

if(EMSCRIPTEN)
    target_link_options(cxx-frontend PUBLIC
    "SHELL:-s EXIT_RUNTIME=1"
    "SHELL:-s NODERAWFS=1"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    )
endif()

add_custom_target(unit
    DEPENDS tests/unit/source.cc
    COMMAND clang++ -E -stdlib=libc++ -xc++ -std=c++20 ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/source.cc | grep -v ^\# > unit.cc
)

add_custom_target(unit.tokens
    DEPENDS unit cxx-frontend
    COMMAND cxx-frontend --dump-tokens unit.cc > unit.tokens
)

enable_testing()
