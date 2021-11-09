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

find_package(fmt 8.0.1 CONFIG QUIET)

set(CMAKE_CXX_STANDARD 20)

if(NOT fmt_FOUND)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        7.1.3
    GIT_SHALLOW     1
)

FetchContent_MakeAvailable(fmt)
endif()

FetchContent_Declare(
    utfcpp
    GIT_REPOSITORY  https://github.com/nemtrif/utfcpp
    GIT_TAG         v3.2.1
    GIT_SHALLOW     1
)

FetchContent_MakeAvailable(utfcpp)

FetchContent_Declare(
    nlohmann_json
    URL             https://github.com/nlohmann/json/releases/download/v3.10.4/include.zip
    URL_HASH        SHA256=62c585468054e2d8e7c2759c0d990fd339d13be988577699366fe195162d16cb
)

FetchContent_MakeAvailable(nlohmann_json)

add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_SOURCE_DIR}/single_include")

add_subdirectory(tools/kwgen)

aux_source_directory(src/cxx SOURCES)

add_library(CPlusPlus ${SOURCES}
    # generated files
    keywords-priv.h
)

target_include_directories(CPlusPlus PUBLIC src)
target_include_directories(CPlusPlus PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(CPlusPlus fmt::fmt-header-only utf8::cpp)
target_link_libraries(CPlusPlus nlohmann_json::nlohmann_json)

add_custom_command(OUTPUT keywords-priv.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/src/cxx/keywords.kwgen > keywords-priv.h
    DEPENDS kwgen src/cxx/keywords.kwgen
    COMMENT "Generate keywords-priv.h")

aux_source_directory(src/frontend FRONTEND_SOURCES)

add_executable(cxx-frontend ${FRONTEND_SOURCES})
target_link_libraries(cxx-frontend CPlusPlus)

if(EMSCRIPTEN)
    target_link_options(cxx-frontend PUBLIC
    "SHELL:-s EXIT_RUNTIME=1"
    "SHELL:-s NODERAWFS=1"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    )
endif()

if(EMSCRIPTEN)

    add_executable(cxx-js src/cxx-js/api.cc)

    target_link_libraries(cxx-js CPlusPlus)

    target_link_options(cxx-js PUBLIC
        "SHELL:--bind"
        "SHELL:-s ENVIRONMENT=web"
        "SHELL:-s WASM_ASYNC_COMPILATION=1"
        "SHELL:-s MODULARIZE=1"
        "SHELL:-s MALLOC=emmalloc"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s FILESYSTEM=0"
        "SHELL:-s NO_DYNAMIC_EXECUTION=1"
        "SHELL:--extern-post-js ${CMAKE_CURRENT_SOURCE_DIR}/src/cxx-js/suffix.js"
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

add_subdirectory(tests)
