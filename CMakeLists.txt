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
project(cplusplus)

cmake_minimum_required(VERSION 3.10)

find_package(fmt CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(tools/pgen)
add_subdirectory(tools/kwgen)

aux_source_directory(src/cxx SOURCES)

add_library(CPlusPlus ${SOURCES}
    # generated files
    keywords-priv.h
    parser-priv.h
)

target_include_directories(CPlusPlus PUBLIC src)
target_include_directories(CPlusPlus PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(CPlusPlus PUBLIC fmt::fmt)

add_custom_command(OUTPUT parser-priv.h
    COMMAND pgen ${CMAKE_CURRENT_SOURCE_DIR}/src/cxx/parser.pgen -o parser-priv.h
    DEPENDS pgen src/cxx/parser.pgen
    COMMENT "Generate ParserP.h")

add_custom_command(OUTPUT keywords-priv.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/src/cxx/keywords.kwgen > keywords-priv.h
    DEPENDS kwgen src/cxx/keywords.kwgen
    COMMENT "Generate KeywordsP.h")

add_executable(cxx-frontend src/frontend/frontend.cc)
target_link_libraries(cxx-frontend CPlusPlus)

add_custom_target(unit
    DEPENDS tests/unit/source.cc
    COMMAND clang++-10 -E -stdlib=libc++ -xc++ -std=c++20 ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/source.cc | grep -v ^\# > unit.cc
)

add_custom_target(unit.tokens
    DEPENDS unit cxx-frontend
    COMMAND cxx-frontend --dump-tokens unit.cc > unit.tokens
)

enable_testing()
