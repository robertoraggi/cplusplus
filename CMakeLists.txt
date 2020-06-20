# Copyright (c) 2014 Roberto Raggi <roberto.raggi@gmail.com>
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

set(CMAKE_CXX_STANDARD 20)

add_executable(pgen tools/pgen.cpp)
add_executable(kwgen tools/kwgen.cpp)

add_library(CPlusPlus
    src/AST.cpp
    src/AST.h
    src/ASTVisitor.cpp
    src/ASTVisitor.h
    src/Arena.h
    src/Codegen.cpp
    src/Codegen.h
    src/Control.cpp
    src/Control.h
    src/Globals.h
    src/IR.cpp
    src/IR.h
    src/IRfwd.h
    src/Names.cpp
    src/Names.h
    src/ParseContext.cpp
    src/ParseContext.h
    src/Parser.cpp
    src/Symbols.cpp
    src/Symbols.h
    src/Token.cpp
    src/Token.h
    src/TranslationUnit.cpp
    src/TranslationUnit.h
    src/Types.cpp
    src/Types.h
    src/Mangler.cpp
    src/Mangler.h
    # generated files
    KeywordsP.h
    ParserP.h
)

target_include_directories(CPlusPlus PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(OUTPUT ParserP.h
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pgen ${CMAKE_CURRENT_SOURCE_DIR}/src/Parser.pgen -o ParserP.h
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/pgen src/Parser.pgen
    COMMENT "Generate ParserP.h")

add_custom_command(OUTPUT KeywordsP.h
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/src/Keywords.kwgen > KeywordsP.h
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kwgen src/Keywords.kwgen
    COMMENT "Generate KeywordsP.h")

add_executable(cxx-frontend src/Main.cpp)
target_link_libraries(cxx-frontend CPlusPlus)

enable_testing()
