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

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(tools/pgen)

add_executable(kwgen tools/kwgen.cc)

add_library(CPlusPlus
    src/AST.cc
    src/AST.h
    src/ASTVisitor.cc
    src/ASTVisitor.h
    src/Arena.h
    src/Codegen.cc
    src/Codegen.h
    src/Control.cc
    src/Control.h
    src/Globals.h
    src/IR.cc
    src/IR.h
    src/IRfwd.h
    src/Names.cc
    src/Names.h
    src/ParseContext.cc
    src/ParseContext.h
    src/Parser.cc
    src/Symbols.cc
    src/Symbols.h
    src/Token.cc
    src/Token.h
    src/TranslationUnit.cc
    src/TranslationUnit.h
    src/Types.cc
    src/Types.h
    src/Mangler.cc
    src/Mangler.h
    # generated files
    KeywordsP.h
    ParserP.h
)

target_include_directories(CPlusPlus PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(OUTPUT ParserP.h
    COMMAND pgen ${CMAKE_CURRENT_SOURCE_DIR}/src/Parser.pgen -o ParserP.h
    DEPENDS pgen src/Parser.pgen
    COMMENT "Generate ParserP.h")

add_custom_command(OUTPUT KeywordsP.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/src/Keywords.kwgen > KeywordsP.h
    DEPENDS kwgen src/Keywords.kwgen
    COMMENT "Generate KeywordsP.h")

add_executable(cxx-frontend src/Main.cc)
target_link_libraries(cxx-frontend CPlusPlus)

enable_testing()
