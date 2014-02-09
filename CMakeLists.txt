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

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

### our tools
add_executable(pgen tools/pgen.cpp)
add_executable(kwgen tools/kwgen.cpp)

add_library(CPlusPlusObjects OBJECT
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
    ParserP.h)
target_include_directories(CPlusPlusObjects PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(OUTPUT ParserP.h
COMMAND ${CMAKE_CURRENT_BINARY_DIR}/pgen ${CMAKE_CURRENT_SOURCE_DIR}/src/Parser.pgen -o ParserP.h
DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/pgen src/Parser.pgen
COMMENT "Generate ParserP.h")

add_custom_command(OUTPUT KeywordsP.h
COMMAND ${CMAKE_CURRENT_BINARY_DIR}/kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/src/Keywords.kwgen > KeywordsP.h
DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kwgen src/Keywords.kwgen
COMMENT "Generate KeywordsP.h")

add_executable(cplusplus src/Main.cpp $<TARGET_OBJECTS:CPlusPlusObjects>)

add_library(GTestObjects OBJECT
    3rdparty/gtest-1.7.0/src/gtest-all.cc
    3rdparty/gtest-1.7.0/src/gtest_main.cc)
target_include_directories(GTestObjects PRIVATE src 3rdparty/gtest-1.7.0/include 3rdparty/gtest-1.7.0)
target_compile_definitions(GTestObjects PRIVATE "-DGTEST_HAS_PTHREAD=0")

function(add_cplusplus_test TARGET FILES)
add_executable(${TARGET} ${FILES}
    $<TARGET_OBJECTS:GTestObjects>
    $<TARGET_OBJECTS:CPlusPlusObjects>)
target_include_directories(${TARGET} PRIVATE src 3rdparty/gtest-1.7.0/include 3rdparty/gtest-1.7.0)
add_test(${TARGET} ${TARGET})
endfunction(add_cplusplus_test)


enable_testing()
add_cplusplus_test(test-frontend tests/unittest/test-frontend.cpp)
add_cplusplus_test(test-ir tests/unittest/test-ir.cpp)

##
## symbols
##
FILE(GLOB SymbolTests tests/symbols/*.cpp)
foreach(CPP_TEST ${SymbolTests})
    get_filename_component(TEST_NAME ${CPP_TEST} NAME)
        add_test(NAME symbols/${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}
        COMMAND /bin/bash -c "${CMAKE_CURRENT_BINARY_DIR}/cplusplus ${CPP_TEST} --symbols | diff -r ${CPP_TEST}.symbols -")
endforeach(CPP_TEST)

##
## IR
##
FILE(GLOB IRTests tests/ir/*.cpp)
foreach(CPP_TEST ${IRTests})
    get_filename_component(TEST_NAME ${CPP_TEST} NAME)
        add_test(NAME ir/${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}
        COMMAND /bin/bash -c "${CMAKE_CURRENT_BINARY_DIR}/cplusplus ${CPP_TEST} --ir | diff -r ${CPP_TEST}.ir -")
endforeach(CPP_TEST)
