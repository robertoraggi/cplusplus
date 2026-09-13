# Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    return()
endif()

if (EMSCRIPTEN)
    # set the executable suffix to .cjs
    set(CMAKE_EXECUTABLE_SUFFIX ".cjs")
endif()

add_executable(test_api
    test_arena.cc
    test_control.cc
    test_diagnostics.cc
    test_external_names.cc
    test_lexer.cc
    test_pch.cc
    test_translation_unit.cc
    test_type_printer.cc
    test_utf8.cc
)

target_link_libraries(test_api
    GTest::gtest_main
    cxx-parser)

gtest_discover_tests(test_api DISCOVERY_MODE PRE_TEST)

if (CXX_ENABLE_MLIR)
    add_executable(test_mlir_emitter test_mlir_emitter.cc)

    target_link_libraries(test_mlir_emitter
        GTest::gtest_main
        cxx-mlir)

    gtest_discover_tests(test_mlir_emitter DISCOVERY_MODE PRE_TEST)
endif()

# Evaluate the target's effective include directories, including dependencies.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/codegen-dependencies-$<CONFIG>.txt"
    CONTENT "$<TARGET_PROPERTY:cxx-codegen,INCLUDE_DIRECTORIES>\n$<TARGET_PROPERTY:cxx-codegen,LINK_LIBRARIES>\n")
add_test(NAME codegen_backend_firewall
    COMMAND ${CMAKE_COMMAND}
        -DCODEGEN_SOURCE_DIR=${PROJECT_SOURCE_DIR}/src/codegen
        -DDEPENDENCIES_FILE=${CMAKE_CURRENT_BINARY_DIR}/codegen-dependencies-$<CONFIG>.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/codegen_backend_firewall.cmake)
