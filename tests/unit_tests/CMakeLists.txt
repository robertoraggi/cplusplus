# Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

find_program(LIT_EXECUTABLE lit)

find_program(FILECHECK_EXECUTABLE filecheck)

if(NOT LIT_EXECUTABLE)
    message(STATUS "lit not found: skip unit tests")
    return()
endif()

message(STATUS "Using lit: ${LIT_EXECUTABLE}")

if(NOT FILECHECK_EXECUTABLE)
    message(STATUS "filecheck not found: skip unit tests")
    return()
endif()

message(STATUS "Using filecheck: ${FILECHECK_EXECUTABLE}")

add_executable(lit::tool IMPORTED)
set_target_properties(lit::tool PROPERTIES IMPORTED_LOCATION ${LIT_EXECUTABLE})

add_executable(filecheck::tool IMPORTED)
set_target_properties(filecheck::tool PROPERTIES IMPORTED_LOCATION ${FILECHECK_EXECUTABLE})

set(CXX_FRONTEND_TARGET_FILE $<TARGET_FILE:cxx-frontend>)

file(READ lit.site.cfg.py.in lit_cfg_in)

string(CONFIGURE "${lit_cfg_in}" lit_cfg @ONLY)

file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    CONTENT "${lit_cfg}")

add_subdirectory(preprocessor)
