# Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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

project(kwgen)

if (KWGEN_EXECUTABLE)
    message(STATUS "Using KWGEN_EXECUTABLE: ${KWGEN_EXECUTABLE}")
    add_executable(kwgen IMPORTED)
    set_target_properties(kwgen PROPERTIES IMPORTED_LOCATION ${KWGEN_EXECUTABLE})
    return()
endif()

set(CMAKE_CXX_STANDARD 20)

aux_source_directory(. SOURCES)
add_executable(kwgen ${SOURCES})

if(EMSCRIPTEN)
    target_link_options(kwgen PUBLIC
        "SHELL:-s EXIT_RUNTIME=1"
        "SHELL:-s NODERAWFS=1"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    )
endif()
