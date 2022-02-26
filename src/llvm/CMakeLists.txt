# Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

if(NOT LLVM_FOUND)
    return()
endif()

file(GLOB CXX_INCLUDE_HEADER_FILES cxx/llvm/*.h)

aux_source_directory(cxx/llvm SOURCES)

add_library(cxx-llvm ${SOURCES})

target_link_libraries(cxx-llvm PUBLIC cxx-parser)

target_include_directories(cxx-llvm
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(cxx-llvm PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(cxx-llvm PUBLIC CXX_WITH_LLVM ${LLVM_DEFINITIONS})
target_link_directories(cxx-llvm PUBLIC ${LLVM_LIBRARY_DIRS})
target_link_libraries(cxx-llvm PUBLIC LLVM)

install(
    FILES ${CXX_INCLUDE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cxx/llvm
)

install(
    TARGETS cxx-llvm
    EXPORT cxxTargets
)
