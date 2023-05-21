# Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

aux_source_directory(cxx SOURCES)

add_executable(cxx ${SOURCES})

target_link_libraries(cxx
    cxx-parser
    cxx-ir
    nlohmann_json::nlohmann_json
)

if (LLVM_FOUND)
    target_link_libraries(cxx cxx-llvm)
endif()

if(EMSCRIPTEN)
    target_link_options(cxx PUBLIC
        "SHELL:-s EXIT_RUNTIME=1"
        "SHELL:-s NODERAWFS=1"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    target_compile_definitions(cxx PUBLIC JSON_HAS_FILESYSTEM=0)
endif()

install(
    TARGETS cxx
    EXPORT cxxTargets
)
