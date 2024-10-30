# Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

target_link_libraries(cxx PRIVATE cxx-parser cxx-lsp)

target_compile_definitions(cxx PRIVATE
    CXX_VERSION="${CMAKE_PROJECT_VERSION}"
)

# if cmake founds the Threads package, link with it
if(Threads_FOUND)
    target_link_libraries(cxx PRIVATE Threads::Threads)
else()
    target_compile_definitions(cxx PRIVATE CXX_NO_THREADS)
endif()

if(EMSCRIPTEN)
    target_link_options(cxx PRIVATE
        "SHELL:-s EXIT_RUNTIME=1"
        "SHELL:-s WASM_BIGINT=1"
        "SHELL:-s NODERAWFS=1"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    set_target_properties(cxx PROPERTIES SUFFIX ".wasm")
endif()

if(CXX_INTERPROCEDURAL_OPTIMIZATION)
    set_target_properties(cxx PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if (CXX_INSTALL_TOOLS)
    install(
        TARGETS cxx
        EXPORT cxxTargets
    )
endif()
