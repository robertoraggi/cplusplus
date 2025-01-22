# Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

file(GLOB CXX_LSP_INCLUDE_HEADER_FILES cxx/lsp/*.h)

aux_source_directory(cxx/lsp LSP_SOURCES)

add_library(cxx-lsp
    ${LSP_SOURCES}
    ${CXX_LSP_INCLUDE_HEADER_FILES}
)

target_link_libraries(cxx-lsp PUBLIC cxx-parser)

# if cmake founds the Threads package, link with it
if(Threads_FOUND)
    target_link_libraries(cxx-lsp PUBLIC Threads::Threads)
else()
    target_compile_definitions(cxx-lsp PUBLIC CXX_NO_THREADS)
endif()

target_include_directories(cxx-lsp
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cxx-lsp PUBLIC nlohmann_json::nlohmann_json)

# disable exception when targeting emscripten
if (EMSCRIPTEN)
    target_compile_options(cxx-lsp PUBLIC -fno-exceptions)
endif()

add_subdirectory(tests)

if(CXX_INSTALL_LSP)

    install(
        FILES ${CXX_LSP_INCLUDE_HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cxx/lsp
    )

    install(
        TARGETS cxx-lsp
        EXPORT cxxTargets
    )

endif()
