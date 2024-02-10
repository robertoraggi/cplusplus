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

file(GLOB CXX_INCLUDE_HEADER_FILES cxx/*.h)

aux_source_directory(cxx SOURCES)

add_library(cxx-parser ${SOURCES}
    # generated files
    keywords-priv.h
)

target_include_directories(cxx-parser
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cxx-parser PUBLIC cxx_std_20)

# disable exception when targeting emscripten
if (EMSCRIPTEN)
    target_compile_options(cxx-parser PUBLIC -fno-exceptions)
endif()

target_link_libraries(cxx-parser
    PUBLIC  $<BUILD_INTERFACE:fmt::fmt-header-only>
            $<BUILD_INTERFACE:utf8cpp::utf8cpp>
)

add_custom_command(OUTPUT keywords-priv.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/cxx/keywords.kwgen > keywords-priv.h
    DEPENDS cxx/keywords.kwgen
    COMMENT "Generate keywords-priv.h")

if (CXX_ENABLE_FLATBUFFERS)
    aux_source_directory(cxx/flatbuffers FLATBUFFERS_SOURCES)
    target_sources(cxx-parser PRIVATE ${FLATBUFFERS_SOURCES})

    # generate flatbuffers decoder for ast.fbs
    flatbuffers_generate_headers(
        TARGET cxx-ast-flatbuffers
        SCHEMAS cxx/ast.fbs
        BINARY_SCHEMAS_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxx
        FLAGS --cpp)

    target_link_libraries(cxx-parser
        PRIVATE $<BUILD_INTERFACE:cxx-ast-flatbuffers>
    )

    target_link_libraries(cxx-parser
        PRIVATE $<BUILD_INTERFACE:flatbuffers::header-only>
    )
else()
    target_compile_definitions(cxx-parser PRIVATE CXX_NO_FLATBUFFERS)
endif()

install(
    FILES ${CXX_INCLUDE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cxx
)

install(
    TARGETS cxx-parser
    EXPORT cxxTargets
)
