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

file(GLOB CXX_INCLUDE_HEADER_FILES cxx/*.h)
file(GLOB CXX_VIEWS_INCLUDE_HEADER_FILES cxx/views/*.h)

add_library(cxx-parser
    cxx/ast_cursor.cc
    cxx/ast_interpreter.cc
    cxx/ast_rewriter.cc
    cxx/ast_slot.cc
    cxx/ast.cc
    cxx/base_classes.cc
    cxx/binder.cc
    cxx/cli.cc
    cxx/const_value.cc
    cxx/control.cc
    cxx/cxx.cc
    cxx/decl_specs.cc
    cxx/decl.cc
    cxx/diagnostic.cc
    cxx/diagnostics_client.cc
    cxx/external_name_encoder.cc
    cxx/gcc_linux_toolchain.cc
    cxx/lexer.cc
    cxx/literals.cc
    cxx/macos_toolchain.cc
    cxx/memory_layout.cc
    cxx/name_lookup.cc
    cxx/name_printer.cc
    cxx/names.cc
    cxx/parser.cc
    cxx/path.cc
    cxx/preprocessor.cc
    cxx/scope.cc
    cxx/source_location.cc
    cxx/symbol_chain_view.cc
    cxx/symbol_instantiation.cc
    cxx/symbol_printer.cc
    cxx/symbols.cc
    cxx/token.cc
    cxx/toolchain.cc
    cxx/translation_unit.cc
    cxx/type_checker.cc
    cxx/type_printer.cc
    cxx/type_traits.cc
    cxx/types.cc
    cxx/util.cc
    cxx/wasm32_wasi_toolchain.cc
    cxx/windows_toolchain.cc

    # generated files
    keywords-priv.h
    pp_keywords-priv.h
    # headers
    ${CXX_INCLUDE_HEADER_FILES}
    ${CXX_VIEWS_INCLUDE_HEADER_FILES}
)

target_compile_definitions(cxx-parser PUBLIC
    CXX_VERSION="${CMAKE_PROJECT_VERSION}"
)

target_include_directories(cxx-parser
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cxx-parser PUBLIC cxx_std_23)

# disable exception when targeting emscripten
if (EMSCRIPTEN)
    target_compile_options(cxx-parser PUBLIC -fno-exceptions)
endif()

target_link_libraries(cxx-parser
    PUBLIC  $<BUILD_INTERFACE:utf8cpp>
)

add_custom_command(OUTPUT keywords-priv.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/cxx/keywords.kwgen > keywords-priv.h
    DEPENDS cxx/keywords.kwgen
    COMMENT "Generate keywords-priv.h")

add_custom_command(OUTPUT pp_keywords-priv.h
    COMMAND kwgen < ${CMAKE_CURRENT_SOURCE_DIR}/cxx/pp_keywords.kwgen > pp_keywords-priv.h
    DEPENDS cxx/pp_keywords.kwgen
    COMMENT "Generate pp_keywords-priv.h")

if (CXX_ENABLE_FLATBUFFERS)
    target_sources(cxx-parser PRIVATE
        cxx/flatbuffers/ast_decoder.cc
        cxx/flatbuffers/ast_encoder.cc
    )

    # generate flatbuffers decoder for ast.fbs
    flatbuffers_generate_headers(
        TARGET cxx-ast-flatbuffers
        SCHEMAS cxx/ast.fbs
        BINARY_SCHEMAS_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxx
        FLAGS --cpp)

    target_link_libraries(cxx-parser
        PRIVATE $<BUILD_INTERFACE:flatbuffers::header-only>
    )

    add_dependencies(cxx-parser GENERATE_cxx-ast-flatbuffers)
else()
    target_compile_definitions(cxx-parser PRIVATE CXX_NO_FLATBUFFERS)
endif()

install(
    FILES ${CXX_INCLUDE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cxx
)

install(
    FILES ${CXX_VIEWS_INCLUDE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cxx/views
)

install(
    TARGETS cxx-parser
    EXPORT cxxTargets
)
