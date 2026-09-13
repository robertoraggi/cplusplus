# Generic lowering depends only on the parser, never on a backend.
add_library(cxx-codegen
    cxx/codegen/codegen_declarations.cc
    cxx/codegen/codegen_declarators.cc
    cxx/codegen/codegen.cc
    cxx/codegen/codegen_specifiers.cc
    cxx/codegen/codegen_statements.cc
    cxx/codegen/codegen_names.cc
    cxx/codegen/codegen_rtti.cc
    cxx/codegen/codegen_expressions.cc
    cxx/codegen/codegen_units.cc
    cxx/codegen/convert_type.cc
)
target_include_directories(cxx-codegen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cxx-codegen PUBLIC cxx-parser)
