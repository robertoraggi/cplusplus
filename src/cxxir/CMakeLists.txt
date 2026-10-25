# A standalone example backend: no LLVM or MLIR dependencies.
add_library(cxx-cxxir cxx/cxxir/cxxir_emitter.cc)
target_include_directories(cxx-cxxir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cxx-cxxir PUBLIC cxx-codegen)
