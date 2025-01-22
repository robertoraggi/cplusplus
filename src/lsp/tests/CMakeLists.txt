if (NOT CXX_BUILD_TESTS)
    return()
endif()

aux_source_directory(. SOURCES)
add_executable(test_lsp ${SOURCES})

target_link_libraries(test_lsp
    GTest::gtest_main
    cxx-lsp)

gtest_discover_tests(test_lsp DISCOVERY_MODE PRE_TEST)
