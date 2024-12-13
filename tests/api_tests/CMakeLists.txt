
if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    return()
endif()

aux_source_directory(. SOURCES)
add_executable(test_api ${SOURCES})

target_link_libraries(test_api
    GTest::gtest_main
    cxx-parser)

gtest_discover_tests(test_api DISCOVERY_MODE PRE_TEST)
