
if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    return()
endif()

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
)

FetchContent_MakeAvailable(googletest)

aux_source_directory(. SOURCES)
add_executable(test_api ${SOURCES})

target_link_libraries(test_api
    GTest::gtest_main
    cxx-parser)

include(GoogleTest)
gtest_discover_tests(test_api)
