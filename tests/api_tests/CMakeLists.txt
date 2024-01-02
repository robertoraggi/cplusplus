
if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    return()
endif()

include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)

FetchContent_MakeAvailable(googletest)

aux_source_directory(. SOURCES)
add_executable(test_api ${SOURCES})

target_link_libraries(test_api
    GTest::gtest_main
    cxx-parser)

include(GoogleTest)
gtest_discover_tests(test_api)
