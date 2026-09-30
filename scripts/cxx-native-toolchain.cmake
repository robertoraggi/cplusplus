if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "The native self-host presets currently require macOS")
endif()

set(cxx_native_target "${CMAKE_HOST_SYSTEM_PROCESSOR}-apple-macosx")
set(CMAKE_C_COMPILER_TARGET "${cxx_native_target}" CACHE STRING "C target triple")
set(CMAKE_CXX_COMPILER_TARGET "${cxx_native_target}" CACHE STRING "C++ target triple")
