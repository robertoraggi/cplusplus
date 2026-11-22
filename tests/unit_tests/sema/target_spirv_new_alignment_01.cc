// RUN: %cxx --target=spirv32-unknown-unknown -E -dM %s | %filecheck %s --check-prefix=SPIRV32
// RUN: %cxx --target=spirv64-unknown-unknown -E -dM %s | %filecheck %s --check-prefix=SPIRV64

// SPIRV32: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8U{{$}}
// SPIRV64: #define __STDCPP_DEFAULT_NEW_ALIGNMENT__ 8UL{{$}}
