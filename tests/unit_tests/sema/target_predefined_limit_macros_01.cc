// RUN: %cxx --target=aarch64-linux-gnu -E -dM %s | %filecheck %s --check-prefixes=CHECK,LINUX,LINUX-AARCH64
// RUN: %cxx --target=x86_64-linux-gnu -E -dM %s | %filecheck %s --check-prefixes=CHECK,LINUX,SIGNED-WCHAR
// RUN: %cxx --target=arm64-apple-macosx14.0 -E -dM %s | %filecheck %s --check-prefixes=CHECK,INT-SIG-ATOMIC,SIGNED-WCHAR,SIGNED-WINT
// RUN: %cxx --target=x86_64-apple-macosx14.0 -E -dM %s | %filecheck %s --check-prefixes=CHECK,INT-SIG-ATOMIC,SIGNED-WCHAR,SIGNED-WINT
// RUN: %cxx --target=x86_64-pc-windows-msvc -E -dM %s | %filecheck %s --check-prefixes=CHECK,INT-SIG-ATOMIC,WINDOWS
// RUN: %cxx --target=aarch64-pc-windows-msvc -E -dM %s | %filecheck %s --check-prefixes=CHECK,INT-SIG-ATOMIC,WINDOWS
// RUN: %cxx --target=wasm32-wasip1 -E -dM %s | %filecheck %s --check-prefixes=CHECK,WASM,SIGNED-WCHAR,SIGNED-WINT
// RUN: %cxx --target=wasm32-unknown-emscripten -E -dM %s | %filecheck %s --check-prefixes=CHECK,WASM,SIGNED-WCHAR,SIGNED-WINT
// RUN: %cxx --target=spirv32-unknown-unknown -E -dM %s | %filecheck %s --check-prefixes=CHECK,INT-SIG-ATOMIC,SIGNED-WCHAR,SIGNED-WINT
// RUN: %cxx --target=spirv64-unknown-unknown -x c -E -dM %s | %filecheck %s --check-prefixes=CHECK,INT-SIG-ATOMIC,SIGNED-WCHAR,SIGNED-WINT

// CHECK-DAG: #define __MEMORY_SCOPE_CLUSTR 5
// CHECK-DAG: #define __SIG_ATOMIC_MIN__ (-__SIG_ATOMIC_MAX__ - 1)

// LINUX-DAG: #define __SIG_ATOMIC_TYPE__ int
// LINUX-DAG: #define __WINT_MIN__ 0U
// LINUX-AARCH64-DAG: #define __WCHAR_MIN__ 0U

// INT-SIG-ATOMIC-DAG: #define __SIG_ATOMIC_TYPE__ int
// WASM-DAG: #define __SIG_ATOMIC_TYPE__ long int

// SIGNED-WCHAR-DAG: #define __WCHAR_MIN__ (-__WCHAR_MAX__ - 1)
// SIGNED-WINT-DAG: #define __WINT_MIN__ (-__WINT_MAX__ - 1)

// WINDOWS-DAG: #define __WCHAR_MIN__ 0
// WINDOWS-DAG: #define __WINT_MIN__ 0
