// RUN: %cxx -emit-ir %s -o - | %filecheck %s --check-prefix=CXX
// RUN: %cxx -emit-llvm %s -o - | %filecheck %s --check-prefix=LLVM

__attribute__((destructor(101))) void early() {}
__attribute__((destructor)) void first() {}
template <int N>
__attribute__((destructor)) void instantiated() {}
template void instantiated<1>();
[[gnu::destructor]] void second() {}

// CXX-DAG: cxx.global_dtor @_Z5earlyv {{.*}}priority = 101{{( : i32)?[}>]}}
// CXX-DAG: cxx.global_dtor @_Z5firstv
// LLVM: @llvm.global_dtors = appending global [4 x { i32, ptr, ptr }]
// LLVM-SAME: { i32 101, ptr @_Z5earlyv, ptr null }
// LLVM-SAME: { i32 65535, ptr @_Z5firstv, ptr null }
// LLVM-SAME: { i32 65535, ptr @_Z12instantiatedILi1EEvv, ptr null }
// LLVM-SAME: { i32 65535, ptr @_Z6secondv, ptr null }
