// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

__attribute__((constructor)) void first() {}
template <int N>
__attribute__((constructor)) void instantiated() {}
template void instantiated<1>();
struct S {
  __attribute__((constructor)) static void member() {}
};
__attribute__((constructor)) void last() {}
__attribute__((constructor(150))) void a() {}
__attribute__((constructor(150))) void b() {}

// CHECK: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }]
// CHECK-SAME: { i32 150, ptr @_Z1av, ptr null }
// CHECK-SAME: { i32 150, ptr @_Z1bv, ptr null }
// CHECK-LABEL: define internal void @_GLOBAL__sub_I_main()
// CHECK-NEXT: call void @_Z5firstv()
// CHECK-NEXT: call void @_Z12instantiatedILi1EEvv()
// CHECK-NEXT: call void @_ZN1S6memberEv()
// CHECK-NEXT: call void @_Z4lastv()
