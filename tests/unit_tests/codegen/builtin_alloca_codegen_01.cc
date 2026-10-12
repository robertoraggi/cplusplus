// RUN: %cxx -emit-ir %s -o - | %filecheck %s
// RUN: %cxx -toolchain macos -emit-ir %s -o - | %filecheck %s

void* do_alloca(int n) { return __builtin_alloca(n); }

// CHECK: cxx.dyn_alloca {{.*}} align 16
// CHECK-NOT: cxx.builtin.call "__builtin_alloca"
