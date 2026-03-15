// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

int x[] = {[2] = 1, [4] = 2, [6] = 3};

// CHECK: @x = {{.*}}[7 x i32]
// CHECK-SAME: 1
// CHECK-SAME: 2
// CHECK-SAME: 3
