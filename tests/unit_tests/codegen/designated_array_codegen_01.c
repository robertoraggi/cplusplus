// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

void test(void) {
  int x[] = {[2] = 1, [4] = 2, [6] = 3};
  (void)x[0];
}

// CHECK: alloca [7 x i32]
// CHECK: call void @llvm.memset
// CHECK: getelementptr i32, {{.*}}, i32 2
// CHECK: store i32 1
// CHECK: getelementptr i32, {{.*}}, i32 4
// CHECK: store i32 2
// CHECK: getelementptr i32, {{.*}}, i32 6
// CHECK: store i32 3
