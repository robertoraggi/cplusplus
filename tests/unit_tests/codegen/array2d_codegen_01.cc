// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

// CHECK: getelementptr [258 x i8], ptr
// CHECK-NOT: getelementptr [258 x i8], ptr {{.*}}, i32 0, i32 %

void test(unsigned char (*p)[258], int i) {
  unsigned char x = p[i][7];
  (void)x;
}
