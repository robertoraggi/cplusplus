// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

int printf(const char* format, ...);

// CHECK-DAG: c"(0,0)
// CHECK-DAG: c"(0,1)
// CHECK-DAG: c"(0,2)
// CHECK-DAG: c"(1,0)
// CHECK-DAG: c"(1,1)
// CHECK-DAG: c"(1,2)
// CHECK: @_ZZ4mainvE3sec = internal global
// CHECK-SAME: ptr @

int main() {
  static char* sec[2][3] = {
      [0][0] = "(0,0)", [0][1] = "(0,1)", [0][2] = "(0,2)",
      [1][0] = "(1,0)", [1][1] = "(1,1)", [1][2] = "(1,2)",
  };
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%s ", sec[i][j]);
    }
    printf("\n");
  }
  return 0;
}
