// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

extern const char SP_NAME[];
extern int arr[];

// CHECK-DAG: @arr = external {{.*}} [0 x i32]
// CHECK-DAG: @SP_NAME = external {{.*}} [0 x i8]

const char *get_name(void) {
  return SP_NAME;
}

char get_first(void) {
  return SP_NAME[0];
}

// CHECK-LABEL: @get_int
int get_int(int i) {
  // CHECK: getelementptr [0 x i32]
  return arr[i];
}
