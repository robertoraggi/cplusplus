// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

union U1 {
  int x;
  char y;
};

union U1 u1 = {42};

struct Inner2 {
  int x, y;
};

union U2 {
  struct Inner2 inner;
  int z;
};

union U2 u2 = {{1, 2}};

union U3 {
  int x;
  struct {
    short a, b;
  } inner;
};

union U3 u3 = {42};

union U4 {
  char c;
  int i;
};

union U4 u4 = {'A'};

union U5inner {
  int a;
  char b;
};

union U5 {
  union U5inner inner;
  float f;
};

union U5 u5 = {{42}};

union U6 {
  char a;
  short b;
};

union U6 u6 = {1};

struct Large {
  long long a, b;
};

union U7 {
  struct Large big;
  int x;
};

union U7 u7 = {{1, 2}};

// CHECK-DAG: %union.U1 = type { i32 }
// CHECK-DAG: %union.U2 = type { %Inner2 }
// CHECK-DAG: %union.U3 = type { i32 }
// CHECK-DAG: %union.U4 = type { i32 }
// CHECK-DAG: %union.U5 = type { %union.U5inner }
// CHECK-DAG: %union.U6 = type { i16 }
// CHECK-DAG: %union.U7 = type { %Large }
// CHECK-DAG: @u1 = global %union.U1 { i32 42 }
// CHECK-DAG: @u2 = global %union.U2 { %Inner2 { i32 1, i32 2 } }
// CHECK-DAG: @u3 = global %union.U3 { i32 42 }
// CHECK-DAG: @u4 = global %union.U4 { i32 65 }
// CHECK-DAG: @u5 = global %union.U5 { %union.U5inner { i32 42 } }
// CHECK-DAG: @u6 = global %union.U6 { i16 1 }
// CHECK-DAG: @u7 = global %union.U7 { %Large { i64 1, i64 2 } }

int main() { return 0; }
