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

union U8 {
  int i;
  const char* p;
};

union U8 u8a = {5};
union U8 u8b = {.p = "hi"};

// CHECK-DAG: %union.U1 = type { i32 }
// CHECK-DAG: %union.U2 = type { %Inner2 }
// CHECK-DAG: %union.U3 = type { i32 }
// CHECK-DAG: %union.U4.constant = type { i8, [3 x i8], [0 x i32] }
// CHECK-DAG: %union.U5 = type { %union.U5inner }
// CHECK-DAG: %union.U6.constant = type { i8, [1 x i8], [0 x i16] }
// CHECK-DAG: %union.U7 = type { %Large }
// CHECK-DAG: @u1 = global %union.U1 { i32 42 }
// CHECK-DAG: @u2 = global %union.U2 { %Inner2 { i32 1, i32 2 } }
// CHECK-DAG: @u3 = global %union.U3 { i32 42 }
// CHECK-DAG: @u4 = global %union.U4.constant { i8 65, [3 x i8] zeroinitializer, [0 x i32] zeroinitializer }, align 4
// CHECK-DAG: @u5 = global %union.U5 { %union.U5inner { i32 42 } }
// CHECK-DAG: @u6 = global %union.U6.constant { i8 1, [1 x i8] zeroinitializer, [0 x i16] zeroinitializer }, align 2
// CHECK-DAG: @u7 = global %union.U7 { %Large { i64 1, i64 2 } }
// CHECK-DAG: %union.U8 = type { ptr }
// CHECK-DAG: %union.U8.constant = type { i32, [4 x i8], [0 x ptr] }
// CHECK-DAG: @u8a = global %union.U8.constant { i32 5, [4 x i8] zeroinitializer, [0 x ptr] zeroinitializer }, align 8
// CHECK-DAG: @u8b = global %union.U8 { ptr @.str{{[0-9]*}} }

int main() { return 0; }
