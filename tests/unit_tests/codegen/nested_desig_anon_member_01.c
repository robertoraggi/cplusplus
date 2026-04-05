// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

// clang-format off

typedef struct {
  int kind;
  union {
    struct { int a; int b; } pair;
    long val;
  };
  int tag;
} Thing;

Thing g1 = {.kind = 1, .pair.a = 10, .pair.b = 20, .tag = 99};
// CHECK-DAG: @g1 = global %Thing { i32 1, %"{{[^"]+}}" { %"{{[^"]+}}" { i32 10, i32 20 } }, i32 99 }

typedef struct {
  int x;
  struct { int a; int b; };
  int y;
} Simple;

Simple s1 = {.x = 5, .a = 11, .b = 22, .y = 7};
// CHECK-DAG: @s1 = global %Simple { i32 5, %"{{[^"]+}}" { i32 11, i32 22 }, i32 7 }
