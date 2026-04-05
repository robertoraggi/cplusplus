// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

// clang-format off

typedef struct {
  int x;
  int y;
  int z;
} Vec3;

Vec3* v1 = &(Vec3){2, 1, 1};
// CHECK-DAG: @.compoundliteral{{[0-9]+}} = internal global %Vec3 { i32 2, i32 1, i32 1 }
// CHECK-DAG: @v1 = global ptr @.compoundliteral

Vec3* v2 = &(Vec3){4, 4, 4};
// CHECK-DAG: @.compoundliteral{{[0-9]+}} = internal global %Vec3 { i32 4, i32 4, i32 4 }
// CHECK-DAG: @v2 = global ptr @.compoundliteral

typedef struct {
  int val;
  void* next;
} Node;
Node* head = &(Node){42, &(Node){99, 0}};
// CHECK-DAG: @.compoundliteral{{[0-9]+}} = internal global %Node { i32 99, ptr null }
// CHECK-DAG: @.compoundliteral{{[0-9]+}} = internal global %Node { i32 42, ptr @.compoundliteral
// CHECK-DAG: @head = global ptr @.compoundliteral

int* arr = (int[]){10, 20, 30};
// CHECK-DAG: @.compoundliteral{{[0-9]+}} = internal global [3 x i32] [i32 10, i32 20, i32 30]
// CHECK-DAG: @arr = global ptr @.compoundliteral

char* msg = (char[]){"hello"};
// CHECK-DAG: @.compoundliteral{{[0-9]+}} = internal global [6 x i8] c"hello\00"
// CHECK-DAG: @msg = global ptr @.compoundliteral

Vec3 p = (Vec3){1, 8, 8};
// CHECK-DAG: @p = global %Vec3 { i32 1, i32 8, i32 8 }
