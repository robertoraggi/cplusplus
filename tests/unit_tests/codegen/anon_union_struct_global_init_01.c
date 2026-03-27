// RUN: %cxx -toolchain macos -emit-llvm %s -o - | %filecheck %s

struct S1 {
  int flags;
  double num;
  union {
    void* ptr;
    int* iptr;
  };
};

struct S1 s1_zero = {0, 0.0, {0}};

// All-zero flat (brace-elision) init
struct S1 s1_flat = {0, 0.0, 0};

struct S2 {
  int tag;
  union {
    char c;
    long long ll;
  };
};

struct S2 s2_zero = {0, {0}};
struct S2 s2_char = {1, {'A'}};

int main(void) { return 0; }

// CHECK-DAG: @s1_zero = {{.*}}zeroinitializer
// CHECK-DAG: @s1_flat = {{.*}}zeroinitializer
// CHECK-DAG: @s2_zero = {{.*}}zeroinitializer
// CHECK-DAG: @s2_char = {{.*}}i32 1{{.*}}
