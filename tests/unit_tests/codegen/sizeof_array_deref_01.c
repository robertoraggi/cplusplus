// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

int is_valid(int kind) {
  int elements[] = {1, 2, 3};
  return kind < (int)(sizeof(elements) / sizeof(*(elements)));
}

// CHECK: define i32 @is_valid(i32 %0)