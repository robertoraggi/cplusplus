// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

// CHECK: call void @llvm.va_start

void test(int n, ...) {
  __builtin_va_list ap;
  __builtin_c23_va_start(ap);
  __builtin_va_end(ap);
}
