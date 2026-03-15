// clang-format off
// RUN: %cxx -verify -emit-ir %s -o %t.ir
// RUN: %cxx -verify -emit-mlir %s -o %t.mlir
// RUN: %cxx -verify -emit-llvm %s -o %t.ll
// RUN: %filecheck %s < %t.ll

// CHECK-LABEL: @_Z12test_runtimei
// CHECK: store i32 0
// CHECK: ret i32

// CHECK-LABEL: @_Z14test_constantsv
// CHECK: store i32 1
// CHECK: ret i32
int test_constants() {
  return __builtin_constant_p(42);
}

int test_runtime(int x) {
  return __builtin_constant_p(x);
}
