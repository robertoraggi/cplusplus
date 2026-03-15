// RUN: %cxx -emit-ir %s -o - | %filecheck %s

double get_nans() { return __builtin_nans(""); }
float get_nansf() { return __builtin_nansf(""); }

// CHECK: arith.constant
// CHECK-NOT: cxx.builtin.call "__builtin_nans"
// CHECK-NOT: cxx.builtin.call "__builtin_nansf"
