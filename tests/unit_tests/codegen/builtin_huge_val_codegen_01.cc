// RUN: %cxx -emit-ir %s -o - | %filecheck %s

double get_huge_val() { return __builtin_huge_val(); }
float get_huge_valf() { return __builtin_huge_valf(); }

// CHECK-DAG: arith.constant 0x7FF0000000000000
// CHECK-DAG: arith.constant 0x7F800000
// CHECK-NOT: cxx.builtin.call "__builtin_huge_val"
// CHECK-NOT: cxx.builtin.call "__builtin_huge_valf"
