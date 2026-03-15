// RUN: %cxx -emit-ir %s -o - | %filecheck %s

void* do_new() { return __builtin_operator_new(16); }

// CHECK: cxx.builtin.call "__builtin_operator_new"
