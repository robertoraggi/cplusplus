// clang-format off
// RUN: %cxx -verify -E -nostdinc %s -o - | %filecheck %s

#include "line_markers_outer.h"
int main_var;

// CHECK: # 1 "{{.*}}line_markers_outer.h" 1
// CHECK-NEXT: typedef int outer_before_type;
// CHECK-NEXT: # 1 "{{.*}}line_markers_nested.h" 1
// CHECK-NEXT: typedef int nested_type;
// CHECK-NEXT: # 3 "{{.*}}line_markers_outer.h" 2
// CHECK-NEXT: typedef int outer_after_type;
// CHECK-NEXT: # 5 "{{.*}}line_markers.001.cc" 2
// CHECK-NEXT: int main_var;
