// clang-format off
// RUN: %cxx -verify -E -nostdinc -M %s | %filecheck %s

#include "line_markers_outer.h"
int x;

// CHECK: line_markers_depfile.001.o: \
// CHECK-NEXT: {{.*}}line_markers_depfile.001.cc \
// CHECK-NEXT: {{.*}}line_markers_outer.h \
// CHECK-NEXT: {{.*}}line_markers_nested.h
