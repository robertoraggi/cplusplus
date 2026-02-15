// clang-format off
// RUN: %cxx -verify -E -nostdinc -MM -isystem %S/sysinclude %s | %filecheck %s

#include <sys_header.h>
#include "line_markers_outer.h"
int x;

// CHECK: line_markers_depfile.002.o: \
// CHECK-NEXT: {{.*}}line_markers_depfile.002.cc \
// CHECK-NEXT: {{.*}}line_markers_outer.h \
// CHECK-NEXT: {{.*}}line_markers_nested.h
// CHECK-NOT: sys_header.h
