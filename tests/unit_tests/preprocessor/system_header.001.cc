// clang-format off
// RUN: %cxx -verify -E -nostdinc -isystem %S/sysinclude %s -o - | %filecheck %s

#include <sys_header.h>
int x;

// CHECK: # 1 "{{.*}}sys_header.h" 1 3
// CHECK-NEXT: typedef int sys_type;
// CHECK-NEXT: # 5 "{{.*}}system_header.001.cc" 2
// CHECK-NEXT: int x;
