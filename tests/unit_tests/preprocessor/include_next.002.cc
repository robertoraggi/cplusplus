// clang-format off
// RUN: %cxx -x c -E -P -nostdinc -isystem %S/include_next_first -isystem %S/include_next_second %s -o - | %filecheck %s

#include <foo.h>
int from_main;

// CHECK: int before_next;
// CHECK-NEXT: int from_second;
// CHECK-NEXT: int after_next;
// CHECK-NEXT: int from_main;
