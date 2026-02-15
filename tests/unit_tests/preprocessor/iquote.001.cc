// clang-format off
// RUN: %cxx -verify -E -nostdinc -iquote %S/quoteinclude %s -o - | %filecheck %s

#include "quote_header.h"
int x;

// CHECK: # 1 "{{.*}}quote_header.h" 1
// CHECK-NEXT: typedef int quote_type;
// CHECK-NEXT: # 5 "{{.*}}iquote.001.cc" 2
// CHECK-NEXT: int x;
