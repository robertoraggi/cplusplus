// RUN: %cxx -xc -verify -fcheck -dump-symbols %s | %filecheck %s

typeof('x') c;
typeof(0) i;

//      CHECK: variable char c
// CHECK-NEXT: variable int i
