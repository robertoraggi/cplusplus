// RUN: %cxx -verify -dump-symbols %s | %filecheck %s

char ch;

typeof(ch) c1;
typeof('x') c2;
typeof(0) i;

//      CHECK:namespace
// CHECK-NEXT:  variable char ch
// CHECK-NEXT:  variable char c1
// CHECK-NEXT:  variable int c2
// CHECK-NEXT:  variable int i
