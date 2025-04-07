// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

auto f(int x) -> decltype(x);
auto g(int x, decltype(x)* y) -> decltype(y);
auto k(int x) -> decltype((x));

// clang-format off
//      CHECK:namespace
// CHECK-NEXT: function int f(int)
// CHECK-NEXT: function int* g(int, int*)
// CHECK-NEXT: function int& k(int)