// RUN: %cxx -verify %s

auto incr(int x) -> int { return x++; }
auto decr(int x) -> int { return x--; }
auto call(int (*f)(int, int), int x) { return (f)(x, x); }
auto subscript(int a[], int n) -> int { return a[n]; }

auto incr_1(int x) -> int { return (x)++; }
auto decr_1(int x) -> int { return (x)--; }

auto subscript_1(int a[], int n) -> int { return (a)[n]; }
