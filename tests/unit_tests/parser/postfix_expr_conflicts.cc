// RUN: %cxx -verify %s

auto incr(int x) -> int { return x++; }
auto decr(int x) -> int { return x--; }
auto call(int (*f)(int, int), int x) { return (f)(x, x); }
auto subscript(int a[], int n) -> int { return a[n]; }

// expected-error@1 {{expected an expression}}
auto incr_1(int x) -> int { return (x)++; }

// expected-error@1 {{expected an expression}}
auto decr_1(int x) -> int { return (x)--; }

// expected-error@1 {{expected lambda declarator}}
auto subscript_1(int a[], int n) -> int { return (a)[n]; }
