// clang-format off
// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

bool f() { return bool(true); }
int g() { return int(42); }
double h() { return double(3.14); }
char k() { return char('x'); }

static_assert(bool(true), "bool(true)");
static_assert(int(1) == 1, "int(1)");
