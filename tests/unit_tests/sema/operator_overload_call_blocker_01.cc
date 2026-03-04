// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct F {
  constexpr int operator()(int) const { return 1; }
  constexpr int operator()(double) const { return 2; }
};

static_assert(F{}(3.14) == 2);
static_assert(F{}(42) == 1);
