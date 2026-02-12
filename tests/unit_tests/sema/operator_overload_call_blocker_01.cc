// RUN: %cxx -verify -fcheck %s

struct F {
  constexpr int operator()(int) const { return 1; }
  constexpr int operator()(double) const { return 2; }
};

// clang-format off
// expected-error@1 {{static assertion expression is not an integral constant expression}}
static_assert(F{}(3.14) == 2);
