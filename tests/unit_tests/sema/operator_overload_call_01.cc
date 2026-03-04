// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Adder {
  constexpr int operator()(int a, int b) const { return a + b; }
  constexpr double operator()(double a, double b) const { return a + b; }
};

static_assert(Adder{}(1, 2) == 3);
static_assert(Adder{}(1.5, 2.5) == 4.0);

struct Mixed {
  constexpr int operator()(int) { return 1; }
  constexpr int operator()(int, int) { return 2; }
};

static_assert(Mixed{}(0) == 1);
static_assert(Mixed{}(0, 0) == 2);

struct Single {
  constexpr int operator()(int x) const { return x * 2; }
};

static_assert(Single{}(5) == 10);
