// RUN: %cxx -verify -fcheck %s

constexpr int add(int a, int b) { return a + b; }

constexpr int factorial(int n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

constexpr int square(int x) { return x * x; }

static_assert(add(1, 2) == 3);
static_assert(add(10, 20) == 30);

static_assert(factorial(1) == 1);
static_assert(factorial(5) == 120);

static_assert(add(square(3), square(4)) == 25);

constexpr int abs_val(int x) {
  if (x < 0) return -x;
  return x;
}

static_assert(abs_val(-5) == 5);
static_assert(abs_val(5) == 5);
static_assert(abs_val(0) == 0);
