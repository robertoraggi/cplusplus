// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct G {
  constexpr int operator()(int x) { return x; }
  constexpr int operator()(int x) const { return x + 100; }
};

static_assert(G{}(1) == 1);

void test_const_overload() {
  const G cg;
  G mg;
  static_assert(__is_same(decltype(cg(1)), int));
  static_assert(__is_same(decltype(mg(1)), int));
}
