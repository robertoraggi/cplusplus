// clang-format off
// RUN: %cxx -verify -fcheck %s

// Using-directive lookup: names made visible by using-directives
// should be found by unqualified lookup.

namespace outer {
namespace inner {
  struct Widget { int x; };
  int helper(Widget w) { return w.x; }
}  // namespace inner
}  // namespace outer

namespace client {
  using namespace outer::inner;

  void test() {
    Widget w{42};
    int r = helper(w);
  }
}  // namespace client

// Transitive using-directives
namespace A {
  struct Foo {};
}

namespace B {
  using namespace A;
}

namespace C {
  using namespace B;
  void test() {
    Foo f;  // found transitively through B -> A
  }
}

// expected-no-diagnostics
