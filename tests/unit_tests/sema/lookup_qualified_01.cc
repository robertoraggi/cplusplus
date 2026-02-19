// clang-format off
// RUN: %cxx -verify -fcheck %s

// Qualified lookup through nested name specifiers with multiple levels.

namespace top {
namespace mid {
namespace bot {
  struct S {
    static int value;
    using type = double;
  };
  int S::value = 99;
}  // namespace bot
}  // namespace mid
}  // namespace top

void test_deep_qualified() {
  top::mid::bot::S s;
  int v = top::mid::bot::S::value;
  top::mid::bot::S::type d = 1.0;
}

// Qualified lookup into class with base class members
struct Base {
  static int base_val;
};

int Base::base_val = 7;

struct Mid : Base {};
struct Leaf : Mid {};

void test_qualified_through_bases() {
  int a = Leaf::base_val;
  int b = Mid::base_val;
}

// expected-no-diagnostics
