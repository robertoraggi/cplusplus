// clang-format off
// RUN: %cxx -verify -fcheck %s

// Type alias qualified lookup: looking up types through
// type aliases and using declarations used as scope qualifiers.

// Using-declaration and nested type as scope qualifier
struct Outer {
  struct Inner {
    static int count;
    using value_type = int;
  };
  using nested = Inner;
};

int Outer::Inner::count = 0;

void test_using_as_qualifier() {
  Outer::nested::value_type v = Outer::nested::count;
}

// expected-no-diagnostics
