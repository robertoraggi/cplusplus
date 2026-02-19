// clang-format off
// RUN: %cxx -verify -fcheck %s

// Base class member lookup: names inherited from base classes
// should be found by unqualified lookup in derived classes.

struct Base {
  int value = 42;
  static int static_value;

  int get() const { return value; }
};

int Base::static_value = 10;

struct Derived : Base {
  int use_base() {
    int a = value;
    int b = static_value;
    int c = get();
    return a + b + c;
  }
};

// Diamond hierarchy — name found through multiple paths
struct A {
  using type = int;
};

struct B : A {};
struct C : A {};
struct D : B, C {
  // Qualified lookup should resolve through a specific path
  using type_b = B::type;
  using type_c = C::type;
};

// expected-no-diagnostics
