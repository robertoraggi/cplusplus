// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

namespace overload_resolution_basic {

void f(int) {}
void f(double) {}

void test() {
  f(42);
  f(3.14);
}

}  // namespace overload_resolution_basic

namespace overload_ctor_selection {

struct S {
  S(int) {}
  S(double) {}
};

void test() {
  S s1(42);
  S s2(3.14);
  S s3 = 42;
}

}  // namespace overload_ctor_selection

namespace overload_implicit_conversion_rank {

void take(int) {}
void take(double) {}

void test_int_preferred_over_double() {
  take(42);
  take(3.14);
}

}  // namespace overload_implicit_conversion_rank

namespace overload_user_defined_conversion {

struct Wrapper {
  Wrapper(int) {}
};

void accept(Wrapper) {}

void test() { accept(42); }

}  // namespace overload_user_defined_conversion

namespace overload_pointer_conversion {

struct Base {};
struct Derived : Base {};

void take(Base*) {}

void test() {
  Derived d;
  take(&d);
}

}  // namespace overload_pointer_conversion

namespace overload_qualification_conversion {

void take(const int*) {}

void test() {
  int x = 0;
  take(&x);
}

}  // namespace overload_qualification_conversion

namespace overload_nullptr_to_pointer {

void take(int*) {}

void test() { take(nullptr); }

}  // namespace overload_nullptr_to_pointer

namespace overload_binary_operator_member {

struct Vec {
  int v;
  Vec operator+(Vec rhs) { return Vec{v + rhs.v}; }
};

void test() {
  Vec a{1};
  Vec b{2};
  Vec c = a + b;
}

}  // namespace overload_binary_operator_member

namespace overload_binary_operator_non_member {

struct Pt {
  int x;
};

Pt operator+(Pt a, Pt b) { return Pt{a.x + b.x}; }

void test() {
  Pt a{1};
  Pt b{2};
  Pt c = a + b;
}

}  // namespace overload_binary_operator_non_member

namespace overload_unary_operator {

struct Counter {
  int n;
  Counter& operator++() {
    ++n;
    return *this;
  }
  Counter operator++(int) {
    Counter tmp = *this;
    ++n;
    return tmp;
  }
};

void test() {
  Counter c{0};
  ++c;
  c++;
}

}  // namespace overload_unary_operator

namespace overload_conversion_function {

struct IntLike {
  operator int() { return 42; }
};

void take(int) {}

void test() {
  IntLike il;
  int x = il;
  take(il);
}

}  // namespace overload_conversion_function

namespace overload_bool_conversion {

struct Truthy {
  operator bool() { return true; }
};

void test() {
  Truthy t;
  if (t) {
  }
}

}  // namespace overload_bool_conversion
