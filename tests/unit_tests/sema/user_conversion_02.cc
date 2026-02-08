// RUN: %cxx -verify -fcheck %s

struct A {
  A(int x) {}
};

struct B {
  B(A a) {}
};

void take_b(B b) {}

void test_chained_conversion() {
  A a = 42;
  B b = a;
}
