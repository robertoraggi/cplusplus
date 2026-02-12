// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Num {
  int v;
  int operator-() const { return -v; }
  bool operator!() const { return v == 0; }
};

int test_unary() {
  Num n{5};
  int a = -n;
  bool b = !n;
  return a + (b ? 1 : 0);
}
