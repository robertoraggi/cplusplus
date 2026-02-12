// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Sum {
  int v;
  Sum& operator+=(int x) {
    v = v + x;
    return *this;
  }
};

int test_compound_assignment() {
  Sum s{3};
  s += 4;
  return s.v;
}
