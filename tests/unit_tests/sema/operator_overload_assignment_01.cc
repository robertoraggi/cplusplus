// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Box {
  int v;
  Box& operator=(int x) {
    v = x;
    return *this;
  }
};

int test_assignment() {
  Box b{0};
  b = 42;
  return b.v;
}
