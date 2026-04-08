// RUN: %cxx -verify %s
// expected-no-diagnostics

struct Counter {
  int v;
  Counter operator++(int) {
    Counter old = *this;
    v = v + 1;
    return old;
  }
};

int test_postfix() {
  Counter c{5};
  Counter old = c++;
  return old.v + c.v;
}
