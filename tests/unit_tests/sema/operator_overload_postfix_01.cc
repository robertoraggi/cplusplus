// RUN: %cxx -verify -fcheck %s

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
  // expected-error@+1 {{cannot increment a value of type '::Counter'}}
  Counter old = c++;
  return old.v + c.v;
}
