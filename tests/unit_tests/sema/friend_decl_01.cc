// clang-format off
// RUN: %cxx -verify -fcheck %s

// Test: hidden friend function found via ADL

struct X {
  friend void found_by_adl(X) {}  // hidden friend, defined inline
};

void test_adl() {
  X x;
  found_by_adl(x);  // OK: ADL finds found_by_adl through X
}
