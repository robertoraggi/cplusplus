// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

struct Buffer {
  int data[4];
  int& operator[](int i) { return data[i]; }
  int operator[](int i) const { return data[i]; }
};

int test_subscript() {
  Buffer b{{1, 2, 3, 4}};
  return b[2];
}

int test_subscript_const() {
  const Buffer b{{1, 2, 3, 4}};
  return b[2];
}
