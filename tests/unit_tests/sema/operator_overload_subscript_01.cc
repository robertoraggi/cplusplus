// RUN: %cxx -verify -fcheck %s

struct Buffer {
  int data[4];
  int& operator[](int i) { return data[i]; }
  int operator[](int i) const { return data[i]; }
};

int test_subscript() {
  Buffer b{{1, 2, 3, 4}};
  // expected-error@1 {{call to overloaded operator '[]' is ambiguous}}
  return b[2];
}
