// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Vec {
  int v;
};

Vec operator+(Vec a, Vec b) { return Vec{a.v + b.v}; }
Vec operator-(Vec a, Vec b) { return Vec{a.v - b.v}; }

int test_binary() {
  Vec a{7};
  Vec b{2};
  Vec c = a + b;
  Vec d = c - b;
  return d.v;
}

static_assert(true);
