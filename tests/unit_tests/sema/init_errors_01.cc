// RUN: %cxx -verify -fcheck %s

// clang-format off

struct Point {
  int x;
  int y;
};

void test_errors() {
  // expected-error@1 {{excess elements in struct initializer}}
  struct Point p1 = {1, 2, 3};

  // expected-error@1 {{excess elements in scalar initializer}}
  int x = {1, 2};

  // expected-error@1 {{designator in initializer for scalar type}}
  int y = {.x = 1};
}

union U {
  int i;
  float f;
};

void test_union_errors() {
  // expected-error@1 {{excess elements in union initializer}}
  union U u = {1, 2};
}
