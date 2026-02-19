// clang-format off
// RUN: %cxx -verify -fcheck %s

// Anonymous struct/union member lookup: members of anonymous
// structs/unions should be found as if they were direct members.

struct HasAnon {
  union {
    int i;
    float f;
  };
  struct {
    int x;
    int y;
  };
};

void test_anon_members() {
  HasAnon h;
  h.i = 1;
  h.f = 2.0f;
  h.x = 3;
  h.y = 4;
}

// Nested anonymous types
struct Nested {
  struct {
    union {
      int a;
      double b;
    };
    int c;
  };
};

void test_nested_anon() {
  Nested n;
  n.a = 1;
  n.b = 2.0;
  n.c = 3;
}

// expected-no-diagnostics
