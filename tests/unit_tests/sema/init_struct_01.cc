// RUN: %cxx -verify -fcheck %s

struct Point {
  int x;
  int y;
};

struct Rect {
  struct Point origin;
  struct Point size;
};

void test_struct_init() {
  // Basic struct init
  struct Point p1 = {1, 2};

  struct Point p2 = {1};

  struct Point p3 = {};

  struct Rect r1 = {{1, 2}, {3, 4}};

  struct Rect r2 = {{1, 2}};
}

union IntOrFloat {
  int i;
  float f;
};

void test_union_init() {
  union IntOrFloat u1 = {42};

  union IntOrFloat u2 = {};
}

void test_scalar_init() {
  int x = {42};

  int y = {};
}
