// RUN: %cxx -verify -fcheck %s

struct Point {
  int x;
  int y;
  int z;
};

struct Rect {
  struct Point origin;
  struct Point size;
};

void test_designated_init() {
  struct Point p1 = {.x = 1, .y = 2, .z = 3};

  struct Point p2 = {.y = 10};

  struct Rect r = {.origin = {1, 2, 3}, .size = {4, 5, 6}};

  struct Rect r2 = {.origin = {.x = 10, .y = 20}};
}

union IntOrFloat {
  int i;
  float f;
};

void test_union_designated() {
  union IntOrFloat u = {.f = 3.14f};
  //
}
