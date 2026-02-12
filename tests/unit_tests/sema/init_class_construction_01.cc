// RUN: %cxx -verify -fcheck %s

struct Point {
  int x;
  int y;

  Point() = default;
  Point(int x, int y) : x(x), y(y) {}
};

auto test_direct_list_init() -> int {
  Point p{1, 2};
  return p.x + p.y;
}

auto test_functional_cast() -> int {
  Point p2 = Point(5, 6);
  return p2.x + p2.y;
}

auto test_braced_type_construction() -> int {
  auto p3 = Point{3, 4};
  return p3.x + p3.y;
}

auto test_member_access_on_temporary() -> int { return Point{3, 4}.x; }

auto test_functional_cast_expression() -> int { return Point(7, 8).x; }
