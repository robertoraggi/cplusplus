// RUN: %cxx -verify -fcheck %s

struct Point {
  int x;
  int y;

  constexpr Point(int x, int y) : x(x), y(y) {}

  constexpr int sum() const { return x + y; }
};

constexpr Point p(3, 4);
static_assert(p.x == 3);
static_assert(p.y == 4);
static_assert(p.sum() == 7);

constexpr int make_and_sum(int a, int b) {
  Point pt(a, b);
  return pt.sum();
}

static_assert(make_and_sum(10, 20) == 30);

constexpr int distance_squared(int x1, int y1, int x2, int y2) {
  Point a(x1, y1);
  Point b(x2, y2);
  int dx = b.x - a.x;
  int dy = b.y - a.y;
  return dx * dx + dy * dy;
}

static_assert(distance_squared(0, 0, 3, 4) == 25);
