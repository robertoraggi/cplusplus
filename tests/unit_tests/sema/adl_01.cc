// RUN: %cxx -verify -fcheck %s

extern "C" int printf(const char*, ...);

namespace ns {

struct Point {
  int x;
  int y;
};

int distance(const Point& a, const Point& b) { return a.x - b.x + a.y - b.y; }

}  // namespace ns

auto main() -> int {
  ns::Point p1{1, 2};
  ns::Point p2{3, 4};
  int d = distance(p1, p2);
  printf("Distance: %d\n", d);
  return 0;
}
