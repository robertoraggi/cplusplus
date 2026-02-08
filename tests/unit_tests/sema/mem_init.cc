// RUN: %cxx -verify -fcheck %s

extern "C" {
int printf(const char* format, ...);
}

namespace std {
using ::printf;
}

struct Point {
  int x;
  int y;

  Point() : x(10), y(20) { std::printf("Point() x=%d, y=%d\n", x, y); }

  Point(int x, int y) : x(x), y(y) {
    std::printf("Point(int, int) x=%d, y=%d\n", x, y);
  }
};

struct Derived : Point {
  int z;

  Derived() : Point(1, 2), z(3) {
    std::printf("Derived() x=%d, y=%d, z=%d\n", x, y, z);
  }
};

auto main() -> int {
  Point p1;
  Point p2(100, 200);
  Derived d;
  return 0;
}
