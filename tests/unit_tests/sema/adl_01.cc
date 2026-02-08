// RUN: %cxx -verify -fcheck %s

extern "C" int printf(const char*, ...);

namespace ns {

struct Vec2 {
  double x;
  double y;
};

constexpr auto dot(const Vec2& a, const Vec2& b) -> double {
  return a.x * b.x + a.y * b.y;
}

}  // namespace ns

void f1() {
  ns::Vec2 p1{1.0, 2.0};
  ns::Vec2 p2{3.0, 4.0};
  printf("dot product: %f\n", dot(p1, p2));
}

#ifdef FIX_TYPE_CONSTRUCTION
void f2() {
  ns::Vec2 p1 = ns::Vec2{1.0, 2.0};
  ns::Vec2 p2 = ns::Vec2{3.0, 4.0};
  printf("dot product: %f\n", ns::dot(p1, p2));
}
#endif

auto main() -> int {
  f1();

#ifdef FIX_TYPE_CONSTRUCTION
  f2();
#endif

  return 0;
}
