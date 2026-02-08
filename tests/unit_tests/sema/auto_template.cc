// RUN: %cxx -verify -fcheck %s

extern "C" {
int printf(const char* format, ...);
}

auto id(auto x) -> decltype(x) { return x; }

auto add(auto x, auto y) -> decltype(x + y) { return x + y; }

auto id_ref(auto& x) -> decltype(x) { return x; }

auto id_cref(const auto& x) -> decltype(x) { return x; }

struct Wrapper {
  auto apply(auto x) -> decltype(x) { return x; }
};

auto main() -> int {
  printf("%d\n", id(1));
  printf("%d\n", add(1, 2));
  printf("%d\n", add(id(1), id(2)));
  printf("%f\n", id(1.0));
  printf("%f\n", add(1.0, 2.0));
  printf("%f\n", add(id(1.0), id(2.0)));
  int v = 42;
  id_ref(v);
  id_cref(v);
  Wrapper w;
  w.apply(42);
  return 0;
}
