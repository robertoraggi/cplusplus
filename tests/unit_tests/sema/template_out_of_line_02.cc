// RUN: %cxx -verify -fcheck %s

template <typename T>
struct Pair {
  T first_;
  T second_;

  template <typename F>
  auto apply(F f) -> decltype(f(first_, second_));
};

template <typename T>
template <typename F>
auto Pair<T>::apply(F f) -> decltype(f(first_, second_)) {
  return f(first_, second_);
}

extern "C" int printf(const char*, ...);

auto add(int a, int b) -> int { return a + b; }

auto main() -> int {
  Pair<int> p{3, 4};
  auto result = p.apply(add);
  printf("%d\n", result);
  return 0;
}
