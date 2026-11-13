// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T, class U>
constexpr bool same = false;
template <class T>
constexpr bool same<T, T> = true;

constexpr auto explicitHead = []<class T, class U>(T t, U u, auto v) {
  static_assert(same<T, int> && same<U, long>);
  return t + u + v;
};
static_assert(explicitHead(1, 2L, 3) == 6);

template <class V>
constexpr V instantiated(V value) {
  auto lambda = [value]<class T, class U>(T t, U u, auto w) {
    return value + t + u + w;
  };
  return lambda(V{1}, short{2}, 3);
}
static_assert(instantiated(4) == 10);
static_assert(instantiated(4L) == 10);

constexpr int nested() {
  auto outer = []<class T>(T t) {
    return [t]<class U>(U u, auto v) { return t + u + v; }(T{2}, 3);
  };
  return outer(1);
}
static_assert(nested() == 6);
