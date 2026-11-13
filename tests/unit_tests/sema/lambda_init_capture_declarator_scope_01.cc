// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T, class U>
constexpr bool same = false;
template <class T>
constexpr bool same<T, T> = true;

constexpr int counter() {
  auto next = [i = 0]() mutable -> decltype(i) { return i++; };
  next();
  return next();
}
static_assert(counter() == 1);

constexpr long parameterType() {
  return [x = 1L](decltype(x) y) { return x + y; }(2);
}
static_assert(parameterType() == 3);

constexpr auto templateHead = [n = 4]<class T = decltype(n)>() { return T{} + n; };
static_assert(same<decltype(templateHead()), int>);

void shadow(double x) {
  [x = 1](decltype(x) y) { static_assert(same<decltype(y), int>); };
  [x](decltype(x) y) { static_assert(same<decltype(y), double>); };
}

template <class T>
constexpr T instantiated(T value) {
  auto lambda = [copy = value](decltype(copy) other) -> decltype(copy) {
    return copy + other;
  };
  return lambda(T{2});
}
static_assert(instantiated(5) == 7);
static_assert(same<decltype(instantiated(5L)), long>);
