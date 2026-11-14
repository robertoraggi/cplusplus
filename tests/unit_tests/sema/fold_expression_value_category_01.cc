// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T, class U>
constexpr bool same = false;
template <class T>
constexpr bool same<T, T> = true;

template <class... Ts>
void folds(Ts... ts) {
  const long first = 1;
  static_assert(same<decltype((first + ... + ts)), long>);
  static_assert(same<decltype((ts + ...)), int>);
  static_assert(same<decltype((... * ts)), int>);
  static_assert(same<decltype((ts << ...)), int>);
  static_assert(same<decltype((ts || ...)), bool>);
  static_assert(same<decltype((ts && ...)), bool>);
  static_assert(same<decltype((... , ts)), int&>);
  static_assert(same<decltype((ts = ...)), int&>);
  static_assert(same<decltype((ts += ...)), int&>);
}
template void folds<int, int>(int, int);

template <class... Ts>
constexpr int sum(Ts... ts) {
  return (0 + ... + ts);
}
static_assert(sum(1, 2, 3) == 6);
