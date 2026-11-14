// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class... Ts>
constexpr int simple(Ts... ts) {
  auto copy = [ts...] { return (0 + ... + ts); };
  auto reference = [&ts...] { return (0 + ... + ts); };
  auto bump = [&ts...] { ((ts += 1), ...); };
  bump();
  return copy() * 100 + reference();
}
static_assert(simple(1, 2, 3) == 609);
static_assert(simple() == 0);

template <class... Ts>
constexpr int init(Ts... ts) {
  auto sum = [... us = ts, first = 100] { return (first + ... + us); };
  auto count = [... us = ts] { return static_cast<int>(sizeof...(us)); };
  auto bump = [&... us = ts] { ((us += 1), ...); };
  bump();
  return sum() + count() + (0 + ... + ts);
}
static_assert(init(1, 2L, short{3}) == 106 + 3 + 9);
static_assert(init() == 100);

template <class... Ts>
constexpr auto nested(Ts... ts) {
  return [... us = ts](auto extra) {
    return [... vs = us, extra] { return (extra + ... + vs); }();
  }(10);
}
static_assert(nested(1, 2, 3) == 16);
