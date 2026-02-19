// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct L {};
struct CL {};
struct R {};

L pick(int&);
CL pick(const int&);
R pick(int&&);

template <typename T>
auto forward_pick(T&& t) -> decltype(pick(static_cast<T&&>(t))) {
  return pick(static_cast<T&&>(t));
}

template <typename T>
void sink(T&&);

int main() {
  int x = 0;
  const int cx = 0;

  static_assert(__is_same(decltype(pick(x)), L));
  static_assert(__is_same(decltype(pick(cx)), CL));
  static_assert(__is_same(decltype(pick(0)), R));
  static_assert(__is_same(decltype(pick(static_cast<int&&>(x))), R));

  static_assert(__is_same(decltype(forward_pick(x)), L));
  static_assert(__is_same(decltype(forward_pick(cx)), CL));
  static_assert(__is_same(decltype(forward_pick(0)), R));

  sink(x);
  sink(cx);
  sink(0);

  return 0;
}
