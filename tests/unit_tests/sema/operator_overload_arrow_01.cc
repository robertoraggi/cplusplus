// RUN: %cxx -verify %s
// expected-no-diagnostics

struct Ptr {
  int value;
  constexpr int* operator->() { return &value; }
  constexpr const int* operator->() const { return &value; }
};

void test_arrow() {
  Ptr p{42};
  const Ptr cp{42};
  static_assert(__is_same(decltype(p.operator->()), int*));
  static_assert(__is_same(decltype(cp.operator->()), const int*));
}
