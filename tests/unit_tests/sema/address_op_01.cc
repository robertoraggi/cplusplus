// RUN: %cxx -verify -fcheck %s

struct X {
  int m{};
  const int ci{};
  const volatile int cvi{};

  void f();
  void g(int a, const void* ptr) const;
};

auto main() -> int {
  static_assert(__is_same(decltype(&X::m), int X::*));
  static_assert(__is_same(decltype(&X::ci), const int X::*));
  static_assert(__is_same(decltype(&X::cvi), const volatile int X::*));
  static_assert(__is_same(decltype(&X::f), void (X::*)()));
  static_assert(
      __is_same(decltype(&X::g), void (X::*)(int, const void*) const));

  char ch{};
  static_assert(__is_same(decltype(&ch), char*));

  int i{};
  static_assert(__is_same(decltype(&i), int*));

  const void* p = nullptr;
  static_assert(__is_same(decltype(&p), const void**));

  int a[2];
  static_assert(__is_same(decltype(&a), int(*)[2]));

  X x;
  static_assert(__is_same(decltype(&x), X*));
  static_assert(__is_same(decltype(*&x), X&));

  const X cx;
  static_assert(__is_same(decltype(&cx), const X*));
  static_assert(__is_same(decltype(*&cx), const X&));

  return 0;
}