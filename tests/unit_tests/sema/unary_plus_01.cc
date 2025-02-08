// RUN: %cxx -verify -fcheck %s

auto main() -> int {
  static_assert(__is_same(decltype(+'a'), int));
  static_assert(__is_same(decltype(+1), int));
  static_assert(__is_same(decltype(+1.0f), float));
  static_assert(__is_same(decltype(+1.0), double));

  short x{};
  static_assert(__is_same(decltype(+x), int));

  short& y = x;
  static_assert(__is_same(decltype(+y), int));

  int a[2];
  static_assert(__is_same(decltype(+a), int*));

  void (*f)() = nullptr;
  static_assert(__is_same(decltype(+f), void (*)()));

  const void* p = nullptr;
  static_assert(__is_same(decltype(+p), const void*));

  return 0;
}