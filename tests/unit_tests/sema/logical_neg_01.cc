// RUN: %cxx -verify -fcheck %s

auto main() -> int {
  static_assert(__is_same(decltype(!true), bool));
  static_assert(__is_same(decltype(!!true), bool));

  static_assert(__is_same(decltype(!1), bool));
  static_assert(__is_same(decltype(!1.0), bool));

  const void* p = nullptr;
  static_assert(__is_same(decltype(!p), bool));

  static_assert(__is_same(decltype(!main), bool));

  const int a[2] = {1, 2};
  static_assert(__is_same(decltype(!a), bool));

  int x{}, &y = x;

  static_assert(__is_same(decltype(!x), bool));
  static_assert(__is_same(decltype(!y), bool));

  return 0;
}