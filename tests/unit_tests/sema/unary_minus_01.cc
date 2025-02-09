// RUN: %cxx -verify -fcheck %s

auto main() -> int {
  static_assert(__is_same(decltype(-'a'), int));
  static_assert(__is_same(decltype(-1), int));
  static_assert(__is_same(decltype(-1l), long));
  static_assert(__is_same(decltype(-1ll), long long));
  static_assert(__is_same(decltype(-1ul), unsigned long));
  static_assert(__is_same(decltype(-1ull), unsigned long long));
  static_assert(__is_same(decltype(-1.0f), float));
  static_assert(__is_same(decltype(-1.0), double));

  short x{};
  static_assert(__is_same(decltype(-x), int));

  short& y = x;
  static_assert(__is_same(decltype(-y), int));

  unsigned u{};
  static_assert(__is_same(decltype(-u), unsigned));

  return 0;
}