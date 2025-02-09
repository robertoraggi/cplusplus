// RUN: %cxx -verify -fcheck %s

auto main() -> int {
  char c{};
  static_assert(__is_same(decltype(++c), char&));
  static_assert(__is_same(decltype(--c), char&));
  static_assert(__is_lvalue_reference(decltype(++c)));
  static_assert(__is_lvalue_reference(decltype(--c)));

  int x{};
  static_assert(__is_same(decltype(++x), int&));
  static_assert(__is_same(decltype(--x), int&));
  static_assert(__is_lvalue_reference(decltype(++x)));
  static_assert(__is_lvalue_reference(decltype(--x)));

  float f{};
  static_assert(__is_same(decltype(++f), float&));
  static_assert(__is_same(decltype(--f), float&));
  static_assert(__is_lvalue_reference(decltype(++f)));
  static_assert(__is_lvalue_reference(decltype(--f)));

  double d{};
  static_assert(__is_same(decltype(++d), double&));
  static_assert(__is_same(decltype(--d), double&));
  static_assert(__is_lvalue_reference(decltype(++d)));
  static_assert(__is_lvalue_reference(decltype(--d)));

  char* p{};
  static_assert(__is_same(decltype(++p), char*&));
  static_assert(__is_same(decltype(--p), char*&));
  static_assert(__is_lvalue_reference(decltype(++p)));
  static_assert(__is_lvalue_reference(decltype(--p)));

  char*& pr = p;
  static_assert(__is_same(decltype(++pr), char*&));
  static_assert(__is_same(decltype(--pr), char*&));
  static_assert(__is_lvalue_reference(decltype(++pr)));
  static_assert(__is_lvalue_reference(decltype(--pr)));

  return 0;
}