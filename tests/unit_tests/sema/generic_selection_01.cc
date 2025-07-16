// RUN: %cxx -verify -fcheck %s

// clang-format off

auto main() -> int {
  static_assert(_Generic(0, int: 123, default: 0) == 123);

  static_assert(_Generic(0.0, int: 123, double: 321) == 321);

  static_assert(_Generic("str", char: 123, const char *: 444) == 444);

  // expected-error@1 {{multiple matching types for _Generic selector of type 'int'}}
  _Generic(0, int: 0, int: 1);

  // expected-error@1 {{multiple default associations in _Generic selection}}
  _Generic(0, default: 0, default: 1);

  // expected-error@1 {{no matching type for _Generic selector of type 'double'}}
  _Generic(0.0, char: 0, int: 1);

  return 0;
}