// RUN: %cxx -verify -fcheck %s

auto main() -> int {
  char* p;
  const char* cp;
  void* vp;
  int i = 0;
  short s = 0;

  static_assert(__is_same(int, decltype('a' - '0')));
  static_assert(__is_same(int, decltype('a' - 0)));
  static_assert(__is_same(int, decltype('a' - s)));
  static_assert(__is_same(long, decltype('a' - 1l)));
  static_assert(__is_same(unsigned, decltype('a' - 1u)));

  static_assert(__is_same(decltype(p), decltype(p - 0)));
  static_assert(__is_same(decltype(p), decltype(p - i)));

  static_assert(__is_same(long, decltype(p - p)));
  static_assert(__is_same(long, decltype(p - cp)));

  // clang-format off

  // expected-error@1 {{'char*' and 'void*' are not pointers to compatible types}}
  p - vp;

  // expected-error@1 {{invalid operands to binary expression 'int' and 'char*'}}
  0 - p;

  // clang-format on

  return 0;
}