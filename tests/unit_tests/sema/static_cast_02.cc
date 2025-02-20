// RUN: %cxx -verify -fcheck -freport-missing-types  %s

// clang-format off

struct B {};
struct D : B {};

auto main() -> int {
  B b;
  const B cb;

  // check casts to void
  static_cast<void>(b);
  static_cast<const void>(b);
  static_cast<volatile void>(b);
  static_cast<void>(123);
  static_cast<void>(cb);

  // check casts to derived class
  D& d = static_cast<D&>(b);
  const D& cd1 = static_cast<const D&>(b);
  const D& cd2 = static_cast<const D&>(cb);
  const volatile D& cd3 = static_cast<const volatile D&>(cb);

  D& d = static_cast<D&>(cb);  // expected-error {{invalid static_cast of 'const ::B' to '::D'}}

  return 0;
}

// clang-format on
