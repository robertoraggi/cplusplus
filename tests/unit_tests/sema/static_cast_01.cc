// RUN: %cxx -verify -fcheck %s

struct X {};

using F = int();

int f() { return 0; }

auto main() -> int {
  X x;

  F&& rf = f;

  // prvalue
  static_assert(__is_reference(decltype(static_cast<X>(x))) == false);

  // lvalue if lvalue reference to object type
  static_assert(__is_lvalue_reference(decltype(static_cast<X&>(x))));

  // rvalue if rvalue reference to object type
  static_assert(__is_rvalue_reference(decltype(static_cast<X&&>(x))));

  // prvalue
  static_assert(__is_reference(decltype(static_cast<F*>(f))) == false);

  // lvalue if lvalue reference to function type
  static_assert(__is_lvalue_reference(decltype(static_cast<F&>(f))));

  // lvalue if rvalue reference to function type
  static_assert(__is_lvalue_reference(decltype(static_cast<F&&>(f))));
}