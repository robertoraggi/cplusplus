// RUN: %cxx -verify -fcheck -freport-missing-types %s

struct X {
  enum E { kValue = 0 };

  static int s_value;
};

auto main() -> int {
  X x;
  static_assert(__is_reference(decltype(x.kValue)) == false);
  static_assert(__is_same(decltype(x.kValue), X::E));

  X& xr = x;
  static_assert(__is_reference(decltype(xr.kValue)) == false);
  static_assert(__is_same(decltype(xr.kValue), X::E));

  const X& cxr = x;
  static_assert(__is_reference(decltype(cxr.kValue)) == false);
  static_assert(__is_same(decltype(cxr.kValue), X::E));

  X* px = &x;

  static_assert(__is_reference(decltype(px->kValue)) == false);
  static_assert(__is_same(decltype(px->kValue), X::E));

  static_assert(__is_reference(decltype(px->s_value)) == false);
  static_assert(__is_same(decltype(px->s_value), int));

  return 0;
}
