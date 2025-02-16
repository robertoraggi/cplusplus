// RUN: %cxx -verify -fcheck -freport-missing-types %s

struct X {
  enum E { kValue = 0 };

  static int s_value;
};

struct W {
  int& r;
  const int& cr;

  const int const_k = 0;
  int k = 0;

  mutable int m = 0;
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

  int i;
  W w{i, i};

  static_assert(__is_lvalue_reference(decltype(w.r)));
  static_assert(__is_same(decltype(w.r), int&));

  static_assert(__is_lvalue_reference(decltype(w.cr)));
  static_assert(__is_same(decltype(w.cr), const int&));

  static_assert(__is_same(decltype((w.const_k)), const int&));
  static_assert(__is_same(decltype((w.k)), int&));

  const W cw{i, i};
  static_assert(__is_same(decltype(cw.const_k), const int));
  static_assert(__is_same(decltype(cw.k), int));

  static_assert(__is_same(decltype((cw.const_k)), const int&));
  static_assert(__is_same(decltype((cw.k)), const int&));
  static_assert(__is_same(decltype((cw.m)), int&));

  return 0;
}
