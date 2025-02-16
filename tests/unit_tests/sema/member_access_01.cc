// RUN: %cxx -verify -fcheck %s

// clang-format off

auto main() -> int {
  using i32 = int;
  using u32 = unsigned int;

  int i;
  const int ci = 0;
  int& ri = i;
  const int& cri = ci;

  i.~i32();
  ci.~i32();
  ri.~i32();
  cri.~i32();

  i.~u32(); // expected-error {{the type of object expression does not match the type being destroyed}}

  int* pi = nullptr;

  pi->~i32();
  pi->~u32(); // expected-error {{the type of object expression does not match the type being destroyed}}

  using nullptr_t = decltype(nullptr);

  (nullptr).~nullptr_t();

  using int_ptr = i32*;

  int_ptr ip = nullptr;
  ip.~int_ptr();
  ip.~nullptr_t(); // expected-error {{the type of object expression does not match the type being destroyed}}

  ip->~i32();
  ip->~u32(); // expected-error {{the type of object expression does not match the type being destroyed}}

  static_assert(__is_same(decltype(i.~i32()), void));

  return 0;
}

// clang-format on
