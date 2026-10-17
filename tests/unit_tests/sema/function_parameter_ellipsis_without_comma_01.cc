// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

template <class R, class A>
using Fn = R(A...);
static_assert(__is_same(Fn<int, char>, int(char, ...)));

template <class R, class... A>
using Fp = R(A...);
static_assert(__is_same(Fp<int, char, long>, int(char, long)));

static_assert(__is_same(void(int...), void(int, ...)));
static_assert(__is_same(void(int*...), void(int*, ...)));
static_assert(__is_same(void(const char*...), void(const char*, ...)));
static_assert(__is_same(void(int...) noexcept, void(int, ...) noexcept));

template <class T>
struct Q {
  using fn = void(T...);
  using fn2 = void(T, ...);
};
static_assert(__is_same(Q<int>::fn, Q<int>::fn2));

template <class T, class L>
struct Shape;

template <class R, class T0, class T1, class L>
struct Shape<R (T0::*)(T1...) const, L> {
  using result = R;
};
template <class R, class T0, class T1, class L>
struct Shape<R (T0::*)(T1, ...) volatile, L> {
  using result = R;
};

struct M {
  void m(int...);
  void n(int, long...) const;
};
void M::m(int, ...) {}
void M::n(int, long, ...) const {}

void c_style(int...);
void c_style_named(int x...);
template <class T>
void tmpl_c_style(T...);

template <class... Ts>
void pack(Ts...);
template <class... Ts>
void pack_ptr(Ts*...);
template <class... Ts>
void pack_ref(const Ts&...);
template <class... Ts>
void pack_fwd(Ts&&...);
template <class... Ts>
struct Tup {};
template <class... Ts>
void pack_nested(Tup<Ts>...);

void abbreviated(auto...);
void abbreviated_ref(const auto&...);

auto lam1 = [](int...) {};
auto lam2 = [](auto...) {};
auto lam3 = [](auto, int...) {};

void h1(void (*)(int...));

void use() {
  c_style(1, 2, 3);
  c_style_named(1, 2);
  tmpl_c_style<int>(1, 2);
  pack(1, 2.0);
  pack_ptr((int*)0, (char*)0);
  pack_ref(1, 2);
  pack_fwd(1, 'c');
  pack_nested(Tup<int>{}, Tup<char>{});
  abbreviated(1, 2);
  abbreviated_ref(1, 2);
  lam1(1, 2);
  lam2(1, 2);
  lam3(1, 2, 3);
}
