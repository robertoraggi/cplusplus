// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T>
struct X {
  struct R;
};

template <>
struct X<int>::R {
  using T = int;
  static constexpr int v = 1;
  struct Deep {
    static constexpr int w = 2;
  };
};

X<int>::R::T t = X<int>::R::v;
static_assert(X<int>::R::v == 1);
static_assert(X<int>::R::Deep::w == 2);

template <class D>
concept HasDeep = requires { D::R::Deep::w; };
static_assert(HasDeep<X<int>>);
