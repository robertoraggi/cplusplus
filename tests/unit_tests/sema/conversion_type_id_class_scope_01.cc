// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T>
struct valarray {};

struct Other {
  typedef long result_type;
};

struct B {
  using T = int;
  struct N {
    using U = long;
  };
  struct Other {
    typedef char result_type;
  };
  operator T();
  operator N::U();
  operator B*();
  operator valarray<Other::result_type>();
};

B::operator T() { return 0; }
B::operator N::U() { return 0; }
B::operator B*() { return this; }
B::operator valarray<Other::result_type>() { return {}; }

static_assert(__is_same(decltype(B{}.operator valarray<char>()),
                        valarray<char>));

template <class E>
struct V {
  using R = E;
  operator R() const;
  operator V*();
};

template <class E>
V<E>::operator R() const {
  return {};
}

template <class E>
V<E>::operator V*() {
  return this;
}

template <class E>
struct val_expr {
  typedef int result_type;
  operator valarray<result_type>() const;
  operator valarray<val_expr*>() const;
};

template <class E>
val_expr<E>::operator valarray<val_expr::result_type>() const {
  return {};
}

template <class E>
val_expr<E>::operator valarray<val_expr*>() const {
  return {};
}

int fromV = int(V<int>{});
valarray<int> fromExpression = val_expr<char>();
