// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

#include <type_traits>
#include <utility>
template <class T> struct A : std::true_type {};
template <class T> struct B : std::true_type {};

template <class C>
struct S {
  struct Inner { int v; };
  using iterator = C*;
  using const_iterator = const C*;
  typedef Inner inner_type;

  S& self();
  const S& cself() const;
  S* ptr();
  S<C>& self2();
  Inner make();
  inner_type make2();
  iterator begin();
  const_iterator cbegin() const;
  std::pair<S, int> pair();
  std::pair<const S&, Inner> pair2() const;
  S (*factory())(S);
  auto trailing() -> S&;
  decltype(sizeof(C)) size() const;
  template <class It> auto ins2(It) -> typename std::enable_if<A<It>::value, S&>::type;
  template <class It> auto ins2(It) -> typename std::enable_if<!A<It>::value, S*>::type;
  template <class U> S<U> convert(U) const;
  template <class U> std::pair<S, U> with(U) const;
  int value() const &;
  int value() const &&;
};

template <class C> S<C>& S<C>::self() { return *this; }
template <class C> const S<C>& S<C>::cself() const { return *this; }
template <class C> S<C>* S<C>::ptr() { return this; }
template <class C> S<C>& S<C>::self2() { return *this; }
template <class C> typename S<C>::Inner S<C>::make() { return {}; }
template <class C> typename S<C>::inner_type S<C>::make2() { return {}; }
template <class C> typename S<C>::iterator S<C>::begin() { return nullptr; }
template <class C> typename S<C>::const_iterator S<C>::cbegin() const { return nullptr; }
template <class C> std::pair<S<C>, int> S<C>::pair() { return {}; }
template <class C> std::pair<const S<C>&, typename S<C>::Inner> S<C>::pair2() const { return {*this, {}}; }
template <class C> S<C> (*S<C>::factory())(S<C>) { return nullptr; }
template <class C> auto S<C>::trailing() -> S<C>& { return *this; }
template <class C> decltype(sizeof(C)) S<C>::size() const { return sizeof(C); }
template <class C> template <class It> auto S<C>::ins2(It) -> typename std::enable_if<A<It>::value, S<C>&>::type { return *this; }
template <class C> template <class It> auto S<C>::ins2(It) -> typename std::enable_if<!A<It>::value, S<C>*>::type { return this; }
template <class C> template <class U> S<U> S<C>::convert(U) const { return {}; }
template <class C> template <class U> std::pair<S<C>, U> S<C>::with(U) const { return {}; }
template <class C> int S<C>::value() const & { return 1; }
template <class C> int S<C>::value() const && { return 2; }

int main() {
  S<int> s;
  s.self(); s.cself(); s.ptr(); s.self2(); s.make(); s.make2(); s.begin(); s.cbegin();
  s.pair(); s.pair2(); s.factory(); s.trailing(); (void)s.size();
  s.ins2(1);
  s.convert(1.0); s.with(1);
  if (s.value() != 1) return 1;
  if (S<int>{}.value() != 2) return 2;
  return 0;
}
