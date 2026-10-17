// RUN: %cxx -verify -fsyntax-only %s

template <class C>
struct S {
  int f();
  S& g();
  typename C::type h();
  template <class T>
  int tf(T);
  const int& cf() const;
};

// expected-error@+1 {{class or namespace has no member named 'f'}}
template <class C> long S<C>::f() { return 0; }
// expected-error@+1 {{class or namespace has no member named 'g'}}
template <class C> S<C>* S<C>::g() { return this; }
// expected-error@+1 {{class or namespace has no member named 'h'}}
template <class C> typename C::other S<C>::h() { return {}; }
// expected-error@+2 {{class or namespace has no member named 'tf'}}
template <class C> template <class T>
long S<C>::tf(T) { return 0; }
// expected-error@+1 {{class or namespace has no member named 'cf'}}
template <class C> int& S<C>::cf() const { static int i; return i; }
