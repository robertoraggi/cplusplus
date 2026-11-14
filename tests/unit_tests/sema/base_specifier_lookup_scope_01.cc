// RUN: %cxx -verify -fsyntax-only %s

struct A {
  struct X {};
};

struct B : A, X {};  // expected-error {{unknown base class 'X'}}

struct C {
  struct Y {};
  struct Z;
};

struct C::Z : Y {};

class P {
 protected:
  struct Q {};
};

struct R : P::Q, P {};

template <class T>
struct W : T {};

template <class T>
struct V {
  struct In {};
  struct Out;
};

template <class T>
struct V<T>::Out : In {};

W<A> w;
V<int>::Out out;
