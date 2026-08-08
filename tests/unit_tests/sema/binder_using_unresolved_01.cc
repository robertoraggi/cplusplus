// RUN: %cxx -verify -fsyntax-only %s

struct Base {};

struct Derived : Base {
  using Base::missing;  // expected-error {{using declaration refers to unresolved name 'missing'}}
};
