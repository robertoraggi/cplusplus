// RUN: %cxx -verify %s

struct Base {};

struct Derived : Base {
  using Base::missing;  // expected-error {{using declaration refers to unresolved name 'missing'}}
};
