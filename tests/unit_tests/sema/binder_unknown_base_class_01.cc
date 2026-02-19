// RUN: %cxx -verify -fcheck %s

struct D : MissingBase {};  // expected-error {{unknown base class 'MissingBase'}}
