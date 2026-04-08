// RUN: %cxx -verify %s

struct D : MissingBase {};  // expected-error {{unknown base class 'MissingBase'}}
