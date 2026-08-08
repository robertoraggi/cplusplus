// RUN: %cxx -verify -fsyntax-only %s

struct D : MissingBase {};  // expected-error {{unknown base class 'MissingBase'}}
