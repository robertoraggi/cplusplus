// RUN: %cxx -verify -fsyntax-only %s

struct S {
  int value;
  int value;  // expected-error {{duplicate member 'value'}}
};
