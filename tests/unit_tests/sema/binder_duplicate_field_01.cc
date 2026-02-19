// RUN: %cxx -verify -fcheck %s

struct S {
  int value;
  int value;  // expected-error {{duplicate member 'value'}}
};
