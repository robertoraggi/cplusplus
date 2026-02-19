// RUN: %cxx -verify -fcheck %s

int Value;
typedef int Value;  // expected-error {{conflicting declaration of 'Value'}}
