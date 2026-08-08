// RUN: %cxx -verify -fsyntax-only %s

int Value;
typedef int Value;  // expected-error {{conflicting declaration of 'Value'}}
