// RUN: %cxx -verify -fsyntax-only %s

typedef int Value;
typedef float Value;  // expected-error {{conflicting declaration of 'Value'}}
