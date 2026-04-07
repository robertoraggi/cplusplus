// RUN: %cxx -verify %s

typedef int Value;
typedef float Value;  // expected-error {{conflicting declaration of 'Value'}}
