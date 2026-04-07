// RUN: %cxx -verify %s

struct S {};

void f() {
  if (S s = S{}) {  // expected-error {{invalid condition expression of type '::S'}}
  }
}
