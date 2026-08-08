// RUN: %cxx -verify -fsyntax-only %s

struct S {};

void f() {
  if (S s = S{}) {  // expected-error {{invalid condition expression of type '::S'}}
  }
}
