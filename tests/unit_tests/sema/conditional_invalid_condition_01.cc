// RUN: %cxx -verify -fcheck %s

struct S {};

int x = S{} ? 1 : 2; // expected-error {{invalid condition expression of type '::S'}}
