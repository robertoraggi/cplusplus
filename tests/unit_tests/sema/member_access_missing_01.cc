// RUN: %cxx -verify -fsyntax-only %s

struct S {};

int x = S{}.missing; // expected-error {{no member named 'missing' in type 'S'}}
