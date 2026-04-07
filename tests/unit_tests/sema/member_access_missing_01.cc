// RUN: %cxx -verify %s

struct S {};

int x = S{}.missing; // expected-error {{no member named 'missing' in type 'S'}}
