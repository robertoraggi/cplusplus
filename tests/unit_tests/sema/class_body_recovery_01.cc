// RUN: %cxx -verify -fcheck %s
// Tests that class body recovery skips to the next semicolon
// and doesn't cascade hundreds of "expected a declaration" errors.

struct S {
  int x;
  // expected-error@+1 {{expected a declaration}}
  @ @ @ int y;
  int z;
  // expected-error@+1 {{expected a declaration}}
  @ @ @ int w;
  int valid_member;
};
