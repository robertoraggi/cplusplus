// RUN: %cxx -verify -fsyntax-only %s

struct S {};

auto x = S{} & 1; // expected-error {{invalid operands to binary expression ('::S' and 'int')}}
