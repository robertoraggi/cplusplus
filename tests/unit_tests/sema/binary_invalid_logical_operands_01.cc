// RUN: %cxx -verify %s

struct S {};

bool b = S{} && true; // expected-error {{invalid operands to binary expression ('::S' and 'bool')}}
