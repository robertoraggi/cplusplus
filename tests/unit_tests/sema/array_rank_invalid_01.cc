// RUN: %cxx -std=c++20 -verify -fsyntax-only %s

auto extra = __array_rank(int, int);  // expected-error {{expected one type argument to '__array_rank'}}
