// RUN: %cxx -verify -fsyntax-only %s

// expected-error@1 {{use of undeclared identifier 'missing'}}
int x = missing(42);
