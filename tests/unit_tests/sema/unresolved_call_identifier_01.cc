// RUN: %cxx -verify -fcheck %s

// expected-error@1 {{use of undeclared identifier 'missing'}}
int x = missing(42);
