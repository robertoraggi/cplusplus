// RUN: %cxx -verify -fcheck %s

struct P {};

int operator==(P, int*);
int operator==(P, void*);

P p;

// expected-error@+1 {{call to overloaded operator '==' is ambiguous}}
int value = (p == nullptr);
