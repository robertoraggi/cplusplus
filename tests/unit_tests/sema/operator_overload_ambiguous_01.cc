// RUN: %cxx -verify -fcheck %s

struct A {};

int operator+(A, short);
int operator+(A, unsigned short);
A a;

// expected-error@+1 {{call to overloaded operator '+' is ambiguous}}
int value = a + 1;
