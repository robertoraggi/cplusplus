// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct A {};

int operator+(A, int);
int operator+(A, long);

A a;
int value = a + (short)1;
