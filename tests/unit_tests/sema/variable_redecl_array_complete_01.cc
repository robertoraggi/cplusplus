// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

extern int data[];
int data[4];

int read0() { return data[0]; }

static_assert(sizeof(data) == 4 * sizeof(int));
