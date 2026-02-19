// RUN: %cxx -verify -fcheck %s

// expected-error@1 {{unknown builtin function '__builtin_shufflevector'}}
int x = __builtin_shufflevector();
