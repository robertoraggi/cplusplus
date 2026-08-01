// RUN: %cxx -verify -fcheck %s

template <typename T>
// expected-note@+1 {{candidate function not viable: template argument deduction failed}}
int fn();

int x = fn<int, int>();  // expected-error {{no matching function for call to 'fn'}}
