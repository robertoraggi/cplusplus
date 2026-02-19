// RUN: %cxx -verify -fcheck %s

template <typename T>
int fn();

int x = fn<int, int>();  // expected-error {{invalid template-id 'fn <>'}}
