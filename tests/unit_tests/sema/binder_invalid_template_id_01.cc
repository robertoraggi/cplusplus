// RUN: %cxx -verify %s

template <typename T>
int fn();

int x = fn<int, int>();  // expected-error {{invalid template-id 'fn <>'}}
