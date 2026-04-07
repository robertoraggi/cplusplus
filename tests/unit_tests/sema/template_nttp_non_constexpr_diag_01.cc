// RUN: %cxx -verify -fstrict-templates %s

template <int N>
struct Box {};

int runtime();

// expected-error@+1 {{template argument is not a constant expression}}
template struct Box<runtime()>;
