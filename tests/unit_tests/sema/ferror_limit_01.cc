// RUN: %cxx -verify -fsyntax-only -ferror-limit 2 %s

template <int>
struct S;

// expected-error@+1 {{expected a declarator}}
S<1>;
// expected-error@+1 {{expected a declarator}}
S<1>;

S<1>;
S<1>;
