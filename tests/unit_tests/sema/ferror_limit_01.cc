// RUN: %cxx -verify -fsyntax-only -ferror-limit 2 %s

template <int>
struct S;

// expected-error@+1 {{expected a declarator}}
S<"bad">;
// expected-error@+1 {{expected a declarator}}
S<"bad">;
// No more errors expected - limit reached
S<"bad">;
S<"bad">;
