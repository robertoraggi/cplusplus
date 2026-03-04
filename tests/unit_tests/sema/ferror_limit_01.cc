// RUN: %cxx -verify -fcheck -ferror-limit 2 %s

template <int>
struct S;

// expected-error@+1 {{expected a declaration}}
S<"bad">;
// expected-error@+1 {{expected a declaration}}
S<"bad">;
// No more errors expected — limit reached
S<"bad">;
S<"bad">;
