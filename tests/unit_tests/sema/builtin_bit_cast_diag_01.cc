// RUN: %cxx -verify -fcheck %s

// expected-error@1 {{__builtin_bit_cast requires source and destination to have the same size}}
auto bad = __builtin_bit_cast(double, 1);
