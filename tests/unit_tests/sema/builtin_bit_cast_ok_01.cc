// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

auto ok = __builtin_bit_cast(float, 1);
