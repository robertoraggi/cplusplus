// RUN: %cxx -verify -fcheck -toolchain wasm32 %s
// expected-no-diagnostics

#include <cstddef>

std::byte f(std::byte a, std::byte b) {
  auto c = a | b;
  c &= a;
  c ^= b;
  return c;
}
