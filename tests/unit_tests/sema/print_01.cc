// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

#include <print>

auto main() -> int {
    std::println("Hello, World!");
    return 0;
}