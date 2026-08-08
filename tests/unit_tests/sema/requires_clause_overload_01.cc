// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

template <bool x>
  requires(x)
void foo() {}

template <bool x>
  requires(!x)
void foo() {}

int main() {
  foo<true>();
  foo<false>();
  return 0;
}
