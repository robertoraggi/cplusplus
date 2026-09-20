// RUN: %cxx -toolchain macos -verify -fsyntax-only %s
// expected-no-diagnostics

#ifdef CXX_HAS_INT128

int main() {
  __int128 a, b, c;

  c = -a;
  c = +a;
  c = ~a;
  c = a++;
  c = a--;
  c = ++a;
  c = --a;

  c = a + b;
  c = a - b;
  c = a * b;
  c = a / b;
  c = a % b;
  c = a & b;
  c = a | b;
  c = a ^ b;

  return 0;
}

#endif