// RUN: %cxx -toolchain macos -verify -fcheck %s
// expected-no-diagnostics

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
