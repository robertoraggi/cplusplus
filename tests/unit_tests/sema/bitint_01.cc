// RUN: %cxx -toolchain macos -verify -fcheck %s
// expected-no-diagnostics

constexpr unsigned N = 128;
static_assert(N <= 128, "BitInt of bit sizes greater than 128 not supported");

int main() {
  unsigned _BitInt(128) a, b, c;

  c = -a;
  c = +a;
  c = ~a;
  c = !a;
  c = a++;
  c = a--;
  c = ++a;
  c = --a;

  (void)sizeof(a);
  (void)alignof(a);

  c = a + b;
  c = a - b;
  c = a * b;
  c = a / b;
  c = a % b;
  c = a << b;
  c = a >> b;
  c = a & b;
  c = a | b;
  c = a ^ b;

  return 0;
}
