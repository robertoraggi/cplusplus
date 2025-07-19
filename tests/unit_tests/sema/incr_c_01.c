// RUN: %cxx -verify -fcheck -xc %s

enum E { a, b, c };

int main() {
  enum E e = a;
  ++e;
  --e;
  e++;
  e--;

  const E ce = a;
  // expected-error@1 {{cannot increment a value of type 'const E'}}
  ++ce;
  // expected-error@1 {{cannot decrement a value of type 'const E'}}
  --ce;
  // expected-error@1 {{cannot increment a value of type 'const E'}}
  ce++;
  // expected-error@1 {{cannot decrement a value of type 'const E'}}
  ce--;
  return 0;
}
