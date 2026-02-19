// clang-format off
// RUN: %cxx -verify -fcheck %s

struct S {
  int a : 3;
  int b : 0; // expected-error {{zero-width bit-field must be unnamed}}
  int : 0; // OK
  int c : -1; // expected-error {{bit-field width is negative}}
  float d : 3; // expected-error {{bit-field has non-integral type}}
  int e : 33; // expected-error {{width of bit-field exceeds width of its type}}
  long long f : 64; // OK
  long long g : 65; // expected-error {{width of bit-field exceeds width of its type}}
  int h : 1 + 2; // OK
};

void f(int n) {
  struct L {
    int x : n; // expected-error {{bit-field width is not a constant expression}}
  };
}
