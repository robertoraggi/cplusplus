// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <bool B>
struct W {
  explicit(B) W() {}
};

W<true> a;
W<false> b;

int main() { return 0; }
