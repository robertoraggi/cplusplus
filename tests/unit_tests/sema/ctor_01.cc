// RUN: %cxx -verify -fsyntax-only %s

struct C {
  struct bar {};

  C() {}
  C(int);
  C(bar);

  C* next;
};

C::C(bar b) {}
