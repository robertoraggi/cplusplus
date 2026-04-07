// RUN: %cxx -verify %s

struct C {
  struct bar {};

  C() {}
  C(int);
  C(bar);

  C* next;
};

C::C(bar b) {}
