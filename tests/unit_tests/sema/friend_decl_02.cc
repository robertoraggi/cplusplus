// clang-format off
// RUN: %cxx -verify -fcheck %s

struct X {
  friend void foo(X*);
  int priv;
};

void foo(X* x) { x->priv = 42; }  // redeclares hidden friend, now visible

void test() {
  X x;
  foo(&x);  // OK: foo is now visible via namespace-scope declaration
}
