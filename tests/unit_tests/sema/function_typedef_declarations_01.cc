// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

typedef int fn_t(int);
fn_t global;
int global(int x) { return x + 1; }

namespace N {
fn_t inner;
}
int N::inner(int x) { return x + 2; }

struct S {
  friend fn_t friended;
  static fn_t stat;
};
int friended(int x) { return x + 3; }
int S::stat(int x) { return x + 4; }

template <class T>
struct Holder {
  typedef T fn(T);
  friend fn hfriend;
  fn member;
};
template <class T>
T Holder<T>::member(T t) { return t; }

int local() {
  fn_t block;
  extern fn_t other;
  return block(1) + other(1);
}
int block(int x) { return x + 5; }
int other(int x) { return x + 6; }

union U {
  fn_t um;
  int i;
};
int U::um(int x) { return x + 7; }

int main() {
  U u;
  Holder<int> h;
  if (global(1) != 2 || N::inner(1) != 3 || friended(1) != 4 || S::stat(1) != 5) return 1;
  if (local() != 13) return 2;
  if (u.um(1) != 8) return 3;
  if (h.member(9) != 9) return 4;
  return 0;
}
