// RUN: %cxx -verify %s

template <typename T>
T foo(T x) {
  return bar(x);
}

struct X {};

X bar(X x) { return x; }

int main() {
  X x;
  foo(x);
  return 0;
}
