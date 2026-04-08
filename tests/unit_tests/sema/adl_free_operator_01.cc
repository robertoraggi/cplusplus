// RUN: %cxx -verify %s

template <typename T>
T foo(T x) {
  return x + x;
}

struct X {};

auto operator+(X, X) -> X { return X{}; }

int main() {
  X x;
  foo(x);
  return 0;
}
