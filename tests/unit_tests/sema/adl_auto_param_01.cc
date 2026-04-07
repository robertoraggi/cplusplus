// RUN: %cxx -verify %s

auto add(auto x, auto y) -> decltype(x + y) { return x + y; }

struct X {};

auto operator+(X, X) -> X { return X{}; }

int main() {
  X x;
  add(x, x);
  return 0;
}
