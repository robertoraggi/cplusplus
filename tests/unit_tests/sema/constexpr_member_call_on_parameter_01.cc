// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

struct two {
  constexpr operator int() const { return 2; }
  constexpr int operator[](int i) const { return i; }
  constexpr int operator()(int i) const { return i; }
  constexpr int operator+(int i) const { return i + 2; }
  constexpr int m() const { return 2; }
};

constexpr int subscript(two n) {
  return n[1];
}

int by_value(two n) {
  constexpr int a = n[1];
  constexpr int b = n + 1;
  constexpr int c = n(1);
  constexpr int d = n.m();
  static_assert(a == 1 && b == 3 && c == 1 && d == 2);
  return a + b + c + d;
}

int by_reference(const two& n) {
  constexpr int a = n[1];
  constexpr int b = n + 1;
  constexpr int c = [&] { return n.m(); }();
  static_assert(a == 1 && b == 3 && c == 2);
  return a + b + c;
}

int by_reference_capture(two& n) {
  constexpr int c = [&] { return n.m(); }();
  static_assert(c == 2);
  return c;
}

int by_value_capture(two n) {
  constexpr int a = [&] { return n - 1; }();
  constexpr int b = [n] { return n.m(); }();
  static_assert(a == 1 && b == 2);
  return a + b;
}

template <class T>
int in_template(T n) {
  constexpr int a = n[1];
  constexpr int b = [&] { return n.m(); }();
  static_assert(a == 1 && b == 2);
  return a + b;
}

int generic_lambda() {
  auto f = [](auto n) {
    constexpr int a = n[1];
    constexpr int b = [&] { return n - 1; }();
    static_assert(a == 1 && b == 1);
    return a + b;
  };
  return f(two{});
}

int main() {
  two t;
  if (by_value(t) != 7) return 1;
  if (by_reference(t) != 6) return 2;
  if (by_reference_capture(t) != 2) return 3;
  if (by_value_capture(t) != 3) return 4;
  if (in_template(t) != 3) return 5;
  if (generic_lambda() != 2) return 6;
  return 0;
}
