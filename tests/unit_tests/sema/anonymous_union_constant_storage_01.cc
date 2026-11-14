// RUN: %cxx -std=c++20 -verify -fsyntax-only %s

struct Value {
  union {
    int integer;
    float real;
  };
  constexpr Value(int n) : integer(n) {}
  constexpr void activate() { real = 1.0f; }
};

constexpr Value values[] = {21, 42};
static_assert(values[0].integer == 21);
static_assert(values[1].integer == 42);

constexpr bool aliases() {
  Value value(1);
  int& reference = value.integer;
  value.integer = 2;
  Value copy = value;
  copy.integer = 3;
  return reference == 2 && value.integer == 2 && copy.integer == 3;
}
static_assert(aliases());

constexpr int inactive() {
  Value value(1);
  value.activate();
  return value.integer;
}
static_assert(inactive() == 1); // expected-error {{static assertion expression is not an integral constant expression}}
