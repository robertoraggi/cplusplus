// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

struct two {
  constexpr operator int() const { return 2; }
  constexpr int operator[](int i) const { return i; }
  constexpr two& operator=(int) { return *this; }
  constexpr two& operator+=(int) { return *this; }
};

int by_reference() {
  constexpr int y = 1;
  auto f = [&](auto n) {
    constexpr int j = n - y;
    static_assert(j == 1);
    return j;
  };
  return f(two{});
}

int no_default() {
  constexpr int y = 1;
  auto f = [](auto n) {
    constexpr int j = n - y;
    static_assert(j == 1);
    return j;
  };
  return f(two{});
}

int unary_operand() {
  constexpr int y = 1;
  auto f = [](auto n) {
    constexpr int j = n + -y;
    static_assert(j == 1);
    return j;
  };
  return f(two{});
}

int parenthesized_operands() {
  constexpr int y = 1;
  auto f = [](auto n) {
    constexpr int j = (n) - (y);
    static_assert(j == 1);
    return j;
  };
  return f(two{});
}

int conditional_operands() {
  constexpr int y = 1;
  auto f = [](auto n) {
    constexpr int a = n ? y : 0;
    constexpr int b = y ? n : 0;
    static_assert(a == 1);
    static_assert(b == 2);
    return a + b;
  };
  return f(two{});
}

int comma_operand() {
  constexpr int y = 1;
  auto f = [](auto n) {
    constexpr int j = n + (two{}, y);
    static_assert(j == 3);
    return j;
  };
  return f(two{});
}

int nested_generic_lambdas() {
  constexpr int y = 1;
  auto f = [](auto n) {
    auto g = [](auto m) {
      constexpr int j = m - y;
      static_assert(j == 1);
      return j;
    };
    return g(n);
  };
  return f(two{});
}

int operator_operands() {
  constexpr int y = 1;
  auto f = [](auto n) {
    n = y;
    n += y;
    return n[y];
  };
  return f(two{});
}

constexpr int add(int a, int b) {
  return a + b;
}

template <class A, class B>
constexpr int template_add(A a, B b) {
  return a + b;
}

template <class T>
constexpr int read(const T& value) {
  return value;
}

int call_arguments() {
  constexpr int y = 1;
  auto f = [](auto n) {
    constexpr int a = add(n, y);
    constexpr int b = template_add(n, y);
    constexpr int c = static_cast<int>(n) + int(y) + (int)y;
    static_assert(a == 3 && b == 3 && c == 4);
    return a + b + c;
  };
  return f(two{});
}

int reference_parameter_default_capture() {
  constexpr int y = 1;
  auto f = [&](auto n) { return read(y) + n; };
  return f(two{});
}

int reference_parameter_copy_capture() {
  constexpr int y = 1;
  auto f = [=](auto n) { return read(y) + n; };
  return f(two{});
}

int address_taken_default_capture() {
  constexpr int y = 1;
  auto f = [&](auto n) {
    const int* p = &y;
    return *p + n;
  };
  return f(two{});
}

int main() {
  if (by_reference() != 1) return 1;
  if (no_default() != 1) return 2;
  if (unary_operand() != 1) return 3;
  if (parenthesized_operands() != 1) return 4;
  if (conditional_operands() != 3) return 5;
  if (comma_operand() != 3) return 6;
  if (nested_generic_lambdas() != 1) return 7;
  if (operator_operands() != 1) return 8;
  if (call_arguments() != 10) return 9;
  if (reference_parameter_default_capture() != 3) return 10;
  if (reference_parameter_copy_capture() != 3) return 11;
  if (address_taken_default_capture() != 3) return 12;
  return 0;
}
