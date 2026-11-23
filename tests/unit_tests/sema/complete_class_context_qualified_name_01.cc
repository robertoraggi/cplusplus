// RUN: %cxx -std=c++26 -verify -fsyntax-only %s

struct S {
  int a = later() + Later::v;
  constexpr int f(int x = later() + Later::v) { return x; }
  static constexpr int later() { return 2; }
  struct Later {
    static constexpr int v = 1;
  };
};

static_assert(S{}.a == 3);
static_assert(S{}.f() == 3);

struct T {
  void g() noexcept(Later::v == 1);
  void h() noexcept(sizeof(Later::Type) == 2);
  struct Later {
    static constexpr int v = 1;
    using Type = char;
  };
};

static_assert(noexcept(T{}.g()));
static_assert(!noexcept(T{}.h()));

class P {
  struct In {
    static constexpr int v = 1;
  };
  static constexpr int w = 2;
};

struct Q {
  int a = P::In::v;  // expected-error {{'In' is a private member of 'P'}}
  int b = P::w;  // expected-error {{'w' is a private member of 'P'}}
  void f(int x = P::In::v)  // expected-error {{'In' is a private member of 'P'}}
      noexcept(P::In::v == 1);  // expected-error {{'In' is a private member of 'P'}}
};
