// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct Value {
  char value = 7;
  constexpr Value() = default;
  constexpr Value(char initial) : value(initial) {}
  constexpr virtual int get() const { return value; }
};

struct Holder : virtual Value {
  int held = 5;
  constexpr int get() const override { return held; }
};

struct Derived : Holder {
  int extra = 9;
};

struct Initializes : virtual Value {
  constexpr Initializes() : Value{3} {}
};

struct MostDerived : Initializes {
  constexpr MostDerived() : Value{4}, Initializes() {}
};

constexpr Holder holder{};
static_assert(holder.value == 7);
static_assert(holder.held == 5);
static_assert(static_cast<const Value&>(holder).get() == 5);

constexpr Derived derived{};
static_assert(derived.value == 7);
static_assert(derived.extra == 9);
static_assert(static_cast<const Value&>(derived).get() == 5);

constexpr Initializes initializes{};
static_assert(initializes.value == 3);

constexpr MostDerived mostDerived{};
static_assert(mostDerived.value == 4);

constinit Holder braced{};
constinit Holder plain;
