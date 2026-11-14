// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

struct Derived;

struct Base {
  Base() = default;
  Base(const Base&) = default;
  Base(const Base&, int) {}
  Base(const Derived&, long = 0) {}
  explicit Base(Base&, ...) {}
};

struct Middle : Base {
  using Base::Base;
};

struct Derived : Middle {
  using Middle::Middle;
  Derived() = default;
};

struct Convertible {
  operator Base() const;
};

static_assert(!__is_constructible(Middle, Convertible));
static_assert(!__is_constructible(Derived, Convertible));
static_assert(!__is_constructible(Derived, const Base&));
static_assert(!__is_constructible(Derived, const Middle&));
static_assert(__is_constructible(Middle, const Base&, int));
static_assert(__is_constructible(Derived, const Base&, int));
static_assert(__is_constructible(Derived, const Derived&, long));
static_assert(__is_constructible(Derived, Base&, int, int));
static_assert(__is_constructible(Derived, const Derived&));
static_assert(__is_constructible(Derived, Derived));

void copies(const Derived& source) {
  Derived copy(source);
  Middle middle(static_cast<const Middle&>(source));
}
