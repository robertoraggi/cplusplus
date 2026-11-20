// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct Polymorphic {
  virtual void f();
};

struct InheritsVirtualFunction : Polymorphic {
  int x;
};

struct VirtualBase {
  int v;
};

struct DirectVirtualBase : virtual VirtualBase {};

struct IndirectVirtualBase : DirectVirtualBase {
  int y;
};

struct Plain {
  int a;
};

struct DerivedPlain : Plain {
  int b;
};

static_assert(!__is_aggregate(Polymorphic));
static_assert(!__is_aggregate(InheritsVirtualFunction));
static_assert(!__is_aggregate(DirectVirtualBase));
static_assert(!__is_aggregate(IndirectVirtualBase));
static_assert(__is_aggregate(Plain));
static_assert(__is_aggregate(DerivedPlain));
