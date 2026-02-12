// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
struct Wrap {};

template <class T>
struct Holder {};

template <class T>
struct S {
  enum { value = 0 };
};

template <class T>
struct S<Holder<T>> {
  enum { value = 2 };
};

template <class T>
struct S<Holder<Wrap<T>>> {
  enum { value = 1 };
};

static_assert(S<Holder<Wrap<int>>>::value == 1);
static_assert(S<Holder<int>>::value == 2);
