// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
struct Wrap {};

template <class T>
struct Holder {};

template <class T>
struct choose {
  enum { value = 0 };
};

template <class T>
struct choose<Holder<T>> {
  enum { value = 2 };
};

template <class T>
struct choose<Holder<Wrap<T>>> {
  enum { value = 1 };
};

static_assert(choose<Holder<int>>::value == 2);
static_assert(choose<Holder<Wrap<int>>>::value == 1,
              "nested specialization should be preferred");
