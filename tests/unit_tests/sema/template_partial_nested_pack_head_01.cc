// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
struct wrap {};

template <class T>
struct holder {};

template <class T>
struct select {
  enum { value = 0 };
};

template <class T>
struct select<holder<T>> {
  enum { value = 2 };
};

template <class T>
struct select<holder<wrap<T>>> {
  enum { value = 1 };
};

static_assert(select<holder<wrap<long>>>::value == 1,
              "nested wrap specialization");
static_assert(select<holder<long>>::value == 2,
              "single-level holder specialization");
