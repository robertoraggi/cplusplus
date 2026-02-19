// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct with_size_type {
  using type = int;
};

template <class T, int = sizeof(typename T::type)>
auto probe(int) -> char;

template <class>
auto probe(...) -> long;

int sfinae_probe_1[sizeof(probe<with_size_type>(0)) == sizeof(char) ? 1 : -1];
int sfinae_probe_2[sizeof(probe<int>(0)) == sizeof(long) ? 1 : -1];
