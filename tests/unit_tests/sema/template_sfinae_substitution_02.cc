// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct with_type {
  using type = int;
};

template <class T, class = typename T::type>
auto select(int) -> char;

template <class>
auto select(...) -> long;

int sfinae_select_1[sizeof(select<with_type>(0)) == sizeof(char) ? 1 : -1];
int sfinae_select_2[sizeof(select<int>(0)) == sizeof(long) ? 1 : -1];
