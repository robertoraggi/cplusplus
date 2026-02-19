// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct has_type_member {
  using type = int;
};

template <class T>
auto probe(int) -> decltype(sizeof(typename T::type), char{});

template <class>
auto probe(...) -> long;

int sfinae_ok_1[sizeof(probe<has_type_member>(0)) == sizeof(char) ? 1 : -1];
int sfinae_ok_2[sizeof(probe<int>(0)) == sizeof(long) ? 1 : -1];
