// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class A, class B>
struct tuple {
  A first;
  B second;
};

template <int N, class A, class B>
A get(tuple<A, B> const&);

template <class U1, class U2>
struct pair {
  U1 first;
  U2 second;

  pair(tuple<U1, U2> const& p) : first(get<0>(p)), second(get<1>(p)) {}
};
