// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class... Ts, int N = sizeof...(Ts)>
auto probe(int) -> decltype(sizeof(int[N]), char{});

template <class...>
auto probe(...) -> long;

int default_pack_nttp_1[sizeof(probe<int>(0)) == sizeof(char) ? 1 : -1];
int default_pack_nttp_2[sizeof(probe<>(0)) == sizeof(long) ? 1 : -1];
