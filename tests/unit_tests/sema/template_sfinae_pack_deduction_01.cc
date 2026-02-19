// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class... Ts, class U>
auto pick(U) -> char;

template <class...>
auto pick(...) -> long;

int sfinae_pack_deduction_1[sizeof(pick<int>(0)) == sizeof(char) ? 1 : -1];
int sfinae_pack_deduction_2[sizeof(pick<>(0)) == sizeof(char) ? 1 : -1];
