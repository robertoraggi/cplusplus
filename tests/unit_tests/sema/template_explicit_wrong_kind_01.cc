// RUN: not %cxx -fcheck %s

template <int... Ns, class U>
auto wrong_kind(U) -> char;

int wrong_kind_explicit_1[sizeof(wrong_kind<int>(0)) == sizeof(char) ? 1 : -1];
