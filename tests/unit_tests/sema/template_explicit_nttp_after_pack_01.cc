// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <int N, class... Ts>
auto pick_nttp_after_pack(int (&)[N], Ts...) -> char;

int values[3];

int explicit_nttp_after_pack_1[
    sizeof(pick_nttp_after_pack<3>(values)) == sizeof(char) ? 1 : -1];
int explicit_nttp_after_pack_2[
    sizeof(pick_nttp_after_pack<3, int>(values, 42)) == sizeof(char) ? 1 : -1];
