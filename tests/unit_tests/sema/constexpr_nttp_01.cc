// RUN: %cxx -verify -fcheck %s

template <int N>
constexpr int identity() {
  return N;
}

constexpr int three = identity<3>();
static_assert(three == 3, "Expected identity<3>() to be 3");

template <int N>
constexpr int add_one() {
  return N + 1;
}

constexpr int four = add_one<3>();
static_assert(four == 4, "Expected add_one<3>() to be 4");

template <int N>
constexpr int times_two() {
  return N * 2;
}

constexpr int six = times_two<3>();
static_assert(six == 6, "Expected times_two<3>() to be 6");

template <int A, int B>
constexpr int add_nttp() {
  return A + B;
}

constexpr int sum = add_nttp<10, 20>();
static_assert(sum == 30, "Expected add_nttp<10, 20>() to be 30");

template <bool B>
constexpr int bool_to_int() {
  if (B) return 1;
  return 0;
}

constexpr int one = bool_to_int<true>();
static_assert(one == 1, "Expected bool_to_int<true>() to be 1");
