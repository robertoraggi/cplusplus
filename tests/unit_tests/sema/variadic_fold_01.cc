// RUN: %cxx -verify -fcheck %s

template <int... Ns>
constexpr int sum_left() {
  return (... + Ns);
}

static_assert(sum_left<1, 2, 3>() == 6, "left fold 1+2+3");

template <int... Ns>
constexpr int sum_right() {
  return (Ns + ...);
}

static_assert(sum_right<10, 20, 30>() == 60, "right fold 10+20+30");

template <int... Ns>
constexpr int sum_binary() {
  return (0 + ... + Ns);
}

static_assert(sum_binary<1, 2, 3, 4>() == 10, "binary fold 0+1+2+3+4");

template <int... Ns>
constexpr int sum_empty() {
  return (0 + ... + Ns);
}

static_assert(sum_empty<>() == 0, "empty fold");

template <int... Ns>
constexpr int product() {
  return (1 * ... * Ns);
}

static_assert(product<2, 3, 4>() == 24, "product fold 2*3*4");

template <int... Ns>
constexpr int sub_left() {
  return (... - Ns);
}

static_assert(sub_left<10, 3, 2>() == 5, "left fold 10-3-2");
