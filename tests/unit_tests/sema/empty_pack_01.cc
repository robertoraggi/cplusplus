// RUN: %cxx -verify -fcheck %s

template <typename... Ts>
constexpr int count(Ts... args) {
  return sizeof...(args);
}

static_assert(count() == 0, "zero args");
static_assert(count(1) == 1, "one arg");
static_assert(count(1, 2, 3) == 3, "three args");

template <typename T, typename... Ts>
constexpr int count_rest(T first, Ts... rest) {
  return sizeof...(rest);
}

static_assert(count_rest(1) == 0, "empty rest");
static_assert(count_rest(1, 2) == 1, "one rest");
static_assert(count_rest(1, 2, 3, 4) == 3, "three rest");

template <typename... Ts>
constexpr int sum(Ts... args) {
  return (0 + ... + args);
}

static_assert(sum() == 0, "empty sum");
static_assert(sum(1) == 1, "single sum");
static_assert(sum(1, 2, 3) == 6, "three sum");

template <typename T, typename... Ts>
constexpr int first_plus_rest(T first, Ts... rest) {
  return first + (0 + ... + rest);
}

static_assert(first_plus_rest(10) == 10, "first only");
static_assert(first_plus_rest(10, 1) == 11, "first + one");
static_assert(first_plus_rest(10, 1, 2, 3) == 16, "first + three");

template <typename... Ts>
void nop(Ts... args) {
  ((void)args, ...);
}

void test_nop() {
  nop();
  nop(1);
  nop(1, 2, 3);
}
