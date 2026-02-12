// RUN: %cxx -verify -fcheck %s

template <typename... Ts>
constexpr int count(Ts... args) {
  return sizeof...(args);
}

static_assert(count(1, 2, 3) == 3, "three args");
static_assert(count(1, 2) == 2, "two args");
static_assert(count(1) == 1, "one arg");
static_assert(count() == 0, "zero args");
static_assert(count(1, 2, 3, 4, 5) == 5, "five args");

template <typename T, typename... Ts>
constexpr int count_rest(T first, Ts... rest) {
  return sizeof...(rest);
}

static_assert(count_rest(1, 2, 3, 4) == 3, "first + three");
static_assert(count_rest(1, 2) == 1, "first + one");
static_assert(count_rest(1) == 0, "first + zero");

template <typename... Ts>
constexpr int sum_args(Ts... args) {
  return (0 + ... + args);
}

static_assert(sum_args(1, 2, 3) == 6, "sum 1+2+3");
static_assert(sum_args(10, 20) == 30, "sum 10+20");
static_assert(sum_args(100) == 100, "sum single");
static_assert(sum_args() == 0, "sum empty");

template <typename... Ts>
void nop(Ts... args) {
  ((void)args, ...);
}

void test_nop() {
  nop(1, 2, 3);
  nop(1);
  nop();
}
