// RUN: %cxx -verify -fcheck %s

constexpr int count_abbrev(auto... args) { return sizeof...(args); }

static_assert(count_abbrev(1, 2, 3) == 3, "abbrev three args");
static_assert(count_abbrev(1, 2) == 2, "abbrev two args");
static_assert(count_abbrev(1) == 1, "abbrev one arg");

constexpr int sum_abbrev(auto... args) { return (0 + ... + args); }

static_assert(sum_abbrev(1, 2, 3) == 6, "abbrev sum 1+2+3");
static_assert(sum_abbrev(10, 20) == 30, "abbrev sum 10+20");
static_assert(sum_abbrev(100) == 100, "abbrev sum single");

constexpr int sum_explicit(auto... args) { return (0 + ... + args); }

static_assert(sum_explicit<int, int, int>(1, 2, 3) == 6, "explicit targs");

void nop_abbrev(auto... args) { ((void)args, ...); }

void test_nop_abbrev() {
  nop_abbrev(1, 2, 3);
  nop_abbrev(1);
}

constexpr int first_abbrev(auto first, auto... rest) {
  return first + (0 + ... + rest);
}

static_assert(first_abbrev(10, 1, 2, 3) == 16, "mixed abbrev");
static_assert(first_abbrev(10, 1) == 11, "mixed abbrev two");
