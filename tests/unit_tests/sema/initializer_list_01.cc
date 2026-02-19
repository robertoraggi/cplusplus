// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

#include <initializer_list>

struct V {
  V(int, int) = delete;
  constexpr V(std::initializer_list<int>) {}
};

constexpr int count(std::initializer_list<int> il) {
  return (int)il.size();
}

V v1{1, 2};
static_assert(count({1, 2, 3}) == 3);

int initializer_list_call_ok[count({4, 5}) == 2 ? 1 : -1];
