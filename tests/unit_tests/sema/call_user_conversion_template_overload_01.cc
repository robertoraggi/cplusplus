// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Source {
  constexpr operator int() const { return 7; }
};

constexpr int choose(int) { return 1; }

template <typename T>
constexpr int choose(T) {
  return 2;
}

static_assert(choose(Source{}) == 2,
              "template exact match should beat non-template requiring user conversion");
