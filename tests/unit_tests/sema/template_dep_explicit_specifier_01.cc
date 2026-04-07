// RUN: %cxx -verify %s
// expected-no-diagnostics

struct check {
  template <int&...>
  static constexpr bool enable() {
    return true;
  }
};

template <class _T1, class _T2>
struct pair {
  template <class _D = check, int = (_D::template enable<>() ? 0 : 0)>
  explicit(!_D::template enable<>()) pair() {}
};

int main() {
  pair<int, int> p;
  (void)p;
  return 0;
}
