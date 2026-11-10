// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

namespace std {
template <class T>
class initializer_list {
  const T* b_;
  unsigned long n_;

 public:
  constexpr const T* begin() const { return b_; }
  constexpr const T* end() const { return b_ + n_; }
};
}  // namespace std

template <class N>
struct lim {
  static constexpr N max() { return N(7); }
};

template <class N>
constexpr int mixed() {
  int s = 0;
  for (N v : {N(0), lim<N>::max()}) s += static_cast<int>(v);
  return s;
}

template <class N>
constexpr int uniform() {
  int s = 0;
  for (N v : {N(1), N(2)}) s += static_cast<int>(v);
  return s;
}

static_assert(mixed<int>() == 7);
static_assert(mixed<long>() == 7);
static_assert(uniform<int>() == 3);
static_assert(uniform<long>() == 3);
