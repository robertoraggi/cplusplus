// RUN: %cxx -std=c++17 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class F>
constexpr unsigned applyOffset(F fn) {
  return fn(3);
}

struct Fragment {
  unsigned size;
  constexpr unsigned getFixedSize() const { return size; }
  constexpr unsigned adjust() const {
    return applyOffset([Fixed = getFixedSize()](unsigned offset) {
      return Fixed + offset;
    });
  }
};
static_assert(Fragment{7}.adjust() == 10);

template <class T>
struct GenericFragment {
  T size;
  constexpr T adjust() const {
    auto fn = [Fixed = size](T offset) { return Fixed + offset; };
    return fn(T{3});
  }
};
static_assert(GenericFragment<int>{7}.adjust() == 10);
static_assert(GenericFragment<unsigned long>{9}.adjust() == 12);

constexpr int referenceCapture() {
  int value = 7;
  auto fn = [&Fixed = value](int offset) { return Fixed += offset; };
  fn(3);
  return fn(2) + value;
}
static_assert(referenceCapture() == 24);

constexpr int mutableCapture() {
  auto fn = [Fixed = 1](int offset) mutable { return Fixed += offset; };
  fn(3);
  return fn(2);
}
static_assert(mutableCapture() == 6);

constexpr int nestedCapture() {
  auto outer = [Fixed = 7](auto offset) {
    return [Fixed = Fixed](int more) { return Fixed + more; }(offset);
  };
  return outer(3);
}
static_assert(nestedCapture() == 10);
