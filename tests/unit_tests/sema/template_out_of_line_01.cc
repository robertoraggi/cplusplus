// RUN: %cxx -verify -fcheck %s

template <typename T>
struct Box {
  T value_;
  auto get() -> T;
  void set(const T& v);
};

template <typename U>
auto Box<U>::get() -> U {
  return value_;
}

template <typename W>
void Box<W>::set(const W& value) {
  value_ = value;
}

auto main() -> int {
  Box<int> b;
  b.set(42);
  auto v = b.get();
  return 0;
}
