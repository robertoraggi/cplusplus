// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

template <typename T>
struct holder {
  static constexpr int value = static_cast<int>(sizeof(T));
};

template <typename T>
int probe() {
  using alias = holder<T>;
  return alias::value;
}

template <typename T>
struct nested {
  using self = nested;
  static int get() { return self::tag; }
  static int tag;
};

template <typename T>
int tag_of() {
  return nested<T>::get();
}

int main() { return probe<int>() == 4 ? 0 : 1; }
