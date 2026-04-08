// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T>
struct is_copy_assignable {
  static constexpr bool value = true;
};

template <bool B, class T>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
  using type = T;
};

template <class T,
          typename enable_if<is_copy_assignable<T>::value, int>::type = 0>
void foo(T) {}

int main() {
  foo(42);
  return 0;
}
