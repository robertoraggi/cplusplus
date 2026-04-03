// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <bool, class T = void>
struct enable_if {};
template <class T>
struct enable_if<true, T> {
  using type = T;
};
template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <class T>
struct is_mv {
  static constexpr bool value = true;
};

template <class T>
using swap_result_t = enable_if_t<is_mv<T>::value>;

// Forward declaration with alias return type.
template <class T>
inline swap_result_t<T> swap(T& x, T& y) noexcept;

template <class T1, class T2>
struct pair {
  T1 first;
  T2 second;
  void swap(pair& p) noexcept {
    ::swap(first, p.first);
    ::swap(second, p.second);
  }
};

// Definition in a later namespace re-opening.
template <class T>
inline swap_result_t<T> swap(T& x, T& y) noexcept {
  T t = static_cast<T&&>(x);
  x = static_cast<T&&>(y);
  y = static_cast<T&&>(t);
}

int main() {
  pair<int, int> a{1, 2}, b{3, 4};
  a.swap(b);
}
