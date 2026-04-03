// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

using size_t = decltype(sizeof(int));
namespace std {
inline namespace __2 {

template <class _T1, class _T2>
struct pair {
  _T1 first;
  _T2 second;
};

template <size_t _Ip, class _T>
struct tuple_element;
template <class _T1, class _T2>
struct tuple_element<0, pair<_T1, _T2>> {
  using type = _T1;
};
template <class _T1, class _T2>
struct tuple_element<1, pair<_T1, _T2>> {
  using type = _T2;
};

// Forward declarations with dependent return type
template <size_t _Ip, class _T1, class _T2>
constexpr typename tuple_element<_Ip, pair<_T1, _T2>>::type& get(
    pair<_T1, _T2>&) noexcept;
template <size_t _Ip, class _T1, class _T2>
constexpr const typename tuple_element<_Ip, pair<_T1, _T2>>::type& get(
    const pair<_T1, _T2>&) noexcept;

// Definitions - must be recognized as redeclarations of the above
template <size_t _Ip, class _T1, class _T2>
inline constexpr typename tuple_element<_Ip, pair<_T1, _T2>>::type& get(
    pair<_T1, _T2>& p) noexcept {
  return p.first;
}
template <size_t _Ip, class _T1, class _T2>
inline constexpr const typename tuple_element<_Ip, pair<_T1, _T2>>::type& get(
    const pair<_T1, _T2>& p) noexcept {
  return p.first;
}

}  // namespace __2
}  // namespace std

void test() {
  std::pair<int, int> p{1, 2};
  int x = std::get<0>(p);
  int y = std::get<1>(p);
  (void)x;
  (void)y;
}
