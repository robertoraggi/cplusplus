// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

#include <type_traits>

template <class T>
struct Forward : std::true_type {};
template <class T>
struct Input : std::true_type {};

template <class C>
struct Str {
  using iterator = C*;
  using const_iterator = const C*;

  template <class It>
  auto insert(const_iterator, It, It) ->
      typename std::enable_if<Forward<It>::value, int>::type;
  template <class It>
  auto insert(const_iterator, It, It) ->
      typename std::enable_if<Input<It>::value && !Forward<It>::value,
                              long>::type;

  template <class It>
  typename std::enable_if<Forward<It>::value, int>::type same_name(It);
  template <class It>
  typename std::enable_if<Input<It>::value && !Forward<It>::value, long>::type
  same_name(It);

  template <class It>
  auto trailing(It) -> typename std::enable_if<(sizeof(It) > 2), int>::type;
  template <class It>
  auto trailing(It) -> typename std::enable_if<(sizeof(It) <= 2), long>::type;
};

struct ForwardOnly {};
struct InputOnly {};
template <>
struct Forward<InputOnly> : std::false_type {};

template <class C>
template <class It>
auto Str<C>::insert(const_iterator, It, It) ->
    typename std::enable_if<Forward<It>::value, int>::type {
  return 1;
}

template <class C>
template <class It>
auto Str<C>::insert(const_iterator, It, It) ->
    typename std::enable_if<Input<It>::value && !Forward<It>::value,
                            long>::type {
  return 2;
}

template <class C>
template <class It>
typename std::enable_if<Forward<It>::value, int>::type Str<C>::same_name(It) {
  return 3;
}

template <class C>
template <class It>
typename std::enable_if<Input<It>::value && !Forward<It>::value, long>::type
Str<C>::same_name(It) {
  return 4;
}

template <class C>
template <class It>
auto Str<C>::trailing(It) -> typename std::enable_if<(sizeof(It) > 2), int>::type {
  return 5;
}

template <class C>
template <class It>
auto Str<C>::trailing(It) -> typename std::enable_if<(sizeof(It) <= 2), long>::type {
  return 6;
}

int main() {
  Str<char> s;
  const char* p = nullptr;
  if (s.insert(p, 1, 2) != 1) return 1;
  if (s.insert(p, ForwardOnly{}, ForwardOnly{}) != 1) return 2;
  if (s.insert(p, InputOnly{}, InputOnly{}) != 2) return 3;
  if (s.same_name(1) != 3) return 4;
  if (s.same_name(InputOnly{}) != 4) return 5;
  if (s.trailing(1) != 5) return 6;
  if (s.trailing('c') != 6) return 7;
  return 0;
}
