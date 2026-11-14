// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T, class U>
constexpr bool same = false;
template <class T>
constexpr bool same<T, T> = true;

template <class... Ts>
void packs(Ts... ts) {
  [... us = ts] { static_assert((same<decltype(us), Ts> && ...)); };
  [... us = ts] { static_assert((same<decltype((us)), const Ts&> && ...)); };
  [... us = ts]() mutable {
    static_assert((same<decltype((us)), Ts&> && ...));
  };
  [&... us = ts] { static_assert((same<decltype(us), Ts&> && ...)); };
  [&... us = ts] { static_assert((same<decltype((us)), Ts&> && ...)); };
  [... us = ts, first = 1L] {
    static_assert(sizeof...(us) == sizeof...(Ts));
  };
  [... us = ts](auto extra) {
    [... vs = us, extra] {
      static_assert((same<decltype((vs)), const Ts&> && ...));
    };
  };
}
template void packs<int, long, short>(int, long, short);
template void packs<>();
