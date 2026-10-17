// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

#include <string>
#include <type_traits>

template <class K> struct trans : std::true_type {};

namespace case_1 {
template <class P> struct R { using type = int; };
struct M { template <class A, class B> struct action; };
template <class A, class B> struct M::action {
  typedef action<A, B> self_t;
  template <class S> typename R<self_t>::type parse(S) const;
};
template <class A, class B> template <class S>
typename R<M::action<A, B>>::type M::action<A, B>::parse(S) const { return 0; }
}  // namespace case_1

namespace case_2 {
template <class P> struct R { using type = int; };
template <class A> struct M {
  typedef M<A> self_t;
  typename R<self_t>::type parse() const;
};
template <class A>
typename R<M<A>>::type M<A>::parse() const { return 0; }
}  // namespace case_2

namespace case_3 {
template <class P> struct R { using type = int; };
template <class A> struct M {
  typedef M self_t;
  typename R<self_t>::type parse() const;
};
template <class A>
typename R<M<A>>::type M<A>::parse() const { return 0; }
}  // namespace case_3

namespace case_4 {
template <class Ch, class Tr, class Alloc> struct B {
  typedef std::basic_string<Ch, Tr, Alloc> string_type;
  typedef typename string_type::size_type size_type;
  size_type size() const;
};
template <class Ch, class Tr, class Alloc>
typename std::basic_string<Ch,Tr,Alloc>::size_type B<Ch, Tr, Alloc>::size() const { return 0; }
}  // namespace case_4

namespace case_5 {
template <class T> struct Str { using size_type = unsigned; };
template <class T> struct B {
  typedef Str<T> string_type;
  typedef typename string_type::size_type size_type;
  size_type size() const;
};
template <class T>
typename Str<T>::size_type B<T>::size() const { return 0; }
}  // namespace case_5

namespace case_6 {
template <class T> struct Str { using size_type = unsigned; };
template <class T> struct B {
  typedef Str<T> string_type;
  typename string_type::size_type size() const;
};
template <class T>
typename Str<T>::size_type B<T>::size() const { return 0; }
}  // namespace case_6

namespace case_7 {

template <class K, class T> struct U {
  typedef T mapped_type;
  template <class Key> typename std::enable_if<trans<Key>::value, mapped_type&>::type op(Key&&);
};
template <class K, class T> template <class Key>
typename std::enable_if<trans<Key>::value, typename U<K, T>::mapped_type&>::type U<K, T>::op(Key&&) { static T t; return t; }
}  // namespace case_7

namespace case_8 {

template <class K, class T> struct U {
  typedef T mapped_type;
  template <class Key> auto op(Key&&) -> typename std::enable_if<trans<Key>::value, mapped_type&>::type;
};
template <class K, class T> template <class Key>
auto U<K, T>::op(Key&&) -> typename std::enable_if<trans<Key>::value, mapped_type&>::type { static T t; return t; }
}  // namespace case_8

namespace case_9 {

template <class K, class T> struct U {
  typedef T mapped_type;
  template <class Key> typename std::enable_if<trans<Key>::value, mapped_type&>::type op(Key&&);
};
template <class K, class T> template <class Key>
typename std::enable_if<trans<Key>::value, T&>::type U<K, T>::op(Key&&) { static T t; return t; }
}  // namespace case_9
