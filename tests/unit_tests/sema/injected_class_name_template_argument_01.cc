// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

template <template <class> class TT> struct UsesTemplate { };
template <class E> struct W { using type = E; };
template <class T> struct V {
  UsesTemplate<V> template_arg;
  W<V> plain;
  W<V*> pointer;
  W<const V&> reference;
  W<V&&> rvalue;
  W<V[2]> array;
  W<V (*)(V)> function_pointer;
  W<V V::*> member_pointer;
  W<V<T>> explicit_id;
  W<const V<T>*> explicit_pointer;
  static_assert(sizeof(T) > 0);
};
V<int> v;
static_assert(__is_same(decltype(v.plain)::type, V<int>));
static_assert(__is_same(decltype(v.pointer)::type, V<int>*));
static_assert(__is_same(decltype(v.reference)::type, const V<int>&));
static_assert(__is_same(decltype(v.rvalue)::type, V<int>&&));
static_assert(__is_same(decltype(v.array)::type, V<int>[2]));
static_assert(__is_same(decltype(v.function_pointer)::type, V<int> (*)(V<int>)));
static_assert(__is_same(decltype(v.member_pointer)::type, V<int> V<int>::*));
static_assert(__is_same(decltype(v.explicit_id)::type, V<int>));
static_assert(__is_same(decltype(v.explicit_pointer)::type, const V<int>*));
static_assert(!__is_same(decltype(v.plain), decltype(v.pointer)));
static_assert(__is_same(decltype(v.pointer), W<V<int>*>));
