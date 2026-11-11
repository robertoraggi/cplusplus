// RUN: %cxx -std=c++20 -verify -fsyntax-only %s

template <class T>
int distinct(T);

template <class T>
int distinct(T*);

auto ambiguous = &distinct<int>;
// expected-error@-1 {{cannot deduce 'auto' from an overload set}}

decltype(auto) ambiguousReference = (distinct<int>);
// expected-error@-1 {{cannot deduce 'decltype(auto)' from an overload set}}

template <class T, class U>
int incomplete(U);

template <class T, class U>
int incomplete(U*);

auto missing = &incomplete<int>;
// expected-error@-1 {{cannot deduce 'auto' from an overload set}}

template <class T>
int constrained(T) requires (sizeof(T) > 0);

template <class T>
int constrained(T) requires (sizeof(T) >= 1);

auto ambiguousConstraints = &constrained<int>;
// expected-error@-1 {{cannot deduce 'auto' from an overload set}}

template <class T>
concept Sized = sizeof(T) > 0;

template <class T>
int ordered(T) requires Sized<T>;

template <class T>
int ordered(T) requires (Sized<T> && sizeof(T) >= sizeof(int));

decltype(auto) overloadedReference = (ordered<int>);
// expected-error@-1 {{cannot deduce 'decltype(auto)' from an overload set}}
