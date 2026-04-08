// RUN: %cxx -verify %s

// Default template arguments from forward declarations should be
// carried over to the definition when not repeated.

template <class T = void>
struct S;

template <class T>
struct S {
  T* ptr = nullptr;
};

S<> a;     // expected-type-is: S<void>
S<int> b;  // expected-type-is: S<int>

// Non-type template parameter default from forward declaration.
template <int N = 42>
struct V;

template <int N>
struct V {
  int value = N;
};

V<> c;
V<0> d;

// Multiple parameters, some with defaults.
template <class A, class B = int, class C = float>
struct M;

template <class A, class B, class C>
struct M {};

M<char> e;
M<char, long> f;
M<char, long, double> g;
