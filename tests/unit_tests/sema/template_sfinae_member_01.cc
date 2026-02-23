// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

// Regression test: when a class template is instantiated, member function
// templates inside it must be bound with a templateDeclaration so that
// subsequent SFINAE / overload-resolution can instantiate them.
//
// The pattern below is the is_referenceable idiom: T& is ill-formed for
// T=void, so SFINAE selects the fallback overload.

struct false_t {};

template <class T>
struct IsReferenceable {
  template <class U>
  static U& test(int);
  template <class U>
  static false_t test(...);

  // value = true iff T is referenceable (i.e. T& is well-formed)
  static constexpr bool value = !__is_same(decltype(test<T>(0)), false_t);
};

// int is referenceable: T& = int& is well-formed.
static_assert(IsReferenceable<int>::value, "int is referenceable");

// void is not referenceable: T& = void& is ill-formed, SFINAE picks fallback.
static_assert(!IsReferenceable<void>::value, "void is not referenceable");
