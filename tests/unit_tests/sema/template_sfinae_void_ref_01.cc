// RUN: %cxx -verify %s
// expected-no-diagnostics

struct false_type {
  static constexpr bool value = false;
};

struct true_type {
  static constexpr bool value = true;
};

// Classic is_reference-via-SFINAE: T& is ill-formed for T=void.
struct ReferenceSFINAE {
  template <class T>
  static T& test(int);
  template <class>
  static false_type test(...);
};

template <class T>
struct is_referenceable {
  static constexpr bool value =
      !__is_same(decltype(ReferenceSFINAE::test<T>(0)), false_type);
};

// void is not referenceable.
static_assert(!is_referenceable<void>::value, "void is not referenceable");

// int is referenceable.
static_assert(is_referenceable<int>::value, "int is referenceable");
