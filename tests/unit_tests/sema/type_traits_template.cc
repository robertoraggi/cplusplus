// clang-format off
// RUN: %cxx -verify -fcheck %s

struct TrivialBoth {
  TrivialBoth& operator=(const TrivialBoth&) = default;
  TrivialBoth& operator=(TrivialBoth&&) = default;
};

struct NonTrivialCopy {
  NonTrivialCopy& operator=(const NonTrivialCopy&) { return *this; }
  NonTrivialCopy& operator=(NonTrivialCopy&&) = default;
};

struct OverloadedVirtual {
  virtual void f(int);
  void f(double);
};

//
// Template with trait on dependent type
//

template <typename T>
struct CheckCopyable {
  static constexpr bool value = __is_trivially_copyable(T);
};

static_assert(CheckCopyable<TrivialBoth>::value, "TrivialBoth in template");
static_assert(CheckCopyable<int>::value, "int in template");
// expected-error@+1 {{"NonTrivialCopy in template"}}
static_assert(CheckCopyable<NonTrivialCopy>::value, "NonTrivialCopy in template");

template <typename T>
struct CheckTrivial {
  static constexpr bool value = __is_trivial(T);
};

static_assert(CheckTrivial<TrivialBoth>::value, "TrivialBoth trivial in template");
// expected-error@+1 {{"NonTrivialCopy trivial in template"}}
static_assert(CheckTrivial<NonTrivialCopy>::value, "NonTrivialCopy trivial in template");

template <typename T>
struct CheckPoly {
  static constexpr bool value = __is_polymorphic(T);
};

static_assert(CheckPoly<OverloadedVirtual>::value, "OverloadedVirtual poly in template");
static_assert(!CheckPoly<TrivialBoth>::value, "TrivialBoth not poly in template");

//
// Template with trait in dependent context, used in abbrev function
//

template <typename T>
constexpr bool is_tc = __is_trivially_copyable(T);

static_assert(is_tc<TrivialBoth>, "var template TrivialBoth");
// expected-error@+1 {{"var template NonTrivialCopy"}}
static_assert(is_tc<NonTrivialCopy>, "var template NonTrivialCopy");

//
// Template with member that has overloaded operators
//

template <typename T>
struct HasMember {
  T member;
};

static_assert(__is_trivially_copyable(HasMember<TrivialBoth>), "HasMember<TrivialBoth>");
// expected-error@+1 {{"HasMember<NonTrivialCopy>"}}
static_assert(__is_trivially_copyable(HasMember<NonTrivialCopy>), "HasMember<NonTrivialCopy>");
