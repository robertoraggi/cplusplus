// RUN: %cxx -toolchain wasm32 -verify -fsyntax-only -fvalidate-ast %s
// expected-no-diagnostics

using Zero = int __attribute__((address_space(0)));
using One = int __attribute__((address_space(1)));
using Two = int __attribute__((__address_space__(2)));
using ClangOne = int [[clang::address_space(1)]];
using VoidOne = void __attribute__((address_space(1)));

static_assert(!__is_same(Zero, int));
static_assert(!__is_same(Zero, One));
static_assert(!__is_same(One, Two));
static_assert(__is_same(ClangOne, One));
static_assert(__is_integral(One));
static_assert(__is_same(__remove_cv(const volatile One), One));
static_assert(__is_same(__remove_const(const One), One));
static_assert(__is_same(__make_unsigned(const One),
                        const unsigned int __attribute__((address_space(1)))));
static_assert(__is_convertible(One*, const One*));
static_assert(__is_convertible(One*, VoidOne*));
static_assert(!__is_convertible(One*, void*));
static_assert(!__is_convertible(Zero*, int*));
static_assert(!__is_convertible(int*, Zero*));
static_assert(!__is_convertible(One*, Two*));
static_assert(!__is_convertible(One**, const Two**));
static_assert(!__is_convertible(One&, int&));
static_assert(__is_same(decltype(*((One*)nullptr) + 1), int));

constexpr int pick(int*) { return 0; }
constexpr int pick(One*) { return 1; }
constexpr int pick(Two*) { return 2; }
constexpr int pick(const One*) { return 3; }
static_assert(pick((int*)nullptr) == 0);
static_assert(pick((One*)nullptr) == 1);
static_assert(pick((Two*)nullptr) == 2);
static_assert(pick((const One*)nullptr) == 3);

template <class T>
constexpr bool captures_space(T*) {
  return __is_same(T, One);
}
static_assert(captures_space((One*)nullptr));

template <class T>
constexpr bool consumes_space(T __attribute__((address_space(1))) *) {
  return __is_same(T, int);
}
static_assert(consumes_space((One*)nullptr));

using Array = One[3][4];
Array global_array;
static_assert(__is_same(decltype(global_array[1][2]), One&));
static_assert(__is_same(decltype(&global_array[1][2]), One*));

struct Record {
  int value;
  int array[3];
};
using RecordOne = Record __attribute__((address_space(1)));
One* member_address(RecordOne* p) { return &p->value; }
One* element_address(RecordOne* p) { return &p->array[1]; }
auto read_value(One& value) { return value; }
static_assert(__is_same(decltype(read_value(*((One*)nullptr))), int));

template <class T>
T copy_value(T value) {
  return value;
}
static_assert(__is_same(decltype(copy_value(*((One*)nullptr))), int));
int copy_from_pointer(One* p) { return copy_value(*p); }
static_assert(__is_same(__decay(One), One));

struct Base {
  int base;
};
struct Derived : Base {
  int derived;
};
using BaseOne = Base __attribute__((address_space(1)));
using DerivedOne = Derived __attribute__((address_space(1)));
static_assert(__is_convertible(DerivedOne*, BaseOne*));
static_assert(!__is_convertible(DerivedOne*, Base*));

char pick_zero(Zero*);
int pick_zero(int*);
static_assert(__is_same(decltype(pick_zero((Zero*)nullptr)), char));
static_assert(__is_same(decltype(pick_zero((int*)nullptr)), int));

template <class L, class R>
concept Equal = requires(L l, R r) { l == r; };
template <class L, class R>
concept Ordered = requires(L l, R r) { l < r; };
template <class L, class R>
concept Difference = requires(L l, R r) { l - r; };
static_assert(Equal<One*, const One*>);
static_assert(!Equal<One*, Two*>);
static_assert(!Equal<Zero*, int*>);
static_assert(!Ordered<One*, Two*>);
static_assert(!Difference<One*, Two*>);
