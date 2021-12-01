// RUN: %cxx -verify -fsyntax-only %s -o -

// __is_void

static_assert(__is_void(int) == false);
static_assert(__is_void(void) == true);
static_assert(__is_void(void*) == false);
static_assert(__is_void(const void) == true);
static_assert(__is_void(volatile void) == true);
static_assert(__is_void(const volatile void) == true);

static_assert(__is_same(decltype(__is_void(void)), bool) == true);
static_assert(__is_same(decltype(__is_void(void)), int) == false);

// __is_integral

static_assert(__is_integral(bool) == true);
static_assert(__is_integral(char8_t) == true);
static_assert(__is_integral(char16_t) == true);
static_assert(__is_integral(char32_t) == true);
static_assert(__is_integral(char) == true);
static_assert(__is_integral(short) == true);
static_assert(__is_integral(int) == true);
static_assert(__is_integral(long) == true);
static_assert(__is_integral(long long) == true);
static_assert(__is_integral(unsigned char) == true);
static_assert(__is_integral(unsigned short) == true);
static_assert(__is_integral(unsigned int) == true);
static_assert(__is_integral(unsigned long) == true);
static_assert(__is_integral(unsigned long long) == true);
static_assert(__is_integral(float) == false);
static_assert(__is_integral(double) == false);
static_assert(__is_integral(long double) == false);
static_assert(__is_integral(void*) == false);

// __is_floating_point

static_assert(__is_floating_point(float) == true);
static_assert(__is_floating_point(double) == true);
static_assert(__is_floating_point(long double) == true);
static_assert(__is_floating_point(double long) == true);
static_assert(__is_floating_point(const float) == true);
static_assert(__is_floating_point(const double) == true);
static_assert(__is_floating_point(const long double) == true);
static_assert(__is_floating_point(float*) == false);
static_assert(__is_floating_point(float&) == false);

// __is_pointer

static_assert(__is_pointer(void*) == true);

int* p;

static_assert(__is_pointer(decltype(p)) == true);

static_assert(__is_pointer(int) == false);

void foo() {}

static_assert(__is_pointer(decltype(foo)) == false);

// __is_const

const int v = 0;

static_assert(__is_const(decltype(v)) == true);

const int* cv = nullptr;

static_assert(__is_const(decltype(cv)) == false);

int* const cv2 = nullptr;

static_assert(__is_const(decltype(cv2)) == true);

static_assert(__is_const(decltype(0)) == false);

// __is_volatile

const int volatile vv = 0;

static_assert(__is_volatile(decltype(vv)));

static_assert(__is_const(decltype(vv)));

int* volatile pv = nullptr;

static_assert(__is_volatile(decltype(pv)));

// __is_null_pointer

static_assert(__is_null_pointer(decltype(nullptr)) == true);

void* ptr = nullptr;

static_assert(__is_null_pointer(decltype(ptr)) == false);

using nullptr_t = decltype(nullptr);

static_assert(__is_null_pointer(nullptr_t) == true);

// __is_signed

static_assert(__is_signed(char) == true);
static_assert(__is_signed(short) == true);
static_assert(__is_signed(int) == true);
static_assert(__is_signed(long) == true);
static_assert(__is_signed(long long) == true);

static_assert(__is_signed(unsigned char) == false);
static_assert(__is_signed(unsigned short) == false);
static_assert(__is_signed(unsigned int) == false);
static_assert(__is_signed(unsigned long) == false);
static_assert(__is_signed(unsigned long long) == false);

// __is_unsigned

static_assert(__is_unsigned(unsigned char) == true);
static_assert(__is_unsigned(unsigned short) == true);
static_assert(__is_unsigned(unsigned int) == true);
static_assert(__is_unsigned(unsigned long) == true);
static_assert(__is_unsigned(unsigned long long) == true);

static_assert(__is_unsigned(char) == false);
static_assert(__is_unsigned(short) == false);
static_assert(__is_unsigned(int) == false);
static_assert(__is_unsigned(long) == false);
static_assert(__is_unsigned(long long) == false);

// __is_enum and __is_scoped_enum

enum ee {};
enum class sc {};

static_assert(__is_enum(ee) == true);
static_assert(__is_enum(sc) == true);

static_assert(__is_scoped_enum(sc) == true);
static_assert(__is_scoped_enum(ee) == false);

// __is_class and __is_union

struct F;
struct S {};
class C {};

union V;
union U {};

static_assert(__is_class(F) == true);
static_assert(__is_class(S) == true);
static_assert(__is_class(C) == true);
static_assert(__is_class(V) == false);
static_assert(__is_class(U) == false);

static_assert(__is_union(S) == false);
static_assert(__is_union(C) == false);
static_assert(__is_union(U) == true);
static_assert(__is_union(V) == true);

static_assert(__is_class(void) == false);

// __is_lvalue_reference

static_assert(__is_lvalue_reference(int&) == true);
static_assert(__is_lvalue_reference(int&&) == false);
static_assert(__is_lvalue_reference(int*&) == true);
static_assert(__is_lvalue_reference(int) == false);

// __is_rvalue_reference

static_assert(__is_rvalue_reference(int&) == false);
static_assert(__is_rvalue_reference(int&&) == true);
static_assert(__is_rvalue_reference(int*&) == false);
static_assert(__is_rvalue_reference(int) == false);

// __is_reference

static_assert(__is_reference(int&) == true);
static_assert(__is_reference(int&&) == true);
static_assert(__is_reference(int*&) == true);
static_assert(__is_reference(int) == false);

// __is_function

static_assert(__is_function(decltype(foo)) == true);
static_assert(__is_function(void) == false);
static_assert(__is_function(int()) == true);
static_assert(__is_function(int (*)()) == false);

// __is_member_object_pointer

struct Class {
  int i;
};

using IntFieldT = int(Class::*);

int(Class::*intField) = nullptr;

static_assert(__is_member_object_pointer(decltype(intField)) == true);
static_assert(__is_member_object_pointer(IntFieldT) == true);
static_assert(__is_member_object_pointer(int(Class::*)) == true);

static_assert(__is_member_object_pointer(int(Class::*)()) == false);

namespace ns {
struct list {
  struct iterator {
    int p;
  };
};
}  // namespace ns

static_assert(__is_member_object_pointer(int(ns::list::iterator::*)) == true);
