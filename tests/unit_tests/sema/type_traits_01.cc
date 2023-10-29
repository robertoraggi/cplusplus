// RUN: %cxx -verify -fstatic-assert %s

//
// is_void trait
//

static_assert(__is_void(void));
static_assert(__is_void(const void));
static_assert(__is_void(volatile void));
static_assert(__is_void(const volatile void));

// expected-error@1 {{static assert failed}}
static_assert(__is_void(int));

// expected-error@1 {{static assert failed}}
static_assert(__is_void(void*));

//
// is_integral trait
//

static_assert(__is_integral(bool));
static_assert(__is_integral(char));
static_assert(__is_integral(signed char));
static_assert(__is_integral(unsigned char));
static_assert(__is_integral(char8_t));
static_assert(__is_integral(char16_t));
static_assert(__is_integral(char32_t));
static_assert(__is_integral(wchar_t));
static_assert(__is_integral(short));
static_assert(__is_integral(unsigned short));
static_assert(__is_integral(int));
static_assert(__is_integral(unsigned int));
static_assert(__is_integral(long));
static_assert(__is_integral(unsigned long));
static_assert(__is_integral(long long));
static_assert(__is_integral(unsigned long long));

// expected-error@1 {{static assert failed}}
static_assert(__is_integral(float));

// expected-error@1 {{static assert failed}}
static_assert(__is_integral(double));

// expected-error@1 {{static assert failed}}
static_assert(__is_integral(long double));

// expected-error@1 {{static assert failed}}
static_assert(__is_integral(void));

// expected-error@1 {{static assert failed}}
static_assert(__is_integral(void*));

//
// is_floating_point trait
//

static_assert(__is_floating_point(float));
static_assert(__is_floating_point(double));
static_assert(__is_floating_point(long double));
static_assert(__is_floating_point(const float));
static_assert(__is_floating_point(const double));
static_assert(__is_floating_point(const long double));

// expected-error@1 {{static assert failed}}
static_assert(__is_floating_point(bool));

// expected-error@1 {{static assert failed}}
static_assert(__is_floating_point(float*));

//
// is_signed trait
//

static_assert(__is_signed(char));
static_assert(__is_signed(const char));
static_assert(__is_signed(signed char));
static_assert(__is_signed(short));
static_assert(__is_signed(int));
static_assert(__is_signed(long));
static_assert(__is_signed(long long));
static_assert(__is_signed(float));
static_assert(__is_signed(double));
static_assert(__is_signed(long double));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(bool));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(unsigned char));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(char8_t));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(char16_t));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(char32_t));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(wchar_t));

// expected-error@1 {{static assert failed}}
static_assert(__is_signed(void*));

//
// is_unsigned trait
//

static_assert(__is_unsigned(bool));
static_assert(__is_unsigned(const unsigned char));
static_assert(__is_unsigned(unsigned));
static_assert(__is_unsigned(unsigned char));
static_assert(__is_unsigned(unsigned short));
static_assert(__is_unsigned(unsigned int));
static_assert(__is_unsigned(unsigned long));
static_assert(__is_unsigned(unsigned long long));
static_assert(__is_unsigned(char8_t));
static_assert(__is_unsigned(char16_t));
static_assert(__is_unsigned(char32_t));
static_assert(__is_unsigned(wchar_t));

// expected-error@1 {{static assert failed}}
static_assert(__is_unsigned(char));

// expected-error@1 {{static assert failed}}
static_assert(__is_unsigned(float));

// expected-error@1 {{static assert failed}}
static_assert(__is_unsigned(double));

// expected-error@1 {{static assert failed}}
static_assert(__is_unsigned(long double));

// expected-error@1 {{static assert failed}}
static_assert(__is_unsigned(const unsigned int*));

//
// is_array trait
//

static_assert(__is_array(int[]));
static_assert(__is_array(int[10]));
static_assert(__is_array(const int[]));
static_assert(__is_array(int* [10]));

// expected-error@1 {{static assert failed}}
static_assert(__is_array(int));

// expected-error@1 {{static assert failed}}
static_assert(__is_array(int (*)[10]));

//
// is_bounded_array trait
//

static_assert(__is_bounded_array(int[10]));
static_assert(__is_bounded_array(const int[10 + 20]));

// expected-error@1 {{static assert failed}}
static_assert(__is_bounded_array(int[]));

//
// is_unbounded_array trait
//

static_assert(__is_unbounded_array(int[]));

// expected-error@1 {{static assert failed}}
static_assert(__is_unbounded_array(int[10]));

// expected-error@1 {{static assert failed}}
static_assert(__is_unbounded_array(int[10 + 20]));

//
// is_const trait
//

static_assert(__is_const(const void));
static_assert(__is_const(const volatile void));
static_assert(__is_const(int* const));

// expected-error@1 {{static assert failed}}
static_assert(__is_const(const void*));

// expected-error@1 {{static assert failed}}
static_assert(__is_const(int));

//
// is_volatile trait
//

static_assert(__is_volatile(volatile void));
static_assert(__is_volatile(const volatile void));
static_assert(__is_volatile(int* volatile));

// expected-error@1 {{static assert failed}}
static_assert(__is_volatile(volatile void*));

// expected-error@1 {{static assert failed}}
static_assert(__is_volatile(int));

//
// is_lvalue_reference trait
//

static_assert(__is_lvalue_reference(int&));
static_assert(__is_lvalue_reference(const int&));

// expected-error@1 {{static assert failed}}
static_assert(__is_lvalue_reference(int&&));

//
// is_rvalue_reference trait
//

static_assert(__is_rvalue_reference(int&&));
static_assert(__is_rvalue_reference(const int&&));

// expected-error@1 {{static assert failed}}
static_assert(__is_rvalue_reference(int&));

//
// is_reference trait
//

static_assert(__is_reference(int&));
static_assert(__is_reference(const int&));
static_assert(__is_reference(int&&));
static_assert(__is_reference(const int&&));

// expected-error@1 {{static assert failed}}
static_assert(__is_reference(int));
