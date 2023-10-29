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
