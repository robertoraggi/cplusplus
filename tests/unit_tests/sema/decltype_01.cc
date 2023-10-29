// RUN: %cxx -verify -fstatic-assert %s

static_assert(__is_null_pointer(decltype(nullptr)));

static_assert(__is_floating_point(decltype(0.1)));
static_assert(__is_floating_point(decltype(0.1f)));

static_assert(__is_integral(decltype('x')));
static_assert(__is_integral(decltype(u8'x')));
static_assert(__is_integral(decltype(0)));
static_assert(__is_integral(decltype(0U)));
static_assert(__is_unsigned(decltype(0U)));
static_assert(__is_signed(decltype(0)));

static_assert(__is_fundamental(decltype(true)));
static_assert(__is_arithmetic(decltype(true)));
static_assert(__is_unsigned(decltype(true)));

static_assert(__is_lvalue_reference(decltype("ciao")));
static_assert(__is_lvalue_reference(decltype(("ciao"))));

static_assert(__is_same(decltype(nullptr), decltype(nullptr)));
static_assert(__is_same(unsigned, unsigned int));
static_assert(__is_same(unsigned int, decltype(0U)));
static_assert(__is_same(int, decltype(0)));
static_assert(__is_same(unsigned long, decltype(0ul)));
static_assert(__is_same(unsigned long long, decltype(0ull)));

static_assert(__is_same(decltype(true), decltype(false)));
static_assert(__is_same(decltype(true), bool));
static_assert(__is_same(decltype('c'), char));
static_assert(__is_same(decltype(u8'c'), char8_t));
static_assert(__is_same(decltype(u'c'), char16_t));
static_assert(__is_same(decltype(U'c'), char32_t));
static_assert(__is_same(decltype(L'c'), wchar_t));

static_assert(__is_same(int (*)(), int (*)()));

static_assert(__is_same(decltype("ciao"), const char (&)[5]));

// expected-error@1 {{static assert failed}}
static_assert(__is_same(char, unsigned char));

// expected-error@1 {{static assert failed}}
static_assert(__is_same(char, signed char));
