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
