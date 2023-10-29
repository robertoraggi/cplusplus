// RUN: %cxx -verify -fstatic-assert %s

static_assert(__is_void(void));
static_assert(__is_void(const void));
static_assert(__is_void(volatile void));
static_assert(__is_void(const volatile void));

// expected-error@1 {{static assert failed}}
static_assert(__is_void(int));

// expected-error@1 {{static assert failed}}
static_assert(__is_void(void*));
