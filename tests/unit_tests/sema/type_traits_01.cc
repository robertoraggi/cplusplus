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
