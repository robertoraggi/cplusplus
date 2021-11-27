// RUN: %cxx -verify -fsyntax-only %s -o -

static_assert(__is_void(int));  // expected-error{{static_assert failed}}

static_assert(__is_void(void));

static_assert(__is_void(void*));  // expected-error{{static_assert failed}}

static_assert(__is_same(decltype(__is_void(void)), bool));

static_assert(__is_same(decltype(__is_void(void)),
                        int));  // expected-error@-1{{static_assert failed}}

static_assert(__is_pointer(void*));

int* p;

static_assert(__is_pointer(decltype(p)));

static_assert(__is_pointer(int));  // expected-error{{static_assert failed}}

void foo() {}

static_assert(
    __is_pointer(decltype(foo)));  // expected-error{{static_assert failed}}

// __is_const

const int v = 0;

static_assert(__is_const(decltype(v)));

volatile int* cv = nullptr;

static_assert(
    __is_const(decltype(cv)));  // expected-error{{static_assert failed}}

int* const cv2 = nullptr;

static_assert(__is_const(decltype(cv2)));

static_assert(
    __is_const(decltype(0)));  // expected-error{{static_assert failed}}

// __is_volatile

const int volatile vv = 0;

static_assert(__is_volatile(decltype(vv)));

static_assert(__is_const(decltype(vv)));

int* volatile pv = nullptr;

static_assert(__is_volatile(decltype(pv)));

// __is_null_pointer

static_assert(__is_null_pointer(decltype(nullptr)));

void* ptr = nullptr;

static_assert(__is_null_pointer(
    decltype(ptr)));  // expected-error@-1{{static_assert failed}}

using nullptr_t = decltype(nullptr);

static_assert(__is_null_pointer(nullptr_t));

// __is_signed

static_assert(__is_signed(char));
static_assert(__is_signed(short));
static_assert(__is_signed(int));
static_assert(__is_signed(long));
static_assert(__is_signed(long long));

static_assert(
    __is_signed(unsigned char));  // expected-error{{static_assert failed}}

static_assert(
    __is_signed(unsigned short));  // expected-error{{static_assert failed}}

static_assert(
    __is_signed(unsigned int));  // expected-error{{static_assert failed}}

static_assert(
    __is_signed(unsigned long));  // expected-error{{static_assert failed}}

static_assert(
    __is_signed(unsigned long long));  // expected-error{{static_assert failed}}

// __is_unsigned

static_assert(__is_unsigned(unsigned char));
static_assert(__is_unsigned(unsigned short));
static_assert(__is_unsigned(unsigned int));
static_assert(__is_unsigned(unsigned long));
static_assert(__is_unsigned(unsigned long long));

static_assert(__is_unsigned(char));   // expected-error{{static_assert failed}}
static_assert(__is_unsigned(short));  // expected-error{{static_assert failed}}
static_assert(__is_unsigned(int));    // expected-error{{static_assert failed}}
static_assert(__is_unsigned(long));   // expected-error{{static_assert failed}}

static_assert(
    __is_unsigned(long long));  // expected-error{{static_assert failed}}
