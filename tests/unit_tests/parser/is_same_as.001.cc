// RUN: %cxx -verify -fsyntax-only %s -o -

// clang-format off

static_assert(__is_same_as(char, int)); // expected-error{{static_assert failed}}

static_assert(__is_same(char, int)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(int, int));

static_assert(__is_same_as(unsigned int, int)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(unsigned int, unsigned int));

typedef unsigned int uint32_t;

static_assert(__is_same_as(uint32_t, unsigned int));

typedef int int32_t;

static_assert(__is_same_as(int32_t, int));

static_assert(__is_same_as(uint32_t, int32_t)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(signed int, int));

typedef void *ptr_t;

static_assert(__is_same_as(ptr_t, void*));

static_assert(__is_same_as(ptr_t*, void*)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(ptr_t*, void**)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(const int, const int));
static_assert(__is_same_as(const int, int)); // expected-error{{static_assert failed}}
static_assert(__is_same_as(const int, volatile int)); // expected-error{{static_assert failed}}
static_assert(__is_same_as(volatile int, volatile int)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(const volatile int, volatile const int));

struct vec2i {
  int x;
  int y;
};

struct string {
  const char* c_str();
};

void test_1() {
  vec2i v;
  static_assert(__is_same_as(decltype(v), vec2i));
  static_assert(__is_same_as(decltype(v.x), int));
  static_assert(__is_same_as(decltype(v.x), unsigned int)); // expected-error{{static_assert failed}}
  static_assert(__is_same_as(decltype(v.y), int));

  string str;
  static_assert(__is_same_as(decltype(str.c_str()), const char*));
  static_assert(__is_same_as(decltype(str.c_str()), char*)); // expected-error{{static_assert failed}}
}
