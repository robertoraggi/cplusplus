// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

static_assert(__builtin_abs(-42) == 42);
static_assert(__builtin_labs(-100L) == 100L);
static_assert(__builtin_llabs(-999LL) == 999LL);

static_assert(__builtin_strlen("hello") == 5);
static_assert(__builtin_strlen("") == 0);

static_assert(__builtin_strcmp("abc", "abc") == 0);
static_assert(__builtin_strcmp("abc", "abd") < 0);
static_assert(__builtin_strcmp("abd", "abc") > 0);

static_assert(__builtin_strncmp("abcdef", "abcxyz", 3) == 0);
static_assert(__builtin_strncmp("abcdef", "abcxyz", 4) != 0);

static_assert(__builtin_memcmp("abc", "abc", 3) == 0);
static_assert(__builtin_memcmp("abc", "abd", 3) < 0);

static_assert(__builtin_bcmp("abc", "abc", 3) == 0);
static_assert(__builtin_bcmp("abc", "abd", 3) != 0);

static_assert(__builtin_expect(42, 1) == 42);

static_assert(__builtin_bswap32(0x01020304) == 0x04030201);
static_assert(__builtin_bswap64(0x0102030405060708ULL) ==
              0x0807060504030201ULL);

static_assert(__builtin_is_constant_evaluated());
