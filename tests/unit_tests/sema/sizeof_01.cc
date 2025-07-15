// RUN: %cxx -toolchain wasm32 -verify -fcheck %s

static_assert(sizeof(bool) == 1);
static_assert(sizeof(signed char) == 1);
static_assert(sizeof(unsigned char) == 1);
static_assert(sizeof(short) == 2);
static_assert(sizeof(unsigned short) == 2);
static_assert(sizeof(int) == 4);
static_assert(sizeof(unsigned int) == 4);
static_assert(sizeof(long long) == 8);
static_assert(sizeof(unsigned long long) == 8);
static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);
static_assert(sizeof(char) == 1);
static_assert(sizeof(wchar_t) == 4);
static_assert(sizeof(char8_t) == 1);
static_assert(sizeof(char16_t) == 2);
static_assert(sizeof(char32_t) == 4);

#ifdef __wasm32__
static_assert(sizeof(long) == 4);
static_assert(sizeof(unsigned long) == 4);
static_assert(sizeof(long double) == 16);
#endif

int elements[] = {1, 2, 3, 4, 5};
static_assert(sizeof(elements) == 5 * sizeof(int));

double d[]{1.0, 2.0, 3.0};
static_assert(sizeof(d) == 3 * sizeof(double));

struct X {
  char i;
  int dat[];
};
static_assert(sizeof(X) == 4);
static_assert(alignof(X) == 4);