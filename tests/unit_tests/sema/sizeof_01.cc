// RUN: %cxx -toolchain wasm32 -verify -fcheck %s

static_assert(sizeof(bool) == 1);
static_assert(sizeof(signed char) == 1);
static_assert(sizeof(unsigned char) == 1);
static_assert(sizeof(short) == 2);
static_assert(sizeof(unsigned short) == 2);
static_assert(sizeof(int) == 4);
static_assert(sizeof(unsigned int) == 4);
static_assert(sizeof(long) == 4);
static_assert(sizeof(unsigned long) == 4);
static_assert(sizeof(long long) == 8);
static_assert(sizeof(unsigned long long) == 8);
static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);
static_assert(sizeof(long double) == 16);
static_assert(sizeof(char) == 1);
static_assert(sizeof(wchar_t) == 4);
static_assert(sizeof(char8_t) == 1);
static_assert(sizeof(char16_t) == 2);
static_assert(sizeof(char32_t) == 4);
