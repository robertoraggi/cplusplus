// RUN: %cxx -verify -fcheck %s

static_assert(__is_same(decltype("hello"), const char (&)[6]));
static_assert(__is_same(decltype(""), const char (&)[1]));
static_assert(__is_same(decltype("a"), const char (&)[2]));
static_assert(sizeof("hello") == 6);
static_assert(sizeof("") == 1);

static_assert(__is_same(decltype(L"hello"), const wchar_t (&)[6]));
static_assert(__is_same(decltype(L""), const wchar_t (&)[1]));
static_assert(__is_same(decltype(L"a"), const wchar_t (&)[2]));
static_assert(sizeof(L"hello") == 6 * sizeof(wchar_t));

static_assert(__is_same(decltype(u8"hello"), const char8_t (&)[6]));
static_assert(__is_same(decltype(u8""), const char8_t (&)[1]));
static_assert(__is_same(decltype(u8"a"), const char8_t (&)[2]));
static_assert(sizeof(u8"hello") == 6);

static_assert(__is_same(decltype(u"hello"), const char16_t (&)[6]));
static_assert(__is_same(decltype(u""), const char16_t (&)[1]));
static_assert(__is_same(decltype(u"a"), const char16_t (&)[2]));
static_assert(sizeof(u"hello") == 6 * sizeof(char16_t));

static_assert(__is_same(decltype(U"hello"), const char32_t (&)[6]));
static_assert(__is_same(decltype(U""), const char32_t (&)[1]));
static_assert(__is_same(decltype(U"a"), const char32_t (&)[2]));
static_assert(sizeof(U"hello") == 6 * sizeof(char32_t));
