// RUN: %cxx -verify -fcheck %s

static_assert(sizeof("\n") == 2);
static_assert(sizeof("\t") == 2);
static_assert(sizeof("\\") == 2);
static_assert(sizeof("\"") == 2);
static_assert(sizeof("\0") == 2);
static_assert(sizeof("\x41") == 2);
static_assert(sizeof("\101") == 2);

static_assert(sizeof("\n\t\r") == 4);

static_assert(sizeof(L"\n") == 2 * sizeof(wchar_t));
static_assert(sizeof(L"\x41") == 2 * sizeof(wchar_t));

static_assert(sizeof(u"\n") == 2 * sizeof(char16_t));
static_assert(sizeof(u"\x41") == 2 * sizeof(char16_t));

static_assert(sizeof(U"\n") == 2 * sizeof(char32_t));
static_assert(sizeof(U"\x41") == 2 * sizeof(char32_t));

static_assert(sizeof(u"\u03B1") == 2 * sizeof(char16_t));
static_assert(sizeof(U"\u03B1") == 2 * sizeof(char32_t));
static_assert(sizeof(L"\u03B1") == 2 * sizeof(wchar_t));

static_assert(sizeof(U"\U0001F600") == 2 * sizeof(char32_t));
static_assert(sizeof(u"\U0001F600") == 3 * sizeof(char16_t));

static_assert(sizeof(R"(hello)") == 6);
static_assert(sizeof(R"(hello world)") == 12);
static_assert(sizeof(R"(line1\nline2)") == 13);

static_assert(sizeof(R"(\n)") == 3);
static_assert(sizeof(R"xyz(hello)xyz") == 6);

static_assert(sizeof(LR"(hello)") == 6 * sizeof(wchar_t));
static_assert(sizeof(u8R"(hello)") == 6);
static_assert(sizeof(uR"(hello)") == 6 * sizeof(char16_t));
static_assert(sizeof(UR"(hello)") == 6 * sizeof(char32_t));
