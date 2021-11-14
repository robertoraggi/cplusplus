// RUN: %cxx -verify -dump-tokens %s -o - | %filecheck %s

// clang-format off

"string1"
// CHECK: STRING_LITERAL '"string1"' [start-of-line]

L"string2"
// CHECK: WIDE_STRING_LITERAL 'L"string2"' [start-of-line]

u8"string3"
// CHECK: UTF8_STRING_LITERAL 'u8"string3"' [start-of-line]

u"string4"
// CHECK: UTF16_STRING_LITERAL 'u"string4"' [start-of-line]

U"string15"
// CHECK: UTF32_STRING_LITERAL 'U"string15"' [start-of-line]
