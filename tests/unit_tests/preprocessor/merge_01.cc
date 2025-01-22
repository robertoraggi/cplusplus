// RUN: %cxx -verify -E -P %s -o - | %filecheck %s

#define STR(prefix) prefix##"/*%empty*/"

auto cstr = STR();
auto wstr = STR(L);
auto ustr = STR(U);

// CHECK: auto cstr = "/*%empty*/";
// CHECK: auto wstr = L"/*%empty*/";
// CHECK: auto ustr = U"/*%empty*/";