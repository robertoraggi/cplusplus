// clang-format off
// RUN: %cxx -E -P -nostdinc -isystem %S/include_next_first -isystem %S/include_next_second %s -o - | %filecheck %s

#if __has_include_next(<foo.h>)
int has_include_next_true;
#else
int has_include_next_false;
#endif

// CHECK: int has_include_next_true;
