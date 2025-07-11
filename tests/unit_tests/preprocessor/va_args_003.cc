// RUN: %cxx -E -P %s -o - | %filecheck %s

#define X(a, ...) static_assert(a, ## __VA_ARGS__)

static_assert(true);
static_assert(true, "it must be true");

//      CHECK: static_assert(true);
// CHECK-NEXT: static_assert(true, "it must be true");

