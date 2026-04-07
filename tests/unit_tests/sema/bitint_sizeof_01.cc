// RUN: %cxx -toolchain macos -verify %s
// expected-no-diagnostics

static_assert(sizeof(_BitInt(8)) == 1);
static_assert(sizeof(_BitInt(16)) == 2);
static_assert(sizeof(_BitInt(32)) == 4);
static_assert(sizeof(_BitInt(64)) == 8);
static_assert(sizeof(_BitInt(128)) == 16);

static_assert(sizeof(unsigned _BitInt(8)) == 1);
static_assert(sizeof(unsigned _BitInt(16)) == 2);
static_assert(sizeof(unsigned _BitInt(32)) == 4);
static_assert(sizeof(unsigned _BitInt(64)) == 8);
static_assert(sizeof(unsigned _BitInt(128)) == 16);
