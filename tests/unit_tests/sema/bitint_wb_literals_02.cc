// RUN: %cxx -toolchain macos -verify -fcheck %s
// expected-no-diagnostics

static_assert(sizeof(3wb)    == 1);
static_assert(sizeof(3uwb)   == 1);
static_assert(sizeof(127wb)  == 1);
static_assert(sizeof(255uwb) == 1);
static_assert(sizeof(0x1fwb) == 1);
