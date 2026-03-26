// RUN: %cxx -toolchain macos -verify -fcheck %s
// expected-no-diagnostics

static_assert(__is_same(decltype(3wb),    _BitInt(3)));
static_assert(__is_same(decltype(3uwb),   unsigned _BitInt(2)));
static_assert(__is_same(decltype(0wb),    _BitInt(2)));
static_assert(__is_same(decltype(0uwb),   unsigned _BitInt(1)));
static_assert(__is_same(decltype(127wb),  _BitInt(8)));
static_assert(__is_same(decltype(255uwb), unsigned _BitInt(8)));
static_assert(__is_same(decltype(3wbu),   unsigned _BitInt(2)));
static_assert(__is_same(decltype(3WBU),   unsigned _BitInt(2)));
static_assert(__is_same(decltype(0x1fwb), _BitInt(6)));
