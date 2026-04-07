// RUN: %cxx -verify %s

static_assert(__builtin_LINE() > 0);

constexpr int get_line() { return __builtin_LINE(); }
static_assert(get_line() > 0);
