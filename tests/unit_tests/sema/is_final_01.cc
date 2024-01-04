// RUN: %cxx -verify -fcheck %s

struct C {};

struct F final {};

static_assert(__is_final(F));
static_assert(__is_final(const F));

static_assert(__is_final(C) == false);