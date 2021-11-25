// RUN: %cxx -verify -fsyntax-only %s -o -

static_assert(__is_same_as(decltype(0 && 0), bool));
static_assert(__is_same_as(decltype(0 || 0), bool));
