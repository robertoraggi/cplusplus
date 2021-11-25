// RUN: %cxx -verify -fsyntax-only %s -o -

// clang-format off

static_assert(__is_same_as(decltype(0 && 0), bool));
static_assert(__is_same_as(decltype(0 || 0), bool));

static_assert(__is_same_as(decltype(true ? true : false), bool));

static_assert(__is_same_as(decltype(true ? true : false), const bool)); // expected-error{{static_assert failed}}

static_assert(__is_same_as(decltype(true ? true : false), int)); // expected-error{{static_assert failed}}
