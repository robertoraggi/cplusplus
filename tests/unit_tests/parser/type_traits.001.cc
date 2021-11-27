// RUN: %cxx -verify -fsyntax-only %s -o -

static_assert(__is_void(int));  // expected-error{{static_assert failed}}

static_assert(__is_void(void));

static_assert(__is_void(void*));  // expected-error{{static_assert failed}}

static_assert(__is_same(decltype(__is_void(void)), bool));

static_assert(__is_same(decltype(__is_void(void)),
                        int));  // expected-error@-1{{static_assert failed}}
