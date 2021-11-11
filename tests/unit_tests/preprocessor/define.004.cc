// RUN: %cxx -verify -E %s -o - | %filecheck %s

#define WANTS_CC 1

#define WANTS_CC 1

static_assert(WANTS_CC == 1);

// CHECK: {{^}}static_assert(1 == 1);{{$}}

#define WANTS_CC 2  // expected-warning {{'WANTS_CC' macro redefined}}

static_assert(WANTS_CC == 2);

// CHECK: {{^}}static_assert(2 == 2);{{$}}
