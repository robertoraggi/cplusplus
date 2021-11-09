// RUN: %cxx -verify -E %s -o - | filecheck %s

#define CONCAT_HELPER(a, b) a ## b

#define CONCAT(a, b) CONCAT_HELPER(a, b)

#define id(x) CONCAT(x, CONCAT(_, __COUNTER__))

const int id(a) = 0;
// CHECK: {{^}}const int a_0 = 0;{{$}}

const int id(b) = 1;
// CHECK: {{^}}const int b_1 = 1;{{$}}

const int id(c) = 2;
// CHECK: {{^}}const int c_2 = 2;{{$}}
