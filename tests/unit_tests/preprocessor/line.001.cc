// RUN: %cxx -verify -E %s -o - | filecheck %s

const bool line_3 = __LINE__ == 3;

// CHECK: {{^}}const bool line_3 = 3 == 3;{{$}}

const bool line_7 = __LINE__ == 7;

// CHECK: {{^}}const bool line_7 = 7 == 7;{{$}}
