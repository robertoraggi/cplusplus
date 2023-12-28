// RUN: %cxx -verify -E %s -o - | %filecheck %s

// clang-format off
aa
#if 0
unreachable
#endif
/**/bb
cc

// CHECK: {{^}}aa{{$}}
// CHECK: {{^}}bb{{$}}
// CHECK: {{^}}cc{{$}}
