// RUN: %cxx -verify -E %s -o - | filecheck %s

const char* file = __FILE__;

// CHECK: {{^}}const char* file = "{{.*}}/file.001.cc";{{$}}

#include "file.001.h"

// CHECK: {{^}}const char* file_h = "{{.*}}/file.001.h";{{$}}

const char* file_cc = __FILE__;

// CHECK: {{^}}const char* file_cc = "{{.*}}/file.001.cc";{{$}}
