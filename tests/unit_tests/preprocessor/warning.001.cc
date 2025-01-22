// RUN: %cxx -verify -E %s -o - | %filecheck %s

#ifndef PLATFORM
#warning "undefined platform" // expected-warning {{#warning "undefined platform"}}
#endif

#define PLATFORM "linux"

#ifdef PLATFORM
const char* platform = PLATFORM;
#endif

// CHECK: const char* platform = "linux" ;
