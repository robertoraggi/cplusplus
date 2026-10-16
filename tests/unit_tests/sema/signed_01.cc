// RUN: %cxx -verify -fsyntax-only -dump-symbols %s | %filecheck %s

__signed char a = -1;
__signed__ char b = -1;

// CHECK:variable signed char a
// CHECK:variable signed char b