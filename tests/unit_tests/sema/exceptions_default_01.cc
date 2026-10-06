// RUN: %cxx -fsyntax-only %s
// RUN: %cxx -fexceptions -fno-exceptions -fsyntax-only %s
// RUN: %cxx -fexceptions -fsyntax-only -DEXCEPTIONS %s
// RUN: %cxx -fno-exceptions -fexceptions -fsyntax-only -DEXCEPTIONS %s

#ifdef EXCEPTIONS
#ifndef __cpp_exceptions
#error exceptions must be enabled
#endif
#else
#ifdef __cpp_exceptions
#error exceptions must be disabled
#endif
#endif
