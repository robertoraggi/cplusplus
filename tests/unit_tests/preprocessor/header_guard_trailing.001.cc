// RUN: %cxx -verify -E -P -I %S %s -o - | %filecheck %s

#include "redef_header.h"
int a = REDEFINE_ME(10);

#include "redef_undef.h"

// REDEFINE_ME is now undefined

#include "redef_header.h"
int b = REDEFINE_ME(20);

// CHECK: int a = 10 + 1 ;
// CHECK: int b = 20 + 1 ;
