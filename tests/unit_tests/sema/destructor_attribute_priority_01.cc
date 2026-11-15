// RUN: %cxx -verify -fsyntax-only %s

constexpr int priority = 101;
__attribute__((destructor(priority))) void accepted() {}

// expected-error@+1 {{'destructor' priority must be an integer constant from 0 to 65535}}
__attribute__((destructor(65536))) void outOfRange() {}

// expected-error@+1 {{'destructor' attribute requires at most one argument}}
__attribute__((destructor(101, 102))) void twoArguments() {}

int runtime;
// expected-error@+1 {{'destructor' priority must be an integer constant from 0 to 65535}}
__attribute__((destructor(runtime))) void notConstant() {}
