// RUN: %cxx -verify -fcheck %s

using info = __builtin_meta_info;

info a = typeid(int);

int value = 42;
info b = typeid(value);
