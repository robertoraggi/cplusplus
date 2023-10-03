// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

using Pair = struct {
  int a, b;
};
// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    alias-declaration
// CHECK-NEXT:      identifier: Pair
// CHECK-NEXT:      type-id: type-id
// CHECK-NEXT:        type-specifier-list
// CHECK-NEXT:          class-specifier
// CHECK-NEXT:            class-key: struct
// CHECK-NEXT:            declaration-list
// CHECK-NEXT:              simple-declaration
// CHECK-NEXT:                decl-specifier-list
// CHECK-NEXT:                  integral-type-specifier
// CHECK-NEXT:                    specifier: int
// CHECK-NEXT:                init-declarator-list
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        declarator-id: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: a
// CHECK-NEXT:                  init-declarator
// CHECK-NEXT:                    declarator: declarator
// CHECK-NEXT:                      core-declarator: id-declarator
// CHECK-NEXT:                        declarator-id: id-expression
// CHECK-NEXT:                          unqualified-id: name-id
// CHECK-NEXT:                            identifier: b
// CHECK-NEXT:        declarator: declarator
