// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

struct Bits {
  int value : 32;
};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: struct
// CHECK-NEXT:          name: simple-name
// CHECK-NEXT:            identifier: Bits
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            simple-declaration
// CHECK-NEXT:              decl-specifier-list
// CHECK-NEXT:                integral-type-specifier
// CHECK-NEXT:                  specifier: int
// CHECK-NEXT:              init-declarator-list
// CHECK-NEXT:                init-declarator
// CHECK-NEXT:                  declarator: declarator
// CHECK-NEXT:                    core-declarator: bitfield-declarator
// CHECK-NEXT:                      identifier: value
// CHECK-NEXT:                      size-expression: int-literal-expression
// CHECK-NEXT:                        literal: 32
