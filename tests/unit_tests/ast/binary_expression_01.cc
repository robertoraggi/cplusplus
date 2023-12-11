// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

int a = 10;
int b = 20;
int c = 30;
int c = a + b * c;

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: a
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: int-literal-expression
// CHECK-NEXT:              literal: 10
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: b
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: int-literal-expression
// CHECK-NEXT:              literal: 20
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: c
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: int-literal-expression
// CHECK-NEXT:              literal: 30
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        integral-type-specifier
// CHECK-NEXT:          specifier: int
// CHECK-NEXT:      init-declarator-list
// CHECK-NEXT:        init-declarator
// CHECK-NEXT:          declarator: declarator
// CHECK-NEXT:            core-declarator: id-declarator
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: c
// CHECK-NEXT:          initializer: equal-initializer
// CHECK-NEXT:            expression: binary-expression
// CHECK-NEXT:              op: +
// CHECK-NEXT:              left-expression: implicit-cast-expression
// CHECK-NEXT:                cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                expression: id-expression
// CHECK-NEXT:                  unqualified-id: name-id
// CHECK-NEXT:                    identifier: a
// CHECK-NEXT:              right-expression: binary-expression
// CHECK-NEXT:                op: *
// CHECK-NEXT:                left-expression: implicit-cast-expression
// CHECK-NEXT:                  cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                  expression: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: b
// CHECK-NEXT:                right-expression: implicit-cast-expression
// CHECK-NEXT:                  cast-kind: lvalue-to-rvalue-conversion
// CHECK-NEXT:                  expression: id-expression
// CHECK-NEXT:                    unqualified-id: name-id
// CHECK-NEXT:                      identifier: c
